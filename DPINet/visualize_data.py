import time
import argparse
import os
import os.path as osp
from DPINet.graph import load_data
from DPINet.data import load_data, prepare_input, normalize, denormalize
import copy
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS
import numpy as np
import pyflex
import torch
import json
from torch.autograd import Variable
from DPINet.data import collate_fn
from DPINet.models import DPINet
from DPINet.graph import ClothDataset
from DPINet.graph_struct import convert_dpi_to_graph
from softgym.utils.visualization import save_numpy_as_gif

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default=None)
parser.add_argument('--data_folder', type=str, default='datasets/ClothFlatten/train')
parser.add_argument('--env_name', type=str, default='ClothFlatten')
parser.add_argument('--n_rollout', type=int, default=1)
parser.add_argument('--save_folder', type=str, default='./dpi_visualization')


def create_env(env_name):
    env_args = copy.deepcopy(env_arg_dicts[env_name])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['action_repeat'] = 2
    env_args['headless'] = False
    if env_name == 'ClothFlatten':
        env_args['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
    return SOFTGYM_ENVS[env_name](**env_args)


def parse_trajectory(traj_folder):
    steps = os.listdir(traj_folder)
    steps = sorted([int(step[:-3]) for step in steps])
    traj_pos = []
    traj_vel = []
    for t in steps:
        pos, vel, scene_params = load_data(['positions', 'velocities', 'scene_params'], osp.join(traj_folder, '{}.h5'.format(t)))
        _, _, _, config_id = scene_params
        traj_pos.append(pos)
        traj_vel.append(vel)
    return traj_pos, traj_vel, int(config_id)


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)
    return


def visualize(env, n_shape, traj_pos, config_id, sample_idx=None):
    env.reset(config_id=config_id)
    frames = []
    for i, pos in enumerate(traj_pos):
        particle_pos = pos[:-n_shape, :]
        shape_pos = pos[-n_shape:, :]
        p = pyflex.get_positions().reshape(-1, 4)
        if sample_idx is None:
            p[:, :3] = particle_pos
        else:
            p[:, :3] =0
            p[sample_idx[:-2], :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        frames.append(env.get_image(720, 720))
    return frames


def get_model_prediction(args, stat, traj_path, vels, dataset, model):
    # State index order: [particles, shape, roots]
    for i in range(args.time_step-1):
        if i == 0:
            attr, state, rels, n_particles, n_shapes, instance_idx, data, sample_idx = dataset.obtain_graph(osp.join(traj_path, str(i) + '.h5'))
            pos = np.vstack([state[:n_particles, :3], state[n_particles:n_particles + n_shapes, :3]])
            pos = denormalize([pos], [stat[0]])[0]  # Unnormalize
            predicted_pos_trajs = [pos.copy()]
            pick_points = dataset.obtain_pick(None, data=data)
        else:
            data[0] = state[:, :3]
            data[1] = state[:, 3:]
            pick_points = dataset.obtain_pick(osp.join(traj_path, str(i) + '.h5'))
            data[4] = pick_points
            attr, state, rels, n_particles, n_shapes, instance_idx, _ = dataset.construct_graph(data, downsample=False)
        graph = convert_dpi_to_graph(attr, state, rels, n_particles, n_shapes, instance_idx)
        st_time = time.time()
        predicted_vel = model(graph, args.phases_dict, args.verbose_model)
        # print('Time forward', time.time() - st_time)
        predicted_vel = denormalize([predicted_vel.data.cpu().numpy()], [stat[1]])[0]
        predicted_vel = np.concatenate([predicted_vel, vels[i+1][-n_shapes:]], 0)  ### Model only outputs predicted particle velocity,

        # predicted_vel[:,:] = vels[i+1][sample_idx, :]
        # Manually set the velocities of the picked points
        for pp in pick_points:
            predicted_vel[pp, :] =  vels[i+1][sample_idx, :][pp, :]

        ### so here we use the ground truth shape velocity. Why doesn't the model also predict the shape velocity?
        ### maybe, the shape velocity is kind of like the control actions specified by the users
        pos = copy.copy(predicted_pos_trajs[-1])
        pos += predicted_vel * 1 / 60.
        predicted_pos_trajs.append(pos)

        # Modify data for next step rollout (state includes positions and velocities)
        state = np.vstack([state[:n_particles, :], state[-n_shapes:, :]])
        pos = denormalize([state[:, :3]], [stat[0]])[0]  # Unnormalize
        pos += predicted_vel * 1 / 60.
        state[:, :3] = pos
        state[:, 3:] = predicted_vel
    return predicted_pos_trajs, sample_idx


class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)


def vv_to_args(vv):
    args = VArgs(vv)
    return args


def prepare_model(model_path):
    variant_path = osp.join(osp.dirname(model_path), 'variant.json')
    with open(variant_path, 'r') as f:
        vv = json.load(f)

    args = vv_to_args(vv)

    if args.env_name == 'ClothFlatten':
        datasets = {phase: ClothDataset(args, phase, args.phases_dict) for phase in ['train', 'valid']}
    else:
        raise NotImplementedError

    for phase in ['train', 'valid']:
        datasets[phase].load_data(args.env_name)

    print("Dataset loaded from", args.dataf)
    use_gpu = torch.cuda.is_available()

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

    # define propagation network
    model = DPINet(args, datasets['train'].stat, args.phases_dict, residual=True, use_gpu=use_gpu)

    model.load_state_dict(torch.load(model_path))

    if use_gpu:
        model = model.cuda()

    stat_path = os.path.join(args.dataf, 'stat.h5')
    stat = load_data(['positions', 'velocities'], stat_path)
    for i in range(len(stat)):
        stat[i] = stat[i][-args.position_dim:, :]

    return args, datasets, model, stat


def main(data_folder, n_rollout, save_folder=None, model_file=None):
    env_name = 'ClothFlatten'
    n_shape = 2
    env = create_env(env_name)
    if model_file is not None:
        args, datasets, model, stats = prepare_model(model_file)
    for idx, traj_id in enumerate(os.listdir(data_folder)):
        if idx > n_rollout:
            break
        traj_folder = osp.join(data_folder, str(traj_id))
        traj_pos, traj_vel, config_id = parse_trajectory(traj_folder)
        frames_gt = visualize(env, n_shape, traj_pos, config_id)
        if model_file is not None:
            with torch.no_grad():
                predicted_traj_pos, sample_idx = get_model_prediction(args, stats, traj_folder, traj_vel, datasets['valid'], model)
            frames_model = visualize(env, n_shape, predicted_traj_pos, config_id, sample_idx)
            combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
            save_numpy_as_gif(np.array(combined_frames), osp.join(save_folder, str(idx) + '.gif'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_folder, args.n_rollout, args.save_folder, args.model_file)
