import time
import argparse
import os
import os.path as osp
from graph import load_data
from data import load_data, prepare_input, normalize, denormalize
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

from softgym.utils.visualization import save_numpy_as_gif

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default=None)
parser.add_argument('--data_folder', type=str, default='datasets/ClothFlatten/train')
parser.add_argument('--env_name', type=str, default='ClothFlatten')
parser.add_argument('--n_rollout', type=int, default=5)
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
        _, cloth_xdim, cloth_ydim, config_id = scene_params
        traj_pos.append(pos)
        traj_vel.append(vel)
    return traj_pos, traj_vel, int(config_id)


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)
    return


def visualize(env, n_shape, traj_pos, config_id):
    env.reset(config_id=config_id)
    frames = [env.get_image(720, 720)]
    for i, pos in enumerate(traj_pos):
        particle_pos = pos[:-n_shape, :]
        shape_pos = pos[-n_shape:, :]
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        frames.append(env.get_image(720, 720))
    return frames


def get_model_prediction(args, stat, traj_path, initial_pos, vels, datasets, model):
    pos_trajs = [initial_pos]
    for i in range(args.time_step):
        if i == 0:
            attr, state, rels, n_particles, n_shapes, instance_idx, data = datasets['train'].obtain_graph(osp.join(traj_path, str(i) + '.h5'))
        else:
            attr, state, rels, n_particles, n_shapes, instance_idx = datasets['train'].construct_graph(data)

        Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

        Rr, Rs = [], []

        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]  # NOTE: values are all just 1
            Rr.append(torch.sparse.FloatTensor(Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
            Rs.append(torch.sparse.FloatTensor(Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))
        data_cpu = copy.copy(data)
        data = [attr, state, Rr, Rs, Ra]

        with torch.set_grad_enabled(False):
            for d in range(len(data)):
                if type(data[d]) == list:
                    for t in range(len(data[d])):
                        data[d][t] = Variable(data[d][t].cuda())
                else:
                    data[d] = Variable(data[d].cuda())

            attr, state, Rr, Rs, Ra = data

            st_time = time.time()
            predicted_vel = model(attr, state, Rr, Rs, Ra, n_particles, node_r_idx, node_s_idx, pstep, instance_idx, args.phases_dict,
                                  args.verbose_model)
            print('Time forward', time.time() - st_time)
        predicted_vel = denormalize([predicted_vel.data.cpu().numpy()], [stat[1]])[0]
        predicted_vel = np.concatenate([predicted_vel, vels[i][n_particles:]], 0)  ### Model only outputs predicted particle velocity,
        ### so here we use the ground truth shape velocity. Why doesn't the model also predict the shape velocity?
        ### maybe, the shape velocity is kind of like the control actions specified by the users
        pos = copy.copy(pos_trajs[-1])
        pos += predicted_vel * 1 / 60.
        pos_trajs.append(pos)

        # Modify data for next step rollout
        data_cpu[0] = data_cpu[0] + predicted_vel * 1 / 60.
        data_cpu[1][:, :3] = predicted_vel
        data = data_cpu

    return pos_trajs


def prepare_model(model_path):
    variant_path = osp.join(osp.dirname(model_path), 'variant.json')
    with open(variant_path, 'r') as f:
        vv = json.load(f)

    def vv_to_args(vv):
        class VArgs(object):
            def __init__(self, vv):
                for key, val in vv.items():
                    setattr(self, key, val)

        args = VArgs(vv)
        return args

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

    # if args.resume_epoch > 0 or args.resume_iter > 0:
    #     model_path = os.path.join(logdir, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    #     print("Loading saved ckp from %s" % model_path)
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
            predicted_traj_pos = get_model_prediction(args, stats, traj_folder, traj_pos[0], traj_vel, datasets, model)
            frames_model = visualize(env, n_shape, predicted_traj_pos, config_id)
            combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
            save_numpy_as_gif(np.array(combined_frames), osp.join(save_folder, str(idx) + '.gif'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_folder, args.n_rollout, args.save_folder, args.model_file)
