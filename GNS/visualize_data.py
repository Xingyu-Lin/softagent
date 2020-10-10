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
        _, _, _, config_id = scene_params
        traj_pos.append(pos)
        traj_vel.append(vel)
    return traj_pos, traj_vel, int(config_id)


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)

def visualize(env, particle_positions, shape_positions, config_id, sample_idx=None):
    env.reset(config_id=config_id)
    frames = []
    for i in range(len(particle_positions)):
        particle_pos = particle_positions[i]
        shape_pos = shape_positions[i]
        p = pyflex.get_positions().reshape(-1, 4)
        if sample_idx is None:
            p[:, :3] = particle_pos
        else:
            p[:, :3] = 0
            p[sample_idx, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        frames.append(env.get_image(720, 720))
    return frames

def get_model_prediction_rollout(args, data_path, rollout_idx, init_timestep, model, dataset):
    encoder_model, processor_model, decoder_model = model
    encoder_model.eval()
    processor_model.eval()
    decoder_model.eval()

    assert init_timestep == args.n_his - 1
   
    data_dir = osp.join(data_path, str(rollout_idx), '{}.h5'.format(init_timestep))
    data_names = ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'shape_positions']

    data = dataset._load_data_file(data_names, data_dir)

    gt_positions = [data[0]]
    predicted_positions = [data[0]]
    shape_positions = [data[-1]]
    config_id = int(data[-2][-1])

    # print("picked points: ", data[4])

    # Get initial velocity history
    vel_his = []
    for i in range(1, args.n_his):
        path = os.path.join(data_path, str(rollout_idx), str(max(0, init_timestep - i)) + '.h5') # max just in case
        data_his = dataset._load_data_file(data_names, path)
        vel_his.append(data_his[1])
        gt_positions.append(data[0])
        predicted_positions.append(data[0])
        shape_positions.append(data[-1])

    data[1] = np.concatenate([data[1]] + vel_his, 1)

    pos_errors = []
    vel_errors = []
    down_sample_idx = None
    gt_positions = gt_positions[::-1]
    predicted_positions = predicted_positions[::-1]
    shape_positions = shape_positions[::-1]
    
    for t in range(args.n_his, args.time_step - 1):
        # print("visualzie step {}".format(t))
        if t == args.n_his:
            node_attr, neighbors, edge_attr, global_feat, sample_idx, picked_particles, down_cloth_x, down_cloth_y \
                 = dataset._prepare_input(data, test=True)
            down_sample_idx = sample_idx
            data[0] = data[0][down_sample_idx]
            data[1] = data[1][down_sample_idx]
            data[-2][1] = down_cloth_x # update cloth dim after downsample
            data[-2][2] = down_cloth_y
            for idx in range(len(gt_positions)):
                gt_positions[idx] = gt_positions[idx][down_sample_idx]
            for idx in range(len(predicted_positions)):
                predicted_positions[idx] = predicted_positions[idx][down_sample_idx]

        else:
            # after initial downsample, no need to do downsample again
            node_attr, neighbors, edge_attr, global_feat, _, picked_particles, _, _ \
                = dataset._prepare_input(data, test=True, downsample=False)
            # print("len of picked particles: ", len(picked_particles))
            # print('picked particles: ', picked_particles)

        
        node_attr = torch.squeeze(node_attr, dim=0).cuda()
        neighbors = torch.squeeze(neighbors, dim=0).cuda()
        edge_attr = torch.squeeze(edge_attr, dim=0).cuda()
        global_feat = torch.squeeze(global_feat, dim=0).cuda()

        # obtain model predictions
        with torch.no_grad():

            node_embedding, edge_embedding = encoder_model(node_attr, edge_attr)

            node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat,
                                                                                    batch=None)
            pred_accel = decoder_model(node_embedding_out)

        pred_vel = data[1][:, :3] + pred_accel.cpu().numpy() * args.dt
        pred_pos = data[0] + pred_vel * args.dt

        # compute prediction error
        data_dir = osp.join(data_path, str(rollout_idx), '{}.h5'.format(t))
        new_data = dataset._load_data_file(data_names, data_dir)
        gt_pos = new_data[0][down_sample_idx]
        gt_vel = new_data[1][down_sample_idx]
        gt_positions.append(gt_pos)
        predicted_positions.append(pred_pos)
        shape_positions.append(new_data[-1])
        pos_error = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=1))
        vel_error = np.mean(np.linalg.norm(pred_vel - gt_vel, axis=1))
        pos_errors.append(pos_error)
        vel_errors.append(vel_error)

        # udpate position and velocity from the model prediction
        old_vel = data[1][:, :-3]
        data[0] = pred_pos
        data[1][:, 3:] = old_vel
        data[1][:, :3] = pred_vel

        # update picker position and picker action, and the particles picked
        data[2] = new_data[2]
        data[3] = new_data[3]
        data[4] = picked_particles

    return pos_errors, vel_errors, gt_positions, predicted_positions, shape_positions, down_sample_idx, config_id

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

    datasets = {x: torch.utils.data.dataset(
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
