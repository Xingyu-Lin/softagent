import time
import argparse
import os
import os.path as osp
from DPINet.data import load_data, prepare_input, normalize, denormalize
import copy
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS
from DPINet.graph_struct import convert_dpi_to_graph
import numpy as np
import pyflex
import torch
import json
from torch.autograd import Variable
from DPINet.data import collate_fn
from DPINet.models import DPINet
from DPINet.graph import ClothDataset

from softgym.utils.visualization import save_numpy_as_gif
import matplotlib.pyplot as plt
from DPINet.visualize_data import get_model_prediction
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default=None)
parser.add_argument('--data_folder', type=str, default='datasets/ClothFlatten')
parser.add_argument('--env_name', type=str, default='ClothFlatten')
parser.add_argument('--n_rollout', type=int, default=10)
parser.add_argument('--save_folder', type=str, default='./dpi_visualization')

# model_paths = [
#     'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0001/',
#     'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0002/',
#     'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0003/',
#     'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0004/',
#     'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0005/'
# ]

model_dir = 'data/autobot/0928_downsample/0928_downsample/'
model_paths = [osp.join(model_dir, dir_name) for dir_name in os.listdir(model_dir)]
model_names = [
    'net_epoch_10_iter_10000.pth',
    'net_epoch_20_iter_10000.pth',
    'net_epoch_30_iter_10000.pth',
    'net_epoch_40_iter_10000.pth',
    'net_epoch_50_iter_10000.pth',
    'net_epoch_60_iter_10000.pth',
]


def create_env(env_name):
    env_args = copy.deepcopy(env_arg_dicts[env_name])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['action_repeat'] = 2
    env_args['headless'] = True
    env_args['render'] = False
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



def prepare_args(model_path):
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
    return args


def prepare_data(model_path):
    args = prepare_args(model_path)
    if args.env_name == 'ClothFlatten':
        datasets = {phase: ClothDataset(args, phase, args.phases_dict) for phase in ['train', 'valid']}
    else:
        raise NotImplementedError

    for phase in ['train', 'valid']:
        datasets[phase].load_data(args.env_name)

    print("Dataset loaded from", args.dataf)

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

    stat_path = os.path.join(args.dataf, 'stat.h5')
    stat = load_data(['positions', 'velocities'], stat_path)
    for i in range(len(stat)):
        stat[i] = stat[i][-args.position_dim:, :]
    return datasets, stat


def prepare_model(model_path, dataset, stage):
    args = prepare_args(model_path)
    use_gpu = torch.cuda.is_available()
    # define propagation network
    model = DPINet(args, dataset[stage].stat, args.phases_dict, residual=True, use_gpu=use_gpu)
    model.load_state_dict(torch.load(model_path))

    if use_gpu:
        model = model.cuda()
    return args, model


def main(data_folder, n_rollout, save_folder=None, stage='train'):
    data_folder = osp.join(data_folder, stage)
    env_name = 'ClothFlatten'
    n_shape = 2
    env = create_env(env_name)
    all_model_performance = {}
    for model_path in model_paths:
        model_losses = {}
        datasets, stats = prepare_data(osp.join(model_path, model_names[0])) # Re-load dataset for each model as the relation dim can vary
        for model_name in model_names:
            model_file = osp.join(model_path, model_name)
            if not osp.isfile(model_file):
                continue
            print('evaluating ' + model_file)
            # global args
            args, model = prepare_model(model_file, datasets, stage)
            losses = []
            for idx, traj_id in enumerate(os.listdir(data_folder)):
                if idx > n_rollout:
                    break
                traj_folder = osp.join(data_folder, str(traj_id))

                traj_pos, traj_vel, config_id = parse_trajectory(traj_folder)
                with torch.no_grad():
                    predicted_traj_pos, sample_idx = get_model_prediction(args, stats, traj_folder, traj_vel, datasets[stage], model)
                # predicted_traj_pos = np.random.random(np.array(traj_pos).shape)
                # frames_model = visualize(env, n_shape, predicted_traj_pos, config_id)
                # combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
                # save_numpy_as_gif(np.array(combined_frames), osp.join(save_folder, str(idx) + '.gif'))
                if sample_idx is not None:
                    losses.append(np.mean((np.array(traj_pos)[:, sample_idx, :] - np.array(predicted_traj_pos)) ** 2, axis=(1, 2)))
                else:
                    losses.append(np.mean((np.array(traj_pos) - np.array(predicted_traj_pos)) ** 2, axis=(1, 2)))
            model_losses[model_name[:12]] = np.mean(np.array(losses), axis=0)
        plt.figure()
        for k, v in model_losses.items():
            plt.plot(range(len(v)), v, label=k)
        plt.legend()
        label = 'Noise: ' + str(args.noise_level) + ' Downsample: ' + str(
            args.down_sample_scale) + ' EdgeType: ' + args.edge_type + ' NeighborRadius: ' + str(args.neighbor_radius)
        save_name = 'rollout_error_{}_{}.png'.format(stage, label)

        print('Saving to {}'.format(osp.join(save_folder, save_name)))
        plt.savefig(osp.join(save_folder, save_name))
        all_model_performance[label] = model_losses
    import pickle
    with open(osp.join(save_folder, 'rollout_dump.pkl'), 'wb') as f:
        pickle.dump(all_model_performance, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_folder, args.n_rollout, args.save_folder, stage='valid')
