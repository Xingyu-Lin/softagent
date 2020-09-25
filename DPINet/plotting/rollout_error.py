import time
import argparse
import os
import os.path as osp
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

from softgym.utils.visualization import save_numpy_as_gif
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default=None)
parser.add_argument('--data_folder', type=str, default='datasets/ClothFlatten')
parser.add_argument('--env_name', type=str, default='ClothFlatten')
parser.add_argument('--n_rollout', type=int, default=10)
parser.add_argument('--save_folder', type=str, default='./dpi_visualization')

model_paths = [
    'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0001/',
    'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0002/',
    'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0003/',
    'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0004/',
    'data/autobot/0908_noise/0908_noise/0908_noise_2020_09_08_20_00_54_0005/'
]

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


def get_model_prediction(args, stage, stat, traj_path, initial_pos, vels, datasets, model):
    pos_trajs = [initial_pos]
    for i in range(args.time_step):
        step_time = time.time()
        if i == 0:
            attr, state, rels, n_particles, n_shapes, instance_idx, data = datasets[stage].obtain_graph(osp.join(traj_path, str(i) + '.h5'))
        else:
            attr, state, rels, n_particles, n_shapes, instance_idx = datasets[stage].construct_graph(data)

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
        print('step time:', time.time() - step_time)

    return pos_trajs[:-1]


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
    datasets, stats = prepare_data(model_paths[0])
    all_model_performance = {}
    for model_path in model_paths:
        model_losses = {}
        for model_name in model_names:
            model_file = osp.join(model_path, model_name)
            global args
            args, model = prepare_model(model_file, datasets, stage)
            losses = []
            for idx, traj_id in enumerate(os.listdir(data_folder)):
                if idx > n_rollout:
                    break
                traj_folder = osp.join(data_folder, str(traj_id))

                traj_pos, traj_vel, config_id = parse_trajectory(traj_folder)
                predicted_traj_pos = get_model_prediction(args, stage, stats, traj_folder, traj_pos[0], traj_vel, datasets, model)
                # predicted_traj_pos = np.random.random(np.array(traj_pos).shape)
                # frames_model = visualize(env, n_shape, predicted_traj_pos, config_id)
                # combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
                # save_numpy_as_gif(np.array(combined_frames), osp.join(save_folder, str(idx) + '.gif'))
                losses.append(np.mean((np.array(traj_pos) - np.array(predicted_traj_pos)) ** 2, axis=(1, 2)))
            model_losses[model_name[:12]] = np.mean(np.array(losses), axis=0)

        plt.figure()
        for k, v in model_losses.items():
            plt.plot(range(len(v)), v, label=k)
        plt.legend()
        noise_level = str(args.noise_level)
        save_name = 'rollout_error_{}_{}.png'.format(stage, noise_level)

        print('Saving to {}'.format(osp.join(save_folder, save_name)))
        plt.savefig(osp.join(save_folder, save_name))
        all_model_performance[noise_level] = model_losses
    import pickle
    with open(osp.join(save_folder, 'rollout_dump.pkl'), 'wb') as f:
        pickle.dump(all_model_performance, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_folder, args.n_rollout, args.save_folder, stage='valid')
