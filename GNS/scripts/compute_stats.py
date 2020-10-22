import numpy as np
import torch
import h5py
import os

def _load_data_file(data_names, path):

    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()

    return data

def combine_stat(stat_0, stat_1):
    '''
    state_i: mean, std, count
    '''
    mean_0, std_0, n_0 = stat_0[0], stat_0[1], stat_0[2]
    mean_1, std_1, n_1 = stat_1[0], stat_1[1], stat_1[2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + \
                   (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return [mean, std, n]

data_dir = 'datasets/ClothFlatten_new'

stats_vel = [np.zeros(3), np.zeros(3), 0]
stats_acc = [np.zeros(3), np.zeros(3), 0]

train_num = 450
valid_num = 50
horizon = 100
dt = 0.01

load_names = ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'shape_positions']

for idx_rollout in range(train_num + valid_num):
    print("rollout: ", idx_rollout)
    vel_stats = [None, None, horizon - 1]
    acc_stats = [None, None, horizon - 1]
    vels = []
    accs = []
    if idx_rollout < train_num:
        folder = 'train'
    else:
        folder = 'valid'
        idx_rollout -= train_num

    for idx_timestep in range(horizon - 1):
        data_path = os.path.join(data_dir, folder, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(data_dir, folder, str(idx_rollout), str(idx_timestep + 1) + '.h5')
        data = _load_data_file(load_names, data_path)
        nxt_data = _load_data_file(load_names, data_nxt_path)

        vels.append(data[1])
        acc = (nxt_data[1] - data[1]) / dt
        accs.append(acc)

    vels = np.concatenate(vels, axis=0)
    accs = np.concatenate(accs, axis=0)

    vel_stats[0] = np.mean(vels, axis=0)
    vel_stats[1] = np.std(vels, axis=0)
    acc_stats[0] = np.mean(accs, axis=0)
    acc_stats[1] = np.std(accs, axis=0)

    stats_vel = combine_stat(stats_vel, vel_stats)
    stats_acc = combine_stat(stats_acc, acc_stats)

torch.save((stats_vel, stats_acc), os.path.join(data_dir, 'stats.pkl'))
