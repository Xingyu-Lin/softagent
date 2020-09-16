import argparse
import numpy as np
import torchvision
import torch
import os.path as osp
import pickle
import json
import os
from envs.env import Env
from softgym.utils.visualization import save_numpy_as_gif
import matplotlib.pyplot as plt

def cem_make_gif(env, initial_states, action_trajs, configs, save_dir, save_name, img_size=128):
    all_traj = []
    for i in range(len(action_trajs)):
        trajs = []
        env.reset(config=configs[i], initial_state=initial_states[i])
        for action in action_trajs[i]:
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            trajs.append(info['normalized_performance'])
            print(info['normalized_performance'])
        all_traj.append(trajs)

    traj = np.mean(np.array(all_traj), axis=0)
    return traj


def get_env(variant_path):
    with open(variant_path, 'r') as f:
        vv = json.load(f)
    if vv['env_name'] !='ClothFlatten':
        return None, vv['env_name']
    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'
    vv['env_kwargs']['render'] = True
    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': vv['max_episode_length'],
                  'action_repeat': 1,
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    env = env_class(**env_kwargs)
    return env, vv['env_name']


def get_variant_file(traj_path):
    traj_dir = osp.dirname(traj_path)
    return osp.join(traj_dir, 'variant.json')


def generate_video(file_paths):
    envs, env_names, paths = [], [], []
    for file_path in file_paths:
        variant_path = get_variant_file(file_path)
        env, env_name = get_env(variant_path)
        if env_name !='ClothFlatten':
            continue
        envs.append(env)
        env_names.append(env_name)
        paths.append(file_path)
    trajs =[]
    for env, env_name, file_path in zip(envs, env_names, paths):
        with open(file_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']
        traj = cem_make_gif(env, initial_states, action_trajs, configs, save_dir='data/videos/cem', save_name=env_name + '.gif')
        trajs.append(traj)
    all_traj = np.vstack(trajs)
    np.save('./data/cem_trajs/cloth_flatten.npy', all_traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()
    generate_video([osp.join(args.exp_dir, subdir, 'cem_traj.pkl') for subdir in os.listdir(args.exp_dir)])
    # file_path = osp.join(args.exp_dir, 'cem_traj.pkl')
    # with open(file_path, 'rb') as f:
    #     traj_dict = pickle.load(f)
    # initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']
    # for action in action_trajs[0]:
    #     print(action)
