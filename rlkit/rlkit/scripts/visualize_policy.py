# Created by Xingyu Lin, 2019-09-19                                                                                  
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.util.video import dump_video
from rlkit.core import logger
import numpy as np
import os.path as osp
import copy
import json
import glob
import cv2
# from scripts.visualize_vae import batch_chw_to_hwc
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import os
filename = str(uuid.uuid4())
from softgym.registered_env import SOFTGYM_ENVS as SOFTGYM_CUSTOM_ENVS

def batch_chw_to_hwc(images):
    rets = []
    for i in range(len(images)):
        rets.append(copy.copy(np.transpose(images[i], [2, 1, 0])[::-1, :, ::-1]))
    return np.array(rets)

def get_image(img, imsize=48):
    img = np.reshape(img, [3, 48, 48])
    img = batch_chw_to_hwc(img[None])[0] * 256
    return img


def load_variants(exp_dir):
    with open(osp.join(exp_dir, 'variant.json'), 'r') as f:
        variants = json.load(f)
    return variants

def heuristic_pour_water(env):
    env.reset()
    
    v = 0.13
    total_rotate = 0.6 * np.pi

    timestep = env.horizon
    move_part = 15
    path = {'full_observations': []}
    for i in range(env.horizon):
        if i < move_part:
            action = np.array([0, 0.01 * 100, 0.])

        elif i > move_part and i < timestep - 20:
            theta = 1 / float(timestep - move_part) * total_rotate
            action = np.array([0, 0, theta * 100])

        else:
            action = np.array([0, 0, 0])

        obs, reward, done, _ = env.step(action)
        path['full_observations'].append(obs)

    return path

def plot_latent_dist(env, policy, rollout_function, horizon, N=10, save_name='./plot.png', heuristic_func=None):
    all_latent_achieved_goal = []
    all_lateng_goal = []
    for _ in range(N):
        if heuristic_func is None:
            path = rollout_function(
                env,
                policy,
                max_path_length=horizon,
                render=False,
            )
        else:
            path = heuristic_func(env)
        latent_achieved_goal = np.array([d['latent_achieved_goal'] for d in path['full_observations']])
        latent_goal = np.array([d['latent_desired_goal'] for d in path['full_observations']])
        all_latent_achieved_goal.append(latent_achieved_goal)
        all_lateng_goal.append(latent_goal)
    all_latent_achieved_goal = np.array(all_latent_achieved_goal)
    all_lateng_goal = np.array(all_lateng_goal)
    dist = np.linalg.norm(all_latent_achieved_goal - all_lateng_goal, axis=-1)
    mean_dist = np.mean(dist, axis=0)
    std_dist = np.std(dist, axis=0)

    print(len(dist))
    print(std_dist)
    markers, caps, bars = plt.errorbar(list(range(horizon)), mean_dist, std_dist) 
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    plt.xlabel('Episode time')
    plt.ylabel('VAE distance to goal')
    plt.savefig(save_name)
    plt.cla()
    plt.clf()



def simulate_policy(args, dir, heuristic):
    variants = load_variants(dir)
    seed = variants['seed']
    exp_name = variants['exp_name']
    variants = variants['skewfit_kwargs']
    max_path_length = variants['skewfit_variant']['max_path_length']
    env_name = variants['env_id']
    

    print(env_name)

    algo_name = "RIG"
    video_name = env_name + '_' + algo_name + '_' + str(seed) + '.gif'
    latent_plot_name = env_name + '_' + algo_name + '_' + str(seed) + '.png'

    data = torch.load(osp.join(dir, 'params.pkl'))
    policy = data['trainer/policy'] # stochastic policy
    policy_determinisitic = data['evaluation/policy'] # deterministic policy
    env = data['evaluation/env']
    env.wrapped_env.wrapped_env.eval_flag = True
    max_path_length = env.horizon
    if not args.no_gpu:
        set_gpu_mode(True, 0)
        policy_determinisitic.stochastic_policy.cuda()
        policy.cuda()

   
    imsize = env.imsize
    env.goal_sampling_mode = 'reset_of_env'
    env.decode_goals = False
    env.reset()
    print("Policy loaded")

    def rollout(*args, **kwargs):
        return multitask_rollout(*args, **kwargs,
                                 observation_key='latent_observation',
                                 desired_goal_key='latent_desired_goal', 
                                 )

    h = 'heuristic' if heuristic else ''
    if args.video:
        print('dump video')
        dump_video(env, policy, './videos/' + exp_name + h +  "_stochastic_" + video_name, 
                    rollout_function=rollout, imsize=imsize,
                   horizon=max_path_length, rows=1, columns=10 if not heuristic else 1,
                   heuristic_func=heuristic_pour_water if heuristic else None)
        dump_video(env, policy_determinisitic, './videos/' + h + exp_name + "_determinisitc_" +  video_name, 
                    rollout_function=rollout, imsize=imsize,
                   horizon=max_path_length, rows=1, columns=10 if not heuristic else 1,
                   heuristic_func=heuristic_pour_water if heuristic else None)

    if args.latent_distance:
        print("plot_latent_dist")
        plot_latent_dist(env, policy, rollout,
                         horizon=max_path_length,
                         save_name='./imgs/' + exp_name + h + "_stochastic_" + latent_plot_name, 
                         heuristic_func=heuristic_pour_water if heuristic else None)
        plot_latent_dist(env, policy_determinisitic, rollout,
                         horizon=max_path_length,
                         save_name='./imgs/' + exp_name + h + "_determinisitc_" + latent_plot_name, 
                         heuristic_func=heuristic_pour_water if heuristic else None)

def simulate_policy_recursive(args, dir, heuristic):
    dirs = os.walk(dir)
    subdirs = [x[0] for x in dirs]
    print(subdirs)

    # policy_files = glob.glob(dir + '/**/**/**/params.pkl', recursive=True)
    # print(policy_files)
    # exit()
    for policy_file in subdirs:
        if '--s' in policy_file:
            # print(osp.dirname(policy_file))
            print(policy_file)
            simulate_policy(args, policy_file + '/', heuristic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='directory to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--no_gpu', type=int, default=0)
    parser.add_argument('-non_r', '--non_recursive', type=int, default=0)
    parser.add_argument('-vid', '--video', type=int, default=10)
    parser.add_argument('-dist', '--latent_distance', type=int, default=10)
    parser.add_argument('--imsize', type=int, default=48)
    parser.add_argument('--heuristic', type=int, default=0)
    args = parser.parse_args()

    # print(args)
    # exit()
    import pyflex
    headless, render, camera_width, camera_height = True, True, 720, 720
    pyflex.init(headless, render, camera_width, camera_height)

    if not args.non_recursive:
        simulate_policy_recursive(args, args.dir, args.heuristic)
    else:
        simulate_policy(args, args.dir, args.heuristic)
