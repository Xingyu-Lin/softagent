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
import torch, torchvision
import matplotlib
from chester.plotting.cplot import *

font = {'size'   : 12}

matplotlib.rc('font', **font)

def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

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

def heuristic_pour_water(env, ret_reward=False):
    env.reset()
    print("config id is: ", env.current_config_id)
    
    path = {'full_observations': []}
    rewards = []

    move_part = 20
    target_y = env.poured_height + 0.2
    target_x = env.glass_distance - env.poured_glass_dis_x / 2 - env.height - 0.1
    action_repeat = env.action_repeat
    print(env.action_repeat)
    for i in range(move_part):
        action = np.array([target_x / action_repeat / move_part , target_y / action_repeat / move_part, 0.])
        action = action * 100
        obs, reward, done, info = env.step(action)
        rewards.append(info['performance'])
        path['full_observations'].append(obs)

    
    rotate_part = 20
    total_rotate = 0.55 * np.pi
    for i in range(rotate_part):
        action = np.array([0.0005, 0.003, total_rotate / rotate_part / action_repeat])
        action = action * 100
        obs, reward, done, info = env.step(action)
        rewards.append(info['performance'])
        path['full_observations'].append(obs)


    stay_part = 60
    for i in range(stay_part):
        action = np.zeros(3)
        obs, reward, done, info = env.step(action)
        rewards.append(info['performance'])
        path['full_observations'].append(obs)

    if not ret_reward:
        return path
    else:
        return path, np.asarray(rewards)

def heuristic_pass_water(env, ret_reward=False):
    env.reset()
    print("config id is: ", env.current_config_id)
    
    path = {'full_observations': []}
    rewards = []

    particle_y = get_particle_max_y()

    # if np.abs(env.height - particle_y) > 0.2: # small water
    #     print("small")
    # elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
    #     print("medium")
    # else:
    #     print("large")

    action_repeat = env.action_repeat
    horizon = env.horizon
    for i in range(horizon):
        if np.abs(env.height - particle_y) > 0.2: # small water
            action = np.array([0.13]) / action_repeat * 100
        elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
            action = np.array([0.08]) / action_repeat * 100
        else:
            action = np.array([0.025]) / action_repeat * 100

        if np.abs(env.glass_x - env.terminal_x) < 0.1:
            action = np.array([0]) 
    
        obs, reward, _, info = env.step(action)
        rewards.append(info['performance'])
        path['full_observations'].append(obs)

    if not ret_reward:
        return path
    else:
        return path, np.asarray(rewards)

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
            path, rewards = heuristic_func(env, ret_reward=True)

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
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1,1,1)
    if heuristic_func is None:
        markers, caps, bars = ax.errorbar(list(range(horizon)), mean_dist, std_dist, 
            color='r', label='latent distance to goal') 
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]

        ax.set_xlabel('Time step')
        ax.set_ylabel('latent distance to goal')
        ax.legend()

        print(save_name)
        # plt.tight_layout()
        plt.savefig(save_name)
        plt.cla()
        plt.clf()
    
    else:
        color = core.color_defaults[2]
        line1 = ax.plot(list(range(horizon)), -mean_dist, color=color, label='latent dist. \n to goal', linewidth=3) 

        twinax = ax.twinx()
        # print(rewards)
        color = core.color_defaults[1]
        line2 = twinax.plot(range(len(rewards)), rewards, color=color, label='GT reward', linewidth=3)

        max_distance_idx = np.argmax(mean_dist)

        indices = [5, 55, 80] # pour water
        indices = [5, 10, 70] # pass water
        twinax.vlines(x=indices[0], ymin=0, ymax=1, linestyle='dashed', linewidth=0.6)
        twinax.vlines(x=max_distance_idx, ymin=0, ymax=1, linestyle='dashed', linewidth=0.4)
        twinax.vlines(x=indices[1], ymin=0, ymax=1, linestyle='dashed', linewidth=0.6)
        twinax.vlines(x=indices[2], ymin=0, ymax=1, linestyle='dashed', linewidth=0.6)

        # twinax.annotate(s='f1', xy=(5, 1.2))
        # twinax.annotate(s='f2', xy=(max_distance_idx + 0.1, 0.95))
        # twinax.annotate(s='f3', xy=(45, 0.95))
        # twinax.annotate(s='f4', xy=(80, 0.9))
        ax3 = twinax.twiny()
        ax3.set_xlim(ax.get_xlim())
        move = 0

        ax3.set_xticks([indices[0] + move, max_distance_idx+ move, indices[1] + move, indices[2]+ move])
        ax3.set_xticklabels(['frame1', 'frame2', 'frame3', 'frame4'], rotation=15, ha='left')


        ax.set_xlabel('Time Step')
        ax.set_ylabel('Negative Latent Dist. to Goal')
        twinax.set_ylabel("GT Reward")
       
        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        # ax.legend(lns, labs, loc=(0.55, 0.5))
        ax.yaxis.label.set_color(line1[0].get_color())
        twinax.yaxis.label.set_color(line2[0].get_color())
        ax.tick_params(axis='y', colors=line1[0].get_color())
        twinax.tick_params(axis='y', colors=line2[0].get_color())

        print(save_name)
        plt.tight_layout() 
        # plt.show()
        plt.savefig(save_name)
        plt.cla()
        plt.clf()

        imgs = []
        img = cv2.imread('./imgs/rollouts/0/goal.png')
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        for _ in range(4):
            imgs.append(torch.from_numpy(img))

        for idx in [indices[0], max_distance_idx, indices[1], indices[2]]:
            img = cv2.imread('./imgs/rollouts/0/{}_obs.png'.format(idx))
            img = img.astype(np.float64)
            img = img.transpose(2, 0, 1)
            imgs.append(torch.from_numpy(img))
        
        for idx in [indices[0], max_distance_idx, indices[1], indices[2]]:
            img = cv2.imread('./imgs/rollouts/0/{}_recons.png'.format(idx))
            img = img.astype(np.float64)
            img = img.transpose(2, 0, 1)
            imgs.append(torch.from_numpy(img))

        grid_imgs = torchvision.utils.make_grid(imgs, padding=5, pad_value=120, nrow=4).data.cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite('./imgs/rollouts/combined.jpg', grid_imgs)




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
    print(type(env))
    print(type(env._wrapped_env))
    print(type(env._wrapped_env._wrapped_env))
    print(type(env._wrapped_env._wrapped_env._wrapped_env))
    env._wrapped_env._wrapped_env._wrapped_env.deterministic= True
    print(env._wrapped_env._wrapped_env._wrapped_env.deterministic)
    env._wrapped_env._wrapped_env._wrapped_env.eval_flag = True
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
        # dump_video(env, policy, './videos/' + exp_name + h +  "_stochastic_" + video_name, 
        #             rollout_function=rollout, imsize=imsize,
        #            horizon=max_path_length, rows=1, columns=2 if not heuristic else 1,
                #    heuristic_func=heuristic_pour_water if heuristic else None)
        dump_video(env, policy_determinisitic, './videos/' + h + exp_name + "_determinisitc_" +  video_name, 
                    rollout_function=rollout, imsize=imsize, dirname_to_save_images='./imgs',
                   horizon=max_path_length, rows=1, columns=1 if not heuristic else 1,
                   heuristic_func=heuristic_pass_water if heuristic else None)

    if args.latent_distance:
        print("plot_latent_dist")
        # plot_latent_dist(env, policy, rollout,
        #                  horizon=max_path_length,
        #                  save_name='./imgs/' + exp_name + h + "_stochastic_" + latent_plot_name, 
        #                  heuristic_func=heuristic_pour_water if heuristic else None)
        plot_latent_dist(env, policy_determinisitic, rollout,
                         horizon=max_path_length, N=1 if heuristic else 10,
                         save_name='./imgs/' + exp_name + h + "_determinisitc_" + latent_plot_name, 
                         heuristic_func=heuristic_pass_water if heuristic else None)

def simulate_policy_recursive(args, dir, heuristic):
    dirs = os.walk(dir)
    subdirs = [x[0] for x in dirs]
    print(subdirs)

    for policy_file in subdirs:
        if '--s' in policy_file:
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
