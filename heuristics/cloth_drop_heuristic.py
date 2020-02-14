import gym
import numpy as np
import pyflex
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.envs.cloth_drop_multitask import ClothDropGoalConditionedEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
import torch, torchvision, cv2
from softgym.registered_env import  env_arg_dict
import argparse




def run_heuristic(args):
    num_picker = 2
    mode = args.mode
    env_name = "ClothDropGoal" if mode == 'visual' or mode == 'animation' else 'ClothDrop' 
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = ClothDropEnv(**dic) if mode == 'test' else ClothDropGoalConditionedEnv(**dic)

    imgs = []
    returns = []
    final_performances = []
    if mode == 'test':
        N = 100
    elif mode == 'visual':
        N = 1
    elif mode == 'animation':
        N = 4

    for _ in range(N):
        total_reward = 0
        if mode == 'test':
            env.eval_flag = True
            env.reset()
        else:
            if mode == 'visual':
                idx = 15
                env.reset(config_id=idx)
            else:
                env.reset()
            state = env.get_state()
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            # env.reset(config_id=idx)
            env.set_state(state)

        fast_move_steps = 5
        for i in range(fast_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = 0.5 / env.action_repeat 
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual'or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        slow_move_steps = 3
        for i in range(slow_move_steps):
            action = np.zeros((num_picker, 4))
            action[:, -1] = 1
            action[:, 0] = -0.6 / env.action_repeat
            action[:, 1] = -0.12 / env.action_repeat
            _, reward, _, info =env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual'or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        let_go_steps = 6
        for i in range(let_go_steps):
            action = np.zeros((num_picker, 4))
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])
            if i == let_go_steps - 1:
                final_performances.append(reward)

        print("episode total reward: ", total_reward)
        returns.append(total_reward)

    # env.close()
    return returns, final_performances, imgs, goal_img

if __name__ == '__main__':
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument("--mode", type=str, default='heuristic', help='visual: generate env images; otherwise, \
            run heuristic policy and evaluate its performance')
    args.add_argument("--headless", type=int, default=0)
    args.add_argument("--obs_mode", type=str, default='cam_rgb')
    run_heuristic(args)
