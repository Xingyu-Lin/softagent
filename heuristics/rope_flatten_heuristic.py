import numpy as np
import pyflex
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.multitask_envs_arxived.rope_manipulate import RopeManipulateEnv
import sys
from softgym.registered_env import  env_arg_dict
import argparse


def run_heuristic(args):
    mode = args.mode
    env_name = 'RopeFlatten' if mode == 'test' else "RopeManipulate"
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = RopeFlattenEnv(**dic) if mode == 'test' else RopeManipulateEnv(**dic)


    imgs = []
    returns = []
    final_performances = []
    if mode == 'visual':
        N = 1
    elif mode == 'test':
        N = 100
    elif mode == 'animation':
        N = 4

    for idx in range(N):
        if mode == 'test':
            env.eval_flag = True
            env.reset()
        else:
            if mode == 'visual':
                idx = 6
                env.reset(config_id=idx)
            else:
                env.reset()
            state = env.get_state()
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            # env.reset(config_id=idx)
            env.set_state(state)

        total_reward = 0
        
        pos = pyflex.get_positions().reshape((-1, 4))
        corner1 = pos[0][:3]
        corner2 = pos[-1][:3]

        steps = 5
        for i in range(steps):
            action = np.zeros((2, 4))
            action[:, 1] = 0.01
            obs, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = corner1 - picker_pos[0]
        diff2 = corner2 - picker_pos[1]

        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])


        picker_pos, _ = env.action_tool._get_pos()
        target_pos_1 = np.array([2.7, 0.05, 0])
        target_pos_2 = np.array([-2.7, 0.05, 0])

        picker_pos, _ = env.action_tool._get_pos()
        diff1 = target_pos_1 - picker_pos[0]
        diff2 = target_pos_2 - picker_pos[1]

        steps = 20
        for i in range(steps):
            action = np.ones((2, 4))
            action[0, :3] = diff1 / steps / env.action_repeat
            action[1, :3] = diff2 / steps / env.action_repeat
            _, reward, _ , info  = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        steps = 35 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((2, 4))
            _, reward, _ , info  = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])
            if i == steps - 1:
                final_performances.append(reward)

        print("episode {} total reward {}".format(idx, total_reward))
        returns.append(total_reward)

    # env.close()
    return returns, final_performances, imgs, goal_img

if __name__ == '__main__':
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument("--mode", type=str, default='test')
    args.add_argument("--headless", type=int, default=1)
    args.add_argument("--obs_mode", type=str, default='cam_rgb')
    args = args.parse_args()
    run_heuristic(args)