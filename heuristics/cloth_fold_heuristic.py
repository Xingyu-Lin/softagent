import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_fold_multitask import ClothFoldGoalConditionedEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize
import torch, torchvision, cv2, time
from softgym.registered_env import env_arg_dict
import argparse,sys



def run_heuristic(args):
    mode = args.mode
    num_picker = 2
    env_name = 'ClothFold' if mode == 'test' else 'ClothFoldGoal'
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic['action_repeat']
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    env = ClothFoldEnv(**dic) if mode == 'test' else ClothFoldGoalConditionedEnv(**dic)


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
        minx = np.min(pos[:, 0])
        maxx = np.max(pos[:, 0])
        minz = np.min(pos[:, 2])
        maxz = np.max(pos[:, 2])

        corner1 = np.array([minx, 0.05, minz])
        corner2 = np.array([minx, 0.05, maxz])

        picker_pos, _ = env.action_tool._get_pos()

        differ1 = corner1 - picker_pos[0]
        differ2 = corner2 - picker_pos[1]

        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[0, :3] = differ1 / steps / action_repeat
            action[1, :3] = differ2 / steps / action_repeat

            obs, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        picker_pos, _ = env.action_tool._get_pos()
        
        steps = 15 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[:,-1] = 1
            action[:, 1] = 0.002
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        pos = pyflex.get_positions().reshape((-1, 4))
        minx = np.min(pos[:, 0])
        maxx = np.max(pos[:, 0])
        minz = np.min(pos[:, 2])
        maxz = np.max(pos[:, 2])
        target_corner_1 = np.array([maxx, 0.10, minz])
        target_corner_2 = np.array([maxx, 0.10, maxz])
        picker_pos, _ = env.action_tool._get_pos()

        differ1 = target_corner_1 - picker_pos[0]
        differ2 = target_corner_2 - picker_pos[1]
        
        steps = 40 if mode != 'visual' else 30
        for i in range(steps):
            action = np.ones((num_picker, 4))
            action[0, :3] = differ1 / steps / action_repeat
            action[1, :3] = differ2 / steps / action_repeat
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])

        steps = 30 if mode != 'visual' else 10
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual' or mode == 'animation':
                imgs.extend(info['flex_env_recorded_frames'])
            if i == steps - 1:
                final_performances.append(reward)

        print("episode {} total rewards {}".format(idx, total_reward))
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