from softgym.envs.pass_water import PassWater1DEnv
from softgym.multitask_envs_arxived.pass_water_multitask import PassWater1DGoalConditionedEnv
import numpy as np
from softgym.registered_env import  env_arg_dict
import argparse, sys


def get_particle_max_y():
    import pyflex
    pos = pyflex.get_positions().reshape((-1, 4))
    return np.max(pos[:, 1])

def run_heuristic(args):
    mode = args.mode
    if mode == 'visual' or mode == 'animation':
        env_name = "PassWaterGoal"
    else:
        env_name = "PassWater"
    dic = env_arg_dict[env_name]
    dic['headless'] = args.headless
    dic['observation_mode'] = args.obs_mode
    action_repeat = dic.get('action_repeat', 8)
    horizon = dic['horizon']
    print("env name {} action repeat {} horizon {}".format(env_name, action_repeat, horizon))
    if mode == 'visual' or mode == 'animation':
        env = PassWater1DGoalConditionedEnv(**dic)
    else:
        env = PassWater1DEnv(**dic)


    returns = []
    final_performances = []
    imgs = []
    
    if mode == 'visual':
        N = 1
    elif mode == 'test':
        N = 100
    elif mode == 'animation':
        N = 4

    for _ in range(N):
        if mode == 'test':
            env.eval_flag = True
            env.reset()
        else:
            if mode == 'visual':
                config_id = 4
                env.reset(config_id=config_id)
            else:
                env.reset()
            state = env.get_state()
            env.set_to_goal(env.get_goal())
            goal_img = env.render('rgb_array')
            # env.reset(config_id=config_id)
            env.set_state(state)

        total_reward = 0
        particle_y = get_particle_max_y()

        if np.abs(env.height - particle_y) > 0.2: # small water
            print("small")
        elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
            print("medium")
        else:
            print("large")

        horizon = env.horizon if mode != 'visual' else 35
        for i in range(horizon):
            if np.abs(env.height - particle_y) > 0.2: # small water
                action = np.array([0.13]) / action_repeat
            elif np.abs(env.height - particle_y) <= 0.2 and np.abs(env.height - particle_y) > 0.1: # medium water:
                if mode != 'visual':
                    action = np.array([0.08]) / action_repeat
                else:
                    action = np.array([0.16]) / action_repeat
            else:
                action = np.array([0.025]) / action_repeat

            if np.abs(env.glass_x - env.terminal_x) < 0.1:
                action = np.array([0]) 
        
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=256)
            total_reward += reward
            if mode == 'visual'  or mode == 'animation':
                imgs.append(env.render(mode='rgb_array'))

            if i == horizon - 1:
                final_performances.append(reward)

        print("episode total reward: ", total_reward)
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

# all_frames = imgs
# all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
# grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

# from os import path as osp
# save_name = 'pass_water_heuristic' + '.gif'
# save_numpy_as_gif(np.array(grid_imgs), osp.join('./data/video/env_demos', save_name))