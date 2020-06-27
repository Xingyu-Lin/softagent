import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif, make_grid
from multiprocessing import Process
import numpy as np
import os.path as osp
import torchvision
import torch
from envs.env import Env

np.set_printoptions(precision=3, suppress=True)


def test_picker(env, num_picker=2, save_dir='./videos'):
    obs = env.reset()
    key_pos1 = obs[:3]
    key_pos2 = obs[3:6]
    picker_pos = obs[-6:].reshape((-1, 3))
    imgs = []
    for _ in range(1):
        env.reset()
        total_reward = 0
        for i in range(60):
            print('step: ', i)
            action = np.zeros((num_picker, 4))
            if i < 12:
                action[0, :3] = (key_pos1 - picker_pos[0, :]) * 2
                action[1, :3] = (key_pos2 - picker_pos[1, :]) * 2
                action[:, 3] = 0
                # print('action:', action[0], action[1])
            elif i < 16:
                action[:, 1] = 0.3
                action[:, 0] = 0.01
                action[:, 3] = 1
            else:
                action[:, 1] = 0.
                action[:, 0] = 2.0
                action[:, 3] = 1
            obs, reward, _, _ = env.step(action.flatten())
            picker_pos = obs[-6:].reshape((-1, 3))
            # print('obs:', obs[-6:])
            # total_reward += reward
            # print('total reward"', total_reward)
            print(reward)
            img = env.render(mode='rgb_array')
            imgs.append(img)
    fp_out = './videos/fold_picker_random_{}.gif'.format(num_picker)
    save_numpy_as_gif(np.array(imgs), fp_out)


def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


def generate_env_state(env_name):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = False
    kwargs['render'] = True
    kwargs['use_cached_states'] = True
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = False
    kwargs['observation_mode'] = 'key_point'

    # Env wrappter
    env = SOFTGYM_ENVS[env_name](**kwargs)
    # env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
    return env


if __name__ == '__main__':
    env = generate_env_state('ClothFold')
    env = normalize(env)
    test_picker(env)
