import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.visualization import save_numpy_to_gif_matplotlib
import time


def test_picker(num_picker=3, save_dir='./videos', script='manual'):
    env = ClothFlattenEnv(
        observation_mode='key_point',
        action_mode='pickerpickplace',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=1,
        render_mode='cloth')

    imgs = []
    for _ in range(1):
        env.reset()
        for i in range(20):
            print('step: ', i)
            action = np.zeros((num_picker, 6))
          
            first_particle_pos = pyflex.get_positions()[:3]
            last_particle_pos = pyflex.get_positions()[-4:-1]

            action[0, :3] = first_particle_pos
            action[1, :3] = last_particle_pos
            action[0, 3:] = np.array([-2, 0.05, -2])
            action[1, 3:] = np.array([2, 0.05, 1])

            img = env.render(mode='rgb_array')
            env.step(action)
            imgs.append(img)

            # time.sleep(5)

    fp_out = './videos/flatten_pickerpickandplace_manual_{}'.format(num_picker)
    # save_numpy_as_gif(np.array(imgs), fp_out, fps=5)
    save_numpy_to_gif_matplotlib(np.array(imgs), fp_out, interval=200)


def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    test_picker(num_picker=2, script='manual')
    # test_picker(num_picker=2, script='random')

    # env = ClothFlattenPointControlEnv(
    #     observation_mode='key_point',
    #     action_mode='picker',
    #     num_picker=num_picker,
    #     render=True,
    #     headless=False,
    #     horizon=75,
    #     action_repeat=8,
    #     render_mode='cloth')
    #
    # test_random(env)
