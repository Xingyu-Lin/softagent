import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.utils.make_gif import save_numpy_as_gif


def test_picker(num_picker=3, save_dir='./videos', script='manual'):
    env = ClothFlattenEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth')

    imgs = []
    for _ in range(5):
        env.reset()
        for i in range(50):
            print('step: ', i)
            action = np.zeros((num_picker, 4))
            if i < 12:
                action[:, 1] = -0.01
                action[:, 3] = 0
            elif i < 30:
                action[:, 1] = 0.01
                action[:, 3] = 1
            elif i < 40:
                action[:, 3] = 0
            if script == 'random':
                action = env.action_space.sample()
            env.step(action)
            img = env.render(mode='rgb_array')
            imgs.append(img)
    fp_out = './videos/flatten_picker_random_{}.gif'.format(num_picker)
    save_numpy_as_gif(np.array(imgs), fp_out)


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
