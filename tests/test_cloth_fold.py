import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize


def test_picker(num_picker=3, save_dir='./videos'):
    env = ClothFoldEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='particle',
        use_cached_states=False,
        save_cache_states=False,
        deterministic=True,
        cached_init_state_path=None)

    obs = env.reset()
    key_pos1 = obs[:3]
    key_pos2 = obs[3:6]
    picker_pos = obs[6:].reshape((-1, 3))
    imgs = []
    for _ in range(1):
        env.reset()
        total_reward = 0
        for i in range(60):
            print('step: ', i)
            action = np.zeros((num_picker, 4))
            if i < 12:
                action[0, :3] = (key_pos1 - picker_pos[0, :]) * 0.01
                action[1, :3] = (key_pos2 - picker_pos[1, :]) * 0.01
                action[:, 3] = 0
            elif i < 42:
                action[:, 1] = 0.005
                action[:, 0] = 0.01
                action[:, 3] = 1
            _, reward, _, _ = env.step(action)
            total_reward += reward
            print('total reward"', total_reward)
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


if __name__ == '__main__':
    # test_picker(num_picker=2)
    env = ClothFoldEnv(
        observation_mode='cam_rgb',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth',
        use_cached_states=False,
        deterministic=True,
        save_cache_states=False)
    env = normalize(env)
    env.start_record()
    env.reset()
    for i in range(100000):
        action = env.action_space.sample()
        print(i, action)
        env.step(action)
    env.end_record(video_path='./test.gif')
    # env.reset()
    # for _ in range(500):
    #     pyflex.step()

    # test_random(env)
