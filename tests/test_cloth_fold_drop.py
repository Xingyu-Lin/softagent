import numpy as np
from softgym.envs.cloth_fold_crumpled import ClothFoldCrumpledEnv
from softgym.envs.cloth_fold_drop import ClothFoldDropEnv
from softgym.utils.normalized_env import normalize


def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    env = ClothFoldDropEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='particle',
        use_cached_states=True,
        save_cache_states=False,
        num_variations=1000,
        deterministic=False)
    env = normalize(env)
    env.start_record()
    for _ in range(20):
        env.reset()
        for i in range(50):
            action = env.action_space.sample()
            # action = np.zeros_like(action)
            action[3] = action[7] = 1.
            env.step(action)
    env.end_record(video_path='./test.gif')
    # env.reset()
    # for _ in range(500):
    #     pyflex.step()

    # test_random(env)
