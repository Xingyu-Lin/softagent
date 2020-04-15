import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.robot_env import RobotEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize



if __name__ == '__main__':
    env = RobotEnv(
        observation_mode='cam_rgb',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth',
        cached_init_state_path=None,
        use_cached_states=False,
        save_cache_states=False,
        num_variations=1)
    env = normalize(env)
    env.start_record()
    env.reset()
    pyflex.loop()
    for i in range(1000):
        action = env.action_space.sample()
        print(i, action)
        env.step(action)
    env.end_record(video_path='./test.gif')
    # env.reset()
    # for _ in range(500):
    #     pyflex.step()

    # test_random(env)
