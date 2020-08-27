import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlattenNew', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop']
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['headless'] = args.headless
    env_kwargs['action_repeat'] = 2
    env_kwargs['render_mode'] = 'particle'
    env_kwargs['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
    env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    env.reset()
    for i in range(20):
        action = (0, -0.005, 0, 1, 0, 0, 0, 0)
        env.step(action)
    for i in range(50):
        if i < 30:
            action = (0, 0.005, 0, 1, 0, 0, 0, 0)
        else:
            action = (0, 0, 0, 1, 0, 0, 0, 0)
        env.step(action)
    while (1):
        pyflex.step(render=True)
if __name__ == '__main__':
    main()
