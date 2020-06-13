from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif, make_grid
from multiprocessing import Process
import numpy as np
import os.path as osp
import torchvision
import torch

from envs.env import Env


def generate_env_state(env_name):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = False
    kwargs['use_cached_states'] = True
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = False

    # Env wrappter
    env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
    return env


if __name__ == '__main__':
    env_names = ['ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop']
    env_names = ['ClothFoldCrumpled']

    env = generate_env_state(env_names[0])
    for i in range(100):
        env.reset()
        for i in range(50):
            action = env.action_space.sample()
            # action =np.zeros_like(action)
            env.step(action)
