from softgym.registered_env import env_arg_dict
from envs.env import Env
import pyflex

env_name = 'ClothFold'
kwargs = env_arg_dict[env_name]
kwargs['headless'] = False
kwargs['use_cached_states'] = False
kwargs['num_variations'] = 10
kwargs['save_cached_states'] = False
kwargs['observation_mode'] = 'cam_rgb'
# Env wrappter
env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
env.reset()
import numpy as np

pos = pyflex.get_positions().reshape((119, 119, 4))

max_dist = np.linalg.norm(pos[0, 0, :3] - pos[-1, -1, :3])
print(max_dist)  # 1.043

