import argparse
import os
import os.path as osp
from graph import load_data
import copy
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS
import numpy as np
import pyflex

parser = argparse.ArgumentParser()
parser.add_argument('data_folder', type=str, default='data/temp_ClothFlatten/train/')
parser.add_argument('--n_rollout', type=int, default=5)


def create_env(env_name):
    env_args = copy.deepcopy(env_arg_dicts[env_name])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['action_repeat'] = 2
    env_args['headless'] = False
    if env_name == 'ClothFlatten':
        env_args['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
    return SOFTGYM_ENVS[env_name](**env_args)


def parse_trajectory(traj_folder):
    steps = os.listdir(traj_folder)
    steps = sorted([int(step[:-3]) for step in steps])
    traj_pos = []
    for t in steps:
        pos, scene_params = load_data(['positions', 'scene_params'], osp.join(traj_folder, '{}.h5'.format(t)))
        _, cloth_xdim, cloth_ydim, config_id = scene_params
        traj_pos.append(pos)
    return traj_pos, int(config_id)


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)
    return


def visualize(env, n_shape, traj_pos, config_id):
    env.reset(config_id=config_id)
    for pos in traj_pos:
        particle_pos = pos[:-n_shape, :]
        shape_pos = pos[-n_shape:, :]
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        pyflex.step(render=True)


def main(data_folder, n_rollout):
    env_name = 'ClothFlatten'
    n_shape = 2
    env = create_env(env_name)
    for idx, traj_id in enumerate(os.listdir(data_folder)):
        if idx > n_rollout:
            break
        traj_folder = osp.join(data_folder, str(traj_id))
        traj_pos, config_id = parse_trajectory(traj_folder)
        visualize(env, n_shape, traj_pos, config_id)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_folder, args.n_rollout)
