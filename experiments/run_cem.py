from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from cem.visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args


cem_plan_horizon = {
    'PassWater': 7,
    'PourWater': 40,
    'PourWaterAmount': 40,
    'ClothFold': 15,
    'ClothFoldCrumpled': 30,
    'ClothFoldDrop': 30,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15,
    'RopeConfiguration': 15,
}


def run_task(vv, log_dir, exp_name):
    mp.set_start_method('spawn')
    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv['plan_horizon'] = cem_plan_horizon[env_name]  # Planning horizon

    vv['population_size'] = vv['timestep_per_decision'] // vv['max_iters']
    if vv['use_mpc']:
        vv['population_size'] = vv['population_size'] // vv['plan_horizon']
    vv['num_elites'] = vv['population_size'] // 10
    vv = update_env_kwargs(vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(vv['seed'])

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': 200,
                  'action_repeat': 1,  # Action repeat for env wrapper is 1 as it is already inside the env
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    env = env_class(**env_kwargs)

    env_kwargs_render = copy.deepcopy(env_kwargs)
    env_kwargs_render['env_kwargs']['render'] = True
    env_render = env_class(**env_kwargs_render)

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=vv['plan_horizon'], max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'])
    # Run policy
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    for i in range(vv['test_episodes']):
        logger.log('episode ' + str(i))
        obs = env.reset()
        policy.reset()
        initial_state = env.get_state()
        action_traj = []
        infos = []
        for j in range(env.horizon):
            logger.log('episode {}, step {}'.format(i, j))
            action = policy.get_action(obs)
            action_traj.append(copy.copy(action))
            obs, reward, _, info = env.step(action)
            infos.append(info)
        all_infos.append(infos)
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

        # Log for each episode
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

    # Dump trajectories
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)

    # Dump video
    cem_make_gif(env_render, initial_states, action_trajs, configs, logger.get_dir(), vv['env_name'] + '.gif')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='cem', type=str)
    parser.add_argument('--env_name', default='ClothFlatten', type=str)
    parser.add_argument('--log_dir', default='./data/cem', type=str)
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)

    # CEM
    parser.add_argument('--max_iters', default=10, type=int)
    parser.add_argument('--timestep_per_decision', default=21000, type=int)
    parser.add_argument('--use_mpc', default=True, type=bool)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=False, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)

    args = parser.parse_args()
    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
