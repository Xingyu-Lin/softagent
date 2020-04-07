from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from experiments.cem.record_cem import cem_make_gif
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


def run_task(arg_vv, log_dir, exp_name):
    mp.set_start_method('spawn')
    vv = arg_vv
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
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': vv['max_episode_length'],
                  'action_repeat': 1,
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
            print('episode {}, step {}'.format(i, j))
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
