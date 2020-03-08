from PDDM.mppi import MPPI
from experiments.pddm.record_pddm import pddm_make_gif
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
import multiprocessing as mp
import time

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv

def run_task(arg_vv, log_dir, exp_name):
    mp.set_start_method('spawn')
    vv = arg_vv
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

    policy = MPPI(env, horizon=vv['plan_horizon'], N=vv['sample_size'], gamma=vv['gamma'], sigma=vv['sigma'], beta=vv['beta'], 
        action_correlation=vv['action_correlation'], env_class=Env, env_kwargs=env_kwargs)
    
    env_kwargs_render = copy.deepcopy(env_kwargs)
    env_kwargs_render['env_kwargs']['render'] = True
    env_render = env_class(**env_kwargs_render)

    # Run policy
    action_trajs, all_infos = [], []
    for i in range(vv['test_episodes']):
        logger.log('episode ' + str(i))
        obs = env.reset(config_id=i)
        initial_state = env.get_state()
        action_traj = []
        infos = []
        policy.reset()
        for _ in range(env.horizon):
            beg = time.time()
            # print("=" * 50, "step ", _, "="*50)
            action = policy.get_action(env_config_id=i)
            # print("=" * 50, "time cost {}".format(time.time() - beg))
            action_traj.append(copy.copy(action))
            obs, reward, _, info = env.step(action)
            infos.append(info)

        all_infos.append(infos)
        action_trajs.append(action_traj.copy())

        # Log for each episode
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

        pddm_make_gif(env_render, [action_traj], logger.get_dir(), vv['env_name'] + str(i) + '.gif', config_ids=[i])
        with open(osp.join(log_dir, 'pddm_traj_{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(action_traj, f)

        # print("episode {} done".format(i))
    
    # Dump trajectories
    with open(osp.join(log_dir, 'pddm_traj.pkl'), 'wb') as f:
        pickle.dump(action_trajs, f)

    # Dump video
    pddm_make_gif(env_render, action_trajs, logger.get_dir(), vv['env_name'] + '.gif', config_ids=[i for i in range(vv['test_episodes'])])
