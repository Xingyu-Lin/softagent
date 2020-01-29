from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json

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

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=env.horizon, max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'])

    # Run policy
    initial_states, action_trajs, configs = [], [], []
    for i in range(vv['test_episodes']):
        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            action_traj.append(copy.copy(action))
            obs, reward, _, _ = env.step(action)
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }

    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)
