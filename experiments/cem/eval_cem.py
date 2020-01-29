from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from envs.env import Env
from chester import logger
import torch
import os
import copy


def run_task(arg_vv, log_dir, exp_name):
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
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=vv['planning_horizon'], max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'])

    # Run policy
    for i in range(vv['test_episodes']):
        initial_states, action_trajs, configs = [], [], []

        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            action_traj.append(copy.copy(action))
            obs, reward, _, _ = env.step(action)
            print('reward:', reward)

    traj_dict = {
        'initial_state': initial_state,
        'action_traj': action_traj
    }

    with open(traj_path, 'wb') as f:
        pickle.dump(traj_dict, f)
