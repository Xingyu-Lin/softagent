from chester import logger
import torch
import os
import os.path as osp
import json
import numpy as np
from ResRL import utils
from ResRL.td3 import TD3
from envs.env import Env

default_vv = {
    "start_timesteps": 1e2,  # Time steps initial random policy is used
    "eval_freq": 200,  # How often (time steps) we evaluate
    "max_timesteps": 1e6,  # Max time steps to run environment
    "expl_noise": 0.1,  # Std of Gaussian exploration noise
    "batch_size": 256,  # Batch size for both actor and critic
    "discount": 0.99,  # Discount factor
    "tau": 0.005,  # Target network update rate
    "policy_noise": 0.2,  # Noise added to target policy during critic update
    "noise_clip": 0.5,  # Range to clip target policy noise
    "policy_freq": 2,  # Frequency of delayed policy updates
    "action_embed_dim": 64,
    "obs_embed_dim": 256,
    "save_model": False,  # Save model and optimizer parameters
    "load_model": ""  # Model load file name, "" doesn't load, "default" uses file_name
}


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward[0]

    avg_reward /= eval_episodes
    return avg_reward


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def run_task(arg_vv, log_dir, exp_name):
    default_vv.update(**arg_vv)
    vv = update_env_kwargs(default_vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    device = torch.device('cpu')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])
    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    image_dim = vv['env_kwargs']['image_dim']
    symbolic = not vv['env_kwargs']['image_observation']
    env = Env(vv['env_name'], symbolic, vv['seed'], vv['max_episode_length'], 1, 8, image_dim, env_kwargs=vv['env_kwargs'])
    eval_env = Env(vv['env_name'], symbolic, vv['seed'], vv['max_episode_length'], 1, 8, image_dim, env_kwargs=vv['env_kwargs'])

    # env = Box1d(**vv['env_kwargs'])

    # Set seeds
    env.seed(vv['seed'])
    torch.manual_seed(vv['seed'])
    np.random.seed(vv['seed'])

    obs_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = dict(image_observation=vv['env_kwargs']['image_observation'],
                  image_dim=env.image_dim,
                  image_c=obs_dim // (env.image_dim * env.image_dim),
                  state_dim=obs_dim,
                  action_dim=action_dim,
                  obs_embed_dim=vv['obs_embed_dim'],
                  action_embed_dim=vv['action_embed_dim'],
                  visual_encoder_name=vv['visual_encoder_name'],
                  max_action=max_action,
                  discount=vv['discount'],
                  tau=vv['tau'],
                  device=device)

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = vv['policy_noise'] * max_action
    kwargs["noise_clip"] = vv['noise_clip'] * max_action
    kwargs["policy_freq"] = vv['policy_freq']
    policy = TD3(**kwargs)

    if vv['load_model'] != "":
        policy_file = vv['load_model']
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(obs_dim, action_dim, device=device)

    # Evaluate untrained policy
    logger.record_tabular('eval_return', eval_policy(policy, eval_env, vv['seed']))

    obs, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(vv['max_timesteps'])):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < vv['start_timesteps']:
            action = env.action_space.sample()
        else:
            action = (
              policy.select_action(np.array(obs))
              + np.random.normal(0, max_action * vv['expl_noise'], size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        print(action)
        env.render()
        # done_bool = float(done) if episode_timesteps < env.horizon else 0

        # Store data in replay buffer.
        replay_buffer.add(obs, action, next_obs, reward, done)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= vv['start_timesteps']:
            policy.train(replay_buffer, vv['batch_size'])

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        # Evaluate episode
        if (t + 1) % vv['eval_freq'] == 0:
            # if vv['save_model']: policy.save(f"./models/{file_name}")
            logger.record_tabular('timesteps', t + 1)
            logger.record_tabular('episode_num', episode_num)
            logger.record_tabular('eval_return', eval_policy(policy, eval_env, vv['seed']))
            logger.dump_tabular()

    # # Run policy
    # initial_states, action_trajs, configs, all_infos = [], [], [], []
    # for i in range(vv['test_episodes']):
    #     logger.log('episode ' + str(i))
    #     obs = env.reset()
    #     initial_state = env.get_state()
    #     action_traj = []
    #     infos = []
    #     for _ in range(env.horizon):
    #         action = policy.get_action(obs)
    #         action_traj.append(copy.copy(action))
    #         obs, reward, _, info = env.step(action)
    #         infos.append(info)
    #     all_infos.append(infos)
    #     initial_states.append(initial_state.copy())
    #     action_trajs.append(action_traj.copy())
    #     configs.append(env.get_current_config().copy())
    #
    #     # Log for each episode
    #     transformed_info = transform_info([infos])
    #     for info_name in transformed_info:
    #         logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
    #         logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
    #         logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
    #     logger.dump_tabular()
