import numpy as np
import torch
import os
import time
import json
import copy

from pouring.GNN_sac import utils
from pouring.GNN_sac.logger import Logger

from pouring.GNN_sac.GNN_SAC import GNNSAC
from pouring.GNN_sac.GNN_default_config import GNN_default_config
from pouring.GNN_sac.curl_default_config import DEFAULT_CONFIG

from experiments.planet.train import update_env_kwargs

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args

def update_config(default_config, luanch_config):
    import copy
    for key in luanch_config:
        default_config_ = default_config
        now_key = copy.deepcopy(key)
        idx = now_key.find('.')
        while idx != -1:
            default_config_ = default_config_[now_key[:idx]]
            now_key = now_key[idx+1:]
            idx = now_key.find('.')

        default_config_[now_key] = luanch_config[key]

    return default_config


def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # update KPConv config
    config = copy.deepcopy(GNN_default_config)
    update_config(config, vv)
    
    # update curl config
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    main(vv_to_args(updated_vv), config)


def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict


def evaluate(env, agent, video_dir, num_episodes, L, step, args, config):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        infos = []
        all_frames = []
        plt.figure()
        for i in range(num_episodes):
            obs = env.reset()
            obs = utils.preprocess_single_obs(obs)
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            while not done:
                # center crop image
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                obs = utils.preprocess_single_obs(obs)
                episode_reward += reward
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(reward)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)
            infos.append(ep_info)

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        plt.savefig(os.path.join(video_dir, '%d.png' % step))
        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(video_dir, '%d.gif' % step))

        for key, val in get_info_stats(infos).items():
            L.log('eval/info_' + prefix + key, val, step)
        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(args, config, device):
    return GNNSAC(
        args=args,
        actor_kwargs=config['actor_kwargs'],
        critic_kwargs=config['critic_kwargs'],
        device=device,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        alpha_fixed=args.alpha_fixed,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        log_interval=args.log_interval,
    )


def main(args, config):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, 128, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.shape[0]
    config['actor_kwargs']['action_dim'] = action_dim
    config['critic_kwargs']['q_kwargs']['action_dim'] = action_dim


    replay_buffer = utils.GraphReplayBuffer(
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        action_dim=action_dim
    )

    agent = make_agent(
        args=args,
        config=config,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    episode, episode_reward, done, ep_info = 0, 0, True, []
    start_time = time.time()
    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0 and step > 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video_dir, args.num_eval_episodes, L, step, args, config)
            if args.save_model and  step % (args.eval_freq *5):
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    for key, val in get_info_stats([ep_info]).items():
                        L.log('train/info_' + key, val, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            obs = utils.preprocess_single_obs(obs)

            # obs_data = obs.to_data_list()[0]
            # print("in train, right after env reset, obs.data.x.device is: ", obs_data.x.device)
            # assert obs_data.x.device != torch.device("cuda:0")

            done = False
            ep_info = []
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
        next_obs, reward, done, info = env.step(action)
        next_obs = utils.preprocess_single_obs(next_obs)


        # allow infinit bootstrap
        ep_info.append(info)
        done_bool = 0 if episode_step + 1 == env.horizon else float(done)
        episode_reward += reward

        # obs_data = obs.to_data_list()[0]
        # next_obs_data = next_obs.to_data_list()[0]
        # print("in train, right before add, obs.data.x.device is: ", obs_data.x.device)
        # print("in train, right before add, next_obs.data.x.device is: ", next_obs_data.x.device)
        # assert obs_data.x.device != torch.device("cuda:0")
        # assert next_obs_data.x.device != torch.device("cuda:0")
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
