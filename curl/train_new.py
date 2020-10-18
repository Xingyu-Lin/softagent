import numpy as np
import torch
import os
import time
import json
import copy

from curl import utils
from curl.logger import Logger

from curl.curl_sac import CurlSacAgent
from curl.default_config import DEFAULT_CONFIG

from experiments.planet.train import update_env_kwargs

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

from pouring.KPConv_sac.config import Modelnet40Config, Modelnet40DeformConfig


def obs_process(obs):
    if isinstance(obs, tuple):
        return obs[0].astype(np.float32), obs[1].astype(np.float32)
    else:
        # obs = obs.astype(np.float32)
        # return torch.from_numpy(obs)
        return obs.astype(np.float32)

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

def update_config_from_vv(config, vv):
    for key, val in vv.items():
        if key.startswith('KPConv_config_'):
            # print(key)
            setattr(config, key[len('KPConv_config_'): ], val)


def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)

    # update KPConv config
    if vv['rim_to_state']:
        KPConv_config = Modelnet40Config if not vv['KPConv_deform'] else Modelnet40DeformConfig
        config = copy.deepcopy(KPConv_config)
        update_config_from_vv(config, vv)
    else:
        config=None

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


def evaluate(env, agent, video_dir, num_episodes, L, step, args, config=None, KPConv_model=None, device=None):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        infos = []
        all_frames = []
        plt.figure()
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)

                if args.rim_to_state:
                    obs = utils.preprocess_single_obs(KPConv_model, obs, config, device)

                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
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


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            alpha_fixed=args.alpha_fixed,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main(args, config=None):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    if args.rim_to_state:
        env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs, obs_process=obs_process)
        # load pretrained KPConv model
        from pouring.KPConv_sac.architectures import KPConvRegression, KPCNN_Encoder
        setattr(config, 'output_dim', args.output_dim)
        encoder = KPCNN_Encoder(config)
        KPConv_model = KPConvRegression(
            encoder=encoder,
            config=config,
        ).to(device)

        KPConv_model.load_state_dict(torch.load(args.KPConv_model_path))
        print("KPconv model loaded from: ", args.KPConv_model_path)
        KPConv_model.eval()

    else:
        env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))


    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
        pre_aug_obs_shape = (3, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        if args.rim_to_state:
            from gym.spaces import Box
            obs_shape = Box(low=np.array([-np.inf] * 9), high=np.array([np.inf] * 9), dtype=np.float32).shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    episode, episode_reward, done, ep_info = 0, 0, True, []
    start_time = time.time()
    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video_dir, args.num_eval_episodes, L, step, args, config=config, KPConv_model=KPConv_model, device=device)
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
            done = False
            ep_info = []
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        if args.rim_to_state:
            obs = utils.preprocess_single_obs(KPConv_model, obs, config, device)

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
        next_obs_ = utils.preprocess_single_obs(KPConv_model, next_obs, config, device)

        # allow infinit bootstrap
        ep_info.append(info)
        done_bool = 0 if episode_step + 1 == env.horizon else float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs_, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
