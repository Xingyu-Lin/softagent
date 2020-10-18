import numpy as np
import torch
import os
import time
import json
import copy

import utils
from curl_sac import CurlSacAgent
from default_config import DEFAULT_CONFIG

import matplotlib.pyplot as plt


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args



def evaluate(env, agent, num_episodes, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        infos = []
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            ep_info = []
            rewards = []
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                ep_info.append(info)
                rewards.append(reward)

            all_ep_rewards.append(episode_reward)

        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)

    run_eval_loop(sample_stochastically=False)


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


def main(args):
    utils.set_seed_everywhere(args.seed)

    symbolic = True
    args.encoder_type = 'identity' if symbolic else 'pixel'

    # TODO: make the real-world env here
    env = None

    # from softgym.registered_env import env_arg_dict
    # from softgym.registered_env import SOFTGYM_ENVS
    # import copy
    # env_args = copy.deepcopy(env_arg_dict['PourWater'])
    # env_args['camera_name'] = 'default_camera'
    # env_args['observation_mode'] = 'key_point_3'
    # env_args['render'] = True
    # env = SOFTGYM_ENVS['PourWater'](**env_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
        pre_aug_obs_shape = (3, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load(args.model_path, args.model_step)
    print("agent sucessfully loaded!")

    evaluate(env, agent, args.num_eval_episodes, args)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv['model_path'] = '.'
    updated_vv['model_step'] = '950000'
    updated_vv['lr_decay'] = None
    main(vv_to_args(updated_vv))
