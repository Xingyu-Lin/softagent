import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from drq.train import run_task

reward_scales = {
    'PourWater': 20.0,
    'PassWater': 20.0,
    'ClothFold': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': None,  # (-3, 3),
    'ClothFlatten': None,  # (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_critic_lr(env_name, obs_mode):
    return 1e-3


def get_alpha_lr(env_name, obs_mode):
    return 1e-3


def get_lr_decay(env_name, obs_mode):
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--exp_name', default='Drq_SAC', type=str)
    parser.add_argument('--env_name', default='ClothFlatten', type=str)
    parser.add_argument('--log_dir', default='./data/drq/', type=str)
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--log_save_tb', default=False)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=True)
    parser.add_argument('--save_model', default=True)  # Save trained models
    parser.add_argument('--log_interval', default=10000, type=int)  # Save trained models

    # Drq
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--im_size', default=128, type=int)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='cam_rgb', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_deterministic', default=False, type=bool)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_num_variations', default=1, type=str)

    args = parser.parse_args()

    args.algorithm = 'Drq'

    # Set env_specific parameters
    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]

    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
