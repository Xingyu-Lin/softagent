from curl.train import run_task
from softgym.registered_env import env_arg_dict
reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
    'ClothFold': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_lr_decay(env_name, obs_mode):
    if env_name == 'RopeFlatten' or (env_name == 'ClothFlatten' and obs_mode == 'cam_rgb'):
        return 0.01
    elif obs_mode == 'point_cloud':
        return 0.01
    else:
        return None


def get_actor_critic_lr(env_name, obs_mode):
    if env_name == 'ClothFold' or (env_name == 'RopeFlatten' and obs_mode == 'point_cloud'):
        if obs_mode == 'cam_rgb':
            return 1e-4
        else:
            return 5e-4
    if obs_mode == 'cam_rgb':
        return 3e-4
    else:
        return 1e-3


def get_alpha_lr(env_name, obs_mode):
    if env_name == 'ClothFold':
        return 2e-5
    else:
        return 1e-3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='CURL_SAC', type=str)
    parser.add_argument('--env_name', default='ClothFlatten', type=str)
    parser.add_argument('--log_dir', default='./data/curl/', type=str)
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--save_tb', default=False)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=True)
    parser.add_argument('--save_model', default=True)  # Save trained models

    # CURL
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']

    args = parser.parse_args()

    args.algorithm = 'CURL'

    # Set env_specific parameters

    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_actor_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]

    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
