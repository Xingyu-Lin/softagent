import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from planet.train import run_task
from softgym.registered_env import env_arg_dict


def main():
    import argparse
    parser = argparse.ArgumentParser()


    # Experiment
    parser.add_argument('--exp_name', default='PlaNet', type=str)
    parser.add_argument('--env_name', default='ClothFlatten')
    parser.add_argument('--log_dir', default='./data/planet/')
    parser.add_argument('--seed', default=100, type=int)

    # PlaNet
    parser.add_argument('--collect_interval', default=100)
    parser.add_argument('--test_interval', default=10)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--train_epoch', default=1200)
    parser.add_argument('--planning_horizon', default=24)
    parser.add_argument('--use_value_function', default=False)
    parser.add_argument('--im_size', default=128)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_deterministic', default=False, type=bool)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']

    args = parser.parse_args()

    args.algorithm = 'planet'

    # Set env_specific parameters
    env_name = args.env_name
    args.env_kwargs = env_arg_dict[env_name]
    args.test_episodes = 900 // env_arg_dict[env_name]['horizon']
    args.episodes_per_loop = 900 // env_arg_dict[env_name]['horizon']
    run_task(args.__dict__, args.log_dir, args.exp_name)

if __name__ == '__main__':
    main()
