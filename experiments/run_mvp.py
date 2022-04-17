from softgym.registered_env import env_arg_dict
from rlpyt_cloth.rlpyt.experiments.scripts.dm_control.qpg.sac.train.softgym_sac import run_task


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--exp_name', default='qpg', type=str)
    parser.add_argument('--env_name', default='ClothFlatten', type=str)  # Only tested in ClothFlatten and ClothFold
    parser.add_argument('--log_dir', default='./data/qpg/', type=str)
    parser.add_argument('--seed', default=100, type=int)

    # QPG
    parser.add_argument('--config_key', default='sac_pixels_cloth_corner_softgym', type=str)  # Config file that contains all parameters for QPG
    parser.add_argument('--sac_module', default='sac_v2', type=str)
    parser.add_argument('--sac_agent_module', default='sac_agent_v2', type=str)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='cam_rgb', type=str)
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int)  # Number of pickers as the action space. Only tested under one picker
    parser.add_argument('--env_kwargs_horizon', default=20, type=int)  # Number of pick-and-places
    parser.add_argument('--env_kwargs_action_mode', default='picker_qpg', type=str)  # Number of pick-and-places

    args = parser.parse_args()

    args.algorithm = 'qpg'

    # Set env_specific parameters
    env_name = args.env_name
    args.env_kwargs = env_arg_dict[env_name]
    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
