from os.path import join
import importlib
import argparse
import json

import torch
import numpy as np

from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.samplers.serial.sampler import SerialSampler
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict, ClothFlattenEnv
from softgym.envs.mvp_wrapper import MVPWrapper

def main():
    """ Using the pick location output by the actor"""
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_dir', type=str)
    parser.add_argument('--vis', type=bool, default=0)
    parser.add_argument('--dump_traj', type=bool, default=0)

    args = parser.parse_args()
    snapshot_file = join(args.snapshot_dir, 'params.pkl')
    config_file = join(args.snapshot_dir, 'params.json')

    params = torch.load(snapshot_file, map_location='cpu')
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['sampler']['batch_B'] = 1
    config['sampler']['eval_n_envs'] = 1
    config['sampler']['eval_max_trajectories'] = 10 if not args.vis else 1

    itr, cum_steps = params['itr'], params['cum_steps']
    print(f'Loading experiment at itr {itr}, cum_steps {cum_steps}')

    agent_state_dict = params['agent_state_dict']
    optimizer_state_dict = params['optimizer_state_dict']

    # Costum env for evaluation
    config['env_kwargs']['headless'] = True
    config['env_kwargs']['horizon'] = 20

    ### Test
    # env = ClothFlattenEnv(
    #     observation_mode='cam_rgb',
    #     action_mode='picker_qpg',
    #     num_picker=1,
    #     render=True,
    #     headless=True,
    #     horizon=20,
    #     action_repeat=1,
    #     render_mode='cloth',
    #     save_cached_states=False)
    # env = QpgWrapper(env)
    # env.start_record()
    # obs = env.reset()
    # for i in range(20):
    #     action = env.action_space.sample()
    #     location = env.sample_location(obs.pixels)
    #     obs, _, _, _ = env.step(action)
    # env.end_record('./random.gif', fps=40, scale=0.3)
    # exit()
    ### End Test
    sac_agent_module = 'rlpyt.agents.qpg.{}'.format(config['sac_agent_module'])
    sac_agent_module = importlib.import_module(sac_agent_module)
    SacAgent = sac_agent_module.SacAgent

    agent = SacAgent(**config["agent"])
    sampler = SerialSampler(
        EnvCls=SOFTGYM_ENVS[config['env_name']],
        env_kwargs=config["env_kwargs"],
        eval_env_kwargs=config["env_kwargs"],
        **config["sampler"]
    )
    sampler.initialize(agent)
    agent.load_state_dict(agent_state_dict)

    agent.to_device(cuda_idx=0)
    agent.eval_mode(0)

    if args.vis:
        sampler.envs[0].start_record()
    traj_infos = sampler.evaluate_agent(0, include_observations=True)
    if args.vis:
        sampler.envs[0].end_record(join(args.snapshot_dir, 'visualization.gif'), fps=40, scale=0.3)
    returns = [traj_info.Return for traj_info in traj_infos]
    lengths = [traj_info.Length for traj_info in traj_infos]
    performance = [traj_info.env_infos[-1].normalized_performance for traj_info in traj_infos]
    print('Performance: {}, Average performance: {}'.format(performance, np.mean(np.array(performance))))
    print('Returns', returns)
    print(f'Average Return {np.mean(returns)}, Average Length {np.mean(lengths)}')

    # for i, traj_info in enumerate(traj_infos):
    #     observations = np.stack(traj_info.Observations)
    #     video_filename = join(args.snapshot_dir, f'episode_{i}.mp4')
    #     save_video(observations, video_filename, fps=10)


def save_video(video_frames, filename, fps=10):
    assert int(fps) == fps, fps
    import skvideo.io
    skvideo.io.vwrite(filename, video_frames, inputdict={'-r': str(int(fps))})


if __name__ == '__main__':
    main()
