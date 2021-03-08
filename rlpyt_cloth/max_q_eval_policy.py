from os.path import join
import importlib
import argparse
import json

import torch
import numpy as np
import os.path as osp
from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.samplers.serial.sampler import SerialSampler
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict, ClothFlattenEnv
from softgym.envs.mvp_wrapper import MVPWrapper
from softgym.utils.visualization import save_numpy_as_gif, make_grid
import cv2 as cv
import copy
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_dir', type=str)
    parser.add_argument('max_q_eval_mode', type=str)
    parser.add_argument('--vis', type=bool, default=0)
    parser.add_argument('--save_folder', type=str, default='./data/qpg_visualization')
    args = parser.parse_args()

    snapshot_file = join(args.snapshot_dir, 'params.pkl')
    config_file = join(args.snapshot_dir, 'params.json')

    params = torch.load(snapshot_file, map_location='cpu')
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['sampler']['batch_B'] = 1
    config['sampler']['eval_n_envs'] = 1
    config['sampler']['eval_max_trajectories'] = 10 if not args.vis else 1
    config['env_kwargs']['maxq'] = True

    itr, cum_steps = params['itr'], params['cum_steps']
    print(f'Loading experiment at itr {itr}, cum_steps {cum_steps}')

    agent_state_dict = params['agent_state_dict']

    config['env_kwargs']['headless'] = True
    config['env_kwargs']['horizon'] = 20

    sac_agent_module = 'rlpyt.agents.qpg.{}'.format(config['sac_agent_module'])
    sac_agent_module = importlib.import_module(sac_agent_module)
    SacAgent = sac_agent_module.SacAgent

    agent = SacAgent(max_q_eval_mode=args.max_q_eval_mode, **config["agent"])
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
        all_traj_infos = []
        all_video_frames = []
        for i in range(4):
            sampler.envs[0].start_record()
            traj_infos = sampler.evaluate_agent(0, include_observations=True)
            all_traj_infos.extend(traj_infos)
            raw_video_frames = sampler.envs[0].video_frames
            video_frames = []
            for j in range(0, len(raw_video_frames), 2):
                video_frames.append(np.array(cv.resize(raw_video_frames[j].astype('float32'), (256, 256)))) # Down sample and resize to save memory
            all_video_frames.append(copy.copy(video_frames))
            sampler.envs[0].end_record()
        max_length = max(len(video_frames) for video_frames in all_video_frames)
        for i in range(len(all_video_frames)):
            pad_length = max_length - len(all_video_frames[i])
            all_video_frames[i] = np.vstack([all_video_frames[i], np.tile(all_video_frames[i][-1][None], [pad_length, 1,1, 1])])
        all_video_frames = np.array(all_video_frames).swapaxes(0, 1)
        grid_image = np.array([make_grid(frame, 1, 4) for frame in all_video_frames])
        save_numpy_as_gif(grid_image, osp.join(args.save_folder, 'vis_{}.gif'.format(config['env_name'])))
        for i in range(6):
            traj_infos = sampler.evaluate_agent(0, include_observations=True)
            all_traj_infos.extend(traj_infos)
        traj_infos = all_traj_infos
    else:
        traj_infos = sampler.evaluate_agent(0, include_observations=True)
    returns = [traj_info.Return for traj_info in traj_infos]
    lengths = [traj_info.Length for traj_info in traj_infos]
    performance = [traj_info.env_infos[-1].normalized_performance for traj_info in traj_infos]
    print('Performance: {}, Average performance: {}'.format(performance, np.mean(np.array(performance))))
    print('Returns', returns)
    print(f'Average Return {np.mean(returns)}, Average Length {np.mean(lengths)}')

    all_performance = np.array([[info.normalized_performance for info in traj_info.env_infos] for traj_info in traj_infos])
    all_steps = np.array([[info.total_steps for info in traj_info.env_infos] for traj_info in traj_infos])
    with open(osp.join(args.save_folder, 'qpg_traj_{}.npy'.format(config['env_name'])), 'wb') as f:
        np.save(f, all_performance)
        np.save(f, all_steps)

if __name__ == '__main__':
    main()
