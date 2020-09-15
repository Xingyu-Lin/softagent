from os.path import join
import importlib
import argparse
import json

import torch
import numpy as np

from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.samplers.serial.sampler import SerialSampler
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict, ClothFlattenEnv
from softgym.envs.qpg_wrapper import QpgWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_dir', type=str)
    parser.add_argument('max_q_eval_mode', type=str)
    parser.add_argument('--n_rollouts', type=int, default=10)
    args = parser.parse_args()

    snapshot_file = join(args.snapshot_dir, 'params.pkl')
    config_file = join(args.snapshot_dir, 'params.json')

    params = torch.load(snapshot_file, map_location='cpu')
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['sampler']['batch_B'] = 1
    config['sampler']['eval_n_envs'] = 1
    config['sampler']['eval_max_trajectories'] = args.n_rollouts
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

    sampler.envs[0].start_record()
    traj_infos = sampler.evaluate_agent(0, include_observations=True)
    sampler.envs[0].end_record(join(args.snapshot_dir, 'episode_{i}.gif'), fps=40, scale=0.3)
    returns = [traj_info.Return for traj_info in traj_infos]
    lengths = [traj_info.Length for traj_info in traj_infos]
    performance = [traj_info.env_infos[-1].normalized_performance for traj_info in traj_infos]
    print('Performance:', performance)
    print('Returns', returns)
    print(f'Average Return {np.mean(returns)}, Average Length {np.mean(lengths)}')




if __name__ == '__main__':
    main()
