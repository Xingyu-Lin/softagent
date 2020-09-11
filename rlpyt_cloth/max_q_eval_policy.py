
from os.path import join
import importlib
import argparse
import json

import torch
import numpy as np

from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.samplers.serial.sampler import SerialSampler


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
    config['env']['task_kwargs']['maxq'] = True

    itr, cum_steps = params['itr'], params['cum_steps']
    print(f'Loading experiment at itr {itr}, cum_steps {cum_steps}')

    agent_state_dict = params['agent_state_dict']

    sac_agent_module = 'rlpyt.agents.qpg.{}'.format(config['sac_agent_module'])
    sac_agent_module = importlib.import_module(sac_agent_module)
    SacAgent = sac_agent_module.SacAgent

    agent = SacAgent(max_q_eval_mode=args.max_q_eval_mode, **config["agent"])
    sampler = SerialSampler(
        EnvCls=DMControlEnv,
        env_kwargs=config["env"],
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
    sampler.initialize(agent)
    agent.load_state_dict(agent_state_dict)

    agent.to_device(cuda_idx=0)
    agent.eval_mode(0)

    traj_infos = sampler.evaluate_agent(0)
    returns = [traj_info.Return for traj_info in traj_infos]
    lengths = [traj_info.Length for traj_info in traj_infos]

    print('Returns', returns)
    print(f'Average Return {np.mean(returns)}, Average Length {np.mean(lengths)}')


if __name__ == '__main__':
    main()
