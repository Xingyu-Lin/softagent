import os
import sys
import numpy as np
import json
from chester import logger
from planet.planet_agent import PlaNetAgent
from planet.config import DEFAULT_PARAMS
from envs.env import Env
import torch
import os.path as osp
import json


def run_task(arg_vv, log_dir, exp_name):
    vv = DEFAULT_PARAMS
    vv.update(**arg_vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.cuda.manual_seed(vv['seed'])
    else:
        device = torch.device('cpu')

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'],
              env_kwargs=vv['env_kwargs'])

    agent = PlaNetAgent(env, vv, device)
    agent.train(train_episode=500)
    env.close()
