from planet.planet_agent import PlaNetAgent
from planet.config import DEFAULT_PARAMS
from envs.env import Env
import torch
import os
import os.path as osp
import json

from chester import logger

if __name__ == '__main__':
    vv = DEFAULT_PARAMS.copy()
    vv['seed'] = 100
    vv['exp_name'] = 'large_vae'
    vv['env_name'] = env_name = 'PourWaterPosControl-v0'

    logger.configure(osp.join('./data', vv['exp_name']), exp_name=vv['exp_name'])
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    env = Env(env_name, vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'])

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.cuda.manual_seed(vv['seed'])
    else:
        device = torch.device('cpu')

    agent = PlaNetAgent(env, vv, device)
    agent.train(train_episode=500)
    env.close()
