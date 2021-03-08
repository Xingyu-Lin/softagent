import os
from chester import logger
from envs.env import Env
import torch
import os.path as osp
import json


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def run_task(arg_vv, log_dir, exp_name):
    if arg_vv['algorithm'] == 'planet':
        from planet.config import DEFAULT_PARAMS
    else:
        raise NotImplementedError

    vv = DEFAULT_PARAMS
    vv.update(**arg_vv)
    vv = update_env_kwargs(vv)
    vv['max_episode_length'] = vv['env_kwargs']['horizon']

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    if torch.cuda.is_available():
        device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])
    else:
        device = torch.device('cpu')

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)
    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'], vv['image_dim'],
              env_kwargs=vv['env_kwargs'])

    if vv['algorithm'] == 'planet':
        from planet.planet_agent import PlaNetAgent
        agent = PlaNetAgent(env, vv, device)
        agent.train(train_epoch=vv['train_epoch'])
        env.close()
    elif vv['algorithm'] == 'dreamer':
        from dreamer.dreamer_agent import DreamerAgent
        agent = DreamerAgent(env, vv, device)
        agent.train(train_episode=vv['train_episode'])
        env.close()
