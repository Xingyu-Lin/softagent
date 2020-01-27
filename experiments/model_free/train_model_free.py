from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import rlkit.torch.pytorch_util as ptu

from experiments.train import update_env_kwargs
from experiments.model_free.models import ConvQ, ConvPolicy
from envs.env import Env, WrapperRlkit
import torch
import os
import os.path as osp
import json


def run_task(arg_vv, log_dir, exp_name):
    vv = arg_vv
    vv = update_env_kwargs(vv)

    # Configure logger
    setup_logger(exp_prefix=vv['exp_name'], base_log_dir=log_dir, variant=vv, exp_id=0, seed=vv['seed'], snapshot_mode='last', snapshot_gap=10)

    # Configure torch
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            ptu.set_gpu_mode(True, 1)
            device = torch.device('cuda:1')
        else:
            ptu.set_gpu_mode(True, 0)
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])
    else:
        device = torch.device('cpu')

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'
    env = WrapperRlkit(Env(vv['env_name'], env_symbolic, vv['seed'], vv['max_episode_length'], 1, 8, vv['image_dim'],
                           env_kwargs=vv['env_kwargs']))
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    M = vv['layer_size']
    if env_symbolic:
        qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M], )
        qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M], )
        target_qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M], )
        target_qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M], )
        policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[M, M], )
    else:
        qf1 = ConvQ(vv['embedding_size'], vv['image_dim'], [M, M], action_dim)
        qf2 = ConvQ(vv['embedding_size'], vv['image_dim'], [M, M], action_dim)
        target_qf1 = ConvQ(vv['embedding_size'], vv['image_dim'], [M, M], action_dim)
        target_qf2 = ConvQ(vv['embedding_size'], vv['image_dim'], [M, M], action_dim)

        policy = ConvPolicy(vv['embedding_size'], vv['image_dim'], [M, M], action_dim)

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(env, eval_policy, )
    expl_path_collector = MdpPathCollector(env, policy, )

    replay_buffer = EnvReplayBuffer(vv['replay_buffer_size'], env, )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **vv['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **vv['algorithm_kwargs']
    )
    algorithm.to(device)
    algorithm.train()