from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.mlp import QofMuMlpModel, AutoregPiMlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2"])


class SacAgent(BaseAgent):

    shared_pi_model = None

    def __init__(
            self,
            ModelCls=AutoregPiMlpModel,  # Pi model.
            QModelCls=QofMuMlpModel,
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None,
            initial_model_state_dict=None,  # Pi model.
            action_squash=1,  # Max magnitude (or None).
            pretrain_std=0.75,  # High value to make near uniform sampling.
            ):
        if isinstance(ModelCls, str):
            ModelCls = eval(ModelCls)
        if isinstance(ModelCls, str):
            QModelCls = eval(QModelCls)

        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[256, 256])
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)  # For async setup.
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

        self.log_alpha = None

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict
        self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)

        self.target_q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())

        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.target_q1_model.to(self.device)
        self.target_q2_model.to(self.device)
        self.log_alpha.to(self.device)

    def data_parallel(self):
        super().data_parallel()
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        self.q1_model = DDP_WRAP(self.q1_model)
        self.q2_model = DDP_WRAP(self.q2_model)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1 = self.target_q1_model(*model_inputs)
        q2 = self.target_q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def pi(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)

        actions, means, log_stds = [], [], []
        log_pi_total = 0
        self.model.start()
        while self.model.has_next():
            mean, log_std = self.model.next(actions, *model_inputs)
            dist_info = DistInfoStd(mean=mean, log_std=log_std)
            action, log_pi = self.distribution.sample_loglikelihood(dist_info)

            log_pi_total += log_pi
            actions.append(action)
            means.append(mean)
            log_stds.append(log_std)

        mean, log_std = torch.cat(means, dim=-1), torch.cat(log_stds, dim=-1)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)

        log_pi_total, dist_info = buffer_to((log_pi_total, dist_info), device="cpu")
        action = torch.cat(actions, dim=-1)
        return action, log_pi_total, dist_info  # Action stays on device for q models.

    def target_v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)

        next_actions, next_log_pis, _ = self.pi(*model_inputs)

        q1, q2 = self.target_q(observation, prev_action, prev_reward, next_actions)
        min_next_q = torch.min(q1, q2)

        target_v = min_next_q - self.log_alpha.exp().detach().cpu() * next_log_pis
        return target_v.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)

        actions, means, log_stds = [], [], []
        self.model.start()
        while self.model.has_next():
            mean, log_std = self.model.next(actions, *model_inputs)
            dist_info = DistInfoStd(mean=mean, log_std=log_std)
            action = self.distribution.sample(dist_info)

            actions.append(action)
            means.append(mean)
            log_stds.append(log_std)

        mean, log_std = torch.cat(means, dim=-1), torch.cat(log_stds, dim=-1)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        agent_info = AgentInfo(dist_info=dist_info)

        action = torch.cat(actions, dim=-1)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_q1_model, self.q1_model.state_dict(), tau)
        update_state_dict(self.target_q2_model, self.q2_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q1=self.q1_model, q2=self.q2_model)

    def parameters(self):
        for model in self.models:
            yield from model.parameters()
        yield self.log_alpha

    def pi_parameters(self):
        return self.model.parameters()

    def q1_parameters(self):
        return self.q1_model.parameters()

    def q2_parameters(self):
        return self.q2_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q1_model.train()
        self.q2_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            q1_model=self.q1_model.state_dict(),
            q2_model=self.q2_model.state_dict(),
            target_q1_model=self.target_q1_model.state_dict(),
            target_q2_model=self.target_q2_model.state_dict(),
            alpha=self.log_alpha.data
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q1_model.load_state_dict(state_dict["q1_model"])
        self.q2_model.load_state_dict(state_dict["q2_model"])
        self.target_q1_model.load_state_dict(state_dict['target_q1_model'])
        self.target_q2_model.load_state_dict(state_dict['target_q2_model'])
        self.log_alpha.data = state_dict['alpha']
