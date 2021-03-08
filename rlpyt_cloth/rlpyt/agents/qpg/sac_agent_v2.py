from collections import namedtuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.mlp import QofMuMlpModel, PiMlpModel
from rlpyt.models.qpg.conv2d import QofMuConvModel, PiConvModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import repeat, batched_index_select

MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = None
i = 0

MaxQInput = None


class SacAgent(BaseAgent):
    shared_pi_model = None

    def __init__(
      self,
      ModelCls=PiMlpModel,  # Pi model.
      QModelCls=QofMuMlpModel,
      model_kwargs=None,  # Pi model.
      q_model_kwargs=None,
      initial_model_state_dict=None,  # Pi model.
      action_squash=1,  # Max magnitude (or None).
      pretrain_std=0.75,  # High value to make near uniform sampling.
      max_q_eval_mode='none',
      n_qs=2,
    ):
        self._max_q_eval_mode = max_q_eval_mode
        if isinstance(ModelCls, str):
            ModelCls = eval(ModelCls)
        if isinstance(QModelCls, str):
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
        print('n_qs', self.n_qs)

        global Models
        Models = namedtuple("Models", ["pi"] + [f"q{i}" for i in range(self.n_qs)])

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict
        self.q_models = [self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
                         for _ in range(self.n_qs)]

        self.target_q_models = [self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
                                for _ in range(self.n_qs)]
        [target_q.load_state_dict(q.state_dict())
         for target_q, q in zip(self.target_q_models, self.q_models)]

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
        [q.to(self.device) for q in self.q_models]
        [q_target.to(self.device) for q_target in self.target_q_models]
        self.log_alpha.to(self.device)

    def data_parallel(self):
        super().data_parallel()
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        self.q_models = [DDP_WRAP(q) for q in self.q_models]

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
        qs = [q(*model_inputs) for q in self.q_models]
        return [q.cpu() for q in qs]

    def target_q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
                                  action), device=self.device)
        qs = [target_q(*model_inputs) for target_q in self.target_q_models]
        return [q.cpu() for q in qs]

    def pi(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    def target_v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)

        next_actions, next_log_pis, _ = self.pi(*model_inputs)

        qs = self.target_q(observation, prev_action, prev_reward, next_actions)
        min_next_q = torch.min(torch.stack(qs, dim=0), dim=0)[0]

        target_v = min_next_q - self.log_alpha.exp().detach().cpu() * next_log_pis
        return target_v.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        threshold = 0.2
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)

        if self._max_q_eval_mode == 'none':
            mean, log_std = self.model(*model_inputs)
            dist_info = DistInfoStd(mean=mean, log_std=log_std)
            action = self.distribution.sample(dist_info)
            agent_info = AgentInfo(dist_info=dist_info)
            action, agent_info = buffer_to((action, agent_info), device="cpu")
            return AgentStep(action=action, agent_info=agent_info)
        else:
            global MaxQInput
            observation, prev_action, prev_reward = model_inputs
            fields = ('location', 'pixels')  # Hardcode
            if 'position' in fields:
                no_batch = len(observation.position.shape) == 1
            else:
                no_batch = len(observation.pixels.shape) == 3
            if no_batch:
                if 'state' in self._max_q_eval_mode:
                    observation = [observation.position.unsqueeze(0)]
                else:
                    observation = [observation.pixels.unsqueeze(0)]
            else:
                if 'state' in self._max_q_eval_mode:
                    observation = [observation.position]
                else:
                    observation = [observation.pixels]

            if self._max_q_eval_mode == 'state_rope':
                locations = np.arange(25).astype('float32')
                locations = locations[:, None]
                locations = np.tile(locations, (1, 50)) / 24
            elif self._max_q_eval_mode == 'state_cloth_corner':
                locations = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                      [0, 0, 1, 0], [0, 0, 0, 1]],
                                     dtype='float32')
                locations = np.tile(locations, (1, 50))
            elif self._max_q_eval_mode == 'state_cloth_point':
                locations = np.mgrid[0:9, 0:9].reshape(2, 81).T.astype('float32')
                locations = np.tile(locations, (1, 50)) / 8
            elif self._max_q_eval_mode == 'pixel_rope':
                image = observation[0].squeeze(0).cpu().numpy()
                locations = np.transpose(np.where(np.all(image > 150, axis=2))).astype('float32')
                if locations.shape[0] == 0:
                    locations = np.array([[-1, -1]], dtype='float32')
                locations = np.tile(locations, (1, 50)) / 63
            elif self._max_q_eval_mode == 'pixel_cloth':
                image = observation[0].squeeze(0).cpu().numpy()
                location_orange = np.transpose(np.where(np.any(image < 50, axis=-1)))
                location_pink = np.transpose(np.where(np.logical_and(image[:, :, 0] > 160, image[:, :, 1] < 180)))
                locations = np.vstack([location_orange, location_pink])
                locations = locations / (image.shape[0] - 1) * 2. - 1.

                locations = np.array([locations[:, 1], locations[:, 0]]).transpose()  # Revert location into uv coordinate
                locations = np.tile(locations, (1, 50)).astype('float32')
            else:
                raise Exception()

            observation_pi = self.model.forward_embedding(observation)
            observation_qs = [q.forward_embedding(observation) for q in self.q_models]

            n_locations = len(locations)
            observation_pi_i = [repeat(o[[i]], [n_locations] + [1] * len(o.shape[1:]))
                                for o in observation_pi]
            observation_qs_i = [[repeat(o, [n_locations] + [1] * len(o.shape[1:]))
                                 for o in observation_q]
                                for observation_q in observation_qs]
            locations = torch.from_numpy(locations).to(self.device)

            if MaxQInput is None:
                MaxQInput = namedtuple('MaxQPolicyInput', fields)

            aug_observation_pi = [locations] + list(observation_pi_i)
            aug_observation_pi = MaxQInput(*aug_observation_pi)
            aug_observation_qs = [[locations] + list(observation_q_i)
                                  for observation_q_i in observation_qs_i]
            aug_observation_qs = [MaxQInput(*aug_observation_q)
                                  for aug_observation_q in aug_observation_qs]
            mean, log_std = self.model.forward_output(aug_observation_pi)  # , prev_action, prev_reward)

            qs = [q.forward_output(aug_obs, mean) for q, aug_obs
                  in zip(self.q_models, aug_observation_qs)]
            q = torch.min(torch.stack(qs, dim=0), dim=0)[0]
            # q = q.view(batch_size, n_locations)

            values, indices = torch.topk(q, math.ceil(threshold * n_locations), dim=-1)

            # vmin, vmax = values.min(dim=-1, keepdim=True)[0], values.max(dim=-1, keepdim=True)[0]
            # values = (values - vmin) / (vmax - vmin)
            # values = F.log_softmax(values, -1)
            #
            # uniform = torch.rand_like(values)
            # uniform = torch.clamp(uniform, 1e-5, 1 - 1e-5)
            # gumbel = -torch.log(-torch.log(uniform))

            # sampled_idx = torch.argmax(values + gumbel, dim=-1)
            sampled_idx = torch.randint(high=math.ceil(threshold * n_locations), size=(1,)).to(self.device)

            actual_idxs = indices[sampled_idx]
            # actual_idxs += (torch.arange(batch_size) * n_locations).to(self.device)
            location = locations[actual_idxs][:, :2]
            delta = torch.tanh(mean[actual_idxs])
            action = torch.cat((location, delta), dim=-1)

            mean, log_std = mean[actual_idxs], log_std[actual_idxs]

            if no_batch:
                action = action.squeeze(0)
                mean = mean.squeeze(0)
                log_std = log_std.squeeze(0)

            dist_info = DistInfoStd(mean=mean, log_std=log_std)
            agent_info = AgentInfo(dist_info=dist_info)

            action, agent_info = buffer_to((action, agent_info), device="cpu")
            return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        [update_state_dict(target_q, q.state_dict(), tau)
         for target_q, q in zip(self.target_q_models, self.q_models)]

    @property
    def models(self):
        return Models(pi=self.model,
                      **{f'p{i}': q for i, q in enumerate(self.q_models)})

    def parameters(self):
        for model in self.models:
            yield from model.parameters()
        yield self.log_alpha

    def pi_parameters(self):
        return self.model.parameters()

    def q_parameters(self):
        return [q.parameters() for q in self.q_models]

    def train_mode(self, itr):
        super().train_mode(itr)
        [q.train() for q in self.q_models]

    def sample_mode(self, itr):
        super().sample_mode(itr)
        [q.eval() for q in self.q_models]
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        [q.eval() for q in self.q_models]
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        rtn = dict(
            model=self.model.state_dict(),  # Pi model.
            alpha=self.log_alpha.data
        )
        rtn.update({f'q{i}_model': q.state_dict()
                    for i, q in enumerate(self.q_models)})
        rtn.update({f'target_q{i}_model': q.state_dict()
                    for i, q in enumerate(self.target_q_models)})
        return rtn

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.log_alpha.data = state_dict['alpha']
        print(state_dict.keys())

        [q.load_state_dict(state_dict[f'q{i}_model'])
         for i, q in enumerate(self.q_models)]
        [q.load_state_dict(state_dict[f'target_q{i}_model'])
         for i, q in enumerate(self.target_q_models)]
