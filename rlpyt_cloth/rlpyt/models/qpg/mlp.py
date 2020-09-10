
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot, select_at_indexes
from rlpyt.models.mlp import MlpModel
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.distributions.categorical import Categorical, DistInfo


MIN_LOG_STD = -20
MAX_LOG_STD = 2


class MuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            output_max=1,
            ):
        super().__init__()
        self._output_max = output_max
        self._obs_ndim = len(observation_shape)
        input_dim = int(np.prod(observation_shape))
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )

    def forward_embedding(self, observation):
        return observation

    def forward_output(self, observation):
        return self(observation, None, None)

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        mu = self._output_max * torch.tanh(self.mlp(observation.view(T * B, -1)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class PiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
       # action_size = 3
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        self._action_size = action_size
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward_embedding(self, observation):
        return observation

    def forward_output(self, observation):
        return self(observation, None, None)

    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class AutoregPiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=50,
            loc_size=2,
            delta_size=3,
    ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))
        self._n_tile = n_tile
        self._loc_size = loc_size
        self._delta_size = delta_size

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        assert action_size == loc_size + delta_size # First 2 (location), then 3 (action)
        self._action_size = action_size

        self.mlp_loc = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=loc_size * 2
        )
        self.mlp_delta = MlpModel(
            input_size=input_dim + loc_size * n_tile,
            hidden_sizes=hidden_sizes,
            output_size=delta_size * 2,
        )

        self._counter = 0

    def start(self):
        self._counter = 0

    def next(self, actions, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
                                               self._obs_ndim)
        input_obs = observation.view(T * B, -1)
        if self._counter == 0:
            output = self.mlp_loc(input_obs)
            mu, log_std = output.chunk(2, dim=-1)
        elif self._counter == 1:
            assert len(actions) == 1
            action_loc = actions[0].view(T * B, -1)
            model_input = torch.cat((input_obs, action_loc.repeat((1, self._n_tile))), dim=-1)
            output = self.mlp_delta(model_input)
            mu, log_std = output.chunk(2, dim=-1)
        else:
            raise Exception('Invalid self._counter', self._counter)
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        self._counter += 1
        return mu, log_std

    def has_next(self):
        return self._counter < 2


GumbelDistInfo = namedtuple('GumbelDistInfo', ['cat_dist', 'delta_dist'])
class GumbelPiMlpModel(torch.nn.Module):
    """For picking corners"""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            all_corners=False
            ):
        super().__init__()
        self._obs_ndim = 1
        self._all_corners = all_corners
        input_dim = int(np.sum(observation_shape))

        print('all corners', self._all_corners)
        delta_dim = 12 if all_corners else 3
        self._delta_dim = delta_dim
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=2 * delta_dim + 4, # 3 for each corners, times two for std, 4 probs
        )

        self.delta_distribution = Gaussian(
            dim=delta_dim,
            squash=True,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical(4)


    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        logits = output[:, :4]
        mu, log_std = output[:, 4:4 + self._delta_dim], output[:, 4 + self._delta_dim:]
        logits, mu, log_std = restore_leading_dims((logits, mu, log_std), lead_dim, T, B)
        return GumbelDistInfo(cat_dist=logits, delta_dist=DistInfoStd(mean=mu, log_std=log_std))

    def sample_loglikelihood(self, dist_info):
        logits, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist

        u = torch.rand_like(logits)
        u = torch.clamp(u, 1e-5, 1 - 1e-5)
        gumbel = -torch.log(-torch.log(u))
        prob = F.softmax((logits + gumbel) / 10, dim=-1)

        cat_sample = torch.argmax(prob, dim=-1)
        cat_loglikelihood = select_at_indexes(cat_sample, prob)

        one_hot = to_onehot(cat_sample, 4, dtype=torch.float32)
        one_hot = (one_hot - prob).detach() + prob # Make action differentiable through prob

        if self._all_corners:
            mu, log_std = delta_dist_info.mean, delta_dist_info.log_std
            mu, log_std = mu.view(-1, 4, 3), log_std.view(-1, 4, 3)
            mu = mu[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            log_std = log_std[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            new_dist_info = DistInfoStd(mean=mu, log_std=log_std)
        else:
            new_dist_info = delta_dist_info

        delta_sample, delta_loglikelihood = self.delta_distribution.sample_loglikelihood(new_dist_info)
        action = torch.cat((one_hot, delta_sample), dim=-1)
        log_likelihood = cat_loglikelihood + delta_loglikelihood
        return action, log_likelihood

    def sample(self, dist_info):
        logits, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist
        u = torch.rand_like(logits)
        u = torch.clamp(u, 1e-5, 1 - 1e-5)
        gumbel = -torch.log(-torch.log(u))
        prob = F.softmax((logits + gumbel) / 10, dim=-1)

        cat_sample = torch.argmax(prob, dim=-1)
        one_hot = to_onehot(cat_sample, 4, dtype=torch.float32)

        if len(prob.shape) == 1: # Edge case for when it gets buffer shapes
            cat_sample = cat_sample.unsqueeze(0)

        if self._all_corners:
            mu, log_std = delta_dist_info.mean, delta_dist_info.log_std
            mu, log_std = mu.view(-1, 4, 3), log_std.view(-1, 4, 3)
            mu = select_at_indexes(cat_sample, mu)
            log_std = select_at_indexes(cat_sample, log_std)

            if len(prob.shape) == 1: # Edge case for when it gets buffer shapes
                mu, log_std = mu.squeeze(0), log_std.squeeze(0)

            new_dist_info = DistInfoStd(mean=mu, log_std=log_std)
        else:
            new_dist_info = delta_dist_info

        if self.training:
            self.delta_distribution.set_std(None)
        else:
            self.delta_distribution.set_std(0)
        delta_sample = self.delta_distribution.sample(new_dist_info)
        return torch.cat((one_hot, delta_sample), dim=-1)


class GumbelAutoregPiMlpModel(torch.nn.Module):
    """For picking corners autoregressively"""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=20,
    ):
        super().__init__()
        self._obs_ndim = 1
        self._n_tile = n_tile
        input_dim = int(np.sum(observation_shape))

        self._action_size = action_size
        self.mlp_loc = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=4
        )
        self.mlp_delta = MlpModel(
            input_size=input_dim + 4 * n_tile,
            hidden_sizes=hidden_sizes,
            output_size=3 * 2,
        )

        self.delta_distribution = Gaussian(
            dim=3,
            squash=True,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical(4)

        self._counter = 0

    def start(self):
        self._counter = 0

    def next(self, actions, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
                                               self._obs_ndim)
        input_obs = observation.view(T * B, -1)
        if self._counter == 0:
            logits = self.mlp_loc(input_obs)
            logits = restore_leading_dims(logits, lead_dim, T, B)
            self._counter += 1
            return logits

        elif self._counter == 1:
            assert len(actions) == 1
            action_loc = actions[0].view(T * B, -1)
            model_input = torch.cat((input_obs, action_loc.repeat((1, self._n_tile))), dim=-1)
            output = self.mlp_delta(model_input)
            mu, log_std = output.chunk(2, dim=-1)
            mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
            self._counter += 1
            return DistInfoStd(mean=mu, log_std=log_std)
        else:
            raise Exception('Invalid self._counter', self._counter)

    def has_next(self):
        return self._counter < 2

    def sample_loglikelihood(self, dist_info):
        if isinstance(dist_info, DistInfoStd):
            action, log_likelihood = self.delta_distribution.sample_loglikelihood(dist_info)
        else:
            logits = dist_info

            u = torch.rand_like(logits)
            u = torch.clamp(u, 1e-5, 1 - 1e-5)
            gumbel = -torch.log(-torch.log(u))
            prob = F.softmax((logits + gumbel) / 10, dim=-1)

            cat_sample = torch.argmax(prob, dim=-1)
            log_likelihood = select_at_indexes(cat_sample, prob)

            one_hot = to_onehot(cat_sample, 4, dtype=torch.float32)
            action = (one_hot - prob).detach() + prob  # Make action differentiable through prob

        return action, log_likelihood

    def sample(self, dist_info):
        if isinstance(dist_info, DistInfoStd):
            if self.training:
                self.delta_distribution.set_std(None)
            else:
                self.delta_distribution.set_std(0)
            action = self.delta_distribution.sample(dist_info)
        else:
            logits = dist_info
            u = torch.rand_like(logits)
            u = torch.clamp(u, 1e-5, 1 - 1e-5)
            gumbel = -torch.log(-torch.log(u))
            prob = F.softmax((logits + gumbel) / 10, dim=-1)

            cat_sample = torch.argmax(prob, dim=-1)
            action = to_onehot(cat_sample, 4, dtype=torch.float32)

        return action



class QofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=1,
            ):
        super().__init__()
        self._obs_ndim = 1
        self._n_tile = n_tile
       # action_size = 3
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        input_dim += action_size * n_tile
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward_embedding(self, observation):
        return observation

    def forward_output(self, observation, action):
        return self(observation, None, None, action)

    def forward(self, observation, prev_action, prev_reward, action):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        action = action.view(T * B, -1).repeat(1, self._n_tile)
        q_input = torch.cat(
            [observation.view(T * B, -1), action], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class VMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size=None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        v = self.mlp(observation.view(T * B, -1)).squeeze(-1)
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
