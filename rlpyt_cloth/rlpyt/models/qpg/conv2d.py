
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dHeadModel, Conv2dModel
from rlpyt.models.qpg.mlp import AutoregPiMlpModel
from rlpyt.models.preprocessor import get_preprocessor
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.distributions.categorical import Categorical, DistInfo


MIN_LOG_STD = -20
MAX_LOG_STD = 2


def _filter_name(fields, name):
    fields = list(fields)
    idx = fields.index(name)
    del fields[idx]
    return fields


class PiConvModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            # SAME padding for odd kernel sizes
            paddings = [ks // 2 for ks in kernel_sizes]
       # action_size = 3

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels
        # print('observation_shape', observation_shape)

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        extra_input_size = sum([getattr(observation_shape, f)[0] for f in fields])
        self._extra_input_size = extra_input_size
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=2 * action_size,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False,
                                    extra_input_size=extra_input_size)

    def forward_embedding(self, observation):
        pixel_obs = observation[0]
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(pixel_obs)

        out = self.conv.forward_embedding(pixel_obs)
        out = restore_leading_dims(out, lead_dim, T, B)
        return [out]

    def forward_output(self, observation, extra_input=None):
        lead_dim, T, B, _ = infer_leading_dims(observation.pixels, 1)
        fields = _filter_name(observation._fields, 'pixels')
        if self._extra_input_size > 0:
            extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                     for f in fields], dim=-1)
        else:
            extra_input = None
        output = self.conv.forward_output(observation.pixels, extra_input=extra_input)

        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

    def forward(self, observation, prev_action, prev_reward):
        pixel_obs = observation.pixels
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(pixel_obs)
        fields = _filter_name(observation._fields, 'pixels')

        if self._extra_input_size > 0:
            extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                     for f in fields], dim=-1)
        else:
            extra_input = None

        output = self.conv(pixel_obs, extra_input=extra_input)

        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class AutoregPiConvModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            n_tile=50,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            # SAME padding for odd kernel sizes
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._n_tile = n_tile
        self._action_size = action_size
        self._image_shape = observation_shape.pixels

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        self.conv = Conv2dModel(in_channels=observation_shape.pixels[-1],
                                channels=channels, kernel_sizes=kernel_sizes,
                                strides=strides, paddings=paddings,
                                nonlinearity=nonlinearity)
        embedding_size = self.conv.conv_out_size(*observation_shape.pixels)
        self.deconv_loc = nn.Sequential(
            torch.nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nonlinearity(),
            torch.nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nonlinearity(),
            torch.nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nonlinearity(),
            torch.nn.Conv2d(channels, 1, 3, padding=1)
        )
        self.mlp_delta = MlpModel(
            input_size=embedding_size + 2 * n_tile,
            hidden_sizes=hidden_sizes,
            output_size=3 * 2
        )

        self._counter = 0
        self.delta_distribution = Gaussian(
            dim=12,
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical()

    def start(self):
        self._counter = 0

    def next(self, actions, observation, prev_action, prev_reward):
        pixel_obs = observation.pixels
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(observation.pixels)
        embedding = self.conv(pixel_obs)

        if self._counter == 0:
            # TODO create segmentation
            seg = None
            prob_map = self.deconv_loc(embedding)
            prob_map = prob_map - (1 - seg) * float('inf')
            prob_map = prob_map.view(prob_map.shape[0], -1)
            prob = F.softmax(prob_map, dim=-1)
            prob = restore_leading_dims(prob, lead_dim, T, B)

            self._counter += 1
            return DistInfo(prob=prob)
        elif self._counter == 1:
            action_loc = actions[0].view(T * B, -1)
            embedding = embedding.view(embedding.shape[0], -1)
            model_input = torch.cat((embedding, action_loc.repeat((1, self._n_tile))), dim=-1)
            output = self.mlp_delta(model_input)
            mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
            mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)

            self._counter += 1
            return DistInfoStd(mean=mu, log_std=log_std)
        else:
            raise Exception('Invalid self._counter', self._counter)

    def has_next(self):
        return self._counter < 2

    def sample_loglikelihood(self, dist_info):
        if isinstance(dist_info, DistInfo):
            sample, log_likelihood = self.cat_distribution.sample_loglikelihood(dist_info)
            sample = sample.unsqueeze(-1)
            one_hot = torch.zeros_like(dist_info.prob)
            one_hot.scatter_(-1, sample, 1)
            rtn = (one_hot - dist_info.prob).detach() + dist_info.prob
        elif isinstance(dist_info, DistInfoStd):
            rtn = self.delta_distribution.sample(dist_info)
        else:
            raise Exception('Invalid dist_info', type(dist_info))
        return rtn

    def sample(self, dist_info):
        if isinstance(dist_info, DistInfo):
            if self.training:
                sample = self.cat_distribution.sample(dist_info)
            else:
                sample = torch.max(dist_info.prob, dim=-1)[1].view(-1)
            sample = sample.unsqueeze(-1)
            one_hot = torch.zeros_like(dist_info.prob)
            one_hot.scatter_(-1, sample, 1)
            sample = one_hot
        elif isinstance(dist_info, DistInfoStd):
            if self.training:
                self.delta_distribution.set_std(None)
            else:
                self.delta_distribution.set_std(0)
            sample = self.delta_distribution.sample(dist_info)
        return sample


class GumbelPiConvModel(torch.nn.Module):
    """ For picking corners """

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            # SAME padding for odd kernel sizes
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels

        # print('observation shape', observation_shape)

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        extra_input_size = sum([getattr(observation_shape, f)[0] for f in fields])
        self._extra_input_size = extra_input_size
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=2 * 3 + 4,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False,
                                    extra_input_size=extra_input_size)
        self.delta_distribution = Gaussian(
            dim=3,
            squash=True,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical(4)

    def forward(self, observation, prev_action, prev_reward):
        pixel_obs = observation.pixels
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(observation.pixels)

        if self._extra_input_size > 0:
            fields = _filter_name(observation._fields, 'pixels')
            extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                     for f in fields], dim=-1)
        else:
            extra_input = None

        output = self.conv(pixel_obs, extra_input=extra_input)
        prob = F.softmax(output[:, :4] / 10., dim=-1)
        mu, log_std = output[:, 4:4 + 3], output[:, 4 + 3:]
        prob, mu, log_std = restore_leading_dims((prob, mu, log_std), lead_dim, T, B)
        return DistInfo(prob=prob), DistInfoStd(mean=mu, log_std=log_std)

    def sample_loglikelihood(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info
        cat_sample, cat_loglikelihood = self.cat_distribution.sample_loglikelihood(cat_dist_info)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        one_hot.scatter_(-1, cat_sample, 1)
        one_hot = (one_hot - cat_dist_info.prob).detach() + cat_dist_info.prob # Make action differentiable through prob

        delta_sample, delta_loglikelihood = self.delta_distribution.sample_loglikelihood(delta_dist_info)
        action = torch.cat((one_hot, delta_sample), dim=-1)
        log_likelihood = cat_loglikelihood + delta_loglikelihood
        return action, log_likelihood

    def sample(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info
        if self.training:
            cat_sample = self.cat_distribution.sample(cat_dist_info)
        else:
            cat_sample = torch.max(cat_dist_info.prob, dim=-1)[1].view(-1)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        one_hot.scatter_(-1, cat_sample, 1)

        if self.training:
            self.delta_distribution.set_std(None)
        else:
            self.delta_distribution.set_std(0)
        delta_sample = self.delta_distribution.sample(delta_dist_info)
        return torch.cat((one_hot, delta_sample), dim=-1)


class QofMuConvModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU,
            n_tile=1,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            paddings = [ks // 2 for ks in kernel_sizes]
        #action_size = 3

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels
        self._n_tile = n_tile

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        extra_input_size = sum([getattr(observation_shape, f)[0] for f in fields])
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=1,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False,
                                    extra_input_size=extra_input_size + action_size * n_tile)

    def forward_embedding(self, observation):
        pixel_obs = observation[0]
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(pixel_obs)

        out = self.conv.forward_embedding(pixel_obs)
        out = restore_leading_dims(out, lead_dim, T, B)
        return [out]


    def forward_output(self, observation, action):
        pixel_obs = observation.pixels
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, 1)
        fields = _filter_name(observation._fields, 'pixels')
        action = action.view(T * B, -1).repeat(1, self._n_tile)
        extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                 for f in fields] + [action], dim=-1)
        q = self.conv.forward_output(pixel_obs, extra_input=extra_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

    def forward(self, observation, prev_action, prev_reward, action):
        pixel_obs = observation.pixels
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)
        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        pixel_obs = self.preprocessor(pixel_obs)
        fields = _filter_name(observation._fields, 'pixels')
        action = action.view(T * B, -1).repeat(1, self._n_tile)
        extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                 for f in fields] + [action], dim=-1)

        q = self.conv(pixel_obs, extra_input=extra_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)

        return q
