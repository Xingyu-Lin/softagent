from collections import OrderedDict

import numpy as np
from rlkit.data_management.replay_buffer import ReplayBuffer
import time
import torch
from rlkit.torch.core import np_to_pytorch_batch


# Postprocess an observation from [0, 256) to [-0.5, 0.5]
# def postprocess_obs(obs):
#     return (obs.astype(np.float32) / 256. - 0.5) + np.random.random(obs.shape) / 256.

# Postprocessing on GPU
def postprocess_obs(obs):
    obs.div_(256.).sub_(0.5)  # Quantise to given bit depth and centre
    obs.add_(torch.rand_like(obs).div_(256))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Preprocess an observation for storage (from float32 numpy array [-0.5, 0.5) to uint8 numpy array [0, 256))
def preprocess_obs(obs):
    assert np.alltrue(obs >= -0.5) and np.alltrue(obs <= 0.5)
    return np.clip(np.floor((obs + 0.5) * 256), 0, 255).astype(np.uint8)


class ImageReplayBuffer(ReplayBuffer):

    def __init__(
      self,
      max_replay_buffer_size,
      observation_dim,
      action_dim,
      env_info_sizes,
    ):
        """ Optimized data type for storing images; Optimized indexing using JIT """
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.uint8)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.uint8)
        self._actions = np.zeros((max_replay_buffer_size, action_dim), dtype=np.float32)
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size), dtype=np.float32)
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation, terminal, env_info, **kwargs):
        # print('add sample')
        # assert np.allclose(postprocess_obs(preprocess_obs(observation)), observation)
        self._observations[self._top] = preprocess_obs(observation)
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = preprocess_obs(next_observation)

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        # print('random batch')
        # start_time = time.time()
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        # print(time.time() - start_time)
        batch = np_to_pytorch_batch(batch)  # Transfer to GPU first
        postprocess_obs(batch['observations'])
        postprocess_obs(batch['next_observations'])
        # print(time.time() - start_time)
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
