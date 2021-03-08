import cv2
import numpy as np
import torch
import gym
import softgym
from gym.spaces import Box

from softgym.registered_env import SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize

softgym.register_flex_envs()


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
        0.5)  # Quantise to given bit depth and centre
    observation.add_(torch.rand_like(observation).div_(
        2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth, image_dim, normalize_observation=True):
    dtype = torch.float32 if normalize_observation else torch.uint8
    if images.shape[0] != image_dim:
        images = torch.tensor(cv2.resize(images, (image_dim, image_dim), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                              dtype=dtype)  # Resize and put channel first
    else:
        images = torch.tensor(images.transpose(2, 0, 1), dtype=dtype)  # Resize and put channel first
    if normalize_observation:
        preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class SoftGymEnv(object):
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim, env_kwargs=None,
                 normalize_observation=True, scale_reward=1.0, clip_obs=None, obs_process=None):
        self._env = SOFTGYM_ENVS[env](**env_kwargs)
        self._env = normalize(self._env, scale_reward=scale_reward, clip_obs=clip_obs)
        self.symbolic = symbolic
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.image_dim = image_dim
        if image_dim is None:
            self.image_dim = self._env.observation_space.shape[0]
        if not self.symbolic:
            self.image_c = self._env.observation_space.shape[-1]
        self.normalize_observation = normalize_observation
        self.obs_process = obs_process

    def reset(self, **kwargs):
        self.t = 0  # Reset internal timer
        obs = self._env.reset(**kwargs)
        if self.symbolic:
            if self.obs_process is None:
                if not isinstance(obs, tuple):
                    return torch.tensor(obs, dtype=torch.float32)
                else:
                    return obs
            else:
                return self.obs_process(obs)
        else:
            return _images_to_observation(obs, self.bit_depth, self.image_dim, normalize_observation=self.normalize_observation)

    def step(self, action, **kwargs):
        if not isinstance(action, np.ndarray):
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            obs, reward_k, done, info = self._env.step(action, **kwargs)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            # print('t:', self.t, self.max_episode_length, done)
            if self.symbolic:
                if self.obs_process is None:
                    if not isinstance(obs, tuple):
                        obs = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs = self.obs_process(obs)
            else:
                obs = _images_to_observation(obs, self.bit_depth, self.image_dim, normalize_observation=self.normalize_observation)
            if done:
                break
        return obs, reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        if self.symbolic:
            return self._env.observation_space
        else:
            return Box(low=-np.inf, high=np.inf, shape=(self.image_dim, self.image_dim, self.image_c), dtype=np.float32)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (self.image_c, self.image_dim, self.image_dim)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_env':
            # Prevent recursive call on self._env
            raise AttributeError('_env not initialized yet!')
        return getattr(self._env, name)


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim, env_kwargs=None, normalize_observation=True,
        scale_reward=1.0, clip_obs=None, obs_process=None):
    if env in SOFTGYM_ENVS:
        return SoftGymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim, env_kwargs,
                          normalize_observation=normalize_observation,
                          scale_reward=scale_reward,
                          clip_obs=clip_obs,
                          obs_process=obs_process)
    else:
        raise NotImplementedError


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones))[:,
                    0]  # Done mask to blank out observations and zero rewards for previously terminated environments
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [d or prev_d for d, prev_d in
                 zip(dones, self.dones)]  # Env should remain terminated if previously terminated
        self.dones = dones
        observations, rewards, dones = torch.cat(observations), torch.tensor(rewards,
                                                                             dtype=torch.float32), torch.tensor(dones,
                                                                                                                dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones, {}

    def close(self):
        [env.close() for env in self.envs]


class WrapperRlkit(object):
    """ Wrap the image env environment. Use all numpy returns and flatten the observation """

    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return np.array(obs).flatten()

    def step(self, action, **kwargs):
        obs, reward, done, info = self._env.step(action, **kwargs)
        return np.array(obs).flatten(), reward, done, info

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_env':
            # Prevent recursive call on self._env
            raise AttributeError('_env not initialized yet!')
        return getattr(self._env, name)

    @property
    def imsize(self):
        return self._env.image_dim
