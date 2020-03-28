import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, max_size=int(2e5), device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.ptr] = obs.flatten()
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs.flatten()
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
