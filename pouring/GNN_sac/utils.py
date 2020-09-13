import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from collections import deque

from torch_geometric.data import Data as geometric_data
from torch_geometric.data import Batch as geometric_batch

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class GraphReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, action_dim, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # the proprioceptive obs is stored as float32, pixels obs as uint8

        self.obses = []
        self.next_obses = []
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):

        self.obses.append(obs)
        self.next_obses.append(next_obs)
       
        # obs_data = obs.to_data_list()[0]
        # print("in add, obs.data.x.device is: ", obs_data.x.device)
        # assert obs_data.x.device != torch.device("cuda:0")

        if len(self.obses) > self.capacity:
            self.obses.pop(0)
        if len(self.next_obses) > self.capacity:
            self.next_obses.pop(0)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        sampled_obs = []
        sampled_next_obs = []
        for idx in idxs:
            sampled_obs.append(self.obses[idx].to_data_list()[0].to(torch.device("cpu")))
            sampled_next_obs.append(self.next_obses[idx].to_data_list()[0].to(torch.device("cpu")))

        obses = geometric_batch()
        next_obses = geometric_batch()
        # print(type(sampled_obs[0]))
        # # print(sampled_obs[0].x)
        # print(sampled_obs[0].x.dtype)
        # print(type(sampled_obs[0].x))
        # for idx, x in enumerate(sampled_obs):
        #     print(idx, x.x.device)
        obses = obses.from_data_list(sampled_obs).to(self.device)
        next_obses = next_obses.from_data_list(sampled_next_obs).to(self.device)

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


def preprocess_single_obs(obs):
    all_points, edge_index, edge_attr = obs
    all_points = torch.FloatTensor(all_points)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr)
    data = geometric_data(x=all_points, edge_index=edge_index, edge_attr=edge_attr)
    batch = geometric_batch()
    batch = batch.from_data_list([data])
    return batch