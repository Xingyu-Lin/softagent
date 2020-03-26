import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.env import Env


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class VisualEncoder(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, image_c, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.image_c = image_c
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if image_dim == 128:
            self.conv0 = nn.Conv2d(image_c, 16, 4, stride=2)
            self.conv1 = nn.Conv2d(16, 32, 4, stride=2)
        elif image_dim == 64:
            self.conv1 = nn.Conv2d(image_c, 32, 4, stride=2)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observation):
        observation = observation.view(-1, self.image_c, self.image_dim, self.image_dim)
        if self.image_dim == 128:
            hidden = self.act_fn(self.conv0(observation))
            hidden = self.act_fn(self.conv1(hidden))
        elif self.image_dim == 64:
            hidden = self.act_fn(self.conv1(observation))
        else:
            raise NotImplementedError
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


class ActionEncoder(nn.Module):
    def __init__(self, embedding_size, action_dim, activation_function='relu'):
        super().__init__()
        self.l1 = nn.Linear(action_dim, 64)
        self.l2 = nn.Linear(64, embedding_size)
        self.act_fn = getattr(F, activation_function)

    def forward(self, action):
        fa = self.act_fn(self.l1(action))
        fa = self.act_fn(self.l2(fa))
        return fa


class Actor(nn.Module):
    def __init__(self, obs_embed_dim, action_dim, image_dim, image_c, max_action):
        super(Actor, self).__init__()
        self.visual_encoder = VisualEncoder(obs_embed_dim, image_dim, image_c)
        self.l1 = nn.Linear(obs_embed_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, obs):
        state = self.visual_encoder(obs)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, obs_embed_dim, action_dim, image_dim, image_c, action_embed_dim):
        super(Critic, self).__init__()
        self.visual_encoder1 = VisualEncoder(obs_embed_dim, image_dim, image_c)
        self.action_encoder1 = ActionEncoder(action_embed_dim, action_dim)

        self.visual_encoder2 = VisualEncoder(obs_embed_dim, image_dim, image_c)
        self.action_encoder2 = ActionEncoder(action_embed_dim, action_dim)

        # Q1 architecture
        self.l1 = nn.Linear(obs_embed_dim + action_embed_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(obs_embed_dim + action_embed_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        state = self.visual_encoder1(obs)
        action_embed = self.action_encoder1(action)
        sa = torch.cat([state, action_embed], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        state = self.visual_encoder2(obs)
        action_embed = self.action_encoder2(action)
        sa = torch.cat([state, action_embed], 1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, obs, action):
        state = self.visual_encoder1(obs)
        action_embed = self.action_encoder1(action)
        sa = torch.cat([state, action_embed], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
      self,
      image_dim,
      image_c,
      obs_embed_dim,
      action_dim,
      action_embed_dim,
      max_action,
      discount=0.99,
      tau=0.005,
      policy_noise=0.2,
      noise_clip=0.5,
      policy_freq=2,
      device='cpu'
    ):
        self.device = device
        self.actor = Actor(obs_embed_dim, action_dim, image_dim, image_c, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-4)

        self.critic = Critic(obs_embed_dim, action_dim, image_dim, image_c, action_embed_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, *state.shape[1:])).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        obs, action, next_obs, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
              torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
              self.actor_target(next_obs) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # print(self.actor)
        # for name, param in self.actor.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # exit()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
