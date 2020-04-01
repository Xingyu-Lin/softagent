import copy
import numpy as np
import torch
import torch.nn.functional as F
from ResRL.models import *
from ResRL.residual_models import *


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
class TD3(object):
    def __init__(
      self,
      image_observation,
      image_dim,
      image_c,
      obs_embed_dim,
      state_dim,
      action_dim,
      action_embed_dim,
      max_action,
      visual_encoder_name,
      discount=0.99,
      tau=0.005,
      policy_noise=0.2,
      noise_clip=0.5,
      policy_freq=2,
      device='cpu'
    ):
        self.device = device
        if image_observation:
            if 'Residual' in visual_encoder_name:
                self.actor = ResidualActor1D(image_dim, image_c, action_dim, max_action).to(device)
                self.critic = ResidualCritic1D(image_dim, image_c, action_dim).to(device)
            else:
                self.actor = ConvActor(obs_embed_dim, action_dim, image_dim, image_c, max_action, visual_encoder_name).to(device)
                self.critic = ConvCritic(obs_embed_dim, action_dim, image_dim, image_c, action_embed_dim, visual_encoder_name).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        if hasattr(self.actor_target, 'multihead_attn'):
            self.actor_target.multihead_attn._qkv_same_embed_dim = self.actor.multihead_attn._qkv_same_embed_dim # Hacky fix for pytorch bug
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_target = copy.deepcopy(self.critic)
        if hasattr(self.critic_target, 'critic1'):
            self.critic_target.critic1.multihead_attn._qkv_same_embed_dim = self.critic.critic1.multihead_attn._qkv_same_embed_dim
            self.critic_target.critic2.multihead_attn._qkv_same_embed_dim = self.critic.critic2.multihead_attn._qkv_same_embed_dim
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
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
        torch.save({'critic': self.critic.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'actor': self.actor.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict()}, filename)
        # torch.save(self.critic.state_dict(), filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        #
        # torch.save(self.actor.state_dict(), filename + "_actor")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)
