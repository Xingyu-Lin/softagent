import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from matplotlib import pyplot as plt
import time
import math

from pouring.KPConv_sac import utils
from pouring.KPConv_sac.architectures import KPCNN, KPCNN_actor, KPCNN_critic, KPCNN_Encoder


class KPConvSacAgent(object):
    """SAC with KPConv as policy and Q-network"""

    def __init__(
      self,
      KPconv_config,
      device,
      args,
      discount=0.99,
      init_temperature=0.01,
      alpha_lr=1e-3,
      alpha_beta=0.9,
      alpha_fixed=False,
      actor_lr=1e-3,
      actor_beta=0.9,
      actor_update_freq=2,
      critic_lr=1e-3,
      critic_beta=0.9,
      critic_tau=0.005,
      critic_target_update_freq=2,
      log_interval=100,
    ):
        self.args = args
        self.KPconv_config = KPconv_config
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.alpha_fixed = alpha_fixed

        self.actor_encoder = KPCNN_Encoder(KPconv_config)
        self.actor = KPCNN_actor(KPconv_config, self.actor_encoder).to(device)

        self.q_encoder_1 = KPCNN_Encoder(KPconv_config)
        self.q_encoder_2 = KPCNN_Encoder(KPconv_config)
        self.critic = KPCNN_critic(KPconv_config, self.q_encoder_1, self.q_encoder_2).to(device)

        self.q_encoder_target_1 = KPCNN_Encoder(KPconv_config)
        self.q_encoder_target_2 = KPCNN_Encoder(KPconv_config)
        self.critic_target = KPCNN_critic(KPconv_config, self.q_encoder_target_1, self.q_encoder_target_2).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod([KPconv_config.action_dim])

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)

        self.train()
        self.critic_target.train()

    # def print_module(self, module):
    #     print(module.__class__.__name__)
    #     print(module.training)
    #     for module in module.children():
    #         self.print_module(module)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device)
            # obs = obs.unsqueeze(0)
            # print("in selection action, obs shape is: ", obs.points.shape)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device)
            # obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        reduced_states = obs[1]
        obs = obs[0]

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
            # comp_Q = reward + (not_done * self.discount * torch.min(target_Q1, target_Q2))
            # comp_E = reward + (not_done * self.discount * (- self.alpha.detach() * log_pi))

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action)

        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        
        rs_loss = torch.zeros(1).to(self.device)
        if self.args.rs_loss_coef > 0:
            rs_prediction_1, rs_prediction_2 = self.critic.predict_reduced_state(obs)
            rs_loss = F.mse_loss(reduced_states, rs_prediction_1) + F.mse_loss(reduced_states, rs_prediction_2)
            
        loss = critic_loss + self.args.rs_loss_coef * rs_loss

        if step % self.log_interval == 0:
            L.log('train_critic/critic_loss', critic_loss, step)
            L.log('train_critic/rs_loss', rs_loss, step)
            L.log('train_critic/total_loss', loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        if self.args.lr_decay is not None:
            self.critic_lr_scheduler.step()
            L.log('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], step)

        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        reduced_states = obs[1]
        obs = obs[0]

        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        rs_loss = torch.zeros(1).to(self.device)
        if self.args.rs_loss_coef > 0:
            rs_prediction = self.actor.predict_reduced_state(obs)
            rs_loss = F.mse_loss(reduced_states, rs_prediction)

        loss = actor_loss + self.args.rs_loss_coef * rs_loss

        if step % self.log_interval == 0:
            L.log('train_actor/total_loss', loss, step)
            L.log('train_actor/actor_loss', actor_loss, step)
            L.log('train_actor/rs_loss', rs_loss, step)

        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)


        # self.actor.log(L, step)

        # if step % self.log_interval == 0:
        #     actor_stats = get_optimizer_stats(self.actor_optimizer)
        #     for key, val in actor_stats.items():
        #         L.log('train/actor_optim/' + key, val, step)


        if not self.alpha_fixed:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_pi - self.target_entropy).detach()).mean()
            if step % self.log_interval == 0:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):

        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        start_time = time.time()
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        # print('critic update time:', time.time() - start_time)
        if step % self.actor_update_freq == 0:
            start_time = time.time()
            self.update_actor_and_alpha(obs, L, step)
            # print('actor update time:', time.time() - start_time)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        
        if self.encoder_type == 'pixel':    
            self.CURL.load_state_dict(
                torch.load('%s/curl_%s.pt' % (model_dir, step))
            )