import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from envs.env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from planet.memory import ExperienceReplay
from planet.utils import write_video

from dreamer.models import bottle, bottle3, Encoder, ObservationModel, RewardModel, TransitionModel, ActionModel, ValueModel

from chester import logger


class DreamerAgent(object):
    """
    TODO add documentation
    """

    def __init__(self, env, vv, device):
        self.env = env
        self.vv = vv
        self.device = device
        self.dimo, self.dimu = self.env.observation_size, self.env.action_size
        self.train_episodes = 0
        self.train_steps = 0
        self.test_episodes = 0
        self.D = None  # Replay buffer

        # Initialise model parameters randomly
        self.transition_model = TransitionModel(vv['belief_size'], vv['state_size'], self.dimu, vv['hidden_size'], vv['embedding_size'],
                                                vv['activation_function']).to(device=device)
        self.observation_model = ObservationModel(vv['symbolic_env'], self.dimo, vv['belief_size'], vv['state_size'], vv['embedding_size'],
                                                  vv['activation_function']).to(device=device)
        self.reward_model = RewardModel(vv['belief_size'], vv['state_size'], vv['hidden_size'], vv['activation_function']).to(device=device)

        self.encoder = Encoder(vv['symbolic_env'], self.dimo, vv['embedding_size'], vv['activation_function']).to(device=device)
        self.action_model = ActionModel(vv['belief_size'], vv['state_size'], vv['hidden_size'], self.dimu,
                                        vv['activation_function']).to(device=device)
        self.value_model = ValueModel(vv['belief_size'], vv['state_size'], vv['hidden_size'], vv['activation_function']).to(device=device)
        self.value_model_target = ValueModel(vv['belief_size'], vv['state_size'], vv['hidden_size'], vv['activation_function']).to(device=device)

        self.repr_param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(
            self.reward_model.parameters()) + list(self.encoder.parameters())
        self.ac_param_list = list(self.value_model.parameters()) + list(self.action_model.parameters())

        self.repr_optimiser = optim.Adam(self.repr_param_list, lr=vv['learning_rate_repr'], eps=vv['adam_epsilon'])
        self.ac_optimiser = optim.Adam(self.ac_param_list, lr=vv['learning_rate_ac'], eps=vv['adam_epsilon'])

        if vv['saved_models'] is not None and os.path.exists(vv['saved_models']):
            model_dicts = torch.load(vv['saved_models'])
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.observation_model.load_state_dict(model_dicts['observation_model'])
            self.reward_model.load_state_dict(model_dicts['reward_model'])
            self.encoder.load_state_dict(model_dicts['encoder'])
            self.repr_optimiser.load_state_dict(model_dicts['repr_optimiser'])
            self.ac_optimiser.load_state_dict(model_dicts['ac_optimiser'])
            self.value_model.load_state_dict(model_dicts['value_model'])
            print('model loaded from ', vv['saved_models'])

        self.global_prior = Normal(torch.zeros(vv['batch_size'], vv['state_size'], device=device),
                                   torch.ones(vv['batch_size'], vv['state_size'], device=device))  # Global prior N(0, I)
        self.free_nats = torch.full((1,), vv['free_nats'], device=device)  # Allowed deviation in KL divergence

    def _init_replay_buffer(self, experience_replay_path=None):
        if experience_replay_path is not None:
            self.D = torch.load(experience_replay_path)
            # metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
            self.train_episodes += self.D.episodes
            self.train_steps += self.D.steps

        else:
            self.D = ExperienceReplay(self.vv['experience_size'], self.vv['symbolic_env'], self.dimo, self.dimu,
                                      self.vv['bit_depth'], self.device, self.vv['use_value_function'])
            # Initialize dataset D with S random seed episodes
            for s in range(1, self.vv['seed_episodes'] + 1):
                observation, done, t = self.env.reset(), False, 0
                observations, actions, rewards, dones = [], [], [], []
                while not done:
                    action = self.env.sample_random_action()  # TODO is there a reason for using tensor instead of numpy here?
                    next_observation, reward, done = self.env.step(action)
                    observations.append(observation), actions.append(action), rewards.append(reward), dones.append(done)
                    observation = next_observation
                    t += 1
                self.D.append_episode(observations, actions, rewards, dones)
                self.train_steps += t * self.vv['action_repeat']
            self.train_episodes += self.vv['seed_episodes']

    def update_belief_and_act(self, env, belief, posterior_state, action, observation, explore=False):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                                       self.encoder(observation).unsqueeze(
                                                                           dim=0))  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        action = self.action_model.act(belief, posterior_state, deterministic=True)  # Get action from the policy

        if explore:
            action = action + self.vv['action_noise'] * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        next_observation, reward, done = env.step(
            action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
        return belief, posterior_state, action, next_observation, reward, done

    @staticmethod
    def compute_lambda_return(rewards, values, lda, gamma):
        H = rewards.shape[1]
        ret = np.zeros(shape=values.shape, dtype=np.float)
        ret[:, -1] = values[:, -1]
        for i in reversed(range(H - 1)):
            ret[:, i] = rewards[:, i + 1] + (1 - lda) * gamma * values[:, i + 1] + lda * gamma * ret[:, i + 1]
        return ret

    def train(self, train_episode, render=False, experience_replay_path=None):
        logger.info('Warming up ...')
        self._init_replay_buffer(experience_replay_path)
        logger.info('Start training ...')
        for episode in tqdm(range(self.train_episodes, train_episode), total=train_episode, initial=self.train_episodes):
            # Model fitting
            losses = []
            for _ in tqdm(range(self.vv['collect_interval'])):
                # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
                observations, actions, rewards, values, nonterminals = \
                    self.D.sample(self.vv['batch_size'], self.vv['chunk_size'])  # Transitions start at time t = 0
                # Create initial belief and state for time t = 0
                init_belief, init_state = torch.zeros(self.vv['batch_size'], self.vv['belief_size'], device=self.device), \
                                          torch.zeros(self.vv['batch_size'], self.vv['state_size'], device=self.device)
                # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
                beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(
                    init_state, actions[:-1], init_belief, bottle(self.encoder, (observations[1:],)), nonterminals[:-1])
                # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
                observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:],
                                              reduction='none').sum(dim=2 if self.vv['symbolic_env'] else (2, 3, 4)).mean(dim=(0, 1))
                reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))

                # prev_beliefs = torch.cat([init_belief.unsqueeze(dim=0), beliefs[:-1, :, :]])
                # prev_states = torch.cat([init_state.unsqueeze(dim=0), posterior_states[:-1, :, :]])

                # Update representation parameters
                # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2),
                                    self.free_nats).mean(dim=(0, 1))

                if self.vv['global_kl_beta'] != 0:
                    kl_loss += self.vv['global_kl_beta'] * kl_divergence(Normal(posterior_means, posterior_std_devs),
                                                                         self.global_prior).sum(dim=2).mean(dim=(0, 1))
                self.repr_optimiser.zero_grad()
                (observation_loss + reward_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.repr_param_list, self.vv['grad_clip_norm'], norm_type=2)
                self.repr_optimiser.step()

                losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])  # Logging the losses
                # Update actor-critic parameters
                H = self.vv['imagine_horizon']
                beliefs = beliefs.view((self.vv['chunk_size'] - 1) * self.vv['batch_size'], self.vv['belief_size'])
                posterior_states = posterior_states.view((self.vv['chunk_size'] - 1) * self.vv['batch_size'], self.vv['state_size'])

                beliefs, states, actions = self.transition_model.imagine(posterior_states, beliefs, self.action_model, self.vv['imagine_horizon'])

                beliefs = beliefs.view(H, self.vv['batch_size'], self.vv['chunk_size'] - 1, self.vv['belief_size'])
                states = states.view(H, self.vv['batch_size'], self.vv['chunk_size'] - 1, self.vv['state_size'])

                rewards = bottle3(self.reward_model, (beliefs, states)).transpose(0, 2).reshape(self.vv['batch_size'] * (self.vv['chunk_size'] - 1),
                                                                                                H)
                values = bottle3(self.value_model, (beliefs, states)).transpose(0, 2).reshape(self.vv['batch_size'] * (self.vv['chunk_size'] - 1), H)

                values_target = self.compute_lambda_return(rewards.detach().cpu().numpy(), values.detach().cpu().numpy(), self.vv['lambda'],
                                                           self.vv['gamma'])
                values_target = torch.from_numpy(values_target).float().to(device=self.device).detach()

                print('belief size: {}, state size: {}, action size: {}'.format(beliefs.size(), states.size(), actions.size()))
                # Should be (batch_size x chunk_size x horizon)

                value_loss = F.mse_loss(values, values_target, reduction='none').mean(dim=(0, 1))
                value_target_average = values.mean().item()
                losses[-1].append(value_loss.item())
                actor_loss = -values_target.mean(dim=(0, 1))
                losses[-1].append(actor_loss.item())
                self.ac_optimiser.zero_grad()
                (actor_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.ac_param_list, self.vv['grad_clip_norm'], norm_type=2)
                self.ac_optimiser.step()

            # Data collection (1 episode)
            with torch.no_grad():
                observation, total_reward = self.env.reset(), 0
                observations, actions, rewards, dones = [], [], [], []
                belief, posterior_state, action = torch.zeros(1, self.vv['belief_size'], device=self.device), \
                                                  torch.zeros(1, self.vv['state_size'], device=self.device), \
                                                  torch.zeros(1, self.env.action_size, device=self.device)
                pbar = tqdm(range(self.vv['max_episode_length'] // self.vv['action_repeat']))
                for t in pbar:
                    belief, posterior_state, action, next_observation, reward, done = \
                        self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.device), explore=True)
                    observations.append(observation), actions.append(action.cpu()), rewards.append(reward), dones.append(done)
                    total_reward += reward
                    observation = next_observation
                    if render:
                        self.env.render()
                    if done:
                        pbar.close()
                        break
                self.D.append_episode(observations, actions, rewards, dones)
                self.train_episodes += 1
                self.train_steps += t

            # Log
            losses = np.array(losses)
            logger.record_tabular('observation_loss', np.mean(losses[:, 0]))
            logger.record_tabular('reward_loss', np.mean(losses[:, 1]))
            logger.record_tabular('kl_loss', np.mean(losses[:, 2]))
            logger.record_tabular('value_loss', np.mean(losses[:, 3]))
            logger.record_tabular('value_target_average', value_target_average)
            logger.record_tabular('actor_loss', np.mean(losses[:, 4]))
            logger.record_tabular('train_rewards', total_reward)
            logger.record_tabular('num_episodes', self.train_episodes)
            logger.record_tabular('num_steps', self.train_steps)

            # Test model
            if episode % self.vv['test_interval'] == 0:
                self.set_model_eval()
                # Initialise parallelised test environments
                with torch.no_grad():
                    all_frames, all_frames_reconstr = [], []
                    for _ in range(self.vv['test_episodes']):
                        frames, frames_reconstr = [], []
                        observation, total_reward, observation_reconstr = self.env.reset(), 0, []
                        belief, posterior_state, action = torch.zeros(1, self.vv['belief_size'], device=self.device), \
                                                          torch.zeros(1, self.vv['state_size'], device=self.device), \
                                                          torch.zeros(1, self.env.action_size, device=self.device)
                        pbar = tqdm(range(self.vv['max_episode_length'] // self.vv['action_repeat']))
                        for t in pbar:
                            belief, posterior_state, action, next_observation, reward, done = \
                                self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.device),
                                                           explore=True)
                            total_reward += reward
                            if not self.vv['symbolic_env']:  # Collect real vs. predicted frames for video
                                frames.append(observation)
                                frames_reconstr.append(self.observation_model(belief, posterior_state).cpu())
                            observation = next_observation
                            if render:
                                self.env.render()
                            if done:
                                pbar.close()
                                break
                        # frames = torch.cat([x for x in frames], dim=0)
                        # frames_reconstr = torch.cat([x for x in frames_reconstr], dim=0)
                        all_frames.append(frames)
                        all_frames_reconstr.append(frames_reconstr)

                    video_frames = []
                    for i in range(len(all_frames[0])):
                        frame = torch.cat([x[i] for x in all_frames])
                        frame_reconstr = torch.cat([x[i] for x in all_frames_reconstr])
                        video_frames.append(make_grid(torch.cat([frame, frame_reconstr], dim=3) + 0.5, nrow=4).numpy())

                # Update and plot reward metrics (and write video if applicable) and save metrics
                self.test_episodes += self.vv['test_episodes']

                logger.record_tabular('test_episodes', self.test_episodes)
                logger.record_tabular('test_rewards', total_reward)
                if not self.vv['symbolic_env']:
                    episode_str = str(self.train_episodes).zfill(len(str(train_episode)))
                    write_video(video_frames, 'test_episode_%s' % episode_str, logger.get_dir())  # Lossy compression
                    save_image(torch.as_tensor(video_frames[-1]),
                               os.path.join(logger.get_dir(), 'test_episode_%s.png' % episode_str))
                self.set_model_train()

            # Checkpoint models
            if episode % self.vv['checkpoint_interval'] == 0:
                torch.save(
                    {'transition_model': self.transition_model.state_dict(),
                     'observation_model': self.observation_model.state_dict(),
                     'reward_model': self.reward_model.state_dict(),
                     'value_model': self.value_model.state_dict() if self.value_model is not None else None,
                     'encoder': self.encoder.state_dict(),
                     'repr_optimiser': self.repr_optimiser.state_dict(),
                     'ac_optimiser': self.ac_optimiser.state_dict()}, os.path.join(logger.get_dir(), 'models_%d.pth' % episode))
                if self.vv['checkpoint_experience']:
                    torch.save(self.D,
                               os.path.join(logger.get_dir(), 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes
            logger.dump_tabular()

    def set_model_train(self):
        """ Set model to train mode """
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.value_model.train()
        self.action_model.train()
        self.encoder.train()

    def set_model_eval(self):
        """ Set model to evaluation mode"""
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.value_model.eval()
        self.action_model.eval()
        self.encoder.eval()
