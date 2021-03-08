import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from envs.env import  Env, EnvBatcher
from planet.memory import ExperienceReplay
from planet.models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel
from planet.planner import MPCPlanner
from planet.utils import write_video, transform_info

from chester import logger


class PlaNetAgent(object):
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
        self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(
            self.reward_model.parameters()) + list(self.encoder.parameters())
        if vv['use_value_function']:
            self.value_model = ValueModel(vv['belief_size'], vv['state_size'], vv['hidden_size'], vv['activation_function']).to(device=device)
            self.param_list.extend(self.value_model.parameters())
        else:
            self.value_model = None

        self.optimiser = optim.Adam(self.param_list, lr=0 if vv['learning_rate_schedule'] != 0 else vv['learning_rate'], eps=vv['adam_epsilon'])
        if vv['saved_models'] is not None and os.path.exists(vv['saved_models']):
            model_dicts = torch.load(vv['saved_models'])
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.observation_model.load_state_dict(model_dicts['observation_model'])
            self.reward_model.load_state_dict(model_dicts['reward_model'])
            self.encoder.load_state_dict(model_dicts['encoder'])
            self.optimiser.load_state_dict(model_dicts['optimiser'])
            if vv['use_value_function']:
                self.value_model.load_state_dict(model_dicts['value_model'])
            print('model loaded from ', vv['saved_models'])

        self.planner = MPCPlanner(self.dimu, vv['planning_horizon'], vv['optimisation_iters'], vv['candidates'], vv['top_candidates'],
                                  self.transition_model, self.reward_model, self.value_model)
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
                    action = self.env.sample_random_action()
                    next_observation, reward, done, info = self.env.step(action)
                    observations.append(observation), actions.append(action), rewards.append(reward), dones.append(done)
                    observation = next_observation
                    t += 1
                self.D.append_episode(observations, actions, rewards, dones)
                self.train_steps += t
            self.train_episodes += self.vv['seed_episodes']

    def update_belief_and_act(self, env, belief, posterior_state, action, observation, explore=False):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                                       self.encoder(observation).unsqueeze(
                                                                           dim=0))  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
        if explore:
            action = action + self.vv['action_noise'] * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        next_observation, reward, done, info = env.step(
            action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
        return belief, posterior_state, action, next_observation, reward, done, info

    def train(self, train_epoch, experience_replay_path=None):
        logger.info('Warming up ...')
        self._init_replay_buffer(experience_replay_path)
        logger.info('Start training ...')
        for epoch in tqdm(range(train_epoch)):
            # Model fitting
            losses = []
            for _ in tqdm(range(self.vv['collect_interval'])):
                # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
                if self.value_model is not None:
                    observations, actions, rewards, values, nonterminals = \
                        self.D.sample(self.vv['batch_size'], self.vv['chunk_size'])  # Transitions start at time t = 0
                else:
                    observations, actions, rewards, nonterminals = \
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
                # TODO check why the last one of the reward is not used!!
                if self.value_model is not None:
                    prev_beliefs = torch.cat([init_belief.unsqueeze(dim=0), beliefs[:-1, :, :]])
                    prev_states = torch.cat([init_state.unsqueeze(dim=0), posterior_states[:-1, :, :]])
                    target = (rewards[:-1] + bottle(self.value_model, (beliefs, posterior_states)).detach()) * nonterminals[:-1].squeeze(dim=2)
                    value_loss = F.mse_loss(bottle(self.value_model, (prev_beliefs, prev_states)), target, reduction='none').mean(dim=(0, 1))
                    value_target_average = target.mean().item()

                # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2),
                                    self.free_nats).mean(dim=(0, 1))

                if self.vv['global_kl_beta'] != 0:
                    kl_loss += self.vv['global_kl_beta'] * kl_divergence(Normal(posterior_means, posterior_std_devs),
                                                                         self.global_prior).sum(dim=2).mean(dim=(0, 1))
                # Calculate latent overshooting objective for t > 0
                if self.vv['overshooting_kl_beta'] != 0:
                    raise NotImplementedError  # Need to deal with value function
                    overshooting_vars = []  # Collect variables for overshooting to process in batch
                    for t in range(1, self.vv['chunk_size'] - 1):
                        d = min(t + self.vv['overshooting_distance'], self.vv['chunk_size'] - 1)  # Overshooting distance
                        t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                        # Calculate sequence padding so overshooting terms can be calculated in one batch
                        seq_pad = (0, 0, 0, 0, 0, t - d + self.vv['overshooting_distance'])
                        # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                        overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad),
                                                  F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_],
                                                  F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
                                                  F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                                                  F.pad(torch.ones(d - t, self.vv['batch_size'], self.vv['state_size'],
                                                                   device=self.device),
                                                        seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
                    overshooting_vars = tuple(zip(*overshooting_vars))
                    # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
                    beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(
                        torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1),
                        torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
                    seq_mask = torch.cat(overshooting_vars[7], dim=1)
                    # Calculate overshooting KL loss with sequence mask
                    # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
                    kl_loss += (1 / self.vv['overshooting_distance']) * self.vv['overshooting_kl_beta'] * torch.max((kl_divergence(
                        Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                        Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1)) * (self.vv['chunk_size'] - 1)
                    # Calculate overshooting reward prediction loss with sequence mask
                    if self.vv['overshooting_reward_scale'] != 0:
                        reward_loss += (1 / self.vv['overshooting_distance']) * self.vv['overshooting_reward_scale'] * F.mse_loss(
                            bottle(self.reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                            torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (
                                         self.vv[
                                             'chunk_size'] - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)

                # Apply linearly ramping learning rate schedule
                if self.vv['learning_rate_schedule'] != 0:
                    for group in self.optimiser.param_groups:
                        group['lr'] = min(group['lr'] + self.vv['learning_rate'] / self.vv['learning_rate_schedule'], self.vv['learning_rate'])
                # Update model parameters
                self.optimiser.zero_grad()
                if self.value_model is not None:
                    (observation_loss + reward_loss + value_loss + kl_loss).backward()
                else:
                    (observation_loss + reward_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.param_list, self.vv['grad_clip_norm'], norm_type=2)
                self.optimiser.step()
                # Store (0) observation loss (1) reward loss (2) KL loss
                losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])
                if self.value_model is not None:
                    losses[-1].append(value_loss.item())

            # Data collection
            with torch.no_grad():
                all_total_rewards = []  # Average across all episodes
                for i in range(self.vv['episodes_per_loop']):
                    observation, total_reward = self.env.reset(), 0
                    observations, actions, rewards, dones = [], [], [], []
                    belief, posterior_state, action = torch.zeros(1, self.vv['belief_size'], device=self.device), \
                                                      torch.zeros(1, self.vv['state_size'], device=self.device), \
                                                      torch.zeros(1, self.env.action_size, device=self.device)
                    pbar = tqdm(range(self.env.horizon))
                    for t in pbar:
                        belief, posterior_state, action, next_observation, reward, done, info = \
                            self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.device), explore=True)
                        observations.append(observation), actions.append(action.cpu()), rewards.append(reward), dones.append(done)
                        total_reward += reward
                        observation = next_observation
                        if done:
                            pbar.close()
                            break
                    self.D.append_episode(observations, actions, rewards, dones)
                    self.train_episodes += 1
                    self.train_steps += t
                    all_total_rewards.append(total_reward)

            # Log
            losses = np.array(losses)
            logger.record_tabular('observation_loss', np.mean(losses[:, 0]))
            logger.record_tabular('reward_loss', np.mean(losses[:, 1]))
            logger.record_tabular('kl_loss', np.mean(losses[:, 2]))
            if self.value_model is not None:
                logger.record_tabular('value_loss', np.mean(losses[:, 3]))
                logger.record_tabular('value_target_average', value_target_average)
            logger.record_tabular('train_rewards', np.mean(all_total_rewards))
            logger.record_tabular('num_episodes', self.train_episodes)
            logger.record_tabular('num_steps', self.train_steps)

            # Test model
            if epoch % self.vv['test_interval'] == 0:
                self.evaluate_model(eval_on_held_out=True)
                self.evaluate_model(eval_on_held_out=False)

            # Checkpoint models
            if epoch % self.vv['checkpoint_interval'] == 0:
                torch.save(
                    {'transition_model': self.transition_model.state_dict(),
                     'observation_model': self.observation_model.state_dict(),
                     'reward_model': self.reward_model.state_dict(),
                     'value_model': self.value_model.state_dict() if self.value_model is not None else None,
                     'encoder': self.encoder.state_dict(),
                     'optimiser': self.optimiser.state_dict()}, os.path.join(logger.get_dir(), 'models_%d.pth' % epoch))
                if self.vv['checkpoint_experience']:
                    torch.save(self.D,
                               os.path.join(logger.get_dir(), 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes
            logger.dump_tabular()

    def evaluate_model(self, eval_on_held_out):
        """
        :param eval_on_train: If test on training variations of the environments or the test variations of the environment
        :return:
        """
        if eval_on_held_out:
            prefix = 'train_'
        else:
            prefix = 'eval_'

        self.set_model_eval()
        if eval_on_held_out:
            self.env.eval_flag = True
        # Initialise parallelised test environments
        with torch.no_grad():
            all_total_rewards = []
            all_infos = []
            all_frames, all_frames_reconstr = [], []
            for _ in range(self.vv['test_episodes']):
                frames, frames_reconstr, infos = [], [], []
                observation, total_reward, observation_reconstr = self.env.reset(), 0, []
                belief, posterior_state, action = torch.zeros(1, self.vv['belief_size'], device=self.device), \
                                                  torch.zeros(1, self.vv['state_size'], device=self.device), \
                                                  torch.zeros(1, self.env.action_size, device=self.device)
                pbar = tqdm(range(self.env.horizon))
                for t in pbar:
                    belief, posterior_state, action, next_observation, reward, done, info = \
                        self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.device),
                                                   explore=True)
                    total_reward += reward
                    infos.append(info)
                    if not self.vv['symbolic_env']:  # Collect real vs. predicted frames for video
                        frames.append(observation)
                        frames_reconstr.append(self.observation_model(belief, posterior_state).cpu())
                    observation = next_observation
                    if done:
                        pbar.close()
                        break
                all_frames.append(frames)
                all_frames_reconstr.append(frames_reconstr)
                all_total_rewards.append(total_reward)
                all_infos.append(infos)

            all_frames = all_frames[:8]  # Only take the first 8 episodes to visualize
            all_frames_reconstr = all_frames_reconstr[:8]
            video_frames = []
            for i in range(len(all_frames[0])):
                frame = torch.cat([x[i] for x in all_frames])
                frame_reconstr = torch.cat([x[i] for x in all_frames_reconstr])
                video_frames.append(make_grid(torch.cat([frame, frame_reconstr], dim=3) + 0.5, nrow=4).numpy())

        # Update and plot reward metrics (and write video if applicable) and save metrics
        self.test_episodes += self.vv['test_episodes']

        logger.record_tabular(prefix + 'test_episodes', self.test_episodes)
        logger.record_tabular(prefix + 'test_rewards', np.mean(all_total_rewards))
        transformed_info = transform_info(all_infos)
        for info_name in transformed_info:
            logger.record_tabular(prefix + 'info_' + 'final_' + info_name, np.mean(transformed_info[info_name][:, -1]))
            logger.record_tabular(prefix + 'info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][:, :]))
            logger.record_tabular(prefix + 'info_' + 'sum_' + info_name, np.mean(np.sum(transformed_info[info_name][:, :], axis=-1)))
        if not self.vv['symbolic_env']:
            episode_str = str(self.train_episodes).zfill(len(str(self.train_episodes)))
            write_video(video_frames, prefix + 'test_episode_%s' % episode_str, logger.get_dir())  # Lossy compression
            save_image(torch.as_tensor(video_frames[-1]),
                       os.path.join(logger.get_dir(), prefix + 'test_episode_%s.png' % episode_str))
        self.set_model_train()
        if eval_on_held_out:
            self.env.eval_flag = False

    def set_model_train(self):
        """ Set model and env to train mode """
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        if self.value_model is not None:
            self.value_model.train()
        self.encoder.train()

    def set_model_eval(self):
        """ Set model and env to evaluation mode"""
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        if self.value_model is not None:
            self.value_model.eval()
        self.encoder.eval()
