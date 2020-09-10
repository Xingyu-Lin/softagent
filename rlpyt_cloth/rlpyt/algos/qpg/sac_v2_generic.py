
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done


OptInfo = namedtuple("OptInfo",
    ["q1Loss", "q2Loss", "piLoss", "alphaLoss",
    "q1GradNorm", "q2GradNorm", "piGradNorm",
    "q1", "q2", "alpha", "qMeanDiff"])
SamplesToBuffer = namedarraytuple("SamplesToRepay",
    ["observation", "action", "reward", "done"])


class SAC(RlAlgorithm):

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=256,
            min_steps_learn=int(1e4),
            replay_size=int(1e6),
            replay_ratio=256,  # data_consumption / data_generation
            target_update_tau=0.005,  # tau=1 for hard update.
            target_update_interval=1,  # interval=1000 for hard update.
            learning_rate=3e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,  # for pi only.
            action_prior="uniform",  # or "gaussian"
            policy_output_regularization=0.001,
            reward_scale=1,
            reparameterize=True,
            clip_grad_norm=1e9,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only.
            target_entropy='auto',
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        assert action_prior in ["uniform", "gaussian"]
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Used in basic or synchronous multi-GPU runners, not async."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = int(self.replay_ratio * sampler_bs /
            self.batch_size)
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

        if self.target_entropy == 'auto':
            self.target_entropy = -np.prod(self.agent.env_spaces.action.shape)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only."""
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        agent.give_min_itr_learn(self.min_itr_learn)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called by async runner."""
        self.rank = rank
        self.pi_optimizer = self.OptimCls(self.agent.pi_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q1_optimizer = self.OptimCls(self.agent.q1_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q2_optimizer = self.OptimCls(self.agent.q2_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.alpha_optimizer = self.OptimCls([self.agent.log_alpha],
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.pi_optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.action_prior == "gaussian":
            self.action_prior_distribution = Gaussian(
                dim=self.agent.env_spaces.action.size, std=1.)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        ReplayCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            losses, values = self.loss(samples_from_replay)
            q1_loss, q2_loss, pi_loss, alpha_loss = losses

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(),
                self.clip_grad_norm)
            self.pi_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(),
                self.clip_grad_norm)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(),
                self.clip_grad_norm)
            self.q2_optimizer.step()

            grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        return opt_info

    def samples_to_buffer(self, samples):
        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )

    def loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""
        agent_inputs, target_inputs, action = buffer_to(
            (samples.agent_inputs, samples.target_inputs, samples.action))
        q1, q2 = self.agent.q(*agent_inputs, action)
        with torch.no_grad():
            target_v = self.agent.target_v(*target_inputs).detach()
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * samples.return_ +
             (1 - samples.done_n.float()) * disc * target_v)
        if self.mid_batch_reset and not self.agent.recurrent:
            valid = None  # OR: torch.ones_like(samples.done, dtype=torch.float)
        else:
            valid = valid_from_done(samples.done)

        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)

        new_action, log_pi, _ = self.agent.pi(*agent_inputs)
        if not self.reparameterize:
            new_action = new_action.detach()  # No grad.
        log_target1, log_target2 = self.agent.q(*agent_inputs, new_action)
        min_log_target = torch.min(log_target1, log_target2)
        prior_log_pi = self.get_action_prior(new_action.cpu())

        if self.reparameterize:
            alpha = self.agent.log_alpha.exp().detach()
            pi_losses = alpha * log_pi - min_log_target - prior_log_pi

        pi_loss = valid_mean(pi_losses, valid)

        # Calculate log_alpha loss
        alpha_loss = -valid_mean(self.agent.log_alpha * (log_pi + self.target_entropy).detach())

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, alpha))
        return losses, values


    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        elif self.action_prior == "gaussian":
            prior_log_pi = self.action_prior_distribution.log_likelihood(
                action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """In-place."""
        q1_loss, q2_loss, pi_loss, alpha_loss = losses
        q1_grad_norm, q2_grad_norm, pi_grad_norm = grad_norms
        q1, q2, alpha = values
        opt_info.q1Loss.append(q1_loss.item())
        opt_info.q2Loss.append(q2_loss.item())
        opt_info.piLoss.append(pi_loss.item())
        opt_info.alphaLoss.append(alpha_loss.item())
        opt_info.q1GradNorm.append(q1_grad_norm)
        opt_info.q2GradNorm.append(q2_grad_norm)
        opt_info.piGradNorm.append(pi_grad_norm)
        opt_info.q1.extend(q1[::10].numpy())  # Downsample for stats.
        opt_info.q2.extend(q2[::10].numpy())
        opt_info.alpha.append(alpha.numpy())
        opt_info.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())

    def optim_state_dict(self):
        return dict(
            pi_optimizer=self.pi_optimizer.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        self.pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        self.q1_optimizer.load_state_dict(state_dict["q1_optimizer"])
        self.q2_optimizer.load_state_dict(state_dict["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
