import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
from rlkit.samplers.rollout_functions import multitask_rollout, rollout
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.util.video import dump_video, dump_video_non_goal
from os import path as osp
import numpy as np
from matplotlib import pyplot as plt

def plot_latent_dist(env, policy, rollout_function, horizon, N=10, save_name='./plot.png'):
    '''
    doc
    '''
    all_latent_achieved_goal = []
    all_lateng_goal = []
    for _ in range(N):
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )

        latent_achieved_goal = np.array([d['latent_achieved_goal'] for d in path['full_observations']])
        latent_goal = np.array([d['latent_desired_goal'] for d in path['full_observations']])
        all_latent_achieved_goal.append(latent_achieved_goal)
        all_lateng_goal.append(latent_goal)
    all_latent_achieved_goal = np.array(all_latent_achieved_goal)
    all_lateng_goal = np.array(all_lateng_goal)
    dist = np.linalg.norm(all_latent_achieved_goal - all_lateng_goal, axis=-1)
    mean_dist = np.mean(dist, axis=0)
    std_dist = np.std(dist, axis=0)

    markers, caps, bars = plt.errorbar(list(range(horizon)), mean_dist, std_dist) #, colors[i], label = labels[i])
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    plt.xlabel('Episode time')
    plt.ylabel('VAE distance to goal')
    plt.savefig(save_name)

def video_multitask_rollout(*args, **kwargs):
    return multitask_rollout(*args, **kwargs,
                             observation_key='latent_observation',
                             desired_goal_key='latent_desired_goal', )


def video_rollout(*args, **kwargs):
    return rollout(*args, **kwargs)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
      self,
      trainer,
      exploration_env,
      evaluation_env,
      exploration_data_collector: DataCollector,
      evaluation_data_collector: DataCollector,
      replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        # add by yufei: test goal-conditioned policy, 
        # save trained goal-conditioned policy video
        # save returns
        if epoch % self.dump_video_interval == 0:
            video_name = "{}.gif".format(epoch)
            imsize = self.expl_env.imsize
            env = self.eval_env

            if hasattr(self.eval_env, 'decode_goals'):
                old_decode_goals = self.eval_env.decode_goals
                env = self.eval_env
                env.decode_goals = False  # the path collectors would set decode_goals to be true.
                # and once it is set to be true, image desired goals would be replaced as the decoded images from sampled latents.

            dump_path = logger.get_snapshot_dir()
            policy = self.trainer.policy
            
            if isinstance(env, VAEWrappedEnv):  # indicates skewfit and multitask env
                rollout_func = video_multitask_rollout
            else:
                rollout_func = video_rollout

            if hasattr(self.eval_env, 'decode_goals'):
                dump_video(env, policy, osp.join(dump_path, video_name), rollout_function=rollout_func, imsize=imsize,
                           horizon=self.max_path_length, rows=2, columns=4)
                video_name_2 = "{}_deterministic.gif".format(epoch)
                dump_video(env, self.eval_data_collector._policy, osp.join(dump_path, video_name_2), rollout_function=rollout_func, imsize=imsize,
                           horizon=self.max_path_length, rows=2, columns=4)

                latent_distance_name1 = '{}_latent_distance'.format(epoch)
                latent_distance_name2 = '{}_deterministic_latent_distance'.format(epoch)
                plot_latent_dist(env, policy, save_name=osp.join(dump_path, latent_distance_name1), rollout_function=rollout_func, 
                           horizon=self.max_path_length)
                plot_latent_dist(env, self.eval_data_collector._policy, save_name=osp.join(dump_path, latent_distance_name2), rollout_function=rollout_func, 
                           horizon=self.max_path_length)

                self.eval_env.decode_goals = old_decode_goals
            else:
                dump_video_non_goal(env, self.expl_data_collector._policy, osp.join(dump_path, video_name), rollout_function=rollout_func, imsize=imsize,
                                    horizon=self.max_path_length, rows=2, columns=4)
                video_name_2 = "{}_deterministic.gif".format(epoch)
                dump_video_non_goal(env, self.eval_data_collector._policy, osp.join(dump_path, video_name_2), rollout_function=rollout_func, imsize=imsize,
                           horizon=self.max_path_length, rows=2, columns=4)

        non_discounted_returns = []
        policy = self.trainer.policy
        if hasattr(self.eval_env, 'goal_sampling_mode'):
            old_goal_sampling_mode = self.eval_env.goal_sampling_mode
        env = self.eval_env
        if hasattr(self.eval_env, 'goal_sampling_mode'):
            env.goal_sampling_mode = 'reset_of_env'
        for _ in range(10):
            if isinstance(env, VAEWrappedEnv):
                path = multitask_rollout(
                    env,
                    policy,
                    max_path_length=self.max_path_length,
                    render=False,
                    observation_key='latent_observation',
                    desired_goal_key='latent_desired_goal'
                )
            else:
                path = rollout(
                    env, policy, max_path_length=self.max_path_length, render=False
                )

            env_infos = path['env_infos']
            if env_infos[0].get('performance') is None:
                break

            true_rewards = [d['performance'] for d in env_infos]
            non_discounted_returns.append(np.sum(true_rewards))

        if len(non_discounted_returns) > 0:
            logger.record_tabular('performance', np.mean(non_discounted_returns))
        if hasattr(self.eval_env, 'goal_sampling_mode'):
            self.eval_env.goal_sampling_mode = old_goal_sampling_mode

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
