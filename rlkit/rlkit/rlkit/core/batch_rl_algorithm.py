import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.samplers.rollout_functions import multitask_rollout, rollout
from os import path as osp
from rlkit.core import logger
import numpy as np

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            dump_policy_video_interval=50,
            no_goal_env=None
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.dump_video_interval = dump_policy_video_interval

    def _train(self):
        print("batch RL algorithm starts training!")
        if self.min_num_steps_before_training > 0:
            # print("batch RL algorithm collect_new_paths!")
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print("Training epoch: ", epoch)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch): # defualt is just 1 
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

            # # add by yufei: test goal-conditioned policy, 
            # # save trained goal-conditioned policy video
            # # save returns
            # if epoch % self.dump_video_interval == 0:
            #     video_name = "{}.gif".format(epoch)
            #     imsize = self.expl_env.imsize
            #     old_decode_goals = self.eval_env.decode_goals
            #     env = self.eval_env
            #     env.decode_goals = False # the path collectors would set decode_goals to be true.
            #     # and once it is set to be true, image desired goals would be replaced as the decoded images from sampled latents.

            #     dump_path = logger.get_snapshot_dir()
            #     policy = self.trainer.policy
            #     dump_video(env, policy, osp.join(dump_path, video_name), rollout_function=video_rollout, imsize=imsize,
            #        horizon=self.max_path_length, rows=1, columns=3)

            #     self.eval_env.decode_goals = old_decode_goals
            
            # non_discounted_returns = []
            # policy = self.trainer.policy
            # old_goal_sampling_mode = self.eval_env.goal_sampling_mode
            # env = self.eval_env
            # env.goal_sampling_mode = 'reset_of_env'
            # for _ in range(20):
            #     path = multitask_rollout(
            #         env,
            #         policy,
            #         max_path_length=self.max_path_length,
            #         render=False,
            #         observation_key='latent_observation',
            #         desired_goal_key='latent_desired_goal'
            #     )

            #     env_infos = path['env_infos']
            #     true_rewards = [d['real_task_reward'] for d in env_infos]
            #     non_discounted_returns.append(np.sum(true_rewards))

            # logger.record_tabular('no_goal_env_return', np.mean(non_discounted_returns))
            # self.eval_env.goal_sampling_mode = old_goal_sampling_mode