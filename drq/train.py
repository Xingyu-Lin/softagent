import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np

# import dmc2gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import drq.utils as utils
from drq.logger import Logger
from drq.replay_buffer import ReplayBuffer
from chester import logger
import yaml
import json
from drq.Drq import DRQAgent
from softgym.utils.visualization import save_numpy_as_gif, make_grid
import os
from matplotlib import pyplot as plt
from experiments.planet.train import update_env_kwargs
from envs.env import Env

torch.backends.cudnn.benchmark = True

def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    default_cfg = yaml.load(open('drq/config.yml', 'r'))
    cfg = update_config(default_cfg, vv)
    cfg = update_env_kwargs(cfg)
    workspace = Workspace(vv_to_args(cfg))
    workspace.run()

    # main(vv)

def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict

def make_env(args):
    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.im_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = logger.get_dir()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent='drq',
                             action_repeat=1,
                             chester_log=logger)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        obs_shape = self.env.observation_space.shape
        new_obs_shape = np.zeros_like(obs_shape)
        new_obs_shape[0] = obs_shape[-1]
        new_obs_shape[1] = obs_shape[0]
        new_obs_shape[2] = obs_shape[1]
        cfg.agent['obs_shape'] = cfg.encoder['obs_shape'] = new_obs_shape
        cfg.agent['action_shape'] = self.env.action_space.shape
        cfg.actor['action_shape'] = self.env.action_space.shape
        cfg.critic['action_shape'] = self.env.action_space.shape
        cfg.actor['encoder_cfg'] = cfg.encoder
        cfg.critic['encoder_cfg'] = cfg.encoder
        cfg.agent['action_range'] = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        cfg.agent['encoder_cfg'] = cfg.encoder
        cfg.agent['critic_cfg'] = cfg.critic
        cfg.agent['actor_cfg'] = cfg.actor

        self.agent = DRQAgent(**cfg.agent)

        self.replay_buffer = ReplayBuffer(new_obs_shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_video else None)
        self.step = 0
        self.video_dir = os.path.join(self.work_dir, 'video')
        self.model_dir = os.path.join(self.work_dir, 'model')
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, exist_ok=True)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def evaluate(self):
        average_episode_reward = 0
        infos = []
        all_frames = []
        plt.figure()

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            # print(type(obs))
            # print(obs.shape)
            # print(obs)
            # exit()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            ep_info = []
            frames = [self.env.get_image(128, 128)]
            rewards = []

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                # self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1
                ep_info.append(info)
                frames.append(self.env.get_image(128, 128))
                rewards.append(reward)

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
            infos.append(ep_info)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)

        average_episode_reward /= self.cfg.num_eval_episodes
        for key, val in get_info_stats(infos).items():
            self.logger.log('eval/info_' + key, val, self.step)

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(self.video_dir, '%d.gif' % self.step))
        plt.savefig(os.path.join(self.video_dir, '%d.png' % self.step))


    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        ep_info = []
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            print("step: ", self.step)


            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()
                if self.cfg.save_model and self.step % (self.cfg.eval_frequency *5):
                    self.agent.save(self.model_dir, self.step)

            if done:
                if self.step > 0:
                    if self.step % self.cfg.log_interval == 0:
                        self.logger.log('train/duration',
                                        time.time() - start_time, self.step)
                        for key, val in get_info_stats([ep_info]).items():
                            self.logger.log('train/info_' + key, val, self.step)
                        self.logger.dump(
                            self.step, save=(self.step > self.cfg.num_seed_steps))

                    start_time = time.time()

                if self.step % self.cfg.log_interval == 0:
                    self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                ep_info = []

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            done_no_max = 0
            episode_reward += reward
            ep_info.append(info)

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            # print(self.step)


def update_config(default_config, luanch_config):
    import copy
    for key in luanch_config:
        default_config_ = default_config
        now_key = copy.deepcopy(key)
        idx = now_key.find('.')
        while idx != -1:
            default_config_ = default_config_[now_key[:idx]]
            now_key = now_key[idx+1:]
            idx = now_key.find('.')

        default_config_[now_key] = luanch_config[key]

    return default_config

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args
