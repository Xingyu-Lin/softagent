# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import os, time

# my imports
from PDDM import trajectory_sampler

def get_return(env, actions, init_state):
    '''
    actions: [N x T x action_dim].
    init_state: the initial state of the rollout.
    '''

    N = actions.shape[0]
    T = actions.shape[1]
    # print("traj_num: ", N)
    beg = time.time()
    returns = []
    for i in range(N):
        env.set_state(init_state)
        ret = 0
        for action in actions[i]:
            _, reward, done, _ = env.step(action)
            ret += reward
            if done:
                break
        returns.append(ret)
    # print("time cost: ", time.time() - beg)
    return returns

def get_return_env(args):
    env_class, env_kwargs, actions, config_id, init_state = args
    env = env_class(**env_kwargs)
    env.reset(config_id=config_id)
    r = get_return(env, actions, init_state)
    return r, os.getpid()

def get_return_mp(env_class, env_kwargs, actions, config_id, init_state, num_worker):
    pool = Pool(processes=num_worker)
    per_cpu_num = len(actions) // num_worker
    actions_per_cpu = []
    for i in range(num_worker):
        actions_per_cpu.append(actions[i*per_cpu_num:(i+1)*per_cpu_num])
    actions_per_cpu.append(actions[num_worker*per_cpu_num:])

    rs = pool.map(get_return_env, [(env_class, env_kwargs, a, config_id, init_state) for a in actions_per_cpu])
    rets = [r[0] for r in rs]
    pids = [r[1] for r in rs]
    returns = []
    for r in rets:
        returns += r
    for pid in pids:
        os.system('kill -9 {}'.format(pid))
    return returns

class MPPI(object):

    def __init__(self, env, horizon, N, gamma, sigma, beta, action_correlation=True, env_class=None, env_kwargs=None):
        """
        horizon: planning horizon
        N: # of sampled trajectories
        gamma: gamma in exp for computing weights
        sigma: std for Gaussian noise
        beta: smoothing coefficient.
        """

        ###########
        ## params
        ###########
        self.horizon = horizon ### planning horizon
        self.N = N ### sampling action samples
        self.env = env

        #############
        ## init mppi vars
        #############
        self.ac_dim = self.env.action_space.shape[0]
        self.mppi_kappa = gamma # gamma in the paper
        self.sigma = sigma * np.ones(self.ac_dim) ### guassian noise sigma
        self.beta = beta
        self.mppi_mean = np.zeros((self.horizon, self.ac_dim))  #start mean at 0
        self.action_correlation = action_correlation
        self.env_kwargs = env_kwargs
        self.env_class = env_class


    ###################################################################
    ###################################################################
    #### update action mean using weighted average of the actions (by their resulting scores)
    ###################################################################
    ###################################################################

    def mppi_update(self, scores, all_samples):

        #########################
        ## how each sim's score compares to the best score
        ##########################
        S = np.exp(self.mppi_kappa * (scores - np.max(scores)))  # [N,]
        denom = np.sum(S) + 1e-10

        ##########################
        ## weight all actions of the sequence by that sequence's resulting reward
        ##########################
        S_shaped = np.expand_dims(np.expand_dims(S, 1), 2)  #[N,1,1]
        weighted_actions = (all_samples * S_shaped)  #[N x H x acDim]
        self.mppi_mean = np.sum(weighted_actions, 0) / denom

        ##########################
        ## return 1st element of the mean, which corresps to curr timestep
        ##########################
        return self.mppi_mean[0]

    def get_action(self, env_config_id=0):
        # record initial env state
        env_initial_state = self.env.get_state()

        # remove the 1st entry of mean (mean from past timestep, that was just executed)
        # and copy last entry (starting point, for the next timestep)
        past_action = self.mppi_mean[0].copy()
        self.mppi_mean[:-1] = self.mppi_mean[1:]

        ##############################################
        ## sample candidate action sequences
        ## by creating smooth filtered trajecs (noised around a mean)
        ##############################################

        np.random.seed()  # to get different action samples for each rollout

        #sample noise from normal dist, scaled by sigma
        eps = np.random.normal(
            loc=0, scale=1.0, size=(self.N, self.horizon,
                                    self.ac_dim)) * self.sigma

        # actions = mean + noise... then smooth the actions temporally
        all_samples = eps.copy() # size [N, horizon, action_dim]
        if self.action_correlation:
            for i in range(self.horizon): ### this looks slightly different from the paper.
                if(i==0):
                    all_samples[:, i, :] = self.beta*(self.mppi_mean[i, :] + eps[:, i, :]) + (1-self.beta)*past_action
                else:
                    all_samples[:, i, :] = self.beta*(self.mppi_mean[i, :] + eps[:, i, :]) + (1-self.beta)*all_samples[:, i-1, :]
        else:
            all_samples += self.mppi_mean

        # The resulting candidate action sequences:
        # all_samples : [N, horizon, ac_dim]
        # TODO yufei: remember to normalize the environment
        all_samples = np.clip(all_samples, -1, 1)

        ########################################################################
        ### make each action element be (past K actions) instead of just (curr action)
        ########################################################################

        #all_samples : [N, horizon, ac_dim]

        #################################################
        ### Get result of executing those candidate action sequences
        #################################################

        # returns = get_return(self.env, all_samples, env_initial_state)
        returns = get_return_mp(self.env_class, self.env_kwargs, all_samples, env_config_id, env_initial_state, 5)
        selected_action = self.mppi_update(returns, all_samples)
        
        # recover to initial env state
        self.env.set_state(env_initial_state)

        return selected_action
    
    def reset(self):
        self.mppi_mean = np.zeros((self.horizon, self.ac_dim))  #start mean at 0

if __name__ == '__main__':
    import gym
    import softgym
    import argparse
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS as SOFTGYM_CUSTOM_ENVS
    from softgym.utils.normalized_env import normalize
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=int, default=0)
    parser.add_argument("--env_name", type=str, default='PassWater')
    parser.add_argument("--traj_path", type=str, default="./data/local/pddm/")
    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--N", type=int, default=700)
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--config_id", type=int, default=0)

    args = parser.parse_args()
    traj_path = args.traj_path + args.env_name + str(args.config_id) + '.pkl'
    print(traj_path)

    env_kwargs = env_arg_dict[args.env_name]
    env_kwargs["headless"] = 1 - args.render
    env = SOFTGYM_CUSTOM_ENVS[args.env_name](**env_kwargs)
    env = normalize(env)

    if not args.replay:
        policy = MPPI(env, N=args.N, horizon=args.T, gamma=args.gamma, beta=args.beta, sigma=args.sigma)
        
        # Run policy
        obs = env.reset(config_id=args.config_id)
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action()
            action_traj.append(copy.copy(action))
            obs, reward, _, _ = env.step(action)
            print('reward:', reward)

        with open(traj_path, 'wb') as f:
            pickle.dump(action_traj, f)

    else:
        with open(traj_path, 'rb') as f:
            actions = pickle.load(f)
        # env.start_recorsd(video_path='./data/videos/', video_name='cem_folding.gif')
        env.reset(config_id=args.config_id)
        returns = 0
        for action in actions:
            _, r, _ , _ = env.step(action)
            returns += r

        print("totoal return: ", returns)
        from experiments.pddm.record_pddm import pddm_make_gif
        pddm_make_gif(env, [actions], args.traj_path, args.env_name + str(args.config_id) + '.gif', config_ids=[args.config_id])

