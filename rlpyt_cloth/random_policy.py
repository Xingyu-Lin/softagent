from os.path import join
import json

import torch
import numpy as np

from rlpyt.envs.dm_control_env import DMControlEnv


def cloth_corner_random(obs):
    idx = np.random.randint(0, 4)
    one_hot = np.zeros(4)
    one_hot[idx] = 1

    delta = np.random.rand(3) * 2 - 1
    return np.concatenate((one_hot, delta)).astype(np.float32)


def rope_v2_random(obs):
    return np.random.rand(3) * 2 - 1


def cloth_point_random(obs):
    return np.random.rand(4) * 2 - 1


def simulate_policy():
    policy = cloth_point_random
    env = DMControlEnv(domain='cloth_point', task='easy',
                       max_path_length=120, task_kwargs=dict(random_location=False))

    n_episodes = 40
    returns = []

    for i in range(n_episodes):
        o = env.reset()
        done = False
        reward = 0

        while not done:
            o, r, done, info = env.step(policy(o))
            reward += r

            if done or info.traj_done:
                break
        print(reward)
        returns.append(reward)
        print('Finished episode', i)

    print('Rewards', returns)
    print('Average Reward', np.mean(returns))

if __name__ == '__main__':
    simulate_policy()
