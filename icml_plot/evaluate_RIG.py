import os
from os import path as osp
import json
from rlkit.samplers.rollout_functions import multitask_rollout
import numpy as np
import torch
from rlkit.torch.pytorch_util import set_gpu_mode

data = 'data/yufei_s3_data/RIG-128-0202-all/RIG-128-0202-all_2020_02_02_16_13_09_00'

import pyflex

def load_variants(exp_dir):
    with open(osp.join(exp_dir, 'variant.json'), 'r') as f:
        variants = json.load(f)
    return variants

def evaluate_policy(dir, epoch, num=10, final=False):
    data = torch.load(osp.join(dir, 'itr_{}.pkl'.format(epoch)))
    policy_determinisitic = data['evaluation/policy'] # deterministic policy
    env = data['evaluation/env']
    # print(type(env))
    # print(type(env._wrapped_env))
    # print(type(env._wrapped_env._wrapped_env))
    # print(type(env._wrapped_env._wrapped_env._wrapped_env))
    env._wrapped_env._wrapped_env._wrapped_env.eval_flag = True
    set_gpu_mode(True, 0)
    policy_determinisitic.stochastic_policy.cuda()

    performance = []
    final_performance = []
    for i in range(num):
        path = multitask_rollout(env, policy_determinisitic, max_path_length=100, # hard code  
                                    observation_key='latent_observation',
                                    desired_goal_key='latent_desired_goal', 
                                    )

        infos = path['env_infos']
        per = [info['performance'] for info in infos]
        final_perfor = per[-1]
        final_performance.append(final_perfor)
        performance.append(np.sum(per))
        if not final:
            print("episode {} performance {}".format(i, np.sum(per)))
        else:
            print("episode {} final performance {}".format(i, final_perfor))

    if not final:
        return np.mean(performance), np.std(performance)
    else:
        return np.mean(final_performance), None

def run(final=False):
    epoch_lower = []
    epoch_upper = []
    epoch_median = []
    if not final:
        epoch_list = [i * 20 for i in range(13)]
    else:
        epoch_list = [240]

    for epoch in epoch_list:
        seed_mean = []
        for i in range(21, 26):
            data_path = data + str(i)
            dirs = os.walk(data_path)
            subdirs = [x[0] for x in dirs]
            print(subdirs)

            for policy_file in subdirs:
                if '--s' in policy_file:
                    print(policy_file)
                    mean, std = evaluate_policy(policy_file + '/', epoch, final=final)
        
            seed_mean.append(mean)
            if not final:
                print("seed {} mean {}".format(i, mean))
            else:
                print("seed {} final mean {}".format(i, mean))


        # epoch_mean.append(np.mean(seed_mean))
        # epoch_std.append(np.std(seed_mean))
        lower = np.nanpercentile(
            seed_mean, q=25)
        median = np.nanpercentile(
            seed_mean, q=50)
        upper = np.nanpercentile(
            seed_mean, q=75)

        epoch_median.append(median)
        epoch_lower.append(lower)
        epoch_upper.append(upper)

        print("epoch {} lower {} median {} upper {}".format(epoch, lower, median, upper))

    print("epoch_median: ", epoch_median)
    print("epoch_lower: ", epoch_lower)
    print("epoch_upper: ", epoch_upper)
    suffix = "final" if final else ""
    np.save('tmp/rig_clothflatten_epoch_median_{}.npy'.format(suffix), np.asanyarray(epoch_median))
    np.save('tmp/rig_clothflatten_epoch_lower_{}.npy'.format(suffix), np.asanyarray(epoch_lower))
    np.save('tmp/rig_clothflatten_epoch_upper_{}.npy'.format(suffix), np.asanyarray(epoch_upper))

headless, render, camera_width, camera_height = True, True, 720, 720
pyflex.init(headless, render, camera_width, camera_height)
run(final=True)