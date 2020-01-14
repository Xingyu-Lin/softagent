# Created by Xingyu Lin, 2019-09-19                                                                                  
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.util.video import dump_video
from rlkit.core import logger
import numpy as np
import os.path as osp
import copy
import json
import glob
import cv2
# from scripts.visualize_vae import batch_chw_to_hwc
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
filename = str(uuid.uuid4())

def batch_chw_to_hwc(images):
    rets = []
    for i in range(len(images)):
        rets.append(copy.copy(np.transpose(images[i], [2, 1, 0])[::-1, :, ::-1]))
    return np.array(rets)

def get_image(img, imsize=48):
    img = np.reshape(img, [3, 48, 48])
    img = batch_chw_to_hwc(img[None])[0] * 256
    return img


def load_variants(exp_dir):
    with open(osp.join(exp_dir, 'variant.json'), 'r') as f:
        variants = json.load(f)
    return variants


def plot_latent_dist(env, policy, rollout_function, horizon, N=10, save_name='./plot.png'):
    all_latent_achieved_goal = []
    all_lateng_goal = []
    for _ in range(N):
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )

        # print("len of path: ", len(path))
        # print("path[0] keys are: ", path[0].keys())
        # print("path[0] is: ", path[0])
        # exit()
        latent_achieved_goal = np.array([d['latent_achieved_goal'] for d in path['full_observations']])
        latent_goal = np.array([d['latent_desired_goal'] for d in path['full_observations']])
        all_latent_achieved_goal.append(latent_achieved_goal)
        all_lateng_goal.append(latent_goal)
    all_latent_achieved_goal = np.array(all_latent_achieved_goal)
    all_lateng_goal = np.array(all_lateng_goal)
    dist = np.linalg.norm(all_latent_achieved_goal - all_lateng_goal, axis=-1)
    mean_dist = np.mean(dist, axis=0)
    std_dist = np.std(dist, axis=0)

    # plt.figure()
    # for episode_dist in dist:
    #     plt.plot(list(range(horizon)), episode_dist)
    print(len(dist))
    print(std_dist)
    markers, caps, bars = plt.errorbar(list(range(horizon)), mean_dist, std_dist) #, colors[i], label = labels[i])
    [bar.set_alpha(0.05) for bar in bars]
    [cap.set_alpha(0.05) for cap in caps]
    # plt.plot(list(range(horizon)), mean_dist, linewidth=1)
    # plt.fill_between(list(range(horizon)), mean_dist - std_dist, mean_dist + std_dist, alpha=0.5)
    plt.xlabel('Episode time')
    plt.ylabel('VAE distance to goal')
    plt.savefig(save_name)



def simulate_policy(args, dir):
    variants = load_variants(dir)
    max_path_length = variants['skewfit_variant']['max_path_length']
    env_name = variants['env_id']

    # use_indicator_reward = variants['skewfit_variant']['replay_buffer_kwargs']['use_indicator_reward']
    use_indicator_reward = False
    seed = variants['seed']
    algo_name = 'indicator' if use_indicator_reward else 'skewfit'
    video_name = env_name + '_' + algo_name + '_' + seed + '.gif'
    latent_plot_name = env_name + '_' + algo_name + '_' + seed + '.png'

    data = torch.load(osp.join(dir, 'params.pkl'))
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    if not args.no_gpu:
        print("no gpu!")
        set_gpu_mode(True, 0)
        policy.stochastic_policy.cuda()

    vae = env.vae
    import softgym, gym
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    softgym.register_flex_envs()
    env = gym.make(env_name)

    imsize = variants.get('imsize', 48)
    init_camera = variants.get('init_camera', None)
    non_presampled_goal_img_is_garbage = variants['train_vae_variant']['generate_vae_dataset_kwargs'] .get(
        'non_presampled_goal_img_is_garbage', None)
    non_presampled_goal_img_is_garbage = False
    # print("non-presampled-goal-img-is-garbage: ", non_presampled_goal_img_is_garbage)
    # exit()
    image_env = ImageEnv(
                    env,
                    imsize=imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )

    env_variant = variants['skewfit_variant']
    render = env_variant.get('render', False)
    reward_params = env_variant.get("reward_params", dict())
    vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                **env_variant.get('vae_wrapped_env_kwargs', {})
            )

    env = vae_env
    env.goal_sampling_mode = 'reset_of_env'
    env.reset()
    # img_env = env.wrapped_env
    # raw_env = env.wrapped_env.wrapped_env
    # import cv2
    # from matplotlib import pyplot as plt
    # img_goal = img_env._img_goal
    # print(img_goal.shape)
    # plt.imshow(img_goal.reshape((imsize, imsize, 3)))
    # plt.show()
    # cv2.imshow("img goal", img_goal.reshape((imsize, imsize, 3)))
    # cv2.waitKey(0)
    # exit()

    # print(env.goal_sampling_mode)
    # # print(env.decode_goals)
    # # env = env._wrapped_env

    # # Debug reset images
    # goal_imgs = []
    # reconstruct_imgs = []
    # for i in range(30):
    #     obs = env.reset()
    #     print("after reset", i)
    #     goal_img = obs['image_desired_goal']
    #     reconstruct_img = goal_img
    #     # reconstruct_img = np.clip(env._reconstruct_img(goal_img), 0,1)
    #     goal_img = get_image(goal_img)
    #     reconstruct_img = get_image(reconstruct_img)
    #     goal_imgs.append(copy.copy(goal_img))
    #     reconstruct_imgs.append(copy.copy(reconstruct_img))
    # save_goal_img = np.hstack(goal_imgs)
    # save_recon_img = np.hstack(reconstruct_imgs)
    # save_img = np.vstack([save_goal_img, save_recon_img])
    # cv2.imwrite('./imgs/goal_images.png', save_img)
    # # exit()
    # # print(env)
    # # print(env._goal_sampling_mode)
    # # exit()

    print("Policy loaded")

    def rollout(*args, **kwargs):
        return multitask_rollout(*args, **kwargs,
                                 observation_key='latent_observation',
                                 desired_goal_key='latent_desired_goal', )

    if args.video:
        print('dump video')
        dump_video(env, policy, osp.join('./videos', video_name), rollout_function=rollout, imsize=imsize,
                   horizon=max_path_length, rows=1, columns=1)

    if args.latent_distance:
        print("plot_latent_dist")
        plot_latent_dist(env, policy, rollout,
                         horizon=max_path_length,
                         save_name=osp.join('./latent_space', latent_plot_name))

def simulate_policy_recursive(args, dir):
    policy_files = glob.glob(dir + '/**/params.pkl', recursive=True)
    for policy_file in policy_files:
        simulate_policy(args, osp.dirname(policy_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='directory to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--no_gpu', type=int, default=0)
    parser.add_argument('-non_r', '--non_recursive', type=int, default=1)
    parser.add_argument('-vid', '--video', type=int, default=1)
    parser.add_argument('-dist', '--latent_distance', type=int, default=1)
    parser.add_argument('--imsize', type=int, default=48)
    args = parser.parse_args()

    # print(args)
    # exit()
    if not args.non_recursive:
        simulate_policy_recursive(args, args.dir)
    else:
        simulate_policy(args, args.dir)
