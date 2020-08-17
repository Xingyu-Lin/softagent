import os
import os.path as osp
import time

import numpy as np
import cv2
import skvideo.io
import torchvision
import torch
from softgym.utils.visualization import save_numpy_as_gif
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from softgym.core.image_env import unormalize_image


def dump_video(
  env,
  policy,
  filename,
  rollout_function,
  rows=3,
  columns=6,
  pad_length=0,
  pad_color=255,
  do_timer=True,
  horizon=100,
  dirname_to_save_images=None,
  subdirname="rollouts",
  imsize=84,
  num_channels=3,
  heuristic_func=None,
):
    frames = []
    H = 3 * imsize
    W = imsize
    N = rows * columns
    for i in range(N):
        # print(i)
        start = time.time()
        if heuristic_func is None:
            path = rollout_function(
                env,
                policy,
                max_path_length=horizon,
                render=False,
            )
        else:
            path = heuristic_func(env)
        # print("after rollout")
        is_vae_env = isinstance(env, VAEWrappedEnv)
        l = []
        for d in path['full_observations']:
            if is_vae_env:
                # recon = np.clip(env._reconstruct_img(d['image_observation']), 0,
                #                 1)
                recon = env._reconstruct_img(d['image_observation'])
            else:
                recon = d['image_observation']

            l.append(
                get_image(
                    d['image_desired_goal'],
                    d['image_observation'],
                    recon,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    imsize=imsize,
                )
            )
        frames += l
        # print("here ok")

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-env.horizon:]
            # goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            goal_img = rollout_frames[0][:imsize, :imsize, :]
            cv2.imwrite(rollout_dir + "/goal.png", goal_img[:, :, ::-1])
            # goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            goal_img = rollout_frames[0][2*imsize:, :imsize, :]
            cv2.imwrite(rollout_dir + "/z_goal.png", goal_img[:, :, ::-1])
            for j in range(0, env.horizon, 1):
                # img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)

                img = rollout_frames[j][imsize:2*imsize, :imsize, :]
                cv2.imwrite(rollout_dir + "/" + str(j) + "_obs.png", img[:, :, ::-1])


                img = rollout_frames[j][2 * imsize:, :imsize, :]
                cv2.imwrite(rollout_dir + "/" + str(j) + "_recons.png", img[:, :, ::-1])

        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
      N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length,
                 num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def dump_video_non_goal(
  env,
  policy,
  filename,
  rollout_function,
  rows=3,
  columns=6,
  pad_length=0,
  pad_color=255,
  do_timer=True,
  horizon=100,
  dirname_to_save_images=None,
  subdirname="rollouts",
  imsize=84,
):
    N = rows * columns
    all_frames = []
    for i in range(N):
        frames = []
        obs = env.reset()
        policy.reset()
        frames.append(env.get_image(imsize, imsize))
        for _ in range(env.horizon):
            action, _ = policy.get_action(obs)
            obs, _, _, _ = env.step(action)
            frames.append(env.get_image(imsize, imsize))
        all_frames.append(frames)
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=columns).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_numpy_as_gif(np.array(grid_imgs), filename)


def get_image(goal, obs, recon_obs, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose()
        obs = obs.reshape(-1, imsize, imsize).transpose()
        recon_obs = recon_obs.reshape(-1, imsize, imsize).transpose() 

    # from matplotlib import pyplot as plt
    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.imshow(goal)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.imshow(obs)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax3.imshow(recon_obs)

    # plt.show()

    goal = unormalize_image(goal)
    obs = unormalize_image(obs)
    recon_obs = unormalize_image(recon_obs)

    img = np.concatenate((goal, obs, recon_obs))
    # img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, pad_length, pad_color)
    return img


def add_border(img, pad_length, pad_color, imsize=84):
    H = 3 * imsize
    W = imsize
    img = img.reshape((3 * imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2
