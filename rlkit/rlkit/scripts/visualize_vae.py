# Created by Xingyu Lin, 2019-09-18
import argparse
import torch
import numpy as np
from rlkit.torch.pytorch_util import set_gpu_mode
import rlkit.torch.pytorch_util as ptu
import copy
import cv2


def batch_chw_to_hwc(images):
    rets = []
    for i in range(len(images)):
        rets.append(copy.copy(np.transpose(images[i], [2, 1, 0])[::-1, :, ::-1]))
    return np.array(rets)


def visualize_vae(args):
    data = torch.load(args.file)
    vae = data['vae']
    env = data['exploration/env']
    set_gpu_mode(True)
    obs = env.reset()
    curr_img = obs['image_achieved_goal']
    goal_img = obs['image_desired_goal']

    imgs = batch_chw_to_hwc([curr_img.reshape(3, 48, 48),
                             goal_img.reshape(3, 48, 48)])
    save_img = np.hstack(imgs) * 256
    cv2.imwrite('./latent_space/original.png', save_img)

    latent_distribution_params = vae.encode(ptu.from_numpy(np.array([curr_img, goal_img]).reshape(2, -1)))
    latent_mean, logvar = ptu.get_numpy(latent_distribution_params[0]), \
                          ptu.get_numpy(latent_distribution_params[1])

    curr_latent = latent_mean[0, :]
    goal_latent = latent_mean[1, :]

    # reconstr_imgs = ptu.get_numpy(vae.decode(ptu.from_numpy(latent_mean)))
    # reconstr_curr_img = reconstr_imgs[0, :].reshape([3, 48, 48])
    # reconstr_goal_img = reconstr_imgs[1, :].reshape([3, 48, 48])

    alphas = np.linspace(0, 1, 30)
    interpolate_latents = []

    for alpha in alphas:
        latent = (1 - alpha) * curr_latent + alpha * goal_latent
        interpolate_latents.append(copy.copy(latent))
    # print('debug:', vae.decode(ptu.from_numpy(np.array(interpolate_latents))))

    reconstr_imgs = ptu.get_numpy(vae.decode(ptu.from_numpy(np.array(interpolate_latents)))[0])
    reconstr_imgs = batch_chw_to_hwc(reconstr_imgs.reshape([-1, 3, 48, 48]))

    save_img = np.hstack(reconstr_imgs) * 256
    cv2.imwrite('./latent_space/latent_sapce.png', save_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    visualize_vae(args)
