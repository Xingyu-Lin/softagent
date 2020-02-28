from softgym.utils.visualization import save_numpy_as_gif
import click
import os.path as osp
import numpy as np
import torchvision
import torch
import os
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import cv2


@click.command()
@click.argument('headless', type=bool, default=True)
@click.argument('episode', type=int, default=8)
@click.argument('save_dir', type=str, default='./data/icml')
@click.argument('img_size', type=int, default=720)
@click.argument('use_cached_states', type=bool, default=True)
@click.option('--deterministic/--no-deterministic', default=False)
def main(headless, episode, save_dir, img_size, use_cached_states, deterministic):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    """ Generate demos for all environments with different variations, as well as making generate cached states"""

    envs = []
    for env_name, env_class in SOFTGYM_ENVS.items():
        env_arg_dict[env_name]['render'] = True
        env_arg_dict[env_name]['headless'] = headless
        env_arg_dict[env_name]['observation_mode'] = 'point_cloud'
        env_arg_dict[env_name]['use_cached_states'] = use_cached_states
        env = env_class(**env_arg_dict[env_name])
        envs.append(env)

    all_frames = []
    for env_name, env in zip(SOFTGYM_ENVS.keys(), envs):
        for i in range(episode):
            env.reset()
            all_frames.append(env.get_image(img_size, img_size))
        

    show_imgs = []
    for frame in all_frames:
        img = frame.transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    grid_imgs = torchvision.utils.make_grid(show_imgs, nrow=episode, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs=grid_imgs[:, :, ::-1]
    save_path = osp.join(save_dir, 'task_variation.png')
    print(save_path)
    cv2.imwrite(save_path, grid_imgs)


if __name__ == '__main__':
    main()
