import click
import os
import cv2

from softgym.registered_env import env_arg_dict
import os.path as osp
import torchvision
import torch
from envs.env import Env
import pyflex

@click.command()
@click.argument('headless', type=bool, default=True)
@click.argument('episode', type=int, default=8)
@click.argument('save_dir', type=str, default='./data/plots/')
@click.argument('img_size', type=int, default=720)
@click.argument('use_cached_states', type=bool, default=True)
@click.option('--deterministic/--no-deterministic', default=False)
def main(headless, episode, save_dir, img_size, use_cached_states, deterministic):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    """ Generate demos for all environments with different variations, as well as making generate cached states"""

    # env_names = ['PassWater', 'PourWater', 'RopeFlattenNew','ClothFlatten', 'ClothFold', 'ClothDrop']
    env_names = ['TransportTorus', 'RigidClothFold', 'RigidClothDrop']
    envs = [generate_env(env_name) for env_name in env_names]

    all_frames = []
    for env_name, env in zip(env_names, envs):
        for i in range(episode):
            env.reset()
            pos = pyflex.get_positions()
            pyflex.step()
            pyflex.set_positions(pos)
            all_frames.append(env.get_image(img_size, img_size))

    show_imgs = []
    for frame in all_frames:
        img = frame.transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    grid_imgs = torchvision.utils.make_grid(show_imgs, nrow=episode, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs = grid_imgs[:, :, ::-1]
    save_path = osp.join(save_dir, 'task_variation.png')
    print(save_path)
    cv2.imwrite(save_path, grid_imgs)


def generate_env(env_name):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = True
    kwargs['use_cached_states'] = True
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = False

    # Env wrappter
    env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
    return env


if __name__ == '__main__':
    main()
