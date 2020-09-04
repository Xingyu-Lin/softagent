from softgym.registered_env import env_arg_dict
from softgym.utils.visualization import save_numpy_as_gif
import numpy as np
import os.path as osp
import torchvision
import torch

from envs.env import Env

SAVE_PATH = './data/videos'


def generate_video(env, env_name):
    all_videos = []
    for i in range(8):
        obs = env.reset()
        obs = (obs + 0.5) * 256
        video = [obs]
        for j in range(env.horizon):
            action = env.action_space.sample()
            obs, _, _, info = env.step(action)
            obs = (obs + 0.5) * 256.
            video.append(obs)
        all_videos.append(torch.cat(video, 0))
        print('Env: {}, Eval traj {}'.format(env_name, i))

    # Convert to T x index x C x H x W for pytorch
    all_videos = torch.stack(all_videos, 0).permute(1, 0, 2, 3, 4)
    grid_imgs = np.array(
        [torchvision.utils.make_grid(frame, nrow=4, padding=2, pad_value=120).permute(1, 2, 0).data.cpu().numpy()
         for frame in all_videos])
    save_numpy_as_gif(grid_imgs, osp.join(SAVE_PATH, env_name + '.gif'))
    print('Video generated and save to {}'.format(osp.join(SAVE_PATH, env_name + '.gif')))


def generate_env_state(env_name):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = True
    kwargs['use_cached_states'] = False
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = True
    # Env wrappter
    env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
    return env


if __name__ == '__main__':
    env_names = ['ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop']
    env_names = ['ClothFold']
    envs = [generate_env_state(env_name) for env_name in env_names]

    for (env, env_name) in zip(envs, env_names):
        generate_video(env, env_name)
