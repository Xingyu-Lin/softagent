import cv2
import numpy as np
import torch, torchvision
import imageio
from softgym.utils.visualization import save_numpy_as_gif
from os import path as osp

envs = ['PourWater', 'PassWater', 'ClothFlatten', 'ClothFold', 'ClothDrop', 'RopeFlatten']
algos = ['SAC-cam_rgb', 'PlaNet-cam_rgb', 'SAC-key_point', 'TD3-key_point',  'RIG']
suff = [1, 2]

def downsample(frame, minlen):
    factor = len(frame) / minlen
    print(factor)
    res = []
    for i in range(minlen):
        res.append(frame[int(i * factor)])

    return np.asarray(res)

pics = []
for algo in algos:
    all_frames = []
    for suffix in suff:
        minlen = 100000
        for env in envs:
            frames = []
            file_path = './data/website/{}-{}-{}.gif'.format(env, algo, suffix)
            gif = imageio.get_reader(file_path)
            minlen = min(len(gif), minlen)
            for frame in gif:
                # print(frame.shape)
                # cv2.imwrite("tmp.png", frame[:, :, :3])
                # exit()
                frames.append(frame[:, :, :3])
            
            all_frames.append(frames)

    for idx in range(len(all_frames)):
        all_frames[idx] = downsample(all_frames[idx], minlen)
    all_frames = np.asarray(all_frames)
    print(all_frames.shape)

    all_frames = all_frames.transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=len(envs), padding=5, pad_value=120).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_dir = './data/website'
    save_name =  algo + '.gif'
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))