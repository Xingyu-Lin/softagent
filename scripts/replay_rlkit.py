from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
import numpy as np
import torchvision
import os.path as osp
from softgym.utils.visualization import save_numpy_as_gif

filename = str(uuid.uuid4())


def simulate_policy(args, flex_env):
    data = torch.load(args.file)

    policy = data['evaluation/policy']
    env = data['evaluation/env']
    if flex_env:
        import pyflex
        headless, render, camera_width, camera_height = True, True, 720, 720
        pyflex.init(headless, render, camera_width, camera_height)

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    row, column = 2, 4
    img_size = 256
    save_dir = './data/videos'
    all_frames = []
    for i in range(row * column):
        frames = []
        obs = env.reset()
        frames.append(env.get_image(img_size, img_size))
        for _ in range(env.horizon):
            action, _ = policy.get_action(obs)
            print(action)
            obs, _, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            frames.extend(info['flex_env_recorded_frames'])
        all_frames.append(frames)
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_name = 'temp.gif'
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))

    # while True:
    #     path = rollout(
    #         env,
    #         policy,
    #         max_path_length=args.H,
    #         render=False,
    #     )
    #     if hasattr(env, "log_diagnostics"):
    #         env.log_diagnostics([path])
    #     logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()

    simulate_policy(args, True)
