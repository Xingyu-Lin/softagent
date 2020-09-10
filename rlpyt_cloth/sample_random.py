from rlpyt.envs.dm_control_env import DMControlEnv
import math
import time
import os
from os.path import join, exists
import itertools
from tqdm import tqdm
import numpy as np
import imageio
import multiprocessing as mp


def worker(worker_id, start, end):
    np.random.seed(worker_id)
    # Initialize environment
    env_args = dict(
        domain='rope_sac',
        task='easy',
        max_path_length=5,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        #task_kwargs=dict(random_location=True, pixels_only=True) # to not return positions and only pick location
    )
    env = DMControlEnv(**env_args)
    total = 0

    if worker_id == 0:
        pbar = tqdm(total=end - start)

    for i in range(start, end):
        str_i = str(i)
        run_folder = join(root, 'run{}'.format(str_i.zfill(5)))
        if not exists(run_folder):
            os.makedirs(run_folder)

        actions = []
        o = env.reset()
        np.random.seed(0)
        for t in itertools.count():
            a = env.action_space.sample()
            a = a / np.linalg.norm(a) * 1
            actions.append(np.concatenate((o.location[:2], a)))
            str_t = str(t)
            imageio.imwrite(join(run_folder, 'img_{}.png'.format(str_t.zfill(2))), o.pixels.astype('uint8'))

            o, _, terminal, info = env.step(a)
            if terminal or info.traj_done:
                break

        actions = np.stack(actions, axis=0)
        np.save(join(run_folder, 'actions.npy'), actions)

        if worker_id == 0:
            pbar.update(1)
    if worker_id == 0:
        pbar.close()

if __name__ == '__main__':
    start = time.time()
    root = join('data', 'rope_data')
    if not exists(root):
        os.makedirs(root)

    n_trajectories = 10
    n_chunks = 1
    #n_chunks = mp.cpu_count()
    partition_size = math.ceil(n_trajectories / n_chunks)
    args_list = []
    for i in range(n_chunks):
        args_list.append((i, i * partition_size, min((i + 1) * partition_size, n_trajectories)))
    print('args', args_list)

    ps = [mp.Process(target=worker, args=args) for args in args_list]
    [p.start() for p in ps]
    [p.join() for p in ps]

    elapsed = time.time() - start
    print('Finished in {:.2f} min'.format(elapsed / 60))

