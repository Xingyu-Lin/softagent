from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif, make_grid
from multiprocessing import Process
import numpy as np
import os.path as osp
import torchvision
import torch
import pickle
from envs.env import Env
import json
import matplotlib

data_folders = [
    # './data/corl_data/0715_corl_cem_cloth_flatten_cloth_drop/0715_corl_cem_cloth_flatten_cloth_drop_2020_07_15_19_04_01_0005', # ClothFlatten'
    # './data/corl_data/0715_corl_cem_cloth_flatten_cloth_drop/0715_corl_cem_cloth_flatten_cloth_drop_2020_07_15_19_04_01_0016', # ClothDrop
    # './data/corl_data/0719_corl_cem_cloth_fold/0719_corl_cem_cloth_fold_2020_07_19_13_27_53_0012/',  # ClothFold
    # 'data/corl_data/0715-CoRL-CEM-PassWater-and-Torus/0715-CoRL-CEM-PassWater-and-Torus_2020_07_16_02_13_25_0001/', # PassWater
    # 'data/corl_data/0713-CoRL-CEM-PourWater/0713-CoRL-CEM-PourWater_2020_07_14_02_02_06_0002/' # PourWater: Take the second one
    # 'data/corl_data/0717_corl_cem_cloth_rope/0717_corl_cem_cloth_rope_2020_07_17_20_59_01_0001/',  # RopeFlattenNew

    # Rigid tasks
    # 'data/corl_data/0717_corl_cem_cloth_rope/0717_corl_cem_cloth_rope_2020_07_17_20_59_01_0034/', # Rigid Cloth Drop
    # 'data/corl_data/0720_rigid_cloth_fold_cem/0720_corl_rigid_cloth_fold_2020_07_20_22_39_49_0003/',  # Rigid Cloth Fold
    # 'data/corl_data/0724-CoRL-CEM-TransportTorus-2/0724-CoRL-CEM-TransportTorus-2_2020_07_24_17_56_15_0009/'  # Rigid transport water

]

save_dir = './data/cem_trajs'

visualize_horizon = {
    'ClothFlatten': 20,
    'ClothFold': 15,
    'PourWater': 20,
    'RopeFlattenNew': 10,
    'RigidClothDrop': 6,
    'RigidClothFold': 30,
    'TransportTorus': 15,
}


def generate_env(env_name):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = True
    kwargs['use_cached_states'] = True
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = False

    # Env wrappter
    env = Env(env_name, False, 100, 200, 1, 8, 128, kwargs)
    return env


def cem_make_gif(env, env_name, initial_states, action_trajs, configs, save_dir, save_name, img_size=128):
    if env_name == 'PourWater':
        idx = 1
    else:
        idx = 0
    action_traj = action_trajs[idx]
    config = configs[idx]
    initial_state = initial_states[idx]

    if env_name in visualize_horizon:
        action_traj = action_traj[:visualize_horizon[env_name]]

    frames = []
    env.reset(config=config, initial_state=initial_state)
    frames.append(env.get_image(img_size, img_size))
    for action in action_traj:
        _, reward, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
        frames.append(info['flex_env_recorded_frames'][0])
    frames = np.array(frames)

    idxes = np.round(np.linspace(0, len(frames) - 1, 3)).astype(int)
    frames = frames[idxes, :, :, :]

    grid_imgs = make_grid(frames, nrow=1, padding=5)
    matplotlib.image.imsave(osp.join(save_dir, save_name), grid_imgs)


def visualize_cem():
    for data_folder in data_folders:
        variant_path = osp.join(data_folder, 'variant.json')
        with open(osp.join(variant_path), 'r') as f:
            vv = json.load(f)

        env = generate_env(vv['env_name'])
        file_path = osp.join(data_folder, 'cem_traj.pkl')
        with open(file_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']
        cem_make_gif(env, vv['env_name'], initial_states, action_trajs, configs, save_dir, vv['env_name'] + '.png')


if __name__ == '__main__':
    visualize_cem()
