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
import os

save_dir = './data/cem_trajs'
save_dir_website = './data/website_gifs/'

visualize_horizon = {
    'ClothFlatten': 20,
    'ClothFold': 15,
    'PourWater': 20,
    'RopeFlattenNew': 10,
    'RigidClothDrop': 6,
    'RigidClothFold': 30,
    'TransportTorus': 15,
}


def generate_env(env_name, obs_mode):
    kwargs = env_arg_dict[env_name]
    kwargs['headless'] = True
    kwargs['use_cached_states'] = True
    kwargs['num_variations'] = 1000
    kwargs['save_cached_states'] = False
    kwargs['observation_mode'] = obs_mode
    # Env wrappter
    env = Env(env_name, obs_mode == 'key_point', 100, 200, 1, 8, 128, kwargs)
    return env


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)
    return args


def make_agent(env, vv, symbolic):
    from curl.curl_sac import CurlSacAgent
    args = vv_to_args(vv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_shape = env.action_space.shape
    args.encoder_type = 'identity' if symbolic else 'pixel'
    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
    else:
        obs_shape = env.observation_space.shape
    agent = CurlSacAgent(
        args=args,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        alpha_fixed=args.alpha_fixed,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder,
        curl_latent_dim=args.curl_latent_dim
    )

    return agent


def find_last_model_id(dir):
    onlyfiles = [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]
    models = []
    for f in onlyfiles:
        if 'pt' in f:
            number = f[f.rfind('_') + 1:-3]
            if int(number) < 1000000:
                models.append(int(number))

    models.sort()
    return models[-1]


def curl_collect_traj(env, model_dir, vv, N, img_size=128):
    all_frames = []
    symbolic = vv['env_kwargs_observation_mode'] == 'key_point'
    agent = make_agent(env, vv, symbolic)
    id = find_last_model_id(model_dir)
    agent.load(model_dir, id)

    for i in range(N):
        frames = []
        obs = env.reset()
        frames.append(env.get_image(img_size, img_size))
        for j in range(env.horizon):
            action = agent.sample_action(obs)
            obs, reward, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            frames.extend(info['flex_env_recorded_frames'])
        frames = np.array(frames)
        all_frames.append(frames)
    return all_frames


def visualize_trajectory():
    data_folders = [
        './data/corl_data/0717_cloth_flatten/',
        './data/corl_data/0719_corl_cloth_flatten',
        './data/corl_data/0719_corl_cloth_fold_lr',
        './data/corl_data/0717-corl-cloth-drop',
        './data/corl_data/0718_corl_rope_curl_lr',
        './data/corl_data/0713-CoRL-Curl-PourWater',
        './data/corl_data/0715-Curl-PassWater-and-Torus',

    ]
    env_names = ['PassWater', 'PourWater', 'RopeFlattenNew', 'ClothFlatten', 'ClothFold', 'ClothDrop']
    # env_names = ['ClothFlatten']
    K = 1
    all_env_frames = {}
    envs = {}
    for env_name in env_names:
        for obs_mode in ['key_point', 'cam_rgb']:
            all_env_frames[env_name + '_' + obs_mode] = []
            envs[env_name + '_' + obs_mode] = generate_env(env_name, obs_mode)
    for data_folder in data_folders:
        for folder in sorted(os.listdir(data_folder)):
            exp_folder = osp.join(data_folder, folder)

            print('processing...' + exp_folder)
            variant_path = osp.join(exp_folder, 'variant.json')
            if not os.path.exists(variant_path):
                continue
            with open(osp.join(variant_path), 'r') as f:
                vv = json.load(f)


            exp_name = vv['env_name'] + '_' + vv['env_kwargs_observation_mode']
            if vv['env_kwargs_observation_mode'] not in ['key_point', 'cam_rgb']:
                continue
            print(exp_name)

            if vv['env_name'] not in env_names or len(all_env_frames[exp_name]) >= K:
                continue

            # Excluding some trajectories
            # if vv['env_name'] == 'ClothFlatten':
            #     if exp_folder.endswith('02') or exp_folder.endswith('03'):
            #         print('skipping ' + exp_folder)
            #         continue

            env = envs[exp_name]

            frames = curl_collect_traj(env, os.path.join(exp_folder, 'model'), vv, 4)
            all_env_frames[exp_name].extend(frames)

    for key in all_env_frames:
        if len(all_env_frames[key]) == 0:
            continue
        frames = np.array(all_env_frames[key][:4]).swapaxes(0, 1)
        frames = np.array([make_grid(frame, nrow=1, pad_value=120, padding=5) for frame in frames])
        save_numpy_as_gif(frames, osp.join(save_dir_website, 'curl_' + key + '.gif'))


if __name__ == '__main__':
    visualize_trajectory()
