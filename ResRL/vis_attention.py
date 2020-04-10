from ResRL.td3 import TD3
import argparse, sys
import os.path as osp
import json
import torch
import numpy as np
from envs.env import Env
from softgym.utils.visualization import save_numpy_as_gif, make_grid

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--policy", type=str, default='data/seuss/0405_resRL_hyper_search/0405_resRL_hyper_search/0405_resRL_hyper_search_2020_04_05_11_33_48_0002/model_190000.pth')
args = args.parse_args()


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    info = {}
    all_returns = []
    all_actions = []
    vis_trajs = []
    for _ in range(eval_episodes):
        state, done, ret = eval_env.reset(), False, 0.
        vis_traj = [eval_env.render(mode='rgb_array')]
        while not done:
            action, info = policy.select_action(np.array(state), return_info=True)
            state, reward, done, _ = eval_env.step(action)
            ret += reward[0]
            all_actions.append(action)
            attn_weights = info['attn_action_output_weights'].detach().cpu().numpy()
            # print('weight mean and std:', np.mean(attn_weights), np.std(attn_weights))
            # exit()
            if len(vis_traj) == 1:
                max_attn_weight = np.max(attn_weights)
            attn_weights /= max_attn_weight
            attention_img = np.tile(attn_weights, [50, 1]) * 255
            vis_traj[-1] = np.vstack([vis_traj[-1], attention_img])
            vis_traj.append(eval_env.render(mode='rgb_array'))

        vis_traj[-1] = np.vstack([vis_traj[-1], np.zeros_like(attention_img)])
        vis_trajs.append(vis_traj)
        all_returns.append(ret)

    info['eval_return_mean'] = np.mean(all_returns)
    info['eval_return_std'] = np.std(all_returns)
    info['eval_action_mean'] = np.mean(all_actions)
    info['eval_action_std'] = np.std(all_actions)
    info['eval_abs_action_mean'] = np.mean(np.abs(all_actions))

    idxes = list(reversed(np.argsort(all_returns)))
    all_returns = np.array([all_returns[idx] for idx in idxes])
    vis_trajs = np.array([vis_trajs[idx] for idx in idxes])
    return all_returns, vis_trajs


if __name__ == '__main__':
    policy_file = args.policy
    policy_folder = osp.dirname(policy_file)
    variant_file = osp.join(policy_folder, 'variant.json')
    with open(variant_file, 'r') as f:
        vv = json.load(f)

    # Configure torch
    device = torch.device('cpu')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])

    image_dim = vv['env_kwargs']['image_dim']
    symbolic = not vv['env_kwargs']['image_observation']
    env = Env(vv['env_name'], symbolic, vv['seed'], vv['max_episode_length'], 1, 8, image_dim, env_kwargs=vv['env_kwargs'])

    # Set seeds
    env.seed(vv['seed'])
    torch.manual_seed(vv['seed'])
    np.random.seed(vv['seed'])

    obs_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = dict(image_observation=vv['env_kwargs']['image_observation'],
                  image_dim=env.image_dim,
                  image_c=obs_dim // (env.image_dim * env.image_dim),
                  state_dim=obs_dim,
                  action_dim=action_dim,
                  obs_embed_dim=vv['obs_embed_dim'],
                  action_embed_dim=vv['action_embed_dim'],
                  visual_encoder_name=vv['visual_encoder_name'],
                  max_action=max_action,
                  discount=vv['discount'],
                  tau=vv['tau'],
                  weight_decay=vv['weight_decay'],
                  device=device)

    kwargs["policy_noise"] = vv['policy_noise'] * max_action
    kwargs["noise_clip"] = vv['noise_clip'] * max_action
    kwargs["policy_freq"] = vv['policy_freq']
    policy = TD3(**kwargs)
    policy.load(policy_file)

    _, vis_trajs = eval_policy(policy, env, vv['seed'])
    vis_trajs = np.array(vis_trajs).swapaxes(0, 1)

    vis_imgs = np.array([make_grid(vis_trajs[i], nrow=2, padding=5) for i in range(vis_trajs.shape[0])])

    save_numpy_as_gif(vis_imgs, osp.join('./', 'test.gif'), fps=10, scale=1.)

# initial_states, action_trajs, configs, all_infos = [], [], [], []
# for i in range(vv['test_episodes']):
#     logger.log('episode ' + str(i))
#     obs = env.reset()
#     initial_state = env.get_state()
#     action_traj = []
#     infos = []
#     for _ in range(env.horizon):
#         action = policy.get_action(obs)
#         action_traj.append(copy.copy(action))
#         obs, reward, _, info = env.step(action)
#         infos.append(info)
#     all_infos.append(infos)
#     initial_states.append(initial_state.copy())
#     action_trajs.append(action_traj.copy())
#     configs.append(env.get_current_config().copy())
#
#     # Log for each episode
#     transformed_info = transform_info([infos])
#     for info_name in transformed_info:
#         logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
#         logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
#         logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
#     logger.dump_tabular()
