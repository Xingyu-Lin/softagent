import click
import json
import os
import os.path as osp
import torch
import numpy as np
from planet.planet_agent import PlaNetAgent
from envs.env import Env
from planet.utils import write_video
from torchvision.utils import make_grid, save_image


def get_spaced_idx(n, m):
    # Evenly pick m int from [0, n-1]
    return np.round(np.linspace(0, n - 1, m)).astype(int)


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=8)
@click.option('--render', type=int, default=1)
# Should be ['0', '1', '2'], 0 if no rendering; 1 if using default rendering; 2 if using goal based rendering
@click.option('--save_dir', type=str, default='data/open_loop_videos')
def main(policy_file, seed, n_test_rollouts, render, save_dir):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')
    np.random.seed(seed=seed)
    json_file = os.path.join(os.path.dirname(policy_file), 'variant.json')
    print('Load variants from {}'.format(json_file))
    with open(json_file) as f:
        vv = json.load(f)
    vv['env_kwargs']['headless'] = 1 - render
    vv['saved_models'] = policy_file

    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'], vv['image_dim'],
              env_kwargs=vv['env_kwargs'])
    agent = PlaNetAgent(env, vv, device)

    all_rewards, all_frames, all_frames_reconstr = [], [], []
    agent.set_model_eval()
    with torch.no_grad():
        for i in range(n_test_rollouts):
            observation, total_reward = agent.env.reset(), 0
            belief, posterior_state, action = torch.zeros(1, vv['belief_size'], device=device), \
                                              torch.zeros(1, vv['state_size'], device=device), \
                                              torch.zeros(1, env.action_size, device=device)
            initial_belief, initial_posterior, initial_observation = belief.clone(), posterior_state.clone(), observation.clone()
            recorded_actions = [action]
            frames, frames_reconstr = [observation], [observation]
            for t in range(vv['env_kwargs']['horizon']):
                belief, posterior_state, action, next_observation, reward, done, info = \
                    agent.update_belief_and_act(agent.env, belief, posterior_state, action, observation.to(device=agent.device),
                                                explore=False)
                recorded_actions.append(action)
                total_reward += reward
                observation = next_observation
                frames.append(observation)
                if done:
                    break

            # Re-imagine without observation
            belief, state = initial_belief, initial_posterior
            for idx, action in enumerate(recorded_actions):
                print('idx: ', idx)
                if idx <= 5:
                    belief, _, _, _, state, _, _ = agent.transition_model(state, action.unsqueeze(dim=0), belief,
                                                                          agent.encoder(frames[idx].to(device=agent.device)).unsqueeze(dim=0))
                else:
                    belief, state, _, _, = agent.transition_model(posterior_state, action.unsqueeze(dim=0), belief)
                belief, state = belief.squeeze(dim=0), state.squeeze(dim=0)
                # print('belief size:', belief.size(), 'state size:',  state.size())
                frames_reconstr.append(agent.observation_model(belief, state).cpu())

            print('episode: {}, total reward: {}'.format(i, total_reward))
            all_rewards.append(total_reward)
            all_frames.append(frames)
            all_frames_reconstr.append(frames_reconstr)

            # Pick key frames
            num_key_frames = 5
            key_idx = get_spaced_idx(len(frames[:30]), num_key_frames)
            frame = torch.cat([frames[idx] for idx in key_idx], dim=0) + 0.5
            frame_reconstr = torch.cat([frames_reconstr[idx] for idx in key_idx], dim=0) + 0.5
            image_grid = make_grid(torch.cat([frame, frame_reconstr], dim=0), nrow=num_key_frames, pad_value=0.4706, padding=5)
            save_image(image_grid, osp.join(save_dir, vv['env_name'] + '_{}.png'.format(i)))
            # save_image(torch.as_tensor(frame), osp.join(save_dir, vv['env_name'] + '_gt_{}.png'.format(i)))
            # save_image(torch.as_tensor(frame_reconstr), osp.join(save_dir, vv['env_name'] + '_prediction_{}.png'.format(i)))

    all_frames = all_frames[:8]  # Only take the first 8 episodes to visualize
    all_frames_reconstr = all_frames_reconstr[:8]
    video_frames = []
    for i in range(len(all_frames[0])):
        frame = torch.cat([x[i] for x in all_frames])
        frame_reconstr = torch.cat([x[i] for x in all_frames_reconstr])
        video_frames.append(make_grid(torch.cat([frame, frame_reconstr], dim=3) + 0.5, nrow=4).numpy())
    write_video(video_frames, vv['env_name'], save_dir)  # Lossy compression
    print('Average total reward:', np.mean(np.array(all_rewards)))


if __name__ == '__main__':
    main()