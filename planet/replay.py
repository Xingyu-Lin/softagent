import click
import json
import os
import torch
import numpy as np
from planet.planet_agent import PlaNetAgent
from envs.env import Env


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
# Should be ['0', '1', '2'], 0 if no rendering; 1 if using default rendering; 2 if using goal based rendering
@click.option('--exploit', type=int, default=1)
@click.option('--record_video', type=int, default=0)
def main(policy_file, seed, n_test_rollouts, render, exploit, record_video):
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

    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'],
              env_kwargs=vv['env_kwargs'])
    agent = PlaNetAgent(env, vv, device)

    all_rewards = []
    agent.set_model_eval()
    with torch.no_grad():
        for i in range(n_test_rollouts):
            observation, total_reward = agent.env.reset(), 0
            belief, posterior_state, action = torch.zeros(1, vv['belief_size'], device=device), \
                                              torch.zeros(1, vv['state_size'], device=device), \
                                              torch.zeros(1, env.action_size, device=device)
            for t in range(vv['env_kwargs']['horizon']):
                belief, posterior_state, action, next_observation, reward, done = \
                    agent.update_belief_and_act(agent.env, belief, posterior_state, action, observation.to(device=agent.device),
                                                explore=(exploit != 0))
                total_reward += reward
                observation = next_observation
                if done:
                    break

            print('episode: {}, total reward: {}'.format(i, total_reward))
            all_rewards.append(total_reward)
    print('Average total reward:', np.mean(np.array(all_rewards)))


if __name__ == '__main__':
    main()
