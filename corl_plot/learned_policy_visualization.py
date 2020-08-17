from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
import numpy as np
import torchvision
import os.path as osp
import os
from softgym.utils.visualization import save_numpy_as_gif
import json
from planet.planet_agent import PlaNetAgent
from envs.env import Env

filename = str(uuid.uuid4())

config_idxes = [233, 666, 128, 256, 512]

def simulate_policy_model_free(args):
    print(args.dir)
    dirs = os.walk(args.dir)
    subdirs = [x[0] for x in dirs]
    for policy_file in subdirs:
        print(policy_file)
        if '--s' in policy_file:
            # print(osp.dirname(policy_file))
            print(policy_file)
            break            


    data = torch.load(osp.join(policy_file, 'params.pkl'))

    policy = data['evaluation/policy']
    env = data['evaluation/env']
    env.deterministic = True
  
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    row, column = args.row, args.column
    img_size = args.imsize
    save_dir = './data/website/'
    all_frames = []
    for i in range(row * column):
        frames = []
        obs = env.reset()
        frames.append(env.get_image(img_size, img_size))
        for _ in range(env.horizon):
            action, _ = policy.get_action(obs)
            obs, _, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            frames.extend(info['flex_env_recorded_frames'])
        all_frames.append(frames)
    
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=1).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_name = args.env + '-' + args.policy + '-' + env.observation_mode + '-' + args.suffix + '.gif'
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))



def find_last_model(dir):
    onlyfiles = [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]
    models = []
    for f in onlyfiles:
        if 'pth' in f:
            number = f[f.find('_')+1: -4]
            models.append((f, int(number)))

    models.sort(key=lambda x: x[1])
    return models[-1][0]


def simulate_policy_PlaNet(args):
    policy_file = args.dir
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')
    np.random.seed(seed=seed)

    json_file = os.path.join(policy_file, 'variant.json')
    print(json_file)
    print('Load variants from {}'.format(json_file))
    with open(json_file) as f:
        vv = json.load(f)
    vv['env_kwargs']['headless'] = 1
    last_model = find_last_model(policy_file)
    vv['saved_models'] = os.path.join(policy_file, last_model)
    
    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'], vv['image_dim'],
              env_kwargs=vv['env_kwargs'])
    agent = PlaNetAgent(env, vv, device)

    agent.set_model_eval()
    row, column = args.row, args.column
    n_test_rollouts = row * column
    img_size = args.imsize
    with torch.no_grad():
        all_frames = []
        for i in range(n_test_rollouts):
            frames = []
            observation = agent.env.reset()
            frames.append(env.get_image(img_size, img_size))
            belief, posterior_state, action = torch.zeros(1, vv['belief_size'], device=device), \
                                              torch.zeros(1, vv['state_size'], device=device), \
                                              torch.zeros(1, env.action_size, device=device)
            for t in range(vv['env_kwargs']['horizon']):
                belief, posterior_state, action, next_observation, reward, done, info = \
                    agent.update_belief_and_act(agent.env, belief, posterior_state, action, observation.to(device=agent.device),
                                                explore=False)
                observation = next_observation
                frames.extend(info['flex_env_recorded_frames'])
            
            all_frames.append(frames)

    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=1).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_dir = './data/website'
    save_name = args.env + '-' + args.policy + '-' + agent.env.observation_mode + '-' + args.suffix + '.gif'
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--imsize', type=int, default=256)
    parser.add_argument('--policy', type=str, default='SAC')
    parser.add_argument('--row', type=int, default=1)
    parser.add_argument('--column', type=int, default=1)
    parser.add_argument('--env', type=str, default='PourWater')
    parser.add_argument('--suffix', type=str, default='2')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import pyflex
    headless, render, camera_width, camera_height = True, True, 720, 720
    pyflex.init(headless, render, camera_width, camera_height)

    if args.policy == 'SAC' or args.policy == 'TD3':
        simulate_policy_model_free(args)

    elif args.policy == 'PlaNet':
        simulate_policy_PlaNet(args)










