import numpy as np
import torch
import os
import time
import json
import copy

from pouring.KPConv_sac import utils
from pouring.KPConv_sac.logger import Logger

from pouring.KPConv_sac.KPConv_SAC import KPConvSacAgent
from pouring.KPConv_sac.architectures import KPConvRegression, KPCNN_Encoder
from pouring.KPConv_sac.config import Modelnet40Config, Modelnet40DeformConfig
from pouring.KPConv_sac.curl_default_config import DEFAULT_CONFIG
from pouring.KPConv_sac.utils import classification_inputs, PointCloudCustomBatch

from experiments.planet.train import update_env_kwargs

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args

def update_config_from_vv(config, vv):
    for key, val in vv.items():
        if key.startswith('KPConv_config_'):
            # print(key)
            setattr(config, key[len('KPConv_config_'): ], val)
    # exit()

def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # update KPConv config
    KPConv_config = Modelnet40Config if not vv['KPConv_deform'] else Modelnet40DeformConfig
    config = copy.deepcopy(KPConv_config)
    update_config_from_vv(config, vv)
    
    # update curl config
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    main(vv_to_args(updated_vv), config)


def make_agent(args, config, device):
    setattr(config, 'output_dim', args.output_dim)
    encoder = KPCNN_Encoder(config)
    return KPConvRegression(
        encoder=encoder,
        config=config,
    ).to(device)


def handle_obs_batch(obs_list, config):
    points_list = obs_list
    stacked_points = np.concatenate(points_list, axis=0)
    stack_lengths = np.array([tp.shape[0] for tp in points_list], dtype=np.int32)

    # Input features
    stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)

    #######################
    # Create network inputs
    #######################
    #
    #   Points, neighbors, pooling indices for each layers
    #

    # Get the whole input list
    input_list = classification_inputs(config, 
                                        stacked_points,
                                        stacked_features,
                                        stack_lengths)

    obs_batch = PointCloudCustomBatch(input_list)
    return obs_batch

def main(args, config):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, 128, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: change to be a supervise learning model
    agent = make_agent(
        args=args,
        config=config,
        device=device
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    if args.load_data_path is not None:
        data = torch.load("{}.pkl".format(args.load_data_path))
        data_input, data_label = data
        N = len(data_input)

    else:
        # collect a bunch of data for training
        num_episodes = args.num_episodes
        N = num_episodes * env.horizon
        data_input = [] # rim point cloud
        data_label = [] # reduced state

        cnt = 0
        for episode_idx in range(num_episodes):
            if episode_idx % 10 == 0:
                print("collected trajectories: ", episode_idx)
            obs = env.reset()
            data_input.append(obs[:, :3])
            reduced_state = env.get_cup_reduced_state()
            data_label.append(reduced_state)
            cnt += 1
            for t in range(env.horizon):
                action = env.action_space.sample()
                obs, _, _, _ = env.step(action)
                data_input.append(obs[:, :3])
                reduced_state = env.get_cup_reduced_state()
                data_label.append(reduced_state)

        data_label = np.asarray(data_label, dtype=np.float)
        
        if args.save_data_path is not None:
            if not os.path.exists(args.save_data_path):
                os.makedirs(args.save_data_path, exist_ok=True)

            data = [data_input, data_label]
            torch.save(data, "{}.pkl".format(args.save_data_path))

    print("data collection done!")

    train_num = int(N * 0.9)
    data_input_train = data_input[:train_num]
    data_input_valid = data_input[train_num:]

    data_label_train = data_label[:train_num]
    data_label_valid = data_label[train_num:]

    # do the training
    print("start training!")
    for epoch_idx in range(args.train_epochs):
        agent.train()
        batch_num = train_num // args.batch_size
        loss_epoch = 0
        for batch_idx in range(batch_num):
            start_idx = batch_idx * args.batch_size
            end_idx = (batch_idx + 1) * args.batch_size
            x = []
            for idx in range(start_idx, end_idx):
                x.append(data_input_train[idx])
            label = data_label_train[start_idx:end_idx]

            obs_batch = handle_obs_batch(x, config).to(device)
            label_batch = torch.FloatTensor(label).to(device)

            predictions = agent(obs_batch)
            loss = torch.nn.functional.mse_loss(predictions, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item() if not isinstance(loss, float) else loss
            loss_epoch += loss_value

        print(f'epoch {epoch_idx}: train loss {loss_epoch/batch_num:.6f}')
        logger.record_tabular("training_loss", loss_epoch/batch_num)

        if epoch_idx % args.valid_interval == 0:
            agent.eval()
            with torch.no_grad():
                x = []
                for obs in data_input_valid:
                    x.append(obs)
                label = data_label_valid

                obs_batch = handle_obs_batch(x, config).to(device)
                predictions = agent(obs_batch).cpu().numpy()
                valid_error = np.mean((label - predictions) ** 2)

            logger.record_tabular("validation_loss", valid_error)
            print(f'epoch {epoch_idx}: valid loss {valid_error:.6f}')
            logger.dump_tabular()
                
        if epoch_idx % args.save_interval == 0:
            save_path = logger.get_dir()
            save_name = '{}.pkl'.format(epoch_idx)
            torch.save(agent.state_dict(), os.path.join(save_path, save_name))



if __name__ == '__main__':
    main()
