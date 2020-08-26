from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DPINet.models import DPINet
from DPINet.data import collate_fn
from DPINet.graph import ClothDataset
from DPINet.utils import count_parameters

from softgym.registered_env import env_arg_dict
from softgym.registered_env import SOFTGYM_ENVS
from DPINet.configs import env_configs


def get_default_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pstep', type=int, default=2)
    parser.add_argument('--n_rollout', type=int, default=0)
    parser.add_argument('--time_step', type=int, default=0)
    parser.add_argument('--time_step_clip', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1. / 60.)
    parser.add_argument('--nf_relation', type=int, default=300)
    parser.add_argument('--nf_particle', type=int, default=200)
    parser.add_argument('--nf_effect', type=int, default=200)
    parser.add_argument('--env', default='')
    parser.add_argument('--train_valid_ratio', type=float, default=0.9)
    parser.add_argument('--outf', default='files')
    parser.add_argument('--dataf', default='data')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gen_data', type=int, default=0)
    parser.add_argument('--gen_stat', type=int, default=1)
    parser.add_argument('--log_per_iter', type=int, default=200)
    parser.add_argument('--ckp_per_iter', type=int, default=10000)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--verbose_data', type=int, default=0)
    parser.add_argument('--verbose_model', type=int, default=0)

    parser.add_argument('--n_instance', type=int, default=0)  ### an instance is an object. E.g. for fluid shake, instance is water + glass.
    parser.add_argument('--n_stages', type=int, default=0)
    parser.add_argument('--n_his', type=int, default=0)

    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--forward_times', type=int, default=2)

    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--resume_iter', type=int, default=0)

    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    parser.add_argument('--shape_state_dim', type=int, default=14)

    # object attributes:
    parser.add_argument('--attr_dim', type=int, default=0)

    # object state:
    parser.add_argument('--state_dim', type=int, default=0)
    parser.add_argument('--position_dim', type=int, default=0)

    # relation attr:
    parser.add_argument('--relation_dim', type=int, default=0)

    # Graph construction
    parser.add_argument('--gt_edge', type=bool, default=1)

    args = parser.parse_args([])
    return args


def generate_dataset(args):
    # env_args = env_arg_dict[args.env_name]
    # env_args['render_mode'] = 'particle'
    # env_args['camera_name'] = 'default_camera'
    # env_args['action_repeat'] = 2
    # if args.env_name == 'ClothFlatten':
    #     env_args['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
    # env = SOFTGYM_ENVS[args.env_name](**env_args)
    os.system('mkdir -p ' + args.dataf)
    if args.env_name == 'ClothFlatten':
        datasets = {phase: ClothDataset(args, phase, args.phases_dict) for phase in ['train', 'valid']}
    else:
        raise NotImplementedError

    for phase in ['train', 'valid']:
        datasets[phase].create_dataset()
    print('Dataset generated in', args.dataf)


def train(args):
    # env_args = env_arg_dict[args.env_name]
    # env_args['render_mode'] = 'particle'
    # env_args['camera_name'] = 'default_camera'
    # env_args['action_repeat'] = 2
    # if args.env_name == 'ClothFlatten':
    #     env_args['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
    # env = SOFTGYM_ENVS[args.env_name](**env_args)

    if args.env_name == 'ClothFlatten':
        datasets = {phase: ClothDataset(args, phase, args.phases_dict) for phase in ['train', 'valid']}
    else:
        raise NotImplementedError

    for phase in ['train', 'valid']:
        datasets[phase].load_data(args.env)

    for phase in ['train', 'valid']:
        if args.gen_data:
            datasets[phase].create_dataset()
        else:
            datasets[phase].load_data(args.env)

    use_gpu = torch.cuda.is_available()
    print("data generate done!")

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

    print("dataloaders generate done!")

    # define propagation network
    model = DPINet(args, datasets['train'].stat, args.phases_dict, residual=True, use_gpu=use_gpu)

    print("Number of parameters: %d" % count_parameters(model))

    if args.resume_epoch > 0 or args.resume_iter > 0:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)
        model.load_state_dict(torch.load(model_path))

    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    if use_gpu:
        model = model.cuda()
        criterionMSE = criterionMSE.cuda()

    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf
    time.sleep(5)
    for epoch in range(st_epoch, args.n_epoch):

        phases = ['train', 'valid'] if args.eval == 0 else ['valid']
        for phase in phases:

            model.train(phase == 'train')

            losses = 0.
            for i, data in enumerate(dataloaders[phase]):
                attr, state, rels, n_particles, n_shapes, instance_idx, label = data
                Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

                Rr, Rs = [], []

                # print("rels[0] len: ", len(rels[0]))
                for j in range(len(rels[0])):
                    Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]  # NOTE: values are all just 1
                    # print("node_r_idx[j].shape[0]: ", node_r_idx[j].shape[0])
                    # print("Ra[j].size(0): ", Ra[j].size(0))
                    Rr.append(torch.sparse.FloatTensor(  # NOTE: Ra = np.zeros((rels.shape[0], args.relation_dim)), relation_dim is just 1
                        Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
                    Rs.append(torch.sparse.FloatTensor(
                        Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))
                    # NOTE: this is a matrix of size (n_receiver, n_rel). in each column (relationship), the receiver idx row is set to be 1.

                data = [attr, state, Rr, Rs, Ra, label]

                with torch.set_grad_enabled(phase == 'train'):
                    if use_gpu:
                        for d in range(len(data)):
                            if type(data[d]) == list:
                                for t in range(len(data[d])):
                                    data[d][t] = Variable(data[d][t].cuda())
                            else:
                                data[d] = Variable(data[d].cuda())
                    else:
                        for d in range(len(data)):
                            if type(data[d]) == list:
                                for t in range(len(data[d])):
                                    data[d][t] = Variable(data[d][t])
                            else:
                                data[d] = Variable(data[d])

                    attr, state, Rr, Rs, Ra, label = data

                    # st_time = time.time()
                    predicted = model(
                        attr, state, Rr, Rs, Ra, n_particles,
                        node_r_idx, node_s_idx, pstep,
                        instance_idx, phases_dict, args.verbose_model)
                    # print('Time forward', time.time() - st_time)

                    # print(predicted)
                    # print(label)

                loss = criterionMSE(predicted, label)
                losses += np.sqrt(loss.item())

                if phase == 'train':
                    if i % args.forward_times == 0:
                        # update parameters every args.forward_times
                        if i != 0:
                            loss_acc /= args.forward_times
                            optimizer.zero_grad()
                            loss_acc.backward()
                            optimizer.step()
                        loss_acc = loss
                    else:
                        loss_acc += loss

                if i % args.log_per_iter == 0:
                    n_relations = 0
                    for j in range(len(Ra)):
                        n_relations += Ra[j].size(0)
                    print('%s [%d/%d][%d/%d] n_relations: %d, Loss: %.6f, Agg: %.6f' %
                          (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                           n_relations, np.sqrt(loss.item()), losses / (i + 1)))

                if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                    torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

            losses /= len(dataloaders[phase])
            print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
                  (phase, epoch, args.n_epoch, losses, best_valid_loss))

            if phase == 'valid':
                scheduler.step(losses)
                if (losses < best_valid_loss):
                    best_valid_loss = losses
                    torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))


def run_task(vv, log_dir, exp_name):
    import multiprocessing as mp
    mp.set_start_method('spawn')
    args = get_default_args()
    args.__dict__.update(env_configs[vv['env_name']])
    args.__dict__.update(**vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed(vv['seed'])

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    if vv['gen_data']:
        generate_dataset(args)
    else:
        train(args)
