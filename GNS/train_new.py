import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GNS.models_graph_res import MaterialEncoder, Encoder, Processor, Decoder
from GNS.data_graph import PhysicsFleXDataset, ClothDataset

from GNS.utils import count_parameters

from GNS.env_config import env_configs
from chester import logger
import json
import os.path as osp
from GNS.visualize_data import get_model_prediction_rollout, visualize
from softgym.utils.visualization import save_numpy_as_gif

def get_default_args():

    # Setup configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--pstep', type=int, default=2)
    parser.add_argument('--n_rollout', type=int, default=0)
    parser.add_argument('--time_step', type=int, default=0)
    parser.add_argument('--time_step_clip', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1. / 60.)
    parser.add_argument('--nf_relation', type=int, default=300)
    parser.add_argument('--nf_particle', type=int, default=200)
    parser.add_argument('--nf_effect', type=int, default=200)
    parser.add_argument('--env', default='BoxBath')
    parser.add_argument('--train_valid_ratio', type=float, default=0.9)
    parser.add_argument('--outf', default='files')
    parser.add_argument('--dataf', default='data')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gen_data', type=int, default=1)
    parser.add_argument('--gen_stat', type=int, default=1)
    parser.add_argument('--log_per_iter', type=int, default=1000)
    parser.add_argument('--ckp_per_iter', type=int, default=10000)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--verbose_data', type=int, default=0)
    parser.add_argument('--verbose_model', type=int, default=0)

    parser.add_argument('--n_instance', type=int, default=0)
    parser.add_argument('--n_stages', type=int, default=0)
    parser.add_argument('--n_his', type=int, default=5)

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
    args = parser.parse_args([])
    return args

def train(args, env):
    # Create Models
    # material_encoder = MaterialEncoder(len(phases_dict['material'])) #Commented out for now
    phases_dict = args.phases_dict
    encoder_model = Encoder(args.state_dim + args.attr_dim, args.relation_dim).cuda()
    processor_model = Processor([3 * 128 + 1, 2 * 128 + 1, 2 * 128 + 1], use_global=False).cuda()
    decoder_model = Decoder().cuda()

    logdir = logger.get_dir()

    print("Created models")

    # Create Dataloaders

    datasets = {phase: ClothDataset(args, phase, phases_dict, env, args.verbose_data)
                for phase in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers)
        for x in ['train', 'valid']}

    print("Created dataloaders")

    # Load model from checkpoint (optional)

    if args.resume_epoch > 0 or args.resume_iter > 0:
        # mat_enc_path = os.path.join(logdir, 'mat_enc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        enc_path = os.path.join(logdir, 'enc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        proc_path = os.path.join(logdir, 'proc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        dec_path = os.path.join(logdir, 'dec_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))

        # material_encoder.load_state_dict(torch.load(mat_enc_path))
        encoder_model.load_state_dict(torch.load(enc_path))
        processor_model.load_state_dict(torch.load(proc_path))
        decoder_model.load_state_dict(torch.load(dec_path))

        print("Loaded saved ckp from epoch %d iter %d." % (args.resume_epoch, args.resume_iter))

    # Setup criterion and optimizer
    criterionMSE = nn.MSELoss()
    model_parameters = list(encoder_model.parameters()) + \
                       list(processor_model.parameters()) + list(decoder_model.parameters())
    optimizer = optim.Adam(model_parameters, lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    # Training loop

    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf
    for epoch in range(st_epoch, args.n_epoch):

        phases = ['train', 'valid'] if args.eval == 0 else ['valid']
        for phase in phases:

            # material_encoder.train()
            encoder_model.train()
            processor_model.train()
            decoder_model.train()

            losses = 0.
            for i, data in enumerate(dataloaders[phase]):

                node_attr, neighbors, edge_attr, global_feat, gt_accel = data

                node_attr = torch.squeeze(node_attr, dim=0).cuda()
                neighbors = torch.squeeze(neighbors, dim=0).cuda()
                edge_attr = torch.squeeze(edge_attr, dim=0).cuda()
                global_feat = torch.squeeze(global_feat, dim=0).cuda()
                gt_accel = gt_accel.cuda()

                with torch.set_grad_enabled(True):
                    node_embedding, edge_embedding = encoder_model(node_attr, edge_attr)
                    node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat,
                                                                                         batch=None)
                    pred_accel = decoder_model(node_embedding_out)

                loss = criterionMSE(torch.unsqueeze(pred_accel, dim=0), gt_accel)
                losses += np.sqrt(loss.item())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if i > 0 and i % args.log_per_iter == 0 or i == len(dataloaders[phase]) - 1:
                    # print("epoch {} i {}".format(epoch, i))
                    # print('%s [%d/%d][%d/%d] Loss: %.6f, Agg: %.6f' %
                    #       (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                    #        np.sqrt(loss.item()), losses / (i + 1)))
                    logger.record_tabular(phase + '/one_step_loss', np.sqrt(loss.item()))
                    logger.record_tabular(phase + '/one_step_loss_agg', losses / (i+1))

                    eval_time_beg = time.time()
                    nstep_eval_rollout = args.nstep_eval_rollout
                    data_folder = osp.join(args.dataf, phase)
                    traj_ids = np.random.choice(len(os.listdir(data_folder)), nstep_eval_rollout, replace=False)
                    pos_errorss = []
                    vel_errorss = []
                    for idx, traj_id in enumerate(traj_ids):
                        # print("idx: ", idx)
                        with torch.no_grad():
                            pos_errors, vel_errors, gt_positions, predicted_positions, shape_positions, sample_idx, config_id \
                                = get_model_prediction_rollout(args, data_folder, traj_id, args.n_his - 1, 
                                    [encoder_model, processor_model, decoder_model], datasets[phase])

                        pos_errorss.append(pos_errors)
                        vel_errorss.append(vel_errors)

                        if i % args.video_interval == 0 or i == len(dataloaders[phase]) - 1:
                            frames_model = visualize(datasets[phase].env, predicted_positions, 
                                shape_positions, config_id, sample_idx)
                            frames_gt = visualize(datasets[phase].env, gt_positions, 
                                shape_positions, config_id, sample_idx)
                            combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
                            save_numpy_as_gif(np.array(combined_frames), osp.join(logdir, '{}-{}-{}-{}.gif'.format(
                                phase, epoch, i, idx
                            )))

                    pos_errorss = np.vstack(pos_errorss)
                    vel_errorss = np.vstack(vel_errorss)
                    pos_error = np.mean(pos_errorss)
                    vel_error = np.mean(vel_errorss)

                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    axes = axes.reshape(-1)
                    for pos_errors in pos_errorss:
                        axes[0].plot(range(len(pos_errors)), pos_errors)
                    for vel_errors in vel_errorss:
                        axes[0].plot(range(len(vel_errors)), vel_errors)
                    axes[0].set_title("rollout position errors")
                    axes[1].set_title("rollout velocity errors")
                    axes[0].set_xlabel("rollout timestep")
                    axes[0].set_ylabel("error")
                    plt.savefig(osp.join(logdir, 'errors-{}-{}-{}.png'.format(phase, epoch, i)))
                    plt.close('all')

                    logger.record_tabular(phase + '/rollout_eval_time', time.time() - eval_time_beg)
                    logger.record_tabular(phase + '/rollout_pos_error', pos_error)
                    logger.record_tabular(phase + '/rollout_vel_error', vel_error)
                    logger.record_tabular(phase + '/epoch', epoch)
                    logger.record_tabular(phase + '/steps', i)
                    logger.dump_tabular()


                if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                    # mat_enc_path = '%s/mat_enc_net_epoch_%d_iter_%d.pth' % (logdir, epoch, i)
                    enc_path = '%s/enc_net_epoch_%d_iter_%d.pth' % (logdir, epoch, i)
                    proc_path = '%s/proc_net_epoch_%d_iter_%d.pth' % (logdir, epoch, i)
                    dec_path = '%s/dec_net_epoch_%d_iter_%d.pth' % (logdir, epoch, i)

                    # torch.save(material_encoder.state_dict(), mat_enc_path)
                    torch.save(encoder_model.state_dict(), enc_path)
                    torch.save(processor_model.state_dict(), proc_path)
                    torch.save(decoder_model.state_dict(), dec_path)

            losses /= len(dataloaders[phase])
            print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
                  (phase, epoch, args.n_epoch, losses, best_valid_loss))

            if phase == 'valid':

                scheduler.step(losses)

                if (losses < best_valid_loss):
                    best_valid_loss = losses

                    # torch.save(material_encoder.state_dict(), '%s/mat_enc_net_best.pth' % (logdir))
                    torch.save(encoder_model.state_dict(), '%s/enc_net_best.pth' % (logdir))
                    torch.save(processor_model.state_dict(), '%s/proc_net_best.pth' % (logdir))
                    torch.save(decoder_model.state_dict(), '%s/dec_net_best.pth' % (logdir))


def generate_dataset(args, env):
    os.system('mkdir -p ' + args.dataf)
    print("args.env_name: ", args.env_name)
    if args.env_name == 'ClothFlatten':
        datasets = {phase: ClothDataset(args, phase, args.phases_dict, env) for phase in ['train', 'valid']}
    else:
        raise NotImplementedError

    for phase in ['train', 'valid']:
        datasets[phase].create_dataset()
    print('Dataset generated in', args.dataf)


def run_task(vv, log_dir, exp_name):
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    args = get_default_args()
    args.__dict__.update(env_configs[vv['env_name']])
    args.__dict__.update(**vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    seed = vv['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import copy

    env_args = copy.deepcopy(env_arg_dict[args.env_name])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['action_repeat'] = 1
    # env_args['headless'] = False
    if args.env_name == 'ClothFlatten':
        env_args['cached_states_path'] = 'cloth_flatten_init_states_small_2.pkl'
        env_args['num_variations'] = 20
    env = SOFTGYM_ENVS[args.env_name](**env_args)

    if vv['gen_data']:
        generate_dataset(args, env)
    if vv['training']:
        train(args, env)
