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
from GNS.data_graph import PhysicsFleXDataset

from utils import count_parameters

if __name__ == "__main__":

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
    parser.add_argument('--num_workers', type=int, default=10)
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

    args = parser.parse_args()

    phases_dict = dict()

    if args.env == 'BoxBath':
        args.n_rollout = 3000

        # object states:
        # [v_i, v_i-1, v_i-2, v_i-3, v_i-4, distance_to_wall (5)] (3*5 + 5)
        args.state_dim = 20
        args.position_dim = 0

        # object attr:
        # MaterialEncoder(One_Hot_Vector)
        args.attr_dim = 2

        # relation attr:
        # [(x_i - x_j), || x_i - x_j ||] (3 + 1)
        args.relation_dim = 4

        args.time_step = 151
        args.time_step_clip = 0
        args.n_instance = 2
        args.n_stages = 4

        args.neighbor_radius = 0.08

        # ball, fluid
        phases_dict["instance_idx"] = [0, 64, 1024]
        phases_dict["radius"] = 0.08
        phases_dict["instance"] = ['cube', 'fluid']
        phases_dict["material"] = ['rigid', 'fluid']

        args.outf = 'dump_BoxBath_res/' + args.outf

    else:
        raise AssertionError("Unsupported env")

    args.outf = args.outf + '_' + args.env
    args.dataf = 'data/' + args.dataf + '_' + args.env

    os.system('mkdir -p ' + args.outf)
    os.system('mkdir -p ' + args.dataf)

    print("Parsed args")

    # Create Models

    # material_encoder = MaterialEncoder(len(phases_dict['material'])) #Commented out for now
    encoder_model = Encoder(args.state_dim + args.attr_dim, args.relation_dim).cuda()
    processor_model = Processor([3 * 128 + 1, 2 * 128 + 1, 2 * 128 + 1]).cuda()
    decoder_model = Decoder().cuda()

    print("Created models")

    # Create Dataloaders

    datasets = {phase: PhysicsFleXDataset(args, phase, phases_dict, args.verbose_data)
                for phase in ['train', 'valid']}

    for phase in ['train', 'valid']:
        datasets[phase].load_data(args.env)

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers)
        for x in ['train', 'valid']}

    print("Created dataloaders")

    # Load model from checkpoint (optional)

    if args.resume_epoch > 0 or args.resume_iter > 0:
        # mat_enc_path = os.path.join(args.outf, 'mat_enc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        enc_path = os.path.join(args.outf, 'enc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        proc_path = os.path.join(args.outf, 'proc_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        dec_path = os.path.join(args.outf, 'dec_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))

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

                    # st_time = time.time()
                    node_embedding, edge_embedding = encoder_model(node_attr, edge_attr)
                    node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat,
                                                                                         batch=None)
                    pred_accel = decoder_model(node_embedding)
                    # print('Time forward', time.time() - st_time)

                    # print(predicted)
                    # print(label)

                loss = criterionMSE(torch.unsqueeze(pred_accel, dim=0), gt_accel)
                losses += np.sqrt(loss.item())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if i % args.log_per_iter == 0:
                    print('%s [%d/%d][%d/%d] Loss: %.6f, Agg: %.6f' %
                          (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                           np.sqrt(loss.item()), losses / (i + 1)))

                if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                    # mat_enc_path = '%s/mat_enc_net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                    enc_path = '%s/enc_net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                    proc_path = '%s/proc_net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                    dec_path = '%s/dec_net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)

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

                    # torch.save(material_encoder.state_dict(), '%s/mat_enc_net_best.pth' % (args.outf))
                    torch.save(encoder_model.state_dict(), '%s/enc_net_best.pth' % (args.outf))
                    torch.save(processor_model.state_dict(), '%s/proc_net_best.pth' % (args.outf))
                    torch.save(decoder_model.state_dict(), '%s/dec_net_best.pth' % (args.outf))
