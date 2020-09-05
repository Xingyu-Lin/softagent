import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models_graph_res import MaterialEncoder, Encoder, Processor, Decoder
from scipy import spatial

def load_data_file(data_names, path):

    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()

    return data

def prepare_input(data, phases_dict, args):

    num_obj_class = len(phases_dict["material"])
    instance_idxes = phases_dict["instance_idx"]
    radius = phases_dict["radius"]

    if args.env == 'BoxBath':

        # Walls at x= 0, 1.25, y=0, z=0, 0.39

        pos, vel_hist, _ = data

        dist_to_walls = np.stack([np.absolute(pos[:, 0]),
                                    np.absolute(1.25-pos[:, 1]),
                                    np.absolute(pos[:, 1]),
                                    np.absolute(pos[:, 2]),
                                    np.absolute(0.39-pos[:, 2])], axis=1)
        dist_to_walls = np.minimum(radius, dist_to_walls)

        # Generate node attributes (previous C=5 velocity + material features)
        node_type_onehot = np.zeros(shape=(pos.shape[0], num_obj_class))

        for i in range(num_obj_class):
            node_type_onehot[instance_idxes[i]:instance_idxes[i+1], i] = 1.

        f_i = torch.from_numpy(node_type_onehot).float()
        node_attr = torch.cat([torch.from_numpy(vel_hist.astype(np.float32)),
                                torch.from_numpy(dist_to_walls.astype(np.float32)),
                                f_i], axis=1)

        # Calculate undirected edge list and corresponding relative edge attributes (distance vector + magnitude)
        point_tree = spatial.cKDTree(pos)
        undirected_neighbors = np.array(list(point_tree.query_pairs(radius, p=2))).T
        dist_vec = pos[undirected_neighbors[0, :]] - pos[undirected_neighbors[1, :]]
        dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
        edge_attr = np.concatenate([dist_vec, dist], axis=1)

        # Generate directed edge list and corresponding edge attributes
        neighbors = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
        edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr]))

        # Global features are unused
        global_feat = torch.FloatTensor([[0.]])

    else:
        raise AssertionError("Unsupported env")

    return node_attr, neighbors, edge_attr, global_feat

if __name__ == "__main__":
    
    # Setup configs

    parser = argparse.ArgumentParser()
    parser.add_argument('--pstep', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--env', default='BoxBath')
    parser.add_argument('--time_step', type=int, default=0)
    parser.add_argument('--time_step_clip', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1./60.)
    parser.add_argument('--nf_relation', type=int, default=300)
    parser.add_argument('--nf_particle', type=int, default=200)
    parser.add_argument('--nf_effect', type=int, default=200)
    parser.add_argument('--outf', default='files')
    parser.add_argument('--dataf', default='data')
    parser.add_argument('--evalf', default='eval')
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--verbose_data', type=int, default=0)
    parser.add_argument('--verbose_model', type=int, default=0)

    parser.add_argument('--debug', type=int, default=0)

    parser.add_argument('--n_instances', type=int, default=0)
    parser.add_argument('--n_stages', type=int, default=0)
    parser.add_argument('--n_his', type=int, default=5)

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
        env_idx = 1
        data_names = ['positions', 'velocities', 'clusters']

        # object states:
        # [v_i, v_i-1, v_i-2, v_i-3, v_i-4, distance_to_wall (5)]
        args.state_dim = 20
        args.position_dim = 3

        # object attr:
        # MaterialEncoder(One_Hot_Vector)
        args.attr_dim = 2

        # relation attr:
        # [(x_i - x_j), || x_i - x_j ||]
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

        args.outf = 'dump_BoxBath/' + args.outf
        args.evalf = 'dump_BoxBath/' + args.evalf

    else:
        raise AssertionError("Unsupported env")


    args.outf = args.outf + '_' + args.env
    args.evalf = args.evalf + '_' + args.env
    args.dataf = 'data/' + args.dataf + '_' + args.env

    print ("Parsed args")

    stat_path = os.path.join(args.dataf, 'stat.h5')
    hf = h5py.File(stat_path, 'r')
    stat = []
    for i in range(len(data_names[:2])):
        d = np.array(hf.get(data_names[:2][i]))
        stat.append(d)
    hf.close()
    for i in range(len(stat)):
        stat[i] = stat[i][-args.position_dim:, :]
        # print(data_names[i], stat[i].shape)

    print("Loaded stored stat from %s" % args.dataf)

    # Create Models

    #material_encoder = MaterialEncoder(len(phases_dict['material'])) #Commented out for now
    encoder_model = Encoder(args.state_dim+args.attr_dim, args.relation_dim).cuda()
    processor_model = Processor([3*128+1, 2*128+1, 2*128+1]).cuda()
    decoder_model = Decoder().cuda()

    # Load model from checkpoint

    if args.epoch == 0 and args.iter == 0:

        #mat_enc_path = os.path.join(args.outf, 'mat_enc_net_best.pth')
        enc_path = os.path.join(args.outf, 'enc_net_best.pth')
        proc_path = os.path.join(args.outf, 'proc_net_best.pth')
        dec_path = os.path.join(args.outf, 'dec_net_best.pth')

        print("Loaded saved best ckpt.")

    else:

        #mat_enc_path = os.path.join(args.outf, 'mat_enc_net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))
        enc_path = os.path.join(args.outf, 'enc_net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))
        proc_path = os.path.join(args.outf, 'proc_net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))
        dec_path = os.path.join(args.outf, 'dec_net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

        print("Loaded saved ckpt from epoch %d iter %d." % (args.resume_epoch, args.resume_iter))

    #material_encoder.load_state_dict(torch.load(mat_enc_path))
    encoder_model.load_state_dict(torch.load(enc_path))
    processor_model.load_state_dict(torch.load(proc_path))
    decoder_model.load_state_dict(torch.load(dec_path))

    # Set to eval mode

    encoder_model.eval()
    processor_model.eval()
    decoder_model.eval()

    criterionMSE = nn.MSELoss()

    infos = np.arange(5)

    import pyflex
    pyflex.init(True, False)

    recs = []

    for idx in range(len(infos)):

        print("Rollout %d / %d" % (idx, len(infos)))
        des_dir = os.path.join(args.evalf, 'rollout_%d' % idx)
        os.system('mkdir -p ' + des_dir)

        # GT
        for step in range(args.time_step - 1):
            data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step) + '.h5')
            data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')

            data = load_data_file(data_names, data_path)
            data_nxt = load_data_file(data_names, data_nxt_path)
            velocities_nxt = data_nxt[1]

            if step == 0:

                if args.env == 'BoxBath':
                    positions, velocities, _ = data
                    scene_params = np.zeros(1)
                else:
                    raise AssertionError("Unsupported env")

                n_particles = positions.shape[0]

                pos_gt = np.zeros((args.time_step - 1, n_particles, args.position_dim))
                vel_nxt_gt = np.zeros((args.time_step - 1, n_particles, args.position_dim))

                pos_pred = np.zeros((args.time_step - 1, n_particles, args.position_dim))

            pos_gt[step] = positions
            vel_nxt_gt[step] = velocities_nxt

            positions = positions + velocities_nxt * args.dt

        # Prediction
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), '5.h5')
        data = load_data_file(data_names, data_path)
        pos_pred[0:5] = pos_gt[0:5]

        for step in range(5, args.time_step - 1):

            if step % 10 == 0:
                print("Step %d / %d" % (step, args.time_step - 1))

            pos_pred[step] = data[0]

            vel_his = []
            for i in range(args.n_his):
                vel_his.append(vel_nxt_gt[step-i, :, :])
            data[1] = np.concatenate(vel_his, axis=1)

            node_attr, neighbors, edge_attr, global_feat = prepare_input(data, phases_dict, args)

            with torch.set_grad_enabled(False):
                    
                node_attr = node_attr.cuda()
                neighbors = neighbors.cuda()
                edge_attr = edge_attr.cuda()
                global_feat = global_feat.cuda()

                # st_time = time.time()
                node_embedding, edge_embedding = encoder_model(node_attr.float(), edge_attr.float())
                node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat, batch=None)
                pred_accel = decoder_model(node_embedding)
                # print('Time forward', time.time() - st_time)

            pred_vel = vel_nxt_gt[step] + pred_accel.cpu().numpy() * args.dt
            data[0] = data[0] + pred_vel * args.dt

        # Render for GT
        pyflex.set_scene(env_idx, scene_params, 0)

        for step in range(args.time_step - 1):

            mass = np.zeros((n_particles, 1))
            p = np.concatenate([pos_gt[step], mass], 1)

            pyflex.set_positions(p)
            _ = pyflex.render(capture=1, path=os.path.join(des_dir, 'gt_%d.tga' % step))

        # Render for predictions

        pyflex.set_scene(env_idx, scene_params, 0)

        for step in range(args.time_step - 1):

            mass = np.zeros((n_particles, 1))
            p = np.concatenate([pos_pred[step], mass], 1)

            pyflex.set_positions(p)
            pyflex.render(capture=1, path=os.path.join(des_dir, 'pred_%d.tga' % step))

    pyflex.clean()

