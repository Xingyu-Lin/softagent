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

from data import load_data, prepare_input, normalize, denormalize
from models import DPINet
from utils import calc_box_init_FluidShake

from data import SOFTGYM_ENVS
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS as SOFTGYM_CUSTOM_ENVS
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='')
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
parser.add_argument('--n_his', type=int, default=0)

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

if args.env == 'FluidFall':
    env_idx = 4
    data_names = ['positions', 'velocities']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid]
    args.attr_dim = 1

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 121
    args.time_step_clip = 5
    args.n_instance = 1
    args.n_stages = 1

    args.neighbor_radius = 0.08

    phases_dict["instance_idx"] = [0, 189]
    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ['fluid']
    phases_dict["material"] = ['fluid']

    args.outf = 'dump_FluidFall/' + args.outf
    args.evalf = 'dump_FluidFall/' + args.evalf

elif args.env == 'BoxBath':
    env_idx = 1
    data_names = ['positions', 'velocities', 'clusters']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 151
    args.time_step_clip = 0
    args.n_instance = 2
    args.n_stages = 4

    args.neighbor_radius = 0.08

    # ball, fluid
    phases_dict["instance_idx"] = [0, 64, 1024]
    phases_dict["root_num"] = [[8], []]
    phases_dict["root_sib_radius"] = [[0.4], []]
    phases_dict["root_des_radius"] = [[0.08], []]
    phases_dict["root_pstep"] = [[args.pstep], []]
    phases_dict["instance"] = ['cube', 'fluid']
    phases_dict["material"] = ['rigid', 'fluid']

    args.outf = 'dump_BoxBath/' + args.outf
    args.evalf = 'dump_BoxBath/' + args.evalf

elif args.env == 'FluidShake':
    env_idx = 6
    data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']

    height = 1.0
    border = 0.025

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 301
    args.time_step_clip = 0
    args.n_stages = 2

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_FluidShake/' + args.outf
    args.evalf = 'dump_FluidShake/' + args.evalf

elif args.env == 'PassWater':
    data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 75
    args.time_step_clip = 0
    args.n_stages = 2

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_PassWater/' + args.outf
    args.evalf = 'dump_PassWater/' + args.evalf

elif args.env == 'ClothFlatten':
    data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [cloth, root, sphere 0 - 3]
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 100
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_roots = 30

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[args.n_roots]]
    phases_dict["root_sib_radius"] = [[5.0]] # not actually used
    phases_dict["root_des_radius"] = [[0.2]] # not actually used
    phases_dict["root_pstep"] = [[args.pstep]]
    phases_dict["instance"] = ["fluid"] # not actually used
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_ClothFlatten/' + args.outf
    args.evalf = 'dump_ClothFlatten/' + args.evalf

elif args.env == 'RiceGrip':
    env_idx = 5
    data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']

    args.n_his = 3

    # object state:
    # [rest_x, rest_y, rest_z, rest_xdot, rest_ydot, rest_zdot,
    #  x, y, z, xdot, ydot, zdot, quat.x, quat.y, quat.z, quat.w]
    args.state_dim = 16 + 6 * args.n_his
    args.position_dim = 6

    # object attr:
    # [fluid, root, gripper_0, gripper_1,
    #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
    args.attr_dim = 7

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 41
    args.time_step_clip = 0
    args.n_stages = 4

    args.neighbor_radius = 0.08
    args.n_roots = 30

    phases_dict["root_num"] = [[args.n_roots]]
    phases_dict["root_sib_radius"] = [[5.0]]
    phases_dict["root_des_radius"] = [[0.2]]
    phases_dict["root_pstep"] = [[args.pstep]]
    phases_dict["instance"] = ["rice"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_RiceGrip/' + args.outf
    args.evalf = 'dump_RiceGrip/' + args.evalf

else:
    raise AssertionError("Unsupported env")


args.outf = args.outf + '_' + args.env
args.evalf = args.evalf + '_' + args.env
args.dataf = 'data/' + args.dataf + '_' + args.env

print(args)


print("Loading stored stat from %s" % args.dataf)
stat_path = os.path.join(args.dataf, 'stat.h5')
stat = load_data(data_names[:2], stat_path)
for i in range(len(stat)):
    stat[i] = stat[i][-args.position_dim:, :]
    # print(data_names[i], stat[i].shape)


use_gpu = torch.cuda.is_available()

model = DPINet(args, stat, phases_dict, residual=True, use_gpu=use_gpu)

if args.epoch == 0 and args.iter == 0:
    model_file = os.path.join(args.outf, 'net_best.pth')
else:
    model_file = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

print("Loading network from %s" % model_file)
model.load_state_dict(torch.load(model_file))
model.eval()

criterionMSE = nn.MSELoss()

if use_gpu:
    model.cuda()


infos = np.arange(1)

import pyflex
if args.env not in SOFTGYM_ENVS:
    pyflex.init(0, 1, 720, 720)    

recs = []

for idx in range(len(infos)):

    print("Rollout %d / %d" % (idx, len(infos)))
    des_dir = os.path.join(args.evalf, 'rollout_%d' % idx)
    if not os.path.exists(des_dir):
        # os.system('mkdir -p ' + des_dir)
        os.makedirs(des_dir)

    # ground truth
    for step in range(args.time_step - 1):
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step) + '.h5')
        data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')

        data = load_data(data_names, data_path)
        data_nxt = load_data(data_names, data_nxt_path)
        velocities_nxt = data_nxt[1]

        if step == 0:
            if args.env == 'BoxBath':
                positions, velocities, clusters = data
                n_shapes = 0
                scene_params = np.zeros(1)
            elif args.env == 'FluidFall':
                positions, velocities = data
                n_shapes = 0
                scene_params = np.zeros(1)
            elif args.env == 'RiceGrip' or args.env == 'ClothFlatten':
                positions, velocities, shape_quats, clusters, scene_params = data
                n_shapes = shape_quats.shape[0]
            elif args.env == 'FluidShake' :
                positions, velocities, shape_quats, scene_params = data
                n_shapes = shape_quats.shape[0]
            elif args.env == 'PassWater':
                positions, velocities, shape_quats, config_idx = data
                n_shapes = shape_quats.shape[0]
            else:
                raise AssertionError("Unsupported env")

            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            print("n_particles", n_particles)
            print("n_shapes", n_shapes)

            p_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))
            s_gt = np.zeros((args.time_step - 1, n_shapes, args.shape_state_dim))
            v_nxt_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

            p_pred = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

        p_gt[step] = positions[:, -args.position_dim:]
        v_nxt_gt[step] = velocities_nxt[:, -args.position_dim:]

        # print(step, np.sum(np.abs(v_nxt_gt[step, :args.n_particles])))

        if args.env == 'RiceGrip' or args.env == 'FluidShake' or args.env == 'ClothFlatten' or args.env == 'PassWater':
            s_gt[step, :, :3] = positions[n_particles:, :3]
            s_gt[step, :, 3:6] = p_gt[max(0, step-1), n_particles:, :3]
            s_gt[step, :, 6:10] = data[2]
            s_gt[step, :, 10:] = data[2]

        positions = positions + velocities_nxt * args.dt
        print("ground truth positions[0]: ", positions[0])

    # model rollout
    data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), '0.h5')
    data = load_data(data_names, data_path)

    for step in range(args.time_step - 1):
        if step % 10 == 0:
            print("Step %d / %d" % (step, args.time_step - 1))

        p_pred[step] = data[0]

        if args.env == 'RiceGrip' and step == 0:
            data[0] = p_gt[step + 1].copy()
            data[1] = np.concatenate([v_nxt_gt[step]] * (args.n_his + 1), 1)
            continue

        # st_time = time.time()
        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)

        Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

        Rr, Rs = [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        buf = [attr, state, Rr, Rs, Ra]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t].cuda())
                    else:
                        buf[d] = Variable(buf[d].cuda())
            else:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t])
                    else:
                        buf[d] = Variable(buf[d])

            attr, state, Rr, Rs, Ra = buf
            # print('Time prepare input', time.time() - st_time)

            # st_time = time.time()
            vels = model(
                attr, state, Rr, Rs, Ra, n_particles,
                node_r_idx, node_s_idx, pstep,
                instance_idx, phases_dict, args.verbose_model)
            # print('Time forward', time.time() - st_time)

            # print(vels)

            if args.debug:
                data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')
                data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
                label = Variable(torch.FloatTensor(data_nxt[1][:n_particles]).cuda())
                # print(label)
                loss = np.sqrt(criterionMSE(vels, label).item())
                print(loss)

        vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]

        if args.env == 'RiceGrip' or args.env == 'FluidShake' or args.env == 'ClothFlatten' or args.env == 'PassWater':
            vels = np.concatenate([vels, v_nxt_gt[step, n_particles:]], 0) ### Model only outputs predicted particle velocity,
            ### so here we use the ground truth shape velocity. Why doesn't the model also predict the shape velocity?
            ### maybe, the shape velocity is kind of like the control actions specified by the users
        data[0] = data[0] + vels * args.dt

        if args.env == 'RiceGrip':
            # shifting the history
            # positions, restPositions
            data[1][:, args.position_dim:] = data[1][:, :-args.position_dim]
        data[1][:, :args.position_dim] = vels

        if args.debug:
            data[0] = p_gt[step + 1].copy()
            data[1][:, :args.position_dim] = v_nxt_gt[step]

    ##### render for the ground truth
    import time
    print("ground truth rendering!")
    time.sleep(5)
    if args.env not in SOFTGYM_ENVS:
        pyflex.set_scene(env_idx, scene_params, 0)

    if args.env == 'RiceGrip':
        halfEdge = np.array([0.15, 0.8, 0.15])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)
    elif args.env == 'FluidShake':
        x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z = scene_params
        boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)

        x_box = x + (dim_x-1)/2.*0.055

        for box_idx in range(len(boxes) - 1):
            halfEdge = boxes[box_idx][0]
            center = boxes[box_idx][1]
            quat = boxes[box_idx][2]
            pyflex.add_box(halfEdge, center, quat)
    
    elif args.env == 'PassWater':
        tmp_args = copy.deepcopy(env_arg_dicts[args.env])
        tmp_args['headless'] = False
        tmp_args['render_mode'] = 'particle'
        tmp_args['camera_name'] = 'cam_2d'
        softgym_env = SOFTGYM_CUSTOM_ENVS[args.env](**tmp_args)
        softgym_env.reset(config_idx)
        pyflex.pop_box(1)
        print("recorded n_particles {} env actual n_particles {}".format(n_particles, pyflex.get_n_particles()))

    elif args.env == 'ClothFlatten':
        import gym, softgym
        softgym.register_flex_envs()
        env = gym.make("ClothFlattenSphereControl-v0")
        env.reset()

    for step in range(args.time_step - 1):
        print("ground truth render step: ", step)
       
        mass = np.ones((n_particles, 1))
        if args.env == 'RiceGrip':
            p = np.concatenate([p_gt[step, :n_particles, -3:], mass], 1)
        else:
            p = np.concatenate([p_gt[step, :n_particles], mass], 1)

        print(np.mean(p, axis=0))
        pyflex.set_positions(p.copy())
        # pyflex.step()
        # p_after = pyflex.get_positions().reshape(-1, 4)
        # if not np.all(p_after == p):
        #     print("after step, particle positions do change!")
        #     exit()
        if args.env == 'RiceGrip' or args.env == 'ClothFlatten':
            pyflex.set_shape_states(s_gt[step])
        elif args.env == 'FluidShake' or args.env == 'PassWater': # remove the front wall of the glass for rendering
            pyflex.set_shape_states(s_gt[step])

        pyflex.render(capture=1, path=os.path.join(des_dir, 'gt_%d.tga' % step))
        time.sleep(0.1)

    ##### render for the predictions
    print("prediction rendering!")
    time.sleep(5)
    if args.env not in SOFTGYM_ENVS:
        pyflex.set_scene(env_idx, scene_params, 0)
    else:
        softgym_env.reset(config_idx)
        pyflex.pop_box(1)

    if args.env == 'RiceGrip':
        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)
    elif args.env == 'FluidShake':
        for box_idx in range(len(boxes) - 1):
            halfEdge = boxes[box_idx][0]
            center = boxes[box_idx][1]
            quat = boxes[box_idx][2]
            pyflex.add_box(halfEdge, center, quat)

    for step in range(args.time_step - 1):
        if args.env == 'RiceGrip' or args.env == 'ClothFlatten':
            pyflex.set_shape_states(s_gt[step])
        elif args.env == 'FluidShake' or args.env == 'PassWater':
            pyflex.set_shape_states(s_gt[step, :-1])

        mass = np.zeros((n_particles, 1))
        if args.env == 'RiceGrip':
            p = np.concatenate([p_pred[step, :n_particles, -3:], mass], 1)
        else:
            p = np.concatenate([p_pred[step, :n_particles], mass], 1)

        pyflex.set_positions(p)
        pyflex.render(capture=1, path=os.path.join(des_dir, 'pred_%d.tga' % step))
        time.sleep(0.1)

pyflex.clean()

