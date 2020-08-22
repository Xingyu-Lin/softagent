import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import rand_float, rand_int, query_yes_no
from utils import sample_control_RiceGrip, calc_shape_states_RiceGrip
from utils import calc_box_init_FluidShake, calc_shape_states_FluidShake
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS
import copy

import pyflex

from data_loader import store_data, load_data, init_stat, combine_stat, normalize, find_relations_neighbor, make_hierarchy


class PhysicsFleXDataset(Dataset):
    def __init__(self,
                 args,
                 phase,
                 phases_dict):
        self.args = args
        self.phases_dict = phases_dict
        self.phase = phase
        self.env_name = args.env
        self.root_num = phases_dict['root_num']
        self.num_workers = args.num_workers
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.time_step = args.time_step
        self.dt = args.dt
        self.pool = mp.Pool(processes=self.num_workers)
        self.data_names = None  # Task specific

        os.system('mkdir -p ' + self.data_dir)

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def create_env(self):
        env_args = copy.deepcopy(env_arg_dicts[self.env_name])
        env_args['render_mode'] = 'particle'
        env_args['camera_name'] = 'default_camera'
        if self.env_name == 'ClothFlatten':
            env_args['cached_states_path'] = 'cloth_flatten_small_init_states.pkl'
        return SOFTGYM_ENVS[self.env_name](**env_args)

    def create_dataset(self):
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.time_step))

        if os.path.exists(self.data_dir):
            # query_yes_no('Removing directory {}, confirmed?'.format(self.data_dir), default='yes')
            os.system('rm -rf {}'.format(self.data_dir))
        n_rollouts = [self.n_rollout // self.num_workers] * self.num_workers
        thread_idxes = np.arange(self.num_workers)
        data = self.pool.map(self._collect_worker, zip(n_rollouts, thread_idxes))

        if self.phase == 'train' and self.args.gen_stat:
            # store stats (mean, std, count) for position and velocity data
            # positions [x, y, z], velocities[xdot, ydot, zdot]
            if self.env_name == 'RiceGrip':
                self.stat = [init_stat(6), init_stat(6)]
            else:
                self.stat = [init_stat(3), init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:2], self.stat, self.stat_path)
        else:
            print("phase: {}".format(self.phase))
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:2], self.stat_path)

        print("stats warmed done!")

    def load_data(self):
        self.stat = load_data(self.data_names[:2], self.stat_path)
        for i in range(len(self.stat)):
            self.stat[i] = self.stat[i][-3:, :]

    def _construct_graph(self, data, stat, args, phases_dict, var):
        raise NotImplementedError

    def _collect_policy(self, env, timestep):
        """ Policy for collecting data"""
        return env.action_space.sample()

    def _prepare_policy(self, env):
        """ Doing something after env reset but before collecting any data"""
        pass

    def _collect_worker(self, args):
        """ Write data collection function for each task. Use random actions by default"""
        n_rollout, thread_idx = args
        np.random.seed(1000 + thread_idx)  ### NOTE: we might want to fix the seed for reproduction
        stats = [init_stat(3), init_stat(3)]
        env = self.create_env()

        for i in range(n_rollout):
            print("{} / {}".format(i, n_rollout))
            rollout_idx = thread_idx * n_rollout + i
            rollout_dir = os.path.join(self.data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)
            env.reset()
            self._prepare_policy(env)
            n_particles = pyflex.get_n_particles()
            n_shapes = pyflex.get_n_shapes()

            p = pyflex.get_positions().reshape(-1, 4)[:, :3]

            clusters = []
            st_time = time.time()
            kmeans = MiniBatchKMeans(n_clusters=self.root_num, random_state=0).fit(p)
            clusters.append([[kmeans.labels_]])

            positions = np.zeros((self.time_step, n_particles + n_shapes, 3), dtype=np.float32)
            velocities = np.zeros((self.time_step, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((self.time_step, n_shapes, 4), dtype=np.float32)

            positions[0, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
            shape_states = pyflex.get_shape_states().reshape(-1, 14)

            for k in range(n_shapes):
                positions[0, n_particles + k] = shape_states[k, :3]
                shape_quats[0, k] = shape_states[k, 6:10]

            data = [positions[0], velocities[0], shape_quats[0], clusters, [env.cloth_particle_radius]]
            store_data(self.data_names, data, os.path.join(rollout_dir, str(0) + '.h5'))

            for j in range(self.time_step):
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                action = self._collect_policy(env, j)
                env.step(action)
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, 14)

                for k in range(n_shapes):
                    positions[j, n_particles + k] = shape_states[k, :3]
                    shape_quats[j, k] = shape_states[k, 6:10]

                # NOTE: velocity is not directly using particle velocity in Pyflex
                # the main benefit of computing velocity in this way is that we can get the velocity of the shape.
                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / self.dt

                # NOTE: 1) particle + glass wall positions, 2) particle + glass wall velocitys, 3) glass wall rotations, 4) scenen parameters
                data = [positions[j], velocities[j], shape_quats[j], clusters, [env.cloth_particle_radius]]  # NOTE: radii is the sphere radius
                store_data(self.data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            # change dtype for more accurate stat calculation
            # only normalize positions and velocities
            datas = [positions.astype(np.float64), velocities.astype(np.float64)]

            # NOTE: stats is of length 2, for positions and velocities
            for j in range(len(stats)):
                stat = init_stat(stats[j].shape[0])
                stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
                stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
                stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
                stats[j] = combine_stat(stats[j], stat)
        return stats

    def __len__(self):
        return self.n_rollout * (self.time_step - 1)

    def __getitem__(self, idx):
        idx_rollout = idx // (self.time_step - 1)
        idx_timestep = idx % (self.time_step - 1)

        data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')
        data = load_data(self.data_names, data_path)

        # NOTE: include history velocity as part of the state. only used for RiceGrip
        vel_his = []
        for i in range(self.args.n_his):
            path = os.path.join(self.data_dir, str(idx_rollout), str(max(1, idx_timestep - i - 1)) + '.h5')
            data_his = load_data(self.data_names, path)
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)

        attr, state, relations, n_particles, n_shapes, instance_idx = self._construct_graph(data, self.stat, self.args, self.phases_dict, 0)

        ### label
        data_nxt = normalize(load_data(self.data_names, data_nxt_path), self.stat)

        label = torch.FloatTensor(data_nxt[1][:n_particles])  # NOTE: just use velocity at next step as label

        return attr, state, relations, n_particles, n_shapes, instance_idx, label

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']  # Remove pool for pickling
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class ClothDataset(PhysicsFleXDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']

    def _prepare_policy(self, env):
        """ Doing something after env reset but before collecting any data"""
        for i in range(20):
            action = (0, -0.005, 0, 1, 0, 0, 0, 0)
            env.step(action)

    def _collect_policy(self, env, timestep):
        """ Policy for collecting data"""
        if timestep < 20:
            action = (0, 0.005, 0, 1, 0, 0, 0, 0)
        else:
            action = (0, 0, 0, 0, 0, 0, 0, 0)
        return action

    def _construct_graph(self, data, stat, args, phases_dict, var):
        # Arrangement:
        # particles, shapes, roots

        positions, velocities, shape_quats, clusters, scene_params = data
        n_shapes = shape_quats.size(0) if var else shape_quats.shape[0]
        sphere_radius = scene_params[0]

        count_nodes = positions.size(0) if var else positions.shape[0]
        n_particles = count_nodes - n_shapes

        ### instance idx
        instance_idx = [0, n_particles]

        ### object attributes
        #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
        #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
        attr = np.zeros((count_nodes, args.attr_dim))
        # no need to include mass for now
        # attr[:, 6] = positions[:, -1].data.cpu().numpy() if var else positions[:, -1] # mass_inv

        ### construct relations
        Rr_idxs = []  # relation receiver idx list
        Rs_idxs = []  # relation sender idx list
        Ras = []  # relation attributes list
        values = []  # relation value list (should be 1) # NOTE: what is this relation value list? why is it set to be 1?
        node_r_idxs = []  # list of corresponding receiver node idx
        node_s_idxs = []  # list of corresponding sender node idx
        psteps = []  # propagation steps

        ##### add env specific graph components
        rels = []

        # connect each cloth particle to the sphere particle if their distance is smaller than 0.1
        for i in range(n_shapes):
            # object attr:
            # [fluid, root, sphere_0, sphere_1, sphere_2, sphere_3]
            attr[n_particles + i, 2 + i] = 1

            pos = positions.data.cpu().numpy() if var else positions
            sphere_center = pos[n_particles + i, :3]
            dis = np.sqrt(np.sum((pos[:n_particles, :3] - sphere_center) ** 2))
            nodes = np.nonzero(dis < 0.1 + sphere_radius)[0]

            wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]  # NOTE: [receiver, sender, value]
            # NOTE: actually the values are just set to one to construct a sparse receiver-relation matrix

        ##### add relations between leaf particles
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            # FluidShake object attr:
            # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
            if phases_dict['material'][i] == 'fluid':
                attr[st:ed, 0] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")

            # st_time = time.time()
            pos = positions
            pos = pos[:, -3:]
            rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
            # print("Time on neighbor search", time.time() - st_time)

        rels = np.concatenate(rels, 0)
        if rels.shape[0] > 0:
            Rr_idxs.append(torch.LongTensor([rels[:, 0], np.arange(rels.shape[0])]))  # NOTE: why with the np.arange(rels.shape): this
            Rs_idxs.append(torch.LongTensor([rels[:, 1], np.arange(rels.shape[
                                                                       0])]))  # This actually constructs the non-zero entry (row, col) idx for a later sparse matrix with shape (n_receiver, n_rel)
            Ra = np.zeros((rels.shape[0], args.relation_dim))  # NOTE: relation_dim is just 1  for all envs
            Ras.append(torch.FloatTensor(Ra))  # NOTE: why Ras are just 0? So all the attributes are just 0?
            values.append(torch.FloatTensor([1] * rels.shape[
                0]))  # NOTE: why values are just 1? the ones are values filled into the sparse receiver-relation matrix. see line 288 at train.py
            node_r_idxs.append(np.arange(n_particles))
            node_s_idxs.append(np.arange(n_particles + n_shapes))
            psteps.append(args.pstep)

        # add hierarchical relations per instance
        cnt_clusters = 0
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            n_root_level = len(phases_dict["root_num"][i])

            if n_root_level > 0:
                attr, positions, velocities, count_nodes, \
                rels, node_r_idx, node_s_idx, pstep = \
                    make_hierarchy(args.env, attr, positions, velocities, i, st, ed,
                                   phases_dict, count_nodes, clusters[cnt_clusters], 0, var)

                for j in range(len(rels)):
                    Rr_idxs.append(torch.LongTensor([rels[j][:, 0], np.arange(rels[j].shape[0])]))
                    Rs_idxs.append(torch.LongTensor([rels[j][:, 1], np.arange(rels[j].shape[0])]))
                    Ra = np.zeros((rels[j].shape[0], args.relation_dim));
                    Ra[:, 0] = 1
                    Ras.append(torch.FloatTensor(Ra))
                    values.append(torch.FloatTensor([1] * rels[j].shape[0]))
                    node_r_idxs.append(node_r_idx[j])
                    node_s_idxs.append(node_s_idx[j])
                    psteps.append(pstep[j])

                cnt_clusters += 1

        ### normalize data
        data = [positions, velocities]
        data = normalize(data, stat, var)
        positions, velocities = data[0], data[1]

        if var:
            state = torch.cat([positions, velocities], 1)
        else:
            state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))

        attr = torch.FloatTensor(attr)
        relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]  # NOTE: values are just all 1, and Ras are just all 0.

        return attr, state, relations, n_particles, n_shapes, instance_idx  # NOTE: attr are just object attributes, e.g, 0 for fluid, 1 for shape.
        # state = [positions, velocities], relations one line above.
