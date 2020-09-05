import h5py
import numpy as np
import torch
import os
from scipy import spatial

class PhysicsFleXDataset(torch.utils.data.Dataset):

    def __init__(self, args, phase, phases_dict, verbose=False):

        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')

        os.system('mkdir -p ' + self.data_dir)

        if args.env == 'BoxBath':
            self.data_names = ['positions', 'velocities', 'clusters']
        else:
            raise AssertionError("Unsupported env")

        ratio = self.args.train_valid_ratio

        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):

        return self.n_rollout * (self.args.time_step - self.args.n_his) # self.args.time_step - 1 - (self.args.n_his - 1)

    def load_data(self, name):

        self.stat = self._load_data_file(self.data_names[:2], self.stat_path)

        for i in range(len(self.stat)):
            self.stat[i] = self.stat[i][-self.args.position_dim:, :]
    
    def _load_data_file(self, data_names, path):

        hf = h5py.File(path, 'r')
        data = []
        for i in range(len(data_names)):
            d = np.array(hf.get(data_names[i]))
            data.append(d)
        hf.close()

        return data

    def __getitem__(self, idx):

        idx_rollout = idx // (self.args.time_step - self.args.n_his)
        idx_timestep = (self.args.n_his - 1) + idx % (self.args.time_step - self.args.n_his)

        data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')

        data = self._load_data_file(self.data_names, data_path)

        # Get velocity history
        vel_his = []
        for i in range(1, self.args.n_his):
            path = os.path.join(self.data_dir, str(idx_rollout), str(max(0, idx_timestep - i)) + '.h5') # max just in case
            data_his = self._load_data_file(self.data_names, path)
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)

        # Construct Graph
        node_attr, neighbors, edge_attr, global_feat = self._prepare_input(data)

        # Compute GT label: calculate accleration
        data_nxt = self._load_data_file(self.data_names, data_nxt_path)
        gt_accel = torch.FloatTensor((data_nxt[1] - data[1][:, 0:3]) / self.args.dt)

        return node_attr, neighbors, edge_attr, global_feat, gt_accel

    def _prepare_input(self, data):

        num_obj_class = len(self.phases_dict["material"])
        instance_idxes = self.phases_dict["instance_idx"]
        radius = self.phases_dict["radius"]

        if self.args.env == 'BoxBath':

            # Walls at x= 0, 1.25, y=0, z=0, 0.39

            pos, vel_hist, _ = data

            dist_to_walls = np.stack([np.absolute(pos[:, 0]),
                                      np.absolute(1.25-pos[:, 1]),
                                      np.absolute(pos[:, 1]),
                                      np.absolute(pos[:, 2]),
                                      np.absolute(0.39-pos[:, 2])], axis=1)
            dist_to_walls = np.minimum(radius, dist_to_walls)

            # Generate node attributes (previous C=5 velocity + material features)
            node_type_onehot = np.zeros(shape=(pos.shape[0], num_obj_class), dtype=np.float32)

            for i in range(num_obj_class):
                node_type_onehot[instance_idxes[i]:instance_idxes[i+1], i] = 1.

            f_i = torch.from_numpy(node_type_onehot)
            node_attr = torch.cat([torch.from_numpy(vel_hist),
                                   torch.from_numpy(dist_to_walls),
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