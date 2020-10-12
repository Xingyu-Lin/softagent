import h5py
import numpy as np
import torch
import os
from scipy import spatial

import multiprocessing as mp
from softgym.registered_env import env_arg_dict
from softgym.registered_env import SOFTGYM_ENVS
import copy

import pyflex
import scipy

pool = mp.Pool(processes=10)

class PhysicsFleXDataset(torch.utils.data.Dataset):

    def __init__(self, args, phase, phases_dict, env=None, verbose=False):

        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose
        self.env = env
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')

        self.time_step = args.time_step
        self.num_workers = args.num_workers
        
        self.env_name = args.env_name
        self.dt = args.dt

        os.system('mkdir -p ' + self.data_dir)

        # if args.env_name == 'BoxBath':
        #     self.data_names = ['positions', 'velocities', 'clusters']

        ratio = self.args.train_valid_ratio

        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):

        return self.n_rollout * (self.args.time_step - self.args.n_his) # self.args.time_step - 1 - (self.args.n_his - 1)

    def load_stat(self, name):

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

    def store_data(self, data_names, data, path):
        hf = h5py.File(path, 'w')
        for i in range(len(data_names)):
            hf.create_dataset(data_names[i], data=data[i])
        hf.close()

    def __getitem__(self, idx):

        idx_rollout = idx // (self.args.time_step - self.args.n_his)
        idx_timestep = (self.args.n_his - 1) + idx % (self.args.time_step - self.args.n_his)

        data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')

        if not self.args.train_rollout:
            load_names = ['positions', 'velocities', 'picked_points', 'picked_point_positions', 'scene_params']
        else:
            load_names = ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'shape_positions']

        data = self._load_data_file(load_names, data_path)

        # Get velocity history
        vel_his = []
        for i in range(1, self.args.n_his):
            path = os.path.join(self.data_dir, str(idx_rollout), str(max(0, idx_timestep - i)) + '.h5') # max just in case
            data_his = self._load_data_file(load_names, path)
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)

        # Construct Graph
        if not self.args.train_rollout:
            node_attr, neighbors, edge_attr, global_feat, sample_idx = self._prepare_input(data, test=False)
        else:
            node_attr, neighbors, edge_attr, global_feat, sample_idx, _, _, _, _ = self._prepare_input(data, test=True)

        # Compute GT label: calculate accleration
        data_nxt = self._load_data_file(load_names, data_nxt_path)
        if not self.args.predict_vel:
            gt_accel = torch.FloatTensor((data_nxt[1] - data[1][:, 0:3]) / self.args.dt)
            if sample_idx is not None:
                gt_accel = gt_accel[sample_idx]
        else: # predict velocity
            gt_accel = torch.FloatTensor(data_nxt[1])
            if sample_idx is not None:
                gt_accel = gt_accel[sample_idx]


        return node_attr, neighbors, edge_attr, global_feat, gt_accel

    def _prepare_input(self, data):

        num_obj_class = len(self.phases_dict["material"])
        instance_idxes = self.phases_dict["instance_idx"]
        radius = self.phases_dict["radius"]

        if self.args.env_name == 'BoxBath':

            # Walls at x= 0, 1.25, y=0, z=0, 0.39

            pos, vel_hist, _ = data

            dist_to_walls = np.stack([np.absolute(pos[:, 0]),
                                      np.absolute(1.25-pos[:, 1]), # NOTE yufei: should not this be pos[:, 0]?
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

        return node_attr, neighbors, edge_attr, global_feat, None

    def create_env(self):
        env_args = copy.deepcopy(env_arg_dict[self.env_name])
        env_args['render_mode'] = 'particle'
        env_args['camera_name'] = 'default_camera'
        env_args['action_repeat'] = 1
        if self.env_name == 'ClothFlatten':
            env_args['cached_states_path'] = 'cloth_flatten_init_states_small.pkl'
            env_args['num_variations'] = 50
        return SOFTGYM_ENVS[self.env_name](**env_args)

    def create_dataset(self):
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.time_step))

        if os.path.exists(self.data_dir):
            # query_yes_no('Removing directory {}, confirmed?'.format(self.data_dir), default='yes')
            os.system('rm -rf {}'.format(self.data_dir))
        n_rollouts = [self.n_rollout // self.num_workers] * self.num_workers
        thread_idxes = np.arange(self.num_workers)
        # pool.map(self._collect_worker, zip(n_rollouts, thread_idxes))
        self._collect_worker([self.n_rollout, 0])

    def _collect_worker(self, args):
        """ Write data collection function for each task. Use random actions by default"""
        n_rollout, thread_idx = args
        np.random.seed(1000 + thread_idx)  ### NOTE: we might want to fix the seed for reproduction
        env = self.env

        for i in range(n_rollout):
            print("{} / {}".format(i, n_rollout))
            rollout_idx = thread_idx * n_rollout + i
            rollout_dir = os.path.join(self.data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)
            env.reset()
            self._prepare_policy(env)
            n_particles = pyflex.get_n_particles()
            n_shapes = pyflex.get_n_shapes()

            positions = np.zeros((self.time_step, n_particles, 3), dtype=np.float32)
            velocities = np.zeros((self.time_step, n_particles, 3), dtype=np.float32)
            shape_positions = np.zeros((self.time_step, n_shapes, 3), dtype=np.float32)

            positions[0] = pyflex.get_positions().reshape(-1, 4)[:, :3]
            velocities[0] = pyflex.get_velocities().reshape(-1, 3)
            shape_states = pyflex.get_shape_states().reshape(-1, 14)
            for k in range(n_shapes):
                shape_positions[0][k] = shape_states[k, :3]
                
            config = env.get_current_config()
            cloth_xdim, cloth_ydim = config['ClothSize']
            config_id = env.current_config_id
            scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

            for j in range(1, self.time_step):
                picker_position = env.action_tool._get_pos()[0]
                action = self._collect_policy(env, j)
                env.step(action)
                picked_points = env.get_picked_particle()
                intermediate_picked_point_pos = env.get_picked_particle_new_position()

                # Store previous data
                data = [positions[j - 1], velocities[j - 1], picked_points, intermediate_picked_point_pos, 
                    picker_position, action, scene_params, shape_positions[j - 1]]
                self.store_data(self.data_names, data, os.path.join(rollout_dir, str(j - 1) + '.h5'))

                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(n_shapes):
                    shape_positions[j][k] = shape_states[k, :3]

                # NOTE: velocity is not directly using particle velocity in Pyflex
                # the main benefit of computing velocity in this way is that we can get the velocity of the shape.
                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / self.dt

            j = self.time_step - 1
            # the last step has no action
            data = [positions[j], velocities[j], 0, 0, 0, 0, scene_params, shape_positions[j-1]] 
            self.store_data(self.data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))



    def _collect_policy(self, env, timestep):
        """ Policy for collecting data"""
        return env.action_space.sample()

    def _prepare_policy(self, env):
        """ Doing something after env reset but before collecting any data"""
        pass

    
class ClothDataset(PhysicsFleXDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_names = ['positions', 'velocities', 'picked_points', 'picked_point_positions', 
            'picker_position', 'action', 'scene_params', 'shape_positions']

    def _prepare_input(self, data, downsample=True, test=False, noise_scale=None):
        """
        data: positions, vel_history, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return: 
        node_attr: N x (vel_history x 3 + attr_dim)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        global_feat: fixed, not used for now
        """

        args = self.args

        # Arrangement:
        # particles, shapes, roots
        if args.down_sample_scale is not None and downsample:  # First one is vv and the second one is for rollout
            new_data, sample_idx = self._downsample(data, args.down_sample_scale, test=test)
            data = new_data
        else:
            sample_idx = None

        # add noise
        if noise_scale is None:
            noise_scale = args.noise_scale

        position_noise = np.random.normal(loc=0, scale=noise_scale, size=data[0].shape)
        vel_history_noise = np.random.normal(loc=0, scale=noise_scale, size=data[1].shape)

        picked_velocity = []
        picked_pos = []
        if not test:
            positions, velocity_his, picked_points, picked_points_position, scene_params = data
            positions += position_noise
            velocity_his += vel_history_noise

             # modify the position and velocity of the picked particle due to the pick action
            _, cloth_xdim, cloth_ydim, _ = scene_params

            cnt = 0
            for idx in range(len(picked_points)):
                if picked_points[idx] != -1:
                    picked_point = picked_points[idx]
                    picked_point_pos = picked_points_position[cnt]
                    old_pos = positions[picked_point]
                    positions[picked_point] = picked_point_pos
                    new_vel = (picked_point_pos - old_pos) / self.dt

                    tmp_vel_history = velocity_his[picked_point][:-3]
                    velocity_his[picked_point, 3:] = tmp_vel_history
                    velocity_his[picked_point, :3] = new_vel
                    cnt += 1
                    picked_velocity.append(velocity_his[picked_point])
                    picked_pos.append(picked_point_pos)
        else:
            particle_pos, velocity_his, picker_pos, action, picked_particles, scene_params, _ = data
            _, cloth_xdim, cloth_ydim, _ = scene_params
            particle_pos += position_noise
            velocity_his += vel_history_noise

            # print("in data graph, picked particles: ", picked_particles)
            env = self.env
            action = np.reshape(action, [-1, 4])
            pick_flag = action[:, 3] > 0.5
            # print("pick flag is: ", pick_flag)
            new_picker_pos = picker_pos.copy()
            for i in range(env.action_tool.num_picker):
                # print("picker {}".format(i)) 
                new_picker_pos[i, :] = env.action_tool._apply_picker_boundary(picker_pos[i, :] + action[i, :3])
                if pick_flag[i]:
                    if picked_particles[i] == -1:  # No particle is currently picked and thus need to select a particle to pick
                        dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                        idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
                        mask = dists.flatten() <= env.action_tool.picker_threshold * args.down_sample_scale \
                                 + env.action_tool.picker_radius + env.action_tool.particle_radius
                        idx_dists = idx_dists[mask, :].reshape((-1, 2))
                        if idx_dists.shape[0] > 0:
                            pick_id, pick_dist = None, None
                            for j in range(idx_dists.shape[0]):
                                if idx_dists[j, 0] not in picked_particles and (pick_id is None or idx_dists[j, 1] < pick_dist):
                                    pick_id = idx_dists[j, 0]
                                    pick_dist = idx_dists[j, 1]
                            if pick_id is not None: # update picked particles
                                picked_particles[i] = int(pick_id)

                    # update the position and velocity of the picked particle
                    if picked_particles[i] != -1:     

                        old_pos = particle_pos[picked_particles[i]]
                        new_pos = particle_pos[picked_particles[i]] + new_picker_pos[i, :] - picker_pos[i,:]
                        new_vel = (new_pos - old_pos) / self.dt

                        tmp_vel_history = velocity_his[picked_particles[i]][:-3]
                        velocity_his[picked_particles[i], 3:] = tmp_vel_history
                        velocity_his[picked_particles[i], :3] = new_vel

                        particle_pos[picked_particles[i]] = new_pos

                        picked_velocity.append(velocity_his[picked_particles[i]])
                        picked_pos.append(new_pos)
                else:
                    picked_particles[i] = -1
                    
            positions = particle_pos
            picked_points = picked_particles

        # picked particle [0, 1]
        # normal particle [1, 0]
        node_one_hot = np.zeros((len(positions), 2), dtype=np.float32)
        node_one_hot[:, 0] = 1
        for picked in picked_points:
            if picked != -1:
                node_one_hot[picked, 0] = 0
                node_one_hot[picked, 1] = 1
        distance_to_ground = torch.from_numpy(positions[:, 1]).view((-1, 1))
        node_one_hot = torch.from_numpy(node_one_hot)
        node_attr = torch.from_numpy(velocity_his)
        node_attr = torch.cat([node_attr, distance_to_ground, node_one_hot], dim=1)

        ##### add env specific graph components
        ## Edge attributes:
        # [1, 0] Distance based neighbor
        # [0, 1] Mesh edges

        # Calculate undirected edge list and corresponding relative edge attributes (distance vector + magnitude)
        point_tree = spatial.cKDTree(positions)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T
        # print("shape of undirected neighbors: ", undirected_neighbors.shape)
        # print("shape of positions: ", positions.shape)
        # print("undirected_neighbors: ")
        # print(undirected_neighbors)

        if len(undirected_neighbors) > 0:
            dist_vec = positions[undirected_neighbors[0, :]] - positions[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
            edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
            num_distance_edges = edges.shape[1]
        else:
            num_distance_edges = 0

        # Build mesh edges -- both directions
        if self.args.use_mesh_edge:
            mesh_edges = self._get_eight_neighbor(cloth_xdim, cloth_ydim)
            mesh_dist_vec = positions[mesh_edges[0, :]] - positions[mesh_edges[1, :]]
            mesh_dist = np.linalg.norm(mesh_dist_vec, axis=1, keepdims=True)
            mesh_edge_attr = np.concatenate([mesh_dist_vec, mesh_dist], axis=1)
            mesh_edge_attr = torch.from_numpy(mesh_edge_attr)
            num_mesh_edges = mesh_edges.shape[1]
            
            if num_distance_edges > 0:
                edge_attr = torch.cat([edge_attr, mesh_edge_attr], dim=0)
            else:
                edge_attr = mesh_edge_attr

            # Concatenate edge types
            edge_types = np.zeros((num_mesh_edges + num_distance_edges, 2), dtype=np.float32)
            edge_types[:num_distance_edges, 0] = 1.
            edge_types[num_distance_edges:, 1] = 1.
            edge_types = torch.from_numpy(edge_types)

            edge_attr = torch.cat([edge_attr, edge_types], dim=1)

            # concatenate all edges
            mesh_edges = torch.from_numpy(mesh_edges)

            if num_distance_edges > 0:
                edges = torch.cat([edges, mesh_edges], dim=1)
            else:
                edges = mesh_edges

        # Global features are unused
        global_feat = torch.FloatTensor([[0.]])

        if not test:
            return node_attr, edges, edge_attr, global_feat, sample_idx, 
        else:
            return node_attr, edges, edge_attr, global_feat, sample_idx, picked_particles, cloth_xdim, cloth_ydim, (picked_velocity, picked_pos)


    def _prepare_policy(self, env):
        """ Doing something after env reset but before collecting any data"""
        # move one of the picker to be under ground
        shape_states = pyflex.get_shape_states().reshape(-1, 14)
        shape_states[1, :3] = -1

        # move another picker to a randomly chosen particle
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., env.picker_radius, 0.]
        pyflex.set_shape_states(shape_states.flatten())

        # randomly select a move direction and a move distance
        move_direction = np.random.rand(3) - 0.5
        move_direction[1] += 0.52
        self.policy_info = {}
        self.policy_info['move_direction'] = move_direction / np.linalg.norm(move_direction)
        self.policy_info['move_distance'] = np.random.rand() * 0.5
        self.policy_info['move_steps'] = 60
        self.policy_info['delta_move'] = self.policy_info['move_distance'] / self.policy_info['move_steps'] 

    def _collect_policy(self, env, timestep):
        """ Policy for collecting data"""
        # print("timestep: ", timestep)
        # input("enter to continue....")

        if timestep <= self.policy_info['move_steps']:
            delta_move = self.policy_info['delta_move']
            action = np.zeros_like(env.action_space.sample())
            action[3] = 1
            action[:3] = delta_move * self.policy_info['move_direction']
        else:
            action = np.zeros_like(env.action_space.sample())

        return action

        # if timestep == 1 or self.policy_info['status'] == 'finished':  # In the first frame, pick a random start location
        #     self.policy_info = {}
        #     pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        #     pp = np.random.randint(len(pos))
        #     shape_states = pyflex.get_shape_states().reshape(-1, 14)
        #     curr_pos = shape_states[0, :3] = pos[pp] + [0., env.picker_radius, 0.]
        #     shape_states[1, :3] = -1
        #     pyflex.set_shape_states(shape_states.flatten())
        #     delta_pos = (np.random.random(3) - 0.5) 
        #     delta_pos[1] = (delta_pos[1] + 0.55)
        #     delta_pos[[0, 2]] *= 0.5
        #     self.policy_info['delta_pos'] = delta_pos
        #     self.policy_info['target_pos'] = curr_pos + delta_pos
        #     self.policy_info['status'] = 'pick'
        #     return np.zeros_like(env.action_space.sample())

        # if self.policy_info['status'] == 'pick':
        #     curr_pos = pyflex.get_shape_states().reshape(-1, 14)[0, :3]
        #     dist = np.linalg.norm(self.policy_info['target_pos'] - curr_pos)
        #     delta_move = 0.02
        #     num_step = np.ceil(dist / delta_move)

        #     if num_step <= 1:
        #         delta = self.policy_info['target_pos'] - curr_pos
        #         self.policy_info['status'] = 'wait'
        #     else:
        #         delta = (self.policy_info['target_pos'] - curr_pos) / num_step
        #     # print(dist, pp_pos, num_step, self.policy_info['finished'])
        #     action = np.zeros_like(env.action_space.sample())
        #     action[3] = int(self.policy_info['status'] == 'pick')
        #     action[:3] = delta / env.action_repeat
        # else:
        #     action = np.zeros_like(env.action_space.sample())
        #     vel = pyflex.get_velocities()
        #     if np.abs(np.max(vel)) < 0.15:
        #         self.policy_info['status'] = 'finished'
        # return action



    def _get_eight_neighbor(self, cloth_xdim, cloth_ydim):
        # Connect cloth particles based on the ground-truth edges
        # Cloth index looks like the following (For Flex 1.0):
        # 0, 1, ..., cloth_xdim -1
        # ...
        # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
        # Vectorized version
        all_idx = np.arange(cloth_xdim * cloth_ydim).reshape([cloth_ydim, cloth_xdim])

        senders = []
        receivers = []

        # Horizontal connections
        idx_s = all_idx[:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1
        senders.append(idx_s)
        receivers.append(idx_r)

        # Vertical connections
        idx_s = all_idx[:-1, :].reshape(-1, 1)
        idx_r = idx_s + cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        # Diagonal connections
        idx_s = all_idx[:-1, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 + cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        idx_s = all_idx[1:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 - cloth_xdim
        senders.append(idx_s)
        receivers.append(idx_r)

        senders = np.concatenate(senders, axis=0)
        receivers = np.concatenate(receivers, axis=0)
        new_senders = np.concatenate([senders, receivers], axis=0)
        new_receivers = np.concatenate([receivers, senders], axis=0)
        edges = np.concatenate([new_senders, new_receivers], axis=1).T
        assert edges.shape[0] == 2

        return edges

    def _get_cloth_neighbor(self, cloth_xdim, cloth_ydim, relation_dim):
        # Connect cloth particles based on the ground-truth edges
        # Cloth index looks like the following (For Flex 1.0):
        # 0, 1, ..., cloth_xdim -1
        # ...
        # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1
        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
        rels = []

        # Edge types / attr: [0, 0, stretch, bend, shear]
        # Stretch stiffness
        # Vectorized version
        all_idx = np.arange(cloth_xdim * cloth_ydim).reshape([cloth_ydim, cloth_xdim])

        edge_attr = np.zeros((1, relation_dim))
        stretch_edge = edge_attr.copy()
        stretch_edge[0, 2] = 1
        bend_edge = edge_attr.copy()
        bend_edge[0, 3] = 1
        shear_edge = edge_attr.copy()
        shear_edge[0, 4] = 1

        # Horizontal connections
        idx_s = all_idx[:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1
        rels.append(np.hstack([idx_s, idx_r, np.tile(stretch_edge, [len(idx_s), 1])]))

        # Vertical connections
        idx_s = all_idx[:-1, :].reshape(-1, 1)
        idx_r = idx_s + cloth_xdim
        rels.append(np.hstack([idx_s, idx_r, np.tile(stretch_edge, [len(idx_s), 1])]))

        # Diagonal connections
        idx_s = all_idx[:-1, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 + cloth_xdim
        rels.append(np.hstack([idx_s, idx_r, np.tile(bend_edge, [len(idx_s), 1])]))

        idx_s = all_idx[1:, :-1].reshape(-1, 1)
        idx_r = idx_s + 1 - cloth_xdim
        rels.append(np.hstack([idx_s, idx_r, np.tile(bend_edge, [len(idx_s), 1])]))

        # 2-hop connections
        idx_s = all_idx[:, :-2].reshape(-1, 1)
        idx_r = idx_s + 2
        rels.append(np.hstack([idx_s, idx_r, np.tile(shear_edge, [len(idx_s), 1])]))

        idx_s = all_idx[:-2, :].reshape(-1, 1)
        idx_r = idx_s + cloth_xdim + cloth_xdim
        rels.append(np.hstack([idx_s, idx_r, np.tile(shear_edge, [len(idx_s), 1])]))

        rels = np.vstack(rels)  # Directed edges only
        rels_reversed = rels.copy()
        rels_reversed[:, :2] = rels[:, [1, 0]]
        rels = np.vstack([rels, rels_reversed])  # Bi-directional edges
        return rels

        # Horizontal connections
        # for i in range(cloth_ydim):
        #     for j in range(cloth_xdim - 1):
        #         rels.append([i * cloth_xdim + j, i * cloth_xdim + j + 1, 1])
        # Vertical connections
        # for i in range(cloth_ydim - 1):
        #     for j in range(cloth_xdim):
        #         rels.append([i * cloth_ydim + j, (i + 1) * cloth_ydim + j, 1])

    def downsample_mapping(self, cloth_ydim, cloth_xdim, idx, downsample):
        """ Given the down sample scale, map each point index before down sampling to the index after down sampling
        downsample: down sample scale
        """
        y, x = idx // cloth_xdim, idx % cloth_xdim
        down_ydim, down_xdim = (cloth_ydim + downsample - 1) // downsample, (cloth_xdim + downsample - 1) // downsample
        down_y, down_x = y // downsample, x // downsample
        new_idx = down_y * down_xdim + down_x
        return new_idx

    def _downsample(self, data, scale=2, test=False):
        if not test:
            pos, vel_his, picked_points, picked_point_pos, scene_params = data
        else:
            pos, vel_his, pciker_positions, actions, picked_points, scene_params, shape_pos = data
            # print("in downsample, picked points are: ", picked_points)

        sphere_radius, cloth_xdim, cloth_ydim, config_id = scene_params
        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
        original_xdim, original_ydim = cloth_xdim, cloth_ydim
        new_idx = np.arange(cloth_xdim * cloth_ydim).reshape((cloth_ydim, cloth_xdim))
        new_idx = new_idx[::scale, ::scale]
        cloth_ydim, cloth_xdim = new_idx.shape
        new_idx = new_idx.flatten()
        pos = pos[new_idx, :]
        vel_his = vel_his[new_idx, :]

        # Remap picked_points
        pps = []
        for pp in picked_points.astype('int'):
            if pp != -1:
                pps.append(self.downsample_mapping(original_ydim, original_xdim, pp, scale))
                assert pps[-1] < len(pos)
            else:
                pps.append(-1)

        scene_params = sphere_radius, cloth_xdim, cloth_ydim, config_id

        if not test:
            return (pos, vel_his, pps, picked_point_pos, scene_params), new_idx
        else:
            return (pos, vel_his, pciker_positions, actions, pps, scene_params, shape_pos), new_idx

    
