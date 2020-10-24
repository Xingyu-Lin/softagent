import torch
import numpy as np
import os.path as osp
from multiprocessing import Pool
import copy
from GNS.visualize_data import visualize
from scipy.spatial.distance import cdist
from cloth_manipulation.gemo_utils import intrinsic_from_fov, get_rotation_matrix
import matplotlib.pyplot as plt


# from torch.multiprocessing import Pool, set_start_method


# Model-predictive control planner with cross-entropy method and learned transition model
class CEMPlanner():
    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model,
                 reward_model, action_low, action_high, device, num_worker=10):
        """
        cem with mpc
        """

        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.action_low, self.action_high = action_low, action_high
        self.device = device

        self.pool = Pool(processes=num_worker)

    def get_action(self, args, init_data, dataset):
        """
        init_data should be a list that include:
            note: require position, velocity to be already downsampled
            ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params']
        """

        action_mean = np.ones((self.candidates, self.planning_horizon, self.action_size)) * (self.action_high + self.action_low) / 2.
        action_std = np.ones((self.candidates, self.planning_horizon, self.action_size)) * (self.action_high - self.action_low) / 4.

        for opt_iter in range(self.optimisation_iters):
            data = init_data.copy()
            actions = np.clip(np.random.normal(action_mean, action_std), a_min=self.action_low, a_max=self.action_high)
            actions[:, :, 4:] = 0  # we essentially only plan over 1 picker action

            params = [(
                args, data.copy(), self.transition_model, self.reward_model, actions[i], dataset, self.planning_horizon, False
            ) for i in range(self.candidates)]
            results = self.pool.map(parallel_worker, params)

            returns = [x[0] for x in results]
            first_action_picked_particles = [x[1] for x in results]

            # returns = []
            # first_action_picked_particles = []
            # for i in range(self.candidates):
            #     ret, first_picked_particle = parallel_worker((
            #         args, copy.deepcopy(data), self.transition_model, self.reward_model, actions[i], dataset, self.planning_horizon
            #     ))
            #     returns.append(ret)
            #     first_action_picked_particles.append(first_picked_particle)

            sort_returns = np.argsort(returns)
            best = actions[sort_returns][-self.top_candidates:]
            first_action_picked_particles = first_action_picked_particles[sort_returns[-1]]

            action_mean = np.mean(best, axis=0)
            action_std = np.std(best, axis=0)

            action_mean = np.stack([action_mean for _ in range(self.candidates)], axis=0)
            action_std = np.stack([action_std for _ in range(self.candidates)], axis=0)

        return np.clip(action_mean[0][0], self.action_low, self.action_high), first_action_picked_particles


class CEMUVPickandPlacePlanner():
    def __init__(self, num_pick, delta_y, move_distance, stage_1_step, stage_2_step, stage_3_step,
                 transition_model, reward_model, num_worker=10, env=None, downsample_idx=None, uv_sample_method='bounding_box'):
        """
        cem with mpc
        """

        print("self.num_pick is: ", num_pick)
        self.num_pick = num_pick
        self.delta_y = delta_y
        self.move_distance = move_distance
        self.transition_model, self.reward_model = transition_model, reward_model

        self.stage_1, self.stage_2, self.stage_3 = stage_1_step, stage_2_step, stage_3_step

        if num_worker > 0:
            self.pool = Pool(processes=num_worker)
        self.num_worker = num_worker

        ## debug usage
        self.env = env
        self.downsample_idx = downsample_idx

        # Used for transforming u,v to x,y,z
        self.image_size = (env.camera_height, env.camera_width)
        self.cam_pos, self.cam_angle = env.get_camera_params()
        self.uv_sample_method = uv_sample_method
    def _get_depth(self, matrix, vec, height):
        """ Get the depth such that the back-projected point has a fixed height"""
        return (height - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])

    def get_world_coor_from_image(self, u, v):
        height, width = self.image_size
        K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

        cam_x, cam_y, cam_z = self.cam_pos
        cam_x_angle, cam_y_angle, cam_z_angle = self.cam_angle

        # get rotation matrix: from world to camera
        matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
        # matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [np.cos(cam_x_angle), 0, np.sin(cam_x_angle)])
        matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
        rotation_matrix = matrix2 @ matrix1

        # get translation matrix: from world to camera
        translation_matrix = np.zeros((4, 4))
        translation_matrix[0][0] = 1
        translation_matrix[1][1] = 1
        translation_matrix[2][2] = 1
        translation_matrix[3][3] = 1
        translation_matrix[0][3] = - cam_x
        translation_matrix[1][3] = - cam_y
        translation_matrix[2][3] = - cam_z
        matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]
        vec = ((u - u0) / fx, (v - v0) / fy)
        depth = self._get_depth(matrix, vec, 0.00625)  # Height to be the particle radius

        # Loop through each pixel in the image
        # Apply equation in fig 3
        x = (u - u0) * depth / fx
        y = (v - v0) * depth / fy
        z = depth
        cam_coords = np.array([x, y, z, 1])
        cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)

        world_coord = matrix @ cam_coords  # 4 x (height x width)
        world_coord = world_coord.reshape(4)
        return world_coord[:3], np.linalg.inv(matrix)

    def get_target_pos(self, pos, u, v):

        coor, _ = self.get_world_coor_from_image(u, v)
        dists = cdist(coor[None], pos)[0]
        idx = np.argmin(dists)
        return pos[idx] + np.array([0, 0.01, 0])

    def get_target_pos_2d(self, pos, u, v):
        height, width = self.image_size
        K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        coor, cam_mat = self.get_world_coor_from_image(u, v)
        dists = cdist(coor[None], pos)[0]
        idx = np.argmin(dists)
        coor = np.hstack([pos[idx], 1.])
        x, y, z, _ = cam_mat @ coor  # cam coordinate
        u = x * fx / z + u0
        v = y * fy / z + v0
        return int(u), int(v)

    def project_3d(self, pos):
        height, width = self.image_size
        K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        _, cam_mat = self.get_world_coor_from_image(0, 0)
        coor = np.hstack([pos, np.ones([len(pos), 1])]).T
        cam_pos = (cam_mat @ coor).T  # cam coordinate
        x, y, z = cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2]
        u = x * fx / z + u0
        v = y * fy / z + v0
        return u.astype('int'), v.astype('int')

    def get_action(self, args, init_data, dataset):
        """
        init_data should be a list that include: ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params']
            note: require position, velocity to be already downsampled

        """

        data = init_data.copy()
        data[-2] = [-1, -1]

        stage_1 = self.stage_1
        stage_2 = self.stage_2
        stage_3 = self.stage_3
        actions = np.zeros((self.num_pick, stage_1 + stage_2 + stage_3, 8))

        positions = data[0]
        picker_pos = data[2][0][:3]
        num_particles = len(positions)
        bb_margin = 30
        for pick_try_idx in range(self.num_pick):
            # first stage: move to a random chosen point
            if self.uv_sample_method =='uniform':
                u, v = np.random.randint(0, self.image_size, 2)
            elif self.uv_sample_method == 'bounding_box':
                us, vs = self.project_3d(positions)
                lb_u, lb_v, ub_u, ub_v = int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))
                u = np.random.randint(max(lb_u-bb_margin, 0), min(ub_u+bb_margin, self.image_size[0]))
                v = np.random.randint(max(lb_v-bb_margin, 0), min(ub_v+bb_margin, self.image_size[1]))
            ### Visualize distribution 2d
            # N = 1000
            # us = np.random.randint(max(lb_u - bb_margin, 0), min(ub_u + bb_margin, self.image_size[0]), N)
            # vs = np.random.randint(max(lb_v - bb_margin, 0), min(ub_v + bb_margin, self.image_size[1]), N)
            # img_cloth = np.zeros((self.image_size[0] // 3, self.image_size[1] // 3), dtype=np.int)
            # img_pick = np.zeros((self.image_size[0] // 3, self.image_size[1] // 3), dtype=np.int)
            #
            # for u, v in zip(us, vs):
            #     pos = self.get_target_pos_2d(positions, u, v)
            #     img_pick[pos[0] // 3 - 1:pos[0] // 3 + 1, pos[1] // 3 - 1:pos[1] // 3 + 1] += 1
            #
            # us, vs = self.project_3d(positions)
            # for u, v in zip(us, vs):
            #     img_cloth[u // 3 - 1:u // 3 + 1, v // 3 - 1:v // 3 + 1] += 1
            # plt.imshow(img_pick, cmap='hot')
            # plt.colorbar()
            # plt.show()
            # plt.imshow(img_cloth)
            # plt.show()
            ### End visualization
            target_pos = self.get_target_pos(positions, u, v)
            # target_pos = positions[pick_idx] + np.array([0, 0.01, 0])  # add a particle radius distance
            delta_move = (target_pos - picker_pos) / (stage_1 // 2)
            delta_x_z_move = delta_move.copy()
            delta_x_z_move[1] = 0
            delta_move[0] = 0
            delta_move[2] = 0
            actions[pick_try_idx][:stage_1 // 2, :3] = delta_x_z_move
            actions[pick_try_idx][stage_1 // 2:stage_1, :3] = delta_move

            # second stage: choose a random (x, z) direction, move towards that direction for 30 steps.
            move_direction = np.random.rand(3) - 0.5
            move_direction[1] = self.delta_y
            move_direction = move_direction / np.linalg.norm(move_direction)
            delta_move = self.move_distance / stage_2 * move_direction
            actions[pick_try_idx][stage_1:stage_1 + stage_2, :3] = delta_move
            actions[pick_try_idx][stage_1:stage_1 + stage_2, 3] = 1

            # thrid stage, let go the particle
            move = np.array([0, 0.0, 0])
            actions[pick_try_idx][stage_1 + stage_2:, :3] = move

        actions[:, :, 4:] = 0  # we essentially only plan over 1 picker action

        if self.num_worker > 0:
            params = [(
                args, copy.deepcopy(data), self.transition_model, self.reward_model, actions[i], dataset, actions.shape[1], True
            ) for i in range(self.num_pick)]
            results = self.pool.map(parallel_worker, params)

            returns = [x[0] for x in results]
        else:
            returns = []
            results = []
            for i in range(self.num_pick):
                print("\t i: ", i)
                res = parallel_worker(
                    (args, copy.deepcopy(data), self.transition_model, self.reward_model, actions[i], dataset, actions.shape[1], True))
                results.append(res)
                returns.append(res[0])
                # debug
                # visualize(self.env, res[1],
                #                     res[2], data[-1][-1], self.downsample_idx)

        highest_return_idx = np.argmax(returns)
        action_seq = actions[highest_return_idx]
        model_predict_particle_positions = results[highest_return_idx][1]
        model_predict_shape_positions = results[highest_return_idx][2]

        return action_seq, model_predict_particle_positions, model_predict_shape_positions


class CEMPickandPlacePlanner():
    def __init__(self, num_pick, delta_y, move_distance, stage_1_step, stage_2_step, stage_3_step,
                 transition_model, reward_model, num_worker=10, env=None, downsample_idx=None):
        """
        cem with mpc
        """

        print("self.num_pick is: ", num_pick)
        self.num_pick = num_pick
        self.delta_y = delta_y
        self.move_distance = move_distance
        self.transition_model, self.reward_model = transition_model, reward_model

        self.stage_1, self.stage_2, self.stage_3 = stage_1_step, stage_2_step, stage_3_step

        if num_worker > 0:
            self.pool = Pool(processes=num_worker)
        self.num_worker = num_worker

        ## debug usage
        self.env = env
        self.downsample_idx = downsample_idx

    def get_action(self, args, init_data, dataset):
        """
        init_data should be a list that include:
            note: require position, velocity to be already downsampled
            ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params']
        """

        data = init_data.copy()
        data[-2] = [-1, -1]

        stage_1 = self.stage_1
        stage_2 = self.stage_2
        stage_3 = self.stage_3
        actions = np.zeros((self.num_pick, stage_1 + stage_2 + stage_3, 8))

        positions = data[0]
        picker_pos = data[2][0][:3]
        num_particles = len(positions)
        for pick_try_idx in range(self.num_pick):
            # first stage: move to a random chosen point
            pick_idx = np.random.randint(len(positions))

            target_pos = positions[pick_idx] + np.array([0, 0.01, 0])  # add a particle radius distance
            delta_move = (target_pos - picker_pos) / (stage_1 // 2)
            delta_x_z_move = delta_move.copy()
            delta_x_z_move[1] = 0
            delta_move[0] = 0
            delta_move[2] = 0
            actions[pick_try_idx][:stage_1 // 2, :3] = delta_x_z_move
            actions[pick_try_idx][stage_1 // 2:stage_1, :3] = delta_move

            # second stage: choose a random (x, z) direction, move towards that direction for 30 steps.
            move_direction = np.random.rand(3) - 0.5
            move_direction[1] = self.delta_y
            move_direction = move_direction / np.linalg.norm(move_direction)
            delta_move = self.move_distance / stage_2 * move_direction
            actions[pick_try_idx][stage_1:stage_1 + stage_2, :3] = delta_move
            actions[pick_try_idx][stage_1:stage_1 + stage_2, 3] = 1

            # thrid stage, let go the particle
            move = np.array([0, 0.0, 0])
            actions[pick_try_idx][stage_1 + stage_2:, :3] = move

        actions[:, :, 4:] = 0  # we essentially only plan over 1 picker action

        if self.num_worker > 0:
            params = [(
                args, copy.deepcopy(data), self.transition_model, self.reward_model, actions[i], dataset, actions.shape[1], True
            ) for i in range(self.num_pick)]
            results = self.pool.map(parallel_worker, params)

            returns = [x[0] for x in results]
        else:
            returns = []
            results = []
            for i in range(self.num_pick):
                print("\t i: ", i)
                res = parallel_worker(
                    (args, copy.deepcopy(data), self.transition_model, self.reward_model, actions[i], dataset, actions.shape[1], True))
                results.append(res)
                returns.append(res[0])
                # debug
                # visualize(self.env, res[1], 
                #                     res[2], data[-1][-1], self.downsample_idx)

        highest_return_idx = np.argmax(returns)
        action_seq = actions[highest_return_idx]
        model_predict_particle_positions = results[highest_return_idx][1]
        model_predict_shape_positions = results[highest_return_idx][2]

        return action_seq, model_predict_particle_positions, model_predict_shape_positions


def parallel_worker(args):
    args, data, transition_model, reward_model, actions, dataset, planning_horizon, record_model_predictions = args
    encoder_model, processor_model, decoder_model = transition_model
    encoder_model.eval()
    processor_model.eval()
    decoder_model.eval()

    particle_pos, velocity_his, picker_pos, action, picked_particles, scene_params = data

    first_action_picked_particle = None

    if record_model_predictions:
        model_positions = np.zeros((planning_horizon, len(particle_pos), 3))
        shape_positions = np.zeros((planning_horizon, 2, 3))

    ret = 0
    final_ret = 0
    for t in range(planning_horizon):
        action = actions[t]  # use the planned action

        data = [
            particle_pos, velocity_his, picker_pos, action, picked_particles, scene_params, 0
        ]

        node_attr, neighbors, edge_attr, global_feat, _, picked_particles, _, _, picked_status \
            = dataset._prepare_input(data, test=True, downsample=False, noise_scale=0)

        if t == 0:
            first_action_picked_particle = picked_particles
        if record_model_predictions:
            model_positions[t] = particle_pos
            shape_positions[t] = picker_pos

        picked_particles = [int(x) for x in picked_particles]

        node_attr = torch.squeeze(node_attr, dim=0).cuda()
        neighbors = torch.squeeze(neighbors, dim=0).cuda()
        edge_attr = torch.squeeze(edge_attr, dim=0).cuda()
        global_feat = torch.squeeze(global_feat, dim=0).cuda()

        # obtain model predictions
        with torch.no_grad():

            node_embedding, edge_embedding = encoder_model(node_attr, edge_attr)

            node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat,
                                                                                 batch=None)
            pred_accel = decoder_model(node_embedding_out).cpu().numpy()

        # if args.normalize:
        #     pred_accel = pred_accel * dataset.acc_stats[1] + dataset.acc_stats[0]

        if not args.predict_vel:
            pred_vel = data[1][:, :3] + pred_accel * args.dt
        else:
            pred_vel = pred_accel

        picked_vel, picked_pos, new_picker_pos = picked_status
        pred_pos = particle_pos + pred_vel * args.dt

        # udpate position and velocity from the model prediction
        old_vel = data[1][:, :-3]
        particle_pos = pred_pos
        velocity_his[:, 3:] = old_vel
        velocity_his[:, :3] = pred_vel

        # the picked particles position and velocity should not change
        cnt = 0
        for p_idx in picked_particles:
            if p_idx != -1:
                particle_pos[p_idx] = picked_pos[cnt]
                velocity_his[p_idx] = picked_vel[cnt]
                cnt += 1

        # update picker position, and the particles picked
        picker_pos = new_picker_pos

        # get reward of the new position
        reward = reward_model(pred_pos)
        ret += reward

        if t == planning_horizon - 1:
            final_ret = reward

    if not record_model_predictions:
        return ret, first_action_picked_particle
    else:
        return final_ret, model_positions, shape_positions
