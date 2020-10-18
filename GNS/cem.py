import torch
import numpy as np
import os.path as osp
from multiprocessing import Pool
import copy
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

        # print(self.action_high)
        # print(self.action_low)
        # print((self.action_high + self.action_low) / 2.)
        action_mean = np.ones((self.candidates, self.planning_horizon, self.action_size)) * (self.action_high + self.action_low) / 2.
        action_std = np.ones((self.candidates, self.planning_horizon, self.action_size)) * (self.action_high - self.action_low) / 4.

        for opt_iter in range(self.optimisation_iters):
            data = init_data.copy()
            actions = np.clip(np.random.normal(action_mean, action_std), a_min=self.action_low, a_max=self.action_high)
            actions[:, :, 4:] = 0 # we essentially only plan over 1 picker action

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


class CEMPickandPlacePlanner():
    def __init__(self, num_pick, delta_y, move_distance, stage_1_step, stage_2_step, stage_3_step,
        transition_model, reward_model,  device, num_worker=10):
        """
        cem with mpc
        """

        print("self.num_pick is: ", num_pick)
        self.num_pick = num_pick
        self.delta_y = delta_y
        self.move_distance = move_distance
        self.transition_model, self.reward_model = transition_model, reward_model
        self.device = device

        self.stage_1, self.stage_2, self.stage_3 = stage_1_step, stage_2_step, stage_3_step

        self.pool = Pool(processes=num_worker)

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
            # # zero stage: move up the picker
            # delta_move = np.array([0, 0.01, 0])
            # actions[pick_try_idx][:stage_0][:3] = delta_move

            # first stage: use 10 steps move to a random chosen point
            pick_idx = np.random.randint(num_particles)
            target_pos = positions[pick_idx] + np.array([0, 0.01, 0]) # add a particle radius distance
            delta_move = (target_pos - picker_pos) / (stage_1 // 2)
            delta_x_z_move = delta_move.copy()
            delta_x_z_move[1] = 0
            delta_move[0] = 0
            delta_move[2] = 0
            actions[pick_try_idx][:stage_1 // 2, :3] = delta_x_z_move
            actions[pick_try_idx][stage_1 // 2:stage_1, :3] = delta_move

            # second stage: choose a random (x, z) direction, move towards that direction for 30 steps.
            x = np.random.rand() - 0.5
            z = np.random.rand() - 0.5
            # norm_x_z = np.sqrt(1 - self.delta_y ** 2)
            x = x / np.linalg.norm([x, z]) 
            z = z / np.linalg.norm([x, z])
            y = self.delta_y
            move_direction = np.array([x, y, z])
            # print("move direction: ", move_direction)
            delta_move = self.move_distance / stage_2 * move_direction
            delta_move[1] = y
            print(delta_move)
            actions[pick_try_idx][stage_1:stage_1 + stage_2, :3] = delta_move
            actions[pick_try_idx][stage_1:stage_1 + stage_2, 3] = 1

            # thrid stage, let go the particle
            move = np.array([0, 0.0, 0])
            actions[pick_try_idx][stage_1 + stage_2:, :3] = move


        actions[:, :, 4:] = 0 # we essentially only plan over 1 picker action
        params = [(
            args, data.copy(), self.transition_model, self.reward_model, actions[i], dataset, actions.shape[1], True
        ) for i in range(self.num_pick)]
        results = self.pool.map(parallel_worker, params)

        returns = [x[0] for x in results]

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
        action = actions[t] # use the planned action

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
            pred_accel = decoder_model(node_embedding_out)

        if not args.predict_vel:
            pred_vel = data[1][:, :3] + pred_accel.cpu().numpy() * args.dt
        else:
            pred_vel = pred_accel.cpu().numpy()
        
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