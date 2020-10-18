import torch
import copy
from torch.autograd import Variable
import numpy as np
from DPINet.data import denormalize
from multiprocessing import Pool

import copy
import pickle
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from DPINet.visualize_data import prepare_model, convert_dpi_to_graph
import scipy


class DPIModel(object):
    def __init__(self, env, model, reward_func, stat, datasets, phases_dict):
        self.env = env
        self.model = model
        self.reward_func = reward_func
        self.stat = stat
        self.datasets = datasets
        self.cloth_particle_radius = self.env.cloth_particle_radius
        self.denormalize = self.env._env.denormalize
        self.phases_dict = phases_dict

    def generate_pick_and_place(self, actions, picker_pos):
        """
        :param actions: N x 5 vectors indicating a list of pick and place action in qpg format.
        # :param n_particles: Number of particles.
        :return: A list of control veloctires for both the particles (dummy ones) and the picker and whether the picker is activated
        """
        curr_pos = picker_pos
        model_actions = []
        for action in actions:
            generated_actions, curr_pos = self.env.action_tool.get_model_action(action, curr_pos)
            model_actions.extend(generated_actions)
        model_actions = np.array(model_actions)
        return model_actions[:, :, :3] / 60., model_actions[:, :, 3]

    # def get_pick_points(self, picker_pos, particle_pos):
    #     """ Picker pos: shape 1 x 3, particle pos: shape N x 3"""
    #     picker_threshold = 0.005
    #     picker_radius = 0.05
    #     particle_radius = 0.00625
    #     # Pick new particles and update the mass and the positions
    #     dists = scipy.spatial.distance.cdist(picker_pos, particle_pos[:, :3].reshape((-1, 3)))[0]
    #     idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
    #     mask = np.nonzero(dists.flatten() <= picker_threshold + picker_radius + particle_radius)
    #     print(mask)
    #     exit()
    #     if len(mask[0]) == 0:
    #         return -1
    #     else:
    #         idx_dists = idx_dists[mask, :].reshape((-1, 2))
    #         closest = np.argmin(idx_dists[:, 0])
    #         picked_point = idx_dists[closest]
    #         return picked_point

    def get_single_traj_cost(self, *args):
        actions, positions, velocities, clusters, cloth_xdim, cloth_ydim, config_id, phases_dict = args
        dataset = self.datasets['train']
        n_shapes = 1
        with torch.no_grad():
            control_vels, activated = self.generate_pick_and_place(actions, positions[-1:])
            # control_vels = control_vels[:, 0, :]  # n_shape =1
            # activated = activated[:, 0]  # n_shape =1
            for i in range(control_vels.shape[0]):
                print(i)
                if i == 0:
                    assert control_vels[i].shape[0] == 1, 'More than one shape will cause issue here'
                    # Broadcast control vel to all particles as we do not yet know which particle to pick
                    vel_nxt = np.tile(control_vels[i], [positions.shape[0], 1])
                    data = [positions, velocities, clusters, [self.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id], activated[i],
                            vel_nxt]
                    attr, state, rels, n_particles, n_shapes, instance_idx, sample_idx = dataset.construct_graph(data, downsample=True)

                    pos = np.vstack([state[:n_particles, :3], state[n_particles:n_particles + n_shapes, :3]])
                    pos = denormalize([pos], [self.stat[0]])[0]  # Unnormalize
                    pos_trajs = [pos.copy()]
                else:
                    vel_nxt = np.tile(control_vels[i], [positions.shape[0], 1])
                    data[0] = state[:, :3]
                    data[1] = state[:, 3:]
                    data[4], data[5] = activated[i], vel_nxt
                    attr, state, rels, n_particles, n_shapes, instance_idx, _ = dataset.construct_graph(data, downsample=False)
                graph = convert_dpi_to_graph(attr, state, rels, n_particles, n_shapes, instance_idx)
                predicted_vel = self.model(graph, phases_dict, 0)
                predicted_vel = denormalize([predicted_vel.data.cpu().numpy()], [self.stat[1]])[0]
                predicted_vel = np.concatenate([predicted_vel, vel_nxt[-n_shapes:]],
                                               0)  ### Model only outputs predicted particle velocity,

                # Manually set the velocities of the picked points
                picked_points = dataset.picked_points
                for pp in picked_points:
                    if pp != -1:
                        predicted_vel[pp, :] = vel_nxt[pp, :]

                ### so here we use the ground truth shape velocity. Why doesn't the model also predict the shape velocity?
                ### maybe, the shape velocity is kind of like the control actions specified by the users
                pos = copy.copy(pos_trajs[-1])
                pos += predicted_vel * 1 / 60.
                pos_trajs.append(pos)

                # Modify data for next step rollout (state includes positions and velocities)
                state = np.vstack([state[:n_particles, :], state[-n_shapes:, :]])
                pos = denormalize([state[:, :3]], [self.stat[0]])[0]  # Unnormalize
                pos += predicted_vel * 1 / 60.
                state[:, :3] = pos
                state[:, 3:] = predicted_vel

        rewards = []
        for pos in pos_trajs[1:]:
            reward_pos = np.hstack([pos[:-2, :3], np.zeros([pos.shape[0] - 2, 1])])
            rewards.append(self.reward_func(reward_pos))
        cost = - sum(rewards)
        return cost

    def get_cost(self, clusters, init_state, all_actions):
        costs = []
        for actions in all_actions:
            pos = init_state['particle_pos'].reshape(-1, 4)[:, :3]
            vel = init_state['particle_vel'].reshape(-1, 3)
            shape_states = init_state['shape_pos'].reshape(-1, 14)
            n_shapes = 1
            for k in range(n_shapes):
                pos = np.vstack([pos, shape_states[k, :3][None]])
                vel = np.vstack([vel, np.zeros([1, 3])])
            config = self.env.cached_configs[init_state['config_id']]
            cloth_xdim, cloth_ydim = config['ClothSize']
            costs.append(
                self.get_single_traj_cost(actions, pos, vel, clusters, cloth_xdim, cloth_ydim, init_state['config_id'], self.phases_dict))
            print('costs:', costs, actions.shape)
        return costs


class CEMOptimizer(object):
    def __init__(self, cost_function, solution_dim, max_iters, population_size, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.05):
        """
        :param cost_function: Takes input one or multiple data points in R^{sol_dim}\
        :param solution_dim: The dimensionality of the problem space
        :param max_iters: The maximum number of iterations to perform during optimization
        :param population_size: The number of candidate solutions to be sampled at every iteration
        :param num_elites: The number of top solutions that will be used to obtain the distribution
                            at the next iteration.
        :param upper_bound: An array of upper bounds for the sampled data points
        :param lower_bound: An array of lower bounds for the sampled data points
        :param epsilon: A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.solution_dim, self.max_iters, self.population_size, self.num_elites = \
            solution_dim, max_iters, population_size, num_elites

        self.ub, self.lb = upper_bound.reshape([1, solution_dim]), lower_bound.reshape([1, solution_dim])
        self.epsilon = epsilon

        if num_elites > population_size:
            raise ValueError("Number of elites must be at most the population size.")

        self.cost_function = cost_function

    def obtain_solution(self, clusters, cur_state, init_mean=None, init_var=None):
        """ Optimizes the cost function using the provided initial candidate distribution
        :param cur_state: Full state of the current environment such that the environment can always be reset to this state
        :param init_mean: (np.ndarray) The mean of the initial candidate distribution.
        :param init_var: (np.ndarray) The variance of the initial candidate distribution.
        :return:
        """
        mean = (self.ub + self.lb) / 2. if init_mean is None else init_mean
        var = (self.ub - self.lb) / 4. if init_var is None else init_var
        t = 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        while (t < self.max_iters):  # and np.max(var) > self.epsilon:
            print("inside CEM, iteration {}".format(t))
            samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            costs = self.cost_function(clusters, cur_state, samples)
            sort_costs = np.argsort(costs)

            elites = samples[sort_costs][:self.num_elites]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            t += 1
        sol, solvar = mean, var
        return sol


class CEMPolicy(object):
    """ Use the ground truth dynamics to optimize a trajectory of actions. """

    def __init__(self, env, env_class, env_kwargs, use_mpc, plan_horizon, max_iters, population_size, num_elites, model_path):
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon, self.action_dim = plan_horizon, len(env.action_space.sample())
        self.action_buffer = []
        self.prev_sol = None
        args, datasets, model, stat = prepare_model(model_path)
        # assert args.env_name == 'ClothFlatten'
        self.model = DPIModel(env, model, env._get_current_covered_area, stat, datasets, args.phases_dict)
        self.rollout_worker = ParallelRolloutWorker(self.model, plan_horizon, self.action_dim)

        lower_bound = np.tile(env.action_space.low[None], [self.plan_horizon, 1]).flatten()
        upper_bound = np.tile(env.action_space.high[None], [self.plan_horizon, 1]).flatten()
        self.optimizer = CEMOptimizer(self.rollout_worker.cost_function,
                                      self.plan_horizon * self.action_dim,
                                      max_iters=max_iters,
                                      population_size=population_size,
                                      num_elites=num_elites,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)
        self.clusters = None

    def reset(self, clusters):
        self.prev_sol = None
        self.clusters = clusters

    def get_action(self):
        if len(self.action_buffer) > 0 and self.use_mpc:
            action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
            return action
        self.env.debug = False
        env_state = self.env.get_state()
        soln = self.optimizer.obtain_solution(self.clusters, env_state, self.prev_sol).reshape([-1, self.action_dim])
        if self.use_mpc:
            self.prev_sol = np.vstack([np.copy(soln)[1:, :], np.zeros([1, self.action_dim])]).flatten()
        else:
            self.prev_sol = None
            self.action_buffer = soln[1:]  # self.action_buffer is only needed for the non-mpc case.
        self.env.set_state(env_state)  # Recover the environment
        print("cem finished planning!")
        return soln[0]


class ParallelRolloutWorker(object):
    """ Rollout a set of trajectory in parallel. """

    def __init__(self, model, plan_horizon, action_dim, num_worker=1):
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.model = model

    def cost_function(self, clusters, init_state, action_trajs):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        costs = self.model.get_cost(clusters, init_state, action_trajs)
        return costs
