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
from DPINet.visualize_data import prepare_model


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

    def get_single_traj_cost(self, *args):
        actions, positions, velocities, clusters, cloth_xdim, cloth_ydim, config_id, phases_dict = args
        pos_trajs = [positions]
        data = [positions, velocities, clusters, [self.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]]
        for i in range(actions.shape[0]):
            attr, state, rels, n_particles, n_shapes, instance_idx = self.datasets['train'].construct_graph(data)

            Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

            Rr, Rs = [], []

            for j in range(len(rels[0])):
                Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]  # NOTE: values are all just 1
                Rr.append(torch.sparse.FloatTensor(Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
                Rs.append(torch.sparse.FloatTensor(Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))
            data_cpu = copy.copy(data)
            data = [attr, state, Rr, Rs, Ra]

            with torch.set_grad_enabled(False):
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t].cuda())
                    else:
                        data[d] = Variable(data[d].cuda())

                attr, state, Rr, Rs, Ra = data

                predicted_vel = self.model(attr, state, Rr, Rs, Ra, n_particles, node_r_idx, node_s_idx, pstep, instance_idx, self.phases_dict,False)
            predicted_vel = denormalize([predicted_vel.data.cpu().numpy()], [self.stat[1]])[0]
            action = self.denormalize(actions[i, :])
            # Specifically for cloth flatten
            shape_vel = action.reshape((-1, 4))[:, :3] * 60.
            predicted_vel = np.vstack([predicted_vel, shape_vel])  ### Model only outputs predicted particle velocity,
            ### so here we use the ground truth shape velocity. Why doesn't the model also predict the shape velocity?
            ### maybe, the shape velocity is kind of like the control actions specified by the users
            pos = copy.copy(pos_trajs[-1])
            pos += predicted_vel * 1 / 60.

            # Add back fourth dimension and remove the picker

            pos_trajs.append(pos)

            # Modify data for next step rollout
            data_cpu[0] = data_cpu[0] + predicted_vel * 1 / 60.
            data_cpu[1][:, :3] = predicted_vel
            data = data_cpu
        rewards = []
        for pos in pos_trajs:
            reward_pos = np.hstack([pos[:-2, :3], np.zeros([pos.shape[0] - 2, 1])])
            rewards.append(self.reward_func(reward_pos))
        cost = - sum(rewards)
        return cost

    def get_cost(self, args):
        clusters, init_state, all_actions = args
        costs = []
        for actions in all_actions:
            pos = init_state['particle_pos'].reshape(-1, 4)[:, :3]
            vel = init_state['particle_vel'].reshape(-1, 3)
            shape_states = init_state['shape_pos'].reshape(-1, 14)
            n_shapes = 2
            for k in range(n_shapes):
                pos = np.vstack([pos, shape_states[k, :3][None]])
                vel = np.vstack([vel, np.zeros([1, 3])])
            config = self.env.cached_configs[init_state['config_id']]
            cloth_xdim, cloth_ydim = config['ClothSize']
            costs.append(self.get_single_traj_cost(actions, pos, vel, clusters, cloth_xdim, cloth_ydim, init_state['config_id'], self.datasets['train']))
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

    def __init__(self, model, plan_horizon, action_dim, num_worker=4):
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.pool = Pool(processes=num_worker)
        self.model = model

    def cost_function(self, clusters, init_state, action_trajs):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        ret = self.pool.map(self.model.get_cost, [(clusters, init_state, splitted_action_trajs[i]) for i in range(self.num_worker)])
        flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        return flat_costs
