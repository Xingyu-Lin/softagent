import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

env = None


def get_cost(args):
    init_state, action_trajs, env_class, env_kwargs = args
    global env
    if env is None:
        # Need to create the env inside the function such that the GPU buffer is associated with the child process and avoid any deadlock.
        # Use the global variable to access the child process-specific memory
        env = env_class(**env_kwargs)
        print('Child env created!')
    env.reset(config_id=init_state['config_id'])

    N = action_trajs.shape[0]
    costs = []
    for i in range(N):
        env.set_state(init_state)
        ret = 0
        for action in action_trajs[i, :]:
            _, reward, _, _ = env.step(action)
            ret += reward
        costs.append(-ret)
        # print('get_cost {}: {}'.format(i, ret))
    return costs


class ParallelRolloutWorker(object):
    """ Rollout a set of trajectory in parallel. """

    def __init__(self, env_class, env_kwargs, plan_horizon, action_dim, num_worker=8):
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.env_class, self.env_kwargs = env_class, env_kwargs
        self.pool = Pool(processes=num_worker)

    def cost_function(self, init_state, action_trajs):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        ret = self.pool.map(get_cost, [(init_state, splitted_action_trajs[i], self.env_class, self.env_kwargs) for i in range(self.num_worker)])
        flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        return flat_costs


if __name__ == '__main__':
    # Can be used to benchmark the system
    from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS

    mp.set_start_method('spawn')
    env_name = 'PourWater'
    env_class = SOFTGYM_ENVS[env_name]
    env_kwargs = env_arg_dict[env_name]
    env_kwargs['render'] = False
    env_kwargs['observation_mode'] = 'key_point'
    env = env_class(**env_kwargs)
    env.reset()
    initial_state = env.get_state()
    action_trajs = []
    for i in range(400):
        action = env.action_space.sample()
        action_trajs.append(action)
    action_trajs = np.array(action_trajs).reshape([4, 100, -1])
    rollout_worker = ParallelRolloutWorker(env_class, env_kwargs, 10, 4)
    cost = rollout_worker.cost_function(initial_state, action_trajs)
    print('cost:', cost)
