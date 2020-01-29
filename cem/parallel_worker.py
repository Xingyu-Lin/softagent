import multiprocessing as mp
from multiprocessing import Pool

import numpy as np


def get_cost(args):
    init_state, action_trajs, env_class, env_kwargs = args
    env = env_class(**env_kwargs)

    N = action_trajs.shape[0]
    costs = []
    for i in range(N):
        env.reset()
        ret = 0
        for action in action_trajs[i, :]:
            _, reward, _, _ = env.step(action)
            ret += reward
        costs.append(ret)
    return costs


class ParallelRolloutWorker(object):
    def __init__(self, env_class, env_kwargs, num_worker=2):
        self.num_worker = num_worker
        self.env_class, self.env_kwargs = env_class, env_kwargs
        self.pool = Pool(processes=num_worker)

    def cost_function(self, init_state, action_trajs):
        splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        ret = self.pool.map(get_cost, [(init_state, splitted_action_trajs[i], self.env_class, self.env_kwargs) for i in range(self.num_worker)])
        flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        return flat_costs


# if __name__ == '__main__':
#     # Can be used to benchmark the system
#     from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
#
#     mp.set_start_method('spawn')
#     env_name = 'PourWater'
#     env_class = SOFTGYM_ENVS[env_name]
#     env_kwargs = env_arg_dict[env_name]
#     env_kwargs['render'] = False
#     env_kwargs['observation_mode'] = 'key_point'
#     env = env_class(**env_kwargs)
#     env.reset()
#     initial_state = env.get_state()
#     action_trajs = []
#     for i in range(400):
#         action = env.action_space.sample()
#         action_trajs.append(action)
#     action_trajs = np.array(action_trajs).reshape([4, 100, -1])
#     rollout_worker = ParallelRolloutWorker(env_class, env_kwargs)
#     cost = rollout_worker.get_costs(initial_state, action_trajs)
#     print('cost:', cost)
