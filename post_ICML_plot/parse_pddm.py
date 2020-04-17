import numpy as np

from chester.plotting.cplot import *

def custom_series_splitter(x):
    params = x['flat_params']
    # print("-" * 50)
    # print(params)
    # print("-" * 50)
    # exit()
    if ('env_kwargs.delta_reward' in params and params['env_kwargs.delta_reward'] is True) or (
        params['action_correlation'] == False
    ):
        return 'filtered'
    else:
        if 'RIG' in params['exp_name']: 
            return "RIG" 
        elif (params['algorithm']=='TD3' and params['env_kwargs.observation_mode']=='cam_rgb'):
            return 'filtered'
        else:
            return params['algorithm'] + '_' + params['env_kwargs.observation_mode']

data_path = [
            './data/yufei_seuss_data/0311-pddm-all'
        ]

plot_keys = ['info_sum_performance']
plot_envs = ['PassWater', 'PourWater', 'RopeFlatten', 'ClothFlatten', 'ClothDrop', 'ClothFold']

exps_data, plottable_keys, distinct_params = reload_data(data_path)
group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)

pddm_results_mean = {}
pddm_results_std = {}

print(group_legends)
# exit()
key = "env_name"
for tmp_env_name in plot_envs:
    print(tmp_env_name)
    for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
        progresses = selector.where(key, tmp_env_name)
        exps = progresses.extract()
        print(len(exps))
        progresses = [exp.progress.get(plot_keys[0], np.array([np.nan])) for exp in progresses.extract()]
        max_size = max(len(x) for x in progresses)
        progresses = [np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
        progresses = np.asarray(progresses)
        print("{} {}".format(tmp_env_name, progresses))
        mean = np.median(progresses)
        std = np.std(progresses)
        
        # print(np.mean(y))
        # y = np.mean(y)
        pddm_results_mean[tmp_env_name] = mean
        pddm_results_std[tmp_env_name] = std

    print("=" * 50)

print(pddm_results_mean)
print(pddm_results_std)