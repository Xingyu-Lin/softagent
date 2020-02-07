from chester.plotting.cplot import *

def custom_series_splitter(x):
    params = x['flat_params']
    # print("-" * 50)
    # print(params)
    # print("-" * 50)
    # exit()
    if ('env_kwargs.delta_reward' in params and params['env_kwargs.delta_reward'] is True):
        return 'filtered'
    else:
        if 'RIG' in params['exp_name']: 
            return "RIG" 
        elif (params['algorithm']=='TD3' and params['env_kwargs.observation_mode']=='cam_rgb'):
            return 'filtered'
        else:
            return params['algorithm'] + '_' + params['env_kwargs.observation_mode']

data_path = [
            '/tmp/0204_cem/0204_cem/'
        ]

plot_keys = ['info_sum_performance']
plot_envs = ['PassWater', 'PourWater', 'RopeFlatten', 'ClothFlatten', 'ClothDrop', 'ClothFold']

exps_data, plottable_keys, distinct_params = reload_data(data_path)
group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)

cem_results = {}

key = "env_name"
for tmp_env_name in plot_envs:
    for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
        y, y_lower, y_upper = get_shaded_curve(selector.where(key, tmp_env_name), plot_keys[0], shade_type='median', )
        print("{} {}".format(tmp_env_name, y))
        print(np.mean(y))
        y = np.mean(y)
        cem_results[tmp_env_name] = y

print(cem_results)