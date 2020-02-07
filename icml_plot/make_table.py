# Created by Xingyu Lin, 2019-04-10
from chester.plotting.cplot import *
from matplotlib.ticker import FormatStrFormatter
import os.path as osp


expert_policy_mean = {
    "ClothFold": -64.50,
    "PourWater": 55.59863877422661,
    "ClothDrop": -15.127672304436565,
    "PassWater": -28.915380035466747,
    "RopeFlatten": 310.184619798474,
    "ClothFlatten": 150.50905745183715
}

expert_policy_std = {
    "ClothFold": 16.81, 
    "PourWater": 3.2476477013360325,
    "ClothDrop": 3.453695976313334,
    "PassWater": 25.340062501810817,
    "RopeFlatten": 53.90255121006629,
    "ClothFlatten": 63.73237114254054
}

cem_mean = {

}

cem_std = {

}

def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def custom_series_splitter(x):
    params = x['flat_params']
    if ('env_kwargs.delta_reward' in params and params['env_kwargs.delta_reward'] is True):
        return 'filtered'
    else:
        if 'RIG' in params['exp_name']: 
            return "RIG" 
        elif (params['algorithm']=='TD3' and params['env_kwargs.observation_mode']=='cam_rgb'):
            return 'filtered'
        else:
            return params['algorithm'] + '_' + params['env_kwargs.observation_mode']


dict_leg2col = {'planet_cam_rgb': 0, 'TD3_key_point': 1,  'SAC_key_point': 2, 'SAC_cam_rgb': 3, "RIG": 4, 'TD3_cam_rgb': 2,}
save_path = './data/icml/'


def filter_legend(group_selectors, group_legends, filtered_legends):
    combined_list = [(selector, legend) for (selector, legend) in zip(group_selectors, group_legends) if
                     legend not in filtered_legends]
    group_selectors, group_legends = map(list, zip(*combined_list))
    return group_selectors, group_legends


def filter_nan(xs, *args):
    non_nan_idx = [i for i in range(len(xs)) if not np.isnan(xs[i])]
    new_lists = [np.array(one_list)[non_nan_idx] for one_list in args]
    return np.array(xs)[non_nan_idx], new_lists

def get_last_avg(y):
    l = len(y)
    y = y[-int(l // 10):]
    return np.mean(y), np.std(y)


def get_shaded_curve(selector, key, shade_type='variance', horizon = None):
    """
    :param selector: Selector for a group of curves
    :param shade_type: Should be either 'variance' or 'median', indicating how the shades are calculated.
    :return: [y, y_lower, y_upper], representing the mean, upper and lower boundary of the shaded region
    """

    # First, get the progresses
    progresses = [exp.progress.get(key, np.array([np.nan])) for exp in selector.extract()]
    max_size = max(len(x) for x in progresses)
    progresses = [np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
    progresses = np.asarray(progresses) * horizon if horizon is not None else np.asarray(progresses) 

    # Second, calculate the shaded area
    if shade_type == 'median':
        percentile25 = np.nanpercentile(
            progresses, q=25, axis=0)
        percentile50 = np.nanpercentile(
            progresses, q=50, axis=0)
        percentile75 = np.nanpercentile(
            progresses, q=75, axis=0)

        y = list(percentile50)
        y_upper = list(percentile75)
        y_lower = list(percentile25)
    elif shade_type == 'variance':
        means = np.nanmean(progresses, axis=0)
        stds = np.nanstd(progresses, axis=0)

        y = list(means)
        y_upper = list(means + stds)
        y_lower = list(means - stds)
    else:
        raise NotImplementedError

    return y, y_lower, y_upper

# hardcode env horizons
env_horizons = [75, 100, 75, 100, 15, 100]

algo_performances_mean = {}
algo_performances_std = {}

def plot_all():
    data_path = [ './data/yufei_s3_data/PlaNet-0202', 
    './data/yufei_s3_data/RIG-128-0202-all',
                './data/yufei_s3_data/model-free-key-point-0202', 
        './data/yufei_s3_data/model-free-key-point-0203-last-2-seeds', 
        '/tmp/0201_model_free_cam_rgb/0201_model_free_cam_rgb',
        './data/yufei_s3_data/model-free-key-point-0204-ClothFold',
        './data/yufei_s3_data/RIG-128-0204-ClothFold',
        './data/yufei_s3_data/PlaNet-0204-ClothFold',
        './data/yufei_s3_data/model-free-key-point-0205-ClothFlatten',
        './data/yufei_s3_data/PlaNet-0205-ClothFlatten',
        ]

    plot_keys = ['eval_info_sum_performance']
    plot_keys_rlkit = ['evaluation/env_infos/performance Mean']
    plot_ylabels = ['Return']
    plot_envs = ['PassWater', 'PourWater', 'RopeFlatten', 'ClothFlatten', 'ClothDrop', 'ClothFold']
    plot_goal_envs = ['PassWaterGoal', 'PourWaterGoal', 'RopeManipulate', 'ClothManipulate', 'ClothDropGoal', 'ClothFoldGoal']

    for env in plot_envs:
        algo_performances_mean[env] = {}
        algo_performances_std[env] = {}

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])

    # for algo in group_legends:
    #     print(algo)
    #     algo_performances_mean[algo] = {}
    #     algo_performances_std[algo] = {}

    for (plot_key, plot_key_rlkit, plot_ylabel) in zip(plot_keys, plot_keys_rlkit, plot_ylabels):
        for plot_idx, env_name in enumerate(plot_envs):
            tmp_env_name = env_name

            key = 'env_name'
            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
                if len(selector.where(key, tmp_env_name).extract()) == 0:
                    continue
                if 'RIG' in selector._exps_data[-1]['flat_params']['exp_name']: #
                    tmp_env_name = plot_goal_envs[plot_idx]
                    key = 'skewfit_kwargs.env_id'
                    
                if 'env_kwargs' in selector.where(key, tmp_env_name).extract()[0].params:
                    env_horizon = selector.where(key, tmp_env_name).extract()[0].params["env_kwargs"]["horizon"]
                else:
                    env_horizon = env_horizons[plot_idx]

                y, y_lower, y_upper = get_shaded_curve(selector.where(key, tmp_env_name), plot_key, shade_type='median', )
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve(selector.where(key, tmp_env_name), plot_key_rlkit, 
                        shade_type='median', horizon=env_horizon)


                x, _, _ = get_shaded_curve(selector.where(key, tmp_env_name), 'num_episodes')
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve(selector.where(key, tmp_env_name), 'exploration/num steps total')
                else:
                    x = [ele * env_horizon for ele in x]
                
                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)
                if "Rope" in tmp_env_name:
                    y, y_lower, y_upper = -y, -y_lower, -y_upper

                mean, std = get_last_avg(y)
                algo_performances_mean[env_name][legend] = mean
                algo_performances_std[env_name][legend] = std

def make_table(dict_mean, dict_std):
    envs = ["PassWater", "PourWater", "ClothFlatten", "ClothFold", "ClothDrop", "RopeFlatten"]
    algos = ["TD3_key_point", "SAC_key_point", "SAC_cam_rgb", "planet_cam_rgb", "RIG", "CEM", "Heuristic"]
    env_prints = ["PassWater", "PourWater", "SpreadCloth", "FoldCloth", "DropCloth", "StraightenRope"]
    for idx in range(len(envs)):
        env = envs[idx]
        print(env_prints[idx], end = " ")
        for algo in algos:
            if algo == 'SAC_cam_rgb' or algo == 'RIG' or algo == 'CEM' or algo == 'Heuristic':
                print(r"&&", end = ' ')
            else:
                print(r"&", end = ' ')
            print(round(dict_mean[env].get(algo, 0), 1), end = ' ')
            print(r"$\pm$ {}".format(round(dict_std[env].get(algo, 0), 1)), end = ' ')
        print(r"\\")

if __name__ == '__main__':
    plot_all()
    envs = ["PassWater", "PourWater", "ClothFlatten", "ClothFold", "ClothDrop", "RopeFlatten"]
    for env in envs:
        algo_performances_mean[env]["Heuristic"] = expert_policy_mean[env]
        algo_performances_std[env]["Heuristic"] = expert_policy_std[env]
    # algo_performances_mean['Heuristic']
    make_table(algo_performances_mean, algo_performances_std)
    # print(algo_performances_mean)
    # print(algo_performances_std)
