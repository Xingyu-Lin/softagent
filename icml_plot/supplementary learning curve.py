# Created by Xingyu Lin, 2019-04-10
from chester.plotting.cplot import *
from matplotlib.ticker import FormatStrFormatter
import os.path as osp

# colors = ["crimson", "purple", "gold"]
# f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
# handles = [f("s", colors[i]) for i in range(3)]
# labels = colors
# legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

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

RIG_fold_epoch_median =  [136.05282512954537, 114.24484040189607, 122.45100888645725, 148.31479379907105, 111.6825752511653, 130.43628585954556, 111.14634396673293, 114.12847153545995, 141.39887000497384, 119.44937511859125, 117.11791536075664, 128.97492191648053, 104.06639409162923]
RIG_fold_epoch_lower =  [124.06683471324945, 112.78077397479451, 119.28894079604495, 134.83064717436523, 108.17928487690244, 129.5258304051843, 110.68422970554931, 113.39886249156423, 114.43859792595299, 117.72401864323471, 114.89675582091418, 128.30487727976484, 101.61799290375731]
RIG_fold_epoch_upper =  [148.48379799922128, 116.32339466402811, 125.28292749806158, 154.97999398544886, 164.74327856672616, 134.8218171579951, 120.1386075610645, 139.39160222721756, 142.0123297613904, 124.20580895949601, 132.54358399332386, 131.27656762626066, 116.26701665821425]

def export_legend(legend, filename="supply_legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


algo_mapping = {
    'planet_cam_rgb': 'planet',
    'TD3_key_point': 'TD3_feature',
    'SAC_key_point': 'SAC_feature',
    'SAC_cam_rgb': 'SAC_RGB',
    'SAC_point_cloud': 'SAC_full_state', 
    'TD3_point_cloud': 'TD3_full_state',
    'RIG': 'RIG'
}
def custom_series_splitter(x):
    params = x['flat_params']
    if ('env_kwargs.delta_reward' in params and params['env_kwargs.delta_reward'] is True) \
      or ('algorithm' in params and 'CEM' in params['algorithm']):
        return 'filtered'
    else:
        if 'RIG' in params['exp_name']:
            ret = "RIG"
        elif (params['algorithm'] == 'TD3' and params['env_kwargs.observation_mode'] == 'cam_rgb'):
            return 'filtered'
        else:
            ret = params['algorithm'] + '_' + params['env_kwargs.observation_mode']
        return algo_mapping[ret]
dict_leg2col = {'TD3_full_state': 5, 'TD3_feature': 1, 'SAC_feature': 2, 'SAC_RGB': 3, "SAC_full_state": 4, "UpperBound": 0, "Heuristic": 6}

save_path = './data/icml/'


def get_shaded_curve_filter(selector, key, shade_type='variance', interval=10, average=True, horizon=None):
    """
    :param selector: Selector for a group of curves
    :param shade_type: Should be either 'variance' or 'median', indicating how the shades are calculated.
    :return: [y, y_lower, y_upper], representing the mean, upper and lower boundary of the shaded region
    """

    # First, get the progresses
    progresses = [exp.progress.get(key, np.array([np.nan])) for exp in selector.extract()]
    # print("len progresses: ", len(progresses))
    max_size = max(len(x) for x in progresses)
    progresses = [np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]

    num = int(len(progresses[0]) // interval)
    # print("len progresses {} interval {} num {}".format(len(progresses[0]), interval, num))
    ret = np.zeros((len(progresses), num))
    progresses = np.asarray(progresses) * horizon if horizon is not None else np.asarray(progresses)
    # exit()
    for i in range(num):
        if average:
            low = max(0, int(i * interval - interval / 2))
            high = min(progresses.shape[1], int(i * interval + 0.5 * interval))
            # print(np.nanmean(progresses[:, low: high], axis = 1).shape)
            ret[:, i] = np.nanmean(progresses[:, low: high], axis=1)
            # exit()
        else:
            ret[:, i] = progresses[:, i * interval]

    if not average:
        return list(np.nanmean(ret, axis=0)), None, None

    if shade_type == 'variance':
        means = np.nanmean(ret, axis=0)
        stds = np.nanstd(ret, axis=0)

        y = list(means)
        y_upper = list(means + stds)
        y_lower = list(means - stds)
    else:
        percentile25 = np.nanpercentile(
            ret, q=25, axis=0)
        percentile50 = np.nanpercentile(
            ret, q=50, axis=0)
        percentile75 = np.nanpercentile(
            ret, q=75, axis=0)

        y = list(percentile50)
        y_upper = list(percentile75)
        y_lower = list(percentile25)

    return y, y_lower, y_upper


def filter_interval(y, interval=10000, average=True):
    interval = int(interval)
    num = len(y) // interval
    print("len(y) {} interval {} num {} ".format(len(y), interval, num))
    num = int(num)
    ret = np.zeros(num)
    y = np.asarray(y)
    for i in range(num):
        if average:
            ret[i] = np.mean(y[max(0, i * interval - int(interval / 2)): min(i * interval + int(interval / 2), len(y))])
        else:
            ret[i] = y[i * interval]

    std = np.std(ret)
    return ret, ret - std, ret + std


def filter_legend(group_selectors, group_legends, filtered_legends):
    combined_list = [(selector, legend) for (selector, legend) in zip(group_selectors, group_legends) if
                     legend not in filtered_legends]
    group_selectors, group_legends = map(list, zip(*combined_list))
    return group_selectors, group_legends


def filter_nan(xs, *args):
    non_nan_idx = [i for i in range(len(xs)) if not np.isnan(xs[i])]
    new_lists = [np.array(one_list)[non_nan_idx] for one_list in args]
    return np.array(xs)[non_nan_idx], new_lists


# hardcode env horizons
env_horizons = [75, 100, 75, 100, 15, 100]


def plot_all():
    data_path = [
                # './data/yufei_s3_data/PlaNet-0202',
                #  './data/yufei_s3_data/RIG-128-0202-all',
                 './data/yufei_s3_data/model-free-key-point-0202',
                 './data/yufei_s3_data/model-free-key-point-0203-last-2-seeds',
                 '/tmp/0201_model_free_cam_rgb/0201_model_free_cam_rgb',
                 './data/yufei_s3_data/model-free-key-point-0204-ClothFold',
                #  './data/yufei_s3_data/RIG-128-0204-ClothFold',
                #  './data/yufei_s3_data/PlaNet-0204-ClothFold',
                 './data/yufei_s3_data/model-free-key-point-0205-ClothFlatten',
                #  './data/yufei_s3_data/PlaNet-0205-ClothFlatten',
                './data/yufei_seuss_data/2-17-model-free-full-state',
                 ]

    plot_keys = ['eval_info_sum_performance']
    plot_keys_rlkit = ['evaluation/env_infos/performance Mean']
    plot_ylabels = ['Return']
    plot_envs = ['ClothFlatten', 'ClothDrop', 'ClothFold'] # 'PassWater', 'PourWater', # 'RopeFlatten'
    plot_goal_envs = ['PassWaterGoal', 'PourWaterGoal', 'RopeManipulate', 'ClothManipulate', 'ClothDropGoal', 'ClothFoldGoal']
    env_titles = ['SpreadCloth', 'DropCloth', 'FoldCloth'] # 'PassWater', 'PourWater','StraightenRope',

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])

    # print("group_legends: ", group_legends)
    # exit()
    for (plot_key, plot_key_rlkit, plot_ylabel) in zip(plot_keys, plot_keys_rlkit, plot_ylabels):
        fig = plt.figure(figsize=(24, 5))
        plotted_lines = []
        for plot_idx, env_name in enumerate(plot_envs):
            tmp_env_name = env_name
            ax = plt.subplot('13' + str(plot_idx + 1))

            expert_mean = expert_policy_mean[env_name]
            expert_std = expert_policy_std[env_name]

            key = 'env_name'
            max_x = 0
            max_x_list = []
            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
                if len(selector.where(key, tmp_env_name).extract()) == 0:
                    continue

                print(selector._exps_data[-1]['flat_params']['exp_name'])
                

                RIG = False
                if 'RIG' in selector._exps_data[-1]['flat_params']['exp_name']: #
                    tmp_env_name = plot_goal_envs[plot_idx]
                    key = 'skewfit_kwargs.env_id'
                    RIG = True

                Full_state = False
                if 'full-state' in selector._exps_data[-1]['flat_params']['exp_name']:
                    Full_state = True

                # print("key is: ", key)
                # print("tmp_env_name is: ", tmp_env_name)
                if 'env_kwargs' in selector.where(key, tmp_env_name).extract()[0].params:
                    env_horizon = selector.where(key, tmp_env_name).extract()[0].params["env_kwargs"]["horizon"]
                else:
                    env_horizon = env_horizons[plot_idx]

                color = core.color_defaults[dict_leg2col[legend]]

                shape = 'median'
                y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, tmp_env_name), plot_key, shade_type=shape, )
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, tmp_env_name), plot_key_rlkit, shade_type='median',
                                                                  horizon=env_horizon)

                x, _, _ = get_shaded_curve_filter(selector.where(key, tmp_env_name), 'num_episodes', average=False)
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve_filter(selector.where(key, tmp_env_name), 'exploration/num steps total', average=False)
                else:
                    x = [ele * env_horizon for ele in x]

                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)
                if "Rope" in tmp_env_name and not Full_state:
                    y, y_lower, y_upper = -y, -y_lower, -y_upper

                if "Drop" in tmp_env_name:
                    ax.set_ylim(bottom = -25, top=-10)

                if "Fold" in tmp_env_name:
                    ax.set_ylim(bottom = -120, top=-25)
                
                if "Water" in tmp_env_name or "Rope" in tmp_env_name or "Fold" in tmp_env_name:
                    ax.set_xlim(right=2000000)

                if 'Drop' in tmp_env_name or tmp_env_name == 'ClothManipulate' or tmp_env_name == 'ClothFlatten':
                    ax.set_xlim(right=1000000)

                if tmp_env_name == 'ClothManipulate' and RIG:
                    y = RIG_fold_epoch_median
                    y_lower = RIG_fold_epoch_lower
                    y_upper = RIG_fold_epoch_upper
                    x = [i * 20 * 1000 for i in range(len(y))]

                lw = 3.5
                plotted_lines.append(ax.plot(x, y, color=color, label=legend, linewidth=lw))
                ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                                alpha=0.2)

                # record the longest x
                if x[-1] > max_x:
                    max_x = x[-1]
                    max_x_list = x

            # plot performance of heuristic policy
            # ax.hlines(expert_mean, xmin=0, xmax=max_x, color='red', linewidth=3.0, label='Heuristic')
            max_x = int(max_x)
            color = core.color_defaults[dict_leg2col["Heuristic"]]
            ax.plot(range(max_x), np.ones(max_x) * expert_mean, color=color, linewidth=lw, label='Heuristic')
            # expert_low = [expert_mean - expert_std for i in range(len(max_x_list))]
            # expert_high = [expert_mean + expert_std for i in range(len(max_x_list))]
            # ax.fill_between(max_x_list, expert_low, expert_high, interpolate=True, facecolor='red', linewidth=0.0,
            #     alpha=0.1)

            def y_fmt(x, y):
                return str((np.round(x / 1000000.0, 1))) + 'M'

            ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
            ax.grid(True)
            if plot_idx + 1 > 3:
                ax.set_xlabel('Timesteps')

            if plot_idx == 0 or plot_idx == 3:  # only plot y-label for the left-most sub figures
                ax.set_ylabel(plot_ylabel)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes = plt.gca()

            axes.set_xlim(left=0)
            plt.title(env_titles[plot_idx])

            save_name = filter_save_name('supply_learning_curves' + '_' + plot_key)

        # # extrally store a legend
        # loc = 'best'
        # # ax = plt.subplot('231')
        # leg = ax.legend(loc=loc, prop={'size': 16}, ncol=6, labels=group_legends + ["Heuristic"], bbox_to_anchor=(5.02, 1.45))
        # leg.get_frame().set_linewidth(0.0)
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(7.0)
        # export_legend(leg)

        plt.tight_layout()
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')


if __name__ == '__main__':
    plot_all()
