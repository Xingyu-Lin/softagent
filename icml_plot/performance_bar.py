# Created by Xingyu Lin, 2019-04-10
from chester.plotting.cplot import *
from matplotlib.ticker import FormatStrFormatter
import os.path as osp
from softgym.env_performance_bound import env_performance_bound

# colors = ["crimson", "purple", "gold"]
# f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
# handles = [f("s", colors[i]) for i in range(3)]
# labels = colors
# legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

expert_final = {
    "ClothFold": -0.31,
    "PourWater": 0.96,
    "ClothDrop": -0.35,
    "PassWater": -0.27,
    "RopeFlatten": 4.95,
    "ClothFlatten": 1.56
}

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



def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


algo_mapping = {
    'planet_cam_rgb': 'planet',
    'TD3_key_point': 'TD3_feature',
    'SAC_key_point': 'SAC_feature',
    'SAC_cam_rgb': 'SAC_RGB',
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


dict_leg2col = {'planet': 5, 'TD3_feature': 1, 'SAC_feature': 2, 'SAC_RGB': 3, "RIG": 4, "UpperBound": 0, "Heuristic": 6}
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
bottoms = [-2, 0, 0, 0, -1.5, -1.5]


def autolabel(rects, ax, bottom):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height + bottom),
                    xy=(rect.get_x() + rect.get_width() / 2, height + bottom),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_bar():
    data_path = ['./data/yufei_s3_data/PlaNet-0202',
                 './data/yufei_s3_data/RIG-128-0202-all',
                 './data/yufei_s3_data/model-free-key-point-0202',
                 './data/yufei_s3_data/model-free-key-point-0203-last-2-seeds',
                 './data/seuss/0204_cem',
                 './data/yufei_s3_data/model-free-key-point-0204-ClothFold',
                 './data/yufei_s3_data/RIG-128-0204-ClothFold',
                 './data/yufei_s3_data/PlaNet-0204-ClothFold',
                 './data/yufei_s3_data/model-free-key-point-0205-ClothFlatten',
                 './data/yufei_s3_data/PlaNet-0205-ClothFlatten',
                 ]

    plot_keys = ['eval_info_final_performance']
    plot_keys_rlkit = ['evaluation/env_infos/final/performance Mean']
    plot_ylabels = ['Performance']
    plot_envs = ['PassWater', 'PourWater', 'RopeFlatten', 'ClothFlatten', 'ClothDrop', 'ClothFold']
    plot_goal_envs = ['PassWaterGoal', 'PourWaterGoal', 'RopeManipulate', 'ClothManipulate', 'ClothDropGoal', 'ClothFoldGoal']
    env_titles = ['PassWater', 'PourWater', 'StraightenRope', 'SpreadCloth', 'DropCloth', 'FoldCloth']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])

    bar_width = 1.5

    for (plot_key, plot_key_rlkit, plot_ylabel) in zip(plot_keys, plot_keys_rlkit, plot_ylabels):
        plt.figure(figsize=(40, 5))
        for plot_idx, env_name in enumerate(plot_envs):
            ax = plt.subplot('16' + str(plot_idx + 1))
            tmp_env_name = env_name
            upper_bound = env_performance_bound[env_name]
            bottom = bottoms[plot_idx]
            key = 'env_name'
            curr_x = 1

            # Plot UpperBound
            color = core.color_defaults[dict_leg2col["UpperBound"]]
            rects = ax.bar(curr_x, upper_bound - bottom, bar_width, label='UpperBound', bottom=bottom, color=color)
            autolabel(rects, ax, bottom)
            curr_x += bar_width

            ys = [upper_bound]
            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):

                if len(selector.where(key, tmp_env_name).extract()) == 0:
                    continue
                if 'RIG' in selector._exps_data[-1]['flat_params']['exp_name']:  #
                    tmp_env_name = plot_goal_envs[plot_idx]
                    key = 'skewfit_kwargs.env_id'

                if 'env_kwargs' in selector.where(key, tmp_env_name).extract()[0].params:
                    env_horizon = selector.where(key, tmp_env_name).extract()[0].params["env_kwargs"]["horizon"]
                else:
                    env_horizon = env_horizons[plot_idx]

                color = core.color_defaults[dict_leg2col[legend]]

                y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, tmp_env_name), plot_key, shade_type='median', )
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, tmp_env_name), plot_key_rlkit, shade_type='median')

                x, _, _ = get_shaded_curve_filter(selector.where(key, tmp_env_name), 'num_episodes', average=False)
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve_filter(selector.where(key, tmp_env_name), 'exploration/num steps total', average=False)
                else:
                    x = [ele * env_horizon for ele in x]

                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)
                if "Rope" in tmp_env_name:
                    y, y_lower, y_upper = -y, -y_lower, -y_upper

                rects = ax.bar(curr_x, np.mean(y[-10:]) - bottom, bar_width, label=legend, bottom=bottom, color=color)
                ys.append(np.mean(y[-10:]))
                autolabel(rects, ax, bottom)
                # ax.plot(x, y, color=color, label=legend, linewidth=2.0)
                # ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                #                 alpha=0.2)
                curr_x += bar_width
                # record the longest x
                # ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
                # ax.grid(True)
                # if plot_idx + 1 > 3:
                #     ax.set_xlabel('Timesteps')

                # if plot_idx == 0 or plot_idx == 3:  # only plot y-label for the left-most sub figures
                #     ax.set_ylabel(plot_ylabel)
                # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # Plot heuristic
            color = core.color_defaults[dict_leg2col["Heuristic"]]
            expert_mean = expert_final[env_name]
            rects = ax.bar(curr_x, expert_mean - bottom, bar_width, label='Heuristic', bottom=bottom, color=color)
            autolabel(rects, ax, bottom)
            curr_x += bar_width

            ax.set_ylim(top=max(ys) + (max(ys) - bottom) * 0.15)
            ax.set_xlim(left=0., right=9.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            plt.title(env_titles[plot_idx])
            plt.tight_layout()
            plt.subplots_adjust(left=0)
        save_name = filter_save_name('bar_plot')
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')
        # ax.legend(prop={'size': 8})
        # plot performance of heuristic policy
        # ax.hlines(expert_mean, xmin=0, xmax=max_x, color='red', linewidth=2.0, label='heuristic policy')

        # extrally store a legend
        # loc = 'best'
        # ax = plt.subplot('231')
        leg = ax.legend(prop={'size': 16}, ncol=6, bbox_to_anchor=(5.02, 1.45))
        leg.get_frame().set_linewidth(0.0)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(7.0)
        export_legend(leg, osp.join(save_path, 'bar_legend.png'))


if __name__ == '__main__':
    plot_bar()