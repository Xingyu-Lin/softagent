from chester.plotting.cplot import *
from matplotlib.ticker import FormatStrFormatter
import os.path as osp


# cem_policy_mean = {'PassWater': -136.86884220000002, 'PourWater': 77.36095259999999, 'RopeFlatten': 543.3772710000001, 'ClothFlatten': 194.092694,
#                    'ClothDrop': -14.914467849999998, 'ClothFold': -20.4387325}
# cem_policy_std = {'PassWater': 141.91251967433524, 'PourWater': 4.172459790260996, 'RopeFlatten': 28.525632736445466,
#                   'ClothFlatten': 79.32274875977915, 'ClothDrop': 8.972272014085126, 'ClothFold': 4.671872594912007}

#  Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`, and if unique is True, after removing entries with duplicate labels.
def reorderLegend(ax=None, order=None, unique=False):
    if ax is None: ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))  # sort both labels and handles by labels
    if order is not None:  # Sort according to a given list (not necessarily complete)
        keys = dict(zip(order, range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t, keys=keys: keys.get(t[0], np.inf)))
    if unique:  labels, handles = zip(*unique_everseen(zip(labels, handles), key=labels))  # Keep only the first of each handle
    ax.legend(handles, labels)
    return (handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x, k in zip(seq, key) if not (k in seen or seen_add(k))]


def export_legend(legend, filename="./data/plots/legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


algo_mapping = {
    'planet_cam_rgb': 'RGB (PlaNet)',
    'CURL_key_point': 'Reduced State Oracle (SAC)',
    'CURL_cam_rgb': 'RGB (SAC-CURL)',
    'DrQ_cam_rgb': 'RGB (SAC-DrQ)',
    'CEM_key_point': 'Dynamics Oracle (CEM)',
    'CURL_point_cloud': 'Full State Oracle (SAC)'
}


def custom_series_splitter(x):
    params = x['flat_params']
    if 'env_kwargs_observation_mode' in params:
        obs_mode = params['env_kwargs_observation_mode']
    else:
        obs_mode = params['env_kwargs.observation_mode']
    ret = params['algorithm'] + '_' + obs_mode
    return algo_mapping[ret]


from collections import OrderedDict

dict_leg2col = OrderedDict({"Dynamics Oracle (CEM)": 0,
                            'Reduced State Oracle (SAC)': 2,
                            'Full State Oracle (SAC)': 10,
                            'RGB (SAC-CURL)': 1,
                            'RGB (SAC-DrQ)': 16,
                            'RGB (PlaNet)': 9,
                            })

save_path = './data/plots/'


def get_shaded_curve_filter(selector, key, shade_type='variance', interval=1, average=True):
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
    progresses = np.asarray(progresses)
    for i in range(num):
        if average:
            low = max(0, int(i * interval - interval / 2))
            high = min(progresses.shape[1], int(i * interval + 0.5 * interval))
            # print(np.nanmean(progresses[:, low: high], axis = 1).shape)
            ret[:, i] = np.nanmean(progresses[:, low: high], axis=1)
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


def plot_all():
    data_path = [
        './data/corl_data/',
        './data/corl_rebuttal'
    ]

    plot_keys_curl = ['eval/info_normalized_performance_final']
    plot_keys_planet = ['eval_info_final_normalized_performance']
    plot_ylabels = ['Performance']
    plot_envs = ['PassWater', 'PourWater', 'RopeFlattenNew', 'ClothFlatten', 'ClothFold', 'ClothDrop', ]
    env_titles = ['TransportWater', 'PourWater', 'StraightenRope', 'SpreadCloth', 'FoldCloth', 'DropCloth']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])

    for (plot_key_curl, plot_key_planet, plot_ylabel) in zip(plot_keys_curl, plot_keys_planet, plot_ylabels):
        fig = plt.figure(figsize=(24, 10))
        plotted_lines = []
        for plot_idx, env_name in enumerate(plot_envs):
            ax = plt.subplot('23' + str(plot_idx + 1))

            lw = 3.5

            # color = core.color_defaults[dict_leg2col["CEM"]]
            # ax.plot(range(max_x), np.ones(max_x) * cem_mean, color=color, linestyle='dashed', linewidth=lw, label='CEM')

            key = 'env_name'
            for idx, (selector, legend) in enumerate(zip(reversed(group_selectors), reversed(group_legends))):
                if len(selector.where(key, env_name).extract()) == 0:
                    continue

                env_horizon = selector.where(key, env_name).extract()[0].params["env_kwargs"]["horizon"]
                color = core.color_defaults[dict_leg2col[legend]]

                shade = 'median'
                y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_curl, shade_type=shade, )
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_planet, shade_type='median')

                x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'train/episode', average=False)
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'num_steps', average=False)
                else:
                    x = [ele * env_horizon for ele in x]

                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)

                plotted_lines.append(ax.plot(x, y, color=color, label=legend, linewidth=lw))
                ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                                alpha=0.2)

                # # record the longest x
                # if x[-1] > max_x:
                #     max_x = x[-1]
                #     max_x_list = x

                if legend == 'Dynamics Oracle (CEM)':
                    plot_key = 'info_final_normalized_performance'
                    progresses = selector.where(key, env_name)
                    print(env_name)
                    progresses = [exp.progress.get(plot_key, np.array([np.nan])) for exp in progresses.extract()]
                    y = np.mean([ele for ele in np.array(progresses).flatten() if not np.isnan(ele)])
                    # print(y)
                    N = 2000000
                    ax.plot(range(N), np.ones(N) * y, color=color, linewidth=lw)

            # def y_fmt(x, y):
            #     return str((np.round(x / 1000000.0, 1))) + 'M'
            # ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(True)
            if plot_idx + 1 > 3:
                ax.set_xlabel('Timesteps')

            if plot_idx == 0 or plot_idx == 3:  # only plot y-label for the left-most sub figures
                ax.set_ylabel(plot_ylabel)
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes = plt.gca()

            axes.set_xlim(left=0, right=1000000)
            axes.set_ylim(top=1.2)
            if env_name == 'ClothDrop':
                axes.set_ylim(bottom=0.)
            if env_name == 'PassWater':
                axes.set_ylim(bottom=-2)
            if env_name == 'ClothFold':
                axes.set_ylim(bottom=-0.5)
            plt.title(env_titles[plot_idx])
        plt.tight_layout()
        save_name = filter_save_name('learning_curves.png')
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')
        # extrally store a legend
        handles, labels = reorderLegend(ax, list(dict_leg2col.keys()))
        leg = ax.legend(handles, labels, prop={'size': 16}, ncol=6, bbox_to_anchor=(5.02, 1.45))
        leg.get_frame().set_linewidth(0.0)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(7.0)
        export_legend(leg, osp.join(save_path, 'legend.png'))


def autolabel(rects, ax, bottom):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height + bottom),
                    xy=(rect.get_x() + rect.get_width() / 2, height + bottom),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=20)


def plot_rigid_bar():
    data_path = ['./data/corl_data/']

    plot_key_curl = 'eval/info_normalized_performance_final'
    plot_key_planet = 'eval_info_final_normalized_performance'
    plot_ylabel = 'Performance'
    plot_envs = ['TransportTorus', 'PassWater', 'RigidClothFold', 'ClothFold', 'RigidClothDrop', 'ClothDrop']
    env_titles = ['TransportWater-Rigid', 'Transport', 'FoldCloth-Rigid', 'FoldCloth', 'DropCloth-Rigid', 'DropCloth']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])
    bar_width = 1.2

    plt.figure(figsize=(18, 5))
    for plot_idx, env_name in enumerate(plot_envs):
        ax = plt.subplot('13' + str(plot_idx // 2 + 1))
        tmp_env_name = env_name
        upper_bound = 1.0
        bottom = 0.0
        key = 'env_name'
        curr_x = 1.5 + bar_width * 3.5 * (plot_idx % 2)

        # rects = ax.bar(curr_x, upper_bound - bottom, bar_width, label='UpperBound', bottom=bottom, color=color)
        # autolabel(rects, ax, bottom)
        # curr_x += bar_width

        ys = [upper_bound]
        for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
            if legend in ['Dynamics Oracle (CEM)', 'RGB (PlaNet)', 'Full State Oracle (SAC)']:
                continue

            if len(selector.where(key, tmp_env_name).extract()) == 0:
                continue

            env_horizon = selector.where(key, tmp_env_name).extract()[0].params["env_kwargs"]["horizon"]

            color = core.color_defaults[dict_leg2col[legend]]

            y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_curl, shade_type='median')
            if len(y) <= 1:  # Hack
                y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_planet, shade_type='median')

            x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'train/episode', average=False)
            if len(x) <= 1:  # Hack
                x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'num_steps', average=False)
            else:
                x = [ele * env_horizon for ele in x]

            y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)

            clip_len = len([ele for ele in x if ele < 1000000])
            y = y[:clip_len]
            if len(y) < 1:
                y = [1]
            rects = ax.bar(curr_x, np.max(y) - bottom, bar_width, label=legend if plot_idx % 2 == 0 else "", bottom=bottom, color=color)
            ys.append(np.mean(y[-10:]))
            autolabel(rects, ax, bottom)
            curr_x += bar_width
        ax.annotate('Rigid' if plot_idx % 2 == 0 else 'Deformable',
                    xy=(curr_x - 0.5, -0.0),
                    xytext=(-50, -30),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=22)
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


def plot_qpg():
    data_path = [
        './data/corl_data/',
        './data/corl_rebuttal/'
    ]

    plot_keys_curl = ['eval/info_normalized_performance_final']
    plot_keys_planet = ['eval_info_final_normalized_performance']
    plot_ylabels = ['Performance']
    plot_envs = ['ClothFlatten']
    env_titles = ['SpreadCloth']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, ['filtered'])

    bar_width = 1.

    for (plot_key_curl, plot_key_planet, plot_ylabel) in zip(plot_keys_curl, plot_keys_planet, plot_ylabels):
        fig = plt.figure(figsize=(8, 5))
        plotted_lines = []
        for plot_idx, env_name in enumerate(plot_envs):
            ax = plt.subplot('11' + str(plot_idx + 1))

            lw = 3.5

            # color = core.color_defaults[dict_leg2col["CEM"]]
            # ax.plot(range(max_x), np.ones(max_x) * cem_mean, color=color, linestyle='dashed', linewidth=lw, label='CEM')

            key = 'env_name'
            curr_x = 1.
            for idx, (selector, legend) in enumerate(zip(reversed(group_selectors), reversed(group_legends))):
                if len(selector.where(key, env_name).extract()) == 0:
                    continue

                env_horizon = selector.where(key, env_name).extract()[0].params["env_kwargs"]["horizon"]
                color = core.color_defaults[dict_leg2col[legend]]

                shade = 'median'
                y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_curl, shade_type=shade, )
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve_filter(selector.where(key, env_name), plot_key_planet, shade_type='median')

                x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'train/episode', average=False)
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve_filter(selector.where(key, env_name), 'num_steps', average=False)
                else:
                    x = [ele * env_horizon for ele in x]

                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)

                # plotted_lines.append(ax.plot(x, y, color=color, label=legend, linewidth=lw))
                # ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,

                if legend == 'Dynamics Oracle (CEM)':
                    plot_key = 'info_final_normalized_performance'
                    progresses = selector.where(key, env_name)
                    progresses = [exp.progress.get(plot_key, np.array([np.nan])) for exp in progresses.extract()]
                    y = np.mean([ele for ele in np.array(progresses).flatten() if not np.isnan(ele)])
                    bar_height = np.max(y)
                else:
                    bar_height = np.max(y)
                rects = ax.bar(curr_x, bar_height, bar_width, label=legend, color=color)
                autolabel(rects, ax, 0.)
                curr_x += bar_width

            # Plot QPG
            color = core.color_defaults[12]
            rects = ax.bar(curr_x, 0.3154, bar_width, label='Wu et al. 20', color=color)
            autolabel(rects, ax, 0.)
            curr_x += bar_width

            axes = plt.gca()
            ax.set_ylim(top=1.)
            ax.set_xlim(left=0., right=8.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            plt.title(env_titles[plot_idx])
            plt.ylabel('Normalized Performance')
        plt.tight_layout()
        save_name = filter_save_name('qpg.png')

        # extrally store a legend
        # handles, labels = reorderLegend(ax, list(dict_leg2col.keys()))
        # leg = ax.legend(handles, labels, prop={'size': 4}, ncol=1)
        # leg.get_frame().set_linewidth(1.0)
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(1.0)
        ax.legend(prop={'size': 12})
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')
        # export_legend(leg, osp.join(save_path, 'legend_qpg.png'))

if __name__ == '__main__':
    # plot_all()
    # plot_rigid_bar()
    plot_qpg()
