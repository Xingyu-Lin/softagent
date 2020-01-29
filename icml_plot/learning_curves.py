# Created by Xingyu Lin, 2019-04-10
from chester.plotting.cplot import *
from matplotlib.ticker import FormatStrFormatter
import os.path as osp


# colors = ["crimson", "purple", "gold"]
# f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
# handles = [f("s", colors[i]) for i in range(3)]
# labels = colors
# legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def custom_series_splitter(x):
    params = x['flat_params']
    return params['algorithm'] + '_' + params['env_kwargs.observation_mode']
    # if 'use_ae_reward' in params and params['use_ae_reward']:
    #     return 'AE'
    # if 'use_latent_representation' in params and params['use_latent_representation'] is not None:
    #     return params['use_latent_representation']
    # if params['env_kwargs.use_true_reward']:
    #     if 'filter' in params['her_replay_strategy']:
    #         return 'Oracle+Filter'
    #     else:
    #         return 'Oracle'
    # else:
    #     if params['her_replay_strategy'] == 'balance_filter':
    #         return 'Indicator+Balance+Filter'
    #     if params['her_replay_strategy'] == 'balance' or params['her_replay_strategy'] == 'balances':
    #         return 'Indicator+Balance'
    #     if params['her_replay_strategy'] == 'HER_future_filter':
    #         return 'Indicator+Filter'
    #     if params['her_replay_strategy'] == 'balance_filter_mix':
    #         return 'Indicator+Balance+Mix'
    #     return 'Indicator'


dict_leg2col = {'planet_cam_rgb': 0, 'TD3_key_point': 1, 'TD3_cam_rgb': 2}
ablation_legend = ['Oracle+Filter', 'Indicator+Filter', 'Indicator+Balance']
non_ablation_legend = ['Oracle+Filter', 'Oracle']
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


def plot_all():
    data_path = ['./data/seuss/0125_planet', './data/seuss/0126_td3_key_point', './data/seuss/0127_td3_cam_rgb']
    # data_path = ['./data/seuss/0125_planet', './data/seuss/0126_td3_key_point']

    plot_keys = ['info_final_performance']
    plot_keys_rlkit = ['evaluation/env_infos/final/performance Mean']
    plot_ylabels = ['Return']
    plot_envs = ['PassWater', 'PourWater', 'ClothDrop', 'ClothFlatten', 'ClothFold', 'RopeFlatten']
    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    group_selectors, group_legends = filter_legend(group_selectors, group_legends, [])
    for (plot_key, plot_key_rlkit, plot_ylabel) in zip(plot_keys, plot_keys_rlkit, plot_ylabels):
        plt.figure(figsize=(24, 10))
        for plot_idx, env_name in enumerate(plot_envs):
            ax = plt.subplot('23' + str(plot_idx + 1))

            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
                if len(selector.where('env_name', env_name).extract()) ==0:
                    continue
                env_horizon = selector.where('env_name', env_name).extract()[0].params["env_kwargs"]["horizon"]

                color = core.color_defaults[dict_leg2col[legend]]
                y, y_lower, y_upper = get_shaded_curve(selector.where('env_name', env_name), plot_key, shade_type='median')
                if len(y) <= 1:  # Hack
                    y, y_lower, y_upper = get_shaded_curve(selector.where('env_name', env_name), plot_key_rlkit, shade_type='median')

                x, _, _ = get_shaded_curve(selector.where('env_name', env_name), 'num_episodes')
                if len(x) <= 1:  # Hack
                    x, _, _ = get_shaded_curve(selector.where('env_name', env_name), 'exploration/num paths total')
                x = [ele * env_horizon for ele in x]
                y, [y_lower, y_upper, x] = filter_nan(y, y_lower, y_upper, x)
                ax.plot(x, y, color=color, label=legend, linewidth=2.0)

                ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                                alpha=0.2)

            def y_fmt(x, y):
                return str(int(np.round(x / 1000.0))) + 'K'

            ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
            ax.grid(True)
            if plot_idx + 1 > 3:
                ax.set_xlabel('Timesteps')
            ax.set_ylabel(plot_ylabel)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes = plt.gca()

            axes.set_xlim(left=0)
            plt.title(env_name)
            # if env_name == 'Reacher' and plot_key == 'test/goal_dist_final_state' and False:
            loc = 'best'
            leg = ax.legend(loc=loc, prop={'size': 16}, ncol=1, labels=group_legends)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)

            # For saving legendd only
            # if env_name == 'Reacher' and plot_key == 'test/goal_dist_final_state':
            #     loc = 'best'r
            #     leg = ax.legend(loc=loc, prop={'size': 40}, ncol=6, labels=group_legends, bbox_to_anchor=(2.1, 2.05))
            #     leg.get_frame().set_linewidth(0.0)
            #     for legobj in leg.legendHandles:
            #         legobj.set_linewidth(7.0)
            #     export_legend(leg)

            save_name = filter_save_name('learning_curves' + plot_key)
        plt.tight_layout()
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')


if __name__ == '__main__':
    plot_all()
    # plot_state_learning()
    # plot_visual_learning()
    # plot_visual_robot_learning()
    # plot_ablation()
    # Temporary
    # plot_fetch_slide()
