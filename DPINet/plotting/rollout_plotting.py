import matplotlib.pyplot as plt
import pickle
import os.path as osp
import numpy as np
from collections import OrderedDict


def decode_label(str):
    str += ' '
    keys = ['Noise: ', 'Downsample: ', 'EdgeType: ', 'NeighborRadius: ']
    d = {}
    for key in keys:
        st = str.find(key)
        st += len(key)
        en = str.find(' ', st)
        value = str[st:en]
        d[key[:-2]] = value
    return d


if __name__ == '__main__':
    pkl_file = './dpi_visualization/rollout_dump.pkl'
    save_folder = './dpi_visualization'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    # Figure of training curve
    # num_noise = len(data.items())
    # figs, axs = plt.subplots(nrows=1, ncols=num_noise, figsize=(5 * num_noise, 5))
    # for i, model_file in enumerate(data.keys()):
    #     ax = axs[i]
    #     for name, losses in data[model_file].items():
    #         ax.plot(range(len(losses)), losses, label=name)
    #     ax.set_title('training noise level:' + model_file)
    #     ax.legend(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(osp.join(save_folder, 'rollout_error_all.png'))


    num_noise = len(data.items())
    plt.figure(figsize=(15, 15))
    for i, model_file in enumerate(data.keys()):
        d = decode_label(model_file)
        if d['Noise'] != '0.03':
            continue
        curves = OrderedDict(data[model_file])
        avg_loss = [np.mean(losses) for losses in curves.values()]
        idx = np.argmin(avg_loss)
        name = list(curves.keys())[idx]

        loss = curves[name]
        plt.plot(range(len(loss)), loss, label=model_file + '_' + name, linewidth=8)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(osp.join(save_folder, 'rollout_error_best.png'))