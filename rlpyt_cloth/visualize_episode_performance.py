import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os.path as osp
matplotlib.rcParams.update({'font.size': 15})

if __name__ == '__main__':
    cem_data_path = './data/qpg_visualization/cloth_flatten.npy'
    qpg_data_path = './data/qpg_visualization/qpg_traj.npy'

    cem_data = np.load(cem_data_path)
    cem_mean = np.mean(cem_data, axis=0)
    cem_low = np.percentile(cem_data, 25, axis=0)
    cem_high = np.percentile(cem_data, 75, axis=0)
    dt = 1 / 120. * 8
    plt.plot(np.array(range(len(cem_mean))) * dt, cem_mean, label='Dynamics Oracle (CEM)')
    plt.fill_between(np.array(range(len(cem_mean))) * dt, cem_low, cem_high, alpha=0.2)

    with open(qpg_data_path, 'rb') as f:
        qpg_performance = np.load(f)
        qpg_total_steps = np.load(f)
        qpg_total_steps = np.array(qpg_total_steps).astype('float32') / 8.
    max_step = np.max(qpg_total_steps)
    xs = np.arange(0, max_step, 1)
    lines = []
    for (perform, total_step) in zip(qpg_performance, qpg_total_steps):
        line = np.interp(xs, total_step, perform)
        lines.append(line)
    qpg_data = np.array(lines)
    qpg_median = np.median(qpg_data, axis=0)
    qpg_low = np.percentile(qpg_data, 25, axis=0)
    qpg_high = np.percentile(qpg_data, 75, axis=0)
    plt.plot(np.array(range(len(qpg_median))) * dt, qpg_median, label='Wu et al. 20')
    plt.fill_between(np.array(range(len(qpg_median))) * dt, qpg_low, qpg_high, alpha=0.2)
    plt.ylabel('Normalized Performance')
    plt.xlabel('Elapsed Time (s) within Episode')
    plt.title('SpreadCloth - Within an Episode Performance')
    plt.legend()
    plt.savefig(osp.join('./data/qpg_visualization/', 'episode.png'))
