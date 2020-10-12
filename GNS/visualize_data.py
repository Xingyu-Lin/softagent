import time
import argparse
import os
import os.path as osp
from DPINet.graph import load_data
from DPINet.data import load_data, prepare_input, normalize, denormalize
import copy
from softgym.registered_env import env_arg_dict as env_arg_dicts
from softgym.registered_env import SOFTGYM_ENVS
import numpy as np
import pyflex
import torch
import json
from torch.autograd import Variable
from DPINet.data import collate_fn
from DPINet.models import DPINet
from DPINet.graph import ClothDataset
from DPINet.graph_struct import convert_dpi_to_graph
from softgym.utils.visualization import save_numpy_as_gif
from GNS.models_graph_res import MaterialEncoder, Encoder, Processor, Decoder
import json
from GNS.data_graph import PhysicsFleXDataset, ClothDataset
import cv2


def create_env(env_name):
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import copy

    env_args = copy.deepcopy(env_arg_dict[args.env_name])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['action_repeat'] = 1
    # env_args['headless'] = False
    if args.env_name == 'ClothFlatten':
        env_args['cached_states_path'] = 'cloth_flatten_init_states_small_2.pkl'
        env_args['num_variations'] = 20
    env = SOFTGYM_ENVS[args.env_name](**env_args)

    return env

def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)

def visualize(env, particle_positions, shape_positions, config_id, sample_idx=None, picked_particles=None, show=False):
    env.reset(config_id=config_id)
    frames = []
    for i in range(len(particle_positions)):
        particle_pos = particle_positions[i]
        shape_pos = shape_positions[i]
        p = pyflex.get_positions().reshape(-1, 4)
        if sample_idx is None:
            p[:, :3] = particle_pos
        else:
            p[:, :3] = 0
            p[sample_idx, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        frames.append(env.get_image(720, 720))

        if show:
            if i == 0: continue
            picked_point = picked_particles[i]
            phases = np.zeros(pyflex.get_n_particles())
            # print(picked_point)
            for id in picked_point:
                if id != -1:
                    phases[sample_idx[int(id)]] = 1
            pyflex.set_phases(phases)
            img = env.get_image()

            cv2.imshow('pciked particle images', img[:, :, ::-1])
            cv2.waitKey()

    return frames

def get_model_prediction_rollout(args, data_path, rollout_idx, init_timestep, model, dataset, noise_scale=0):
    encoder_model, processor_model, decoder_model = model
    encoder_model.eval()
    processor_model.eval()
    decoder_model.eval()

    assert init_timestep == args.n_his - 1
   
    data_dir = osp.join(data_path, str(rollout_idx), '{}.h5'.format(init_timestep))
    data_names = ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'shape_positions']

    data = dataset._load_data_file(data_names, data_dir)

    gt_positions = [data[0]]
    predicted_positions = [data[0]]
    shape_positions = [data[-1]]
    config_id = int(data[-2][-1])
    all_picked_particles = [data[4]]

    # print("picked points: ", data[4])

    # Get initial velocity history
    vel_his = []
    for i in range(1, args.n_his):
        path = os.path.join(data_path, str(rollout_idx), str(max(0, init_timestep - i)) + '.h5') # max just in case
        data_his = dataset._load_data_file(data_names, path)
        vel_his.append(data_his[1])
        gt_positions.append(data[0])
        predicted_positions.append(data[0])
        shape_positions.append(data[-1])

    data[1] = np.concatenate([data[1]] + vel_his, 1)

    pos_errors = []
    vel_errors = []
    down_sample_idx = None
    gt_positions = gt_positions[::-1]
    predicted_positions = predicted_positions[::-1]
    shape_positions = shape_positions[::-1]
    
    for t in range(args.n_his, args.time_step - 1):
        # print("visualzie step {}".format(t))
        if t == args.n_his:
            node_attr, neighbors, edge_attr, global_feat, sample_idx, picked_particles, down_cloth_x, down_cloth_y, picked_status \
                 = dataset._prepare_input(data, test=True, noise_scale=noise_scale)
            down_sample_idx = sample_idx
            data[0] = data[0][down_sample_idx]
            data[1] = data[1][down_sample_idx]
            data[-2][1] = down_cloth_x # update cloth dim after downsample
            data[-2][2] = down_cloth_y
            for idx in range(len(gt_positions)):
                gt_positions[idx] = gt_positions[idx][down_sample_idx]
            for idx in range(len(predicted_positions)):
                predicted_positions[idx] = predicted_positions[idx][down_sample_idx]

        else:
            # after initial downsample, no need to do downsample again
            node_attr, neighbors, edge_attr, global_feat, _, picked_particles, _, _, picked_status \
                = dataset._prepare_input(data, test=True, downsample=False, noise_scale=noise_scale)
            # print("len of picked particles: ", len(picked_particles))
            # print('picked particles: ', picked_particles)

        # print("t: ", t, " picked particles in get model prediction: ", picked_particles)
        all_picked_particles.append(picked_particles.copy())

        node_attr = torch.squeeze(node_attr, dim=0).cuda()
        neighbors = torch.squeeze(neighbors, dim=0).cuda()
        edge_attr = torch.squeeze(edge_attr, dim=0).cuda()
        global_feat = torch.squeeze(global_feat, dim=0).cuda()

        # obtain model predictions
        with torch.no_grad():

            node_embedding, edge_embedding = encoder_model(node_attr, edge_attr)

            node_embedding_out, edge_embedding_out, global_out = processor_model(node_embedding, neighbors, edge_embedding, global_feat,
                                                                                    batch=None)
            pred_accel = decoder_model(node_embedding_out)

        if not args.predict_vel:
            pred_vel = data[1][:, :3] + pred_accel.cpu().numpy() * args.dt
        else:
            pred_vel = pred_accel.cpu().numpy()
        
        picked_vel, picked_pos = picked_status
        pred_pos = data[0] + pred_vel * args.dt

        # compute prediction error
        data_dir = osp.join(data_path, str(rollout_idx), '{}.h5'.format(t))
        new_data = dataset._load_data_file(data_names, data_dir)
        gt_pos = new_data[0][down_sample_idx]
        gt_vel = new_data[1][down_sample_idx]
        gt_positions.append(gt_pos)
        predicted_positions.append(pred_pos)
        shape_positions.append(new_data[-1])
        pos_error = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=1))
        vel_error = np.mean(np.linalg.norm(pred_vel - gt_vel, axis=1))
        pos_errors.append(pos_error)
        vel_errors.append(vel_error)

        # udpate position and velocity from the model prediction
        old_vel = data[1][:, :-3]
        data[0] = pred_pos
        data[1][:, 3:] = old_vel
        data[1][:, :3] = pred_vel

        # the picked particles position and velocity should not change
        cnt = 0
        for p_idx in picked_particles:
            if p_idx != -1:
                data[0][p_idx] = picked_pos[cnt]
                data[1][p_idx] = picked_vel[cnt]
                cnt += 1

        # update picker position and picker action, and the particles picked
        data[2] = new_data[2]
        data[3] = new_data[3]
        data[4] = picked_particles

    return pos_errors, vel_errors, gt_positions, predicted_positions, shape_positions, down_sample_idx, config_id, all_picked_particles

class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)


def vv_to_args(vv):
    args = VArgs(vv)
    return args

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

# args, data_path, rollout_idx, init_timestep, model, dataset
def main(data_folder, n_rollout, save_folder=None, model_dir=None):
    env_name = 'ClothFlatten'
    env = create_env(env_name)



    vv = json.load(open(osp.join(model_dir, 'variant.json'), 'r'))
    args = vv_to_args(vv)
    rollout_idx = [i  for i in range(n_rollout)]

    args.regularize_picked_vel = True

    encoder_model = Encoder(args.state_dim + args.attr_dim, args.relation_dim).cuda()
    processor_model = Processor([3 * 128 + 1, 2 * 128 + 1, 2 * 128 + 1], use_global=False).cuda()
    decoder_model = Decoder().cuda()

    encoder_model.load_state_dict(torch.load(osp.join(model_dir, 'enc_net_best.pth')))
    processor_model.load_state_dict(torch.load(osp.join(model_dir, 'proc_net_best.pth')))
    decoder_model.load_state_dict(torch.load(osp.join(model_dir, 'dec_net_best.pth')))

    phase = 'train'
    phase_dict = args.phases_dict
    datasets = {phase: ClothDataset(args, phase, phase_dict, env, args.verbose_data)
        for phase in ['train', 'valid']}

    occ = findOccurrences(model_dir, '/')
    save_prefix = model_dir[occ[-2]+1:-1]
    print("save_prefix is: ", save_prefix)

    for r_idx in rollout_idx:
        with torch.no_grad():
            pos_errors, vel_errors, gt_positions, predicted_positions, shape_positions, sample_idx, config_id, picked_points \
                = get_model_prediction_rollout(args, data_folder, r_idx, args.n_his - 1, 
                    [encoder_model, processor_model, decoder_model], datasets[phase], noise_scale=0)

            # print(picked_points)
            # exit()
            frames_model = visualize(datasets[phase].env, predicted_positions, 
                shape_positions, config_id, sample_idx, picked_particles=picked_points, show=False)
            frames_gt = visualize(datasets[phase].env, gt_positions, 
                shape_positions, config_id, sample_idx)
            combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
            save_path = osp.join(save_folder, save_prefix)
            # print(save_path)
            # exit()
            if not osp.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_numpy_as_gif(np.array(combined_frames), osp.join(save_path, 'visual_{}.gif'.format(r_idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--data_folder', type=str, default='datasets/ClothFlatten_small/train')
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
    parser.add_argument('--n_rollout', type=int, default=5)
    parser.add_argument('--save_folder', type=str, default='./datasets/visualizations')

    args = parser.parse_args()
    main(args.data_folder, args.n_rollout, args.save_folder, args.model_dir)
