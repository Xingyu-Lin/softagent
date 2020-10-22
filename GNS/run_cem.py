import numpy as np
import torch
from GNS.cem import CEMPlanner, CEMPickandPlacePlanner
import torchvision
from softgym.utils.visualization import save_numpy_as_gif
from chester import logger
import json
import os.path as osp
import random
import os
from GNS.models_graph_res import MaterialEncoder, Encoder, Processor, Decoder
from GNS.data_graph import PhysicsFleXDataset, ClothDataset
from softgym.registered_env import env_arg_dict
from softgym.registered_env import SOFTGYM_ENVS
import copy
import pyflex
from planet.utils import transform_info
import pickle
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
import multiprocessing as mp
from experiments.planet.train import update_env_kwargs
from GNS.visualize_data import visualize

# from torch.multiprocessing import Pool, set_start_method

def reward_model(pos, cloth_particle_radius=0.00625):
    pos = np.reshape(pos, [-1, 3])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    # just hard code here. TODO: change this according to different cloth size
    # filter out those that exploits the model error
    max_possible_span = (40 * 0.00625) ** 2
    res = np.sum(grid) * span[0] * span[1]
    if res > max_possible_span:
        res = 0
    if np.any(pos[:, 1] > 0.00625 * 15):
        res = 0

    return res

class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)

def vv_to_args(vv):
    args = VArgs(vv)
    return args

def prepare_policy(env, mode):
    print("preparing policy! ")

    # move one of the picker to be under ground
    shape_states = pyflex.get_shape_states().reshape(-1, 14)
    shape_states[1, :3] = -1
    shape_states[1, 7:10] = -1

    if mode == 'low-level':
        # move another picker to a randomly chosen particle
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., env.picker_radius, 0.]
        shape_states[0, 7:10] = pos[pp] + [0., env.picker_radius, 0.]
        pyflex.set_shape_states(shape_states.flatten())
    else:
        # move another picker to be above the cloth
        pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pp = np.random.randint(len(pos))
        shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
        shape_states[0, 7:10] = pos[pp] + [0., 0.06, 0.]
        pyflex.set_shape_states(shape_states.flatten())


def downsample(cloth_xdim, cloth_ydim, scale):
    cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
    original_xdim, original_ydim = cloth_xdim, cloth_ydim
    new_idx = np.arange(cloth_xdim * cloth_ydim).reshape((cloth_ydim, cloth_xdim))
    new_idx = new_idx[::scale, ::scale]
    cloth_ydim, cloth_xdim = new_idx.shape
    new_idx = new_idx.flatten()

    return new_idx, cloth_xdim, cloth_ydim

def cem_make_gif(all_frames, save_dir, save_name):
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=5).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


def run_task(vv, log_dir, exp_name):
    mp.set_start_method('forkserver', force=True)

    vv['candidates'] = vv['timestep_per_decision'] // vv['optimisation_iters']
    vv['candidates'] = vv['candidates'] // vv['planning_horizon']
    vv['top_candidates'] = vv['candidates'] // 10

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    seed = vv['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)


    model_vv = json.load(open(osp.join(vv['model_dir'], 'variant.json')))
    args = vv_to_args(model_vv)
    encoder_model = Encoder(args.state_dim + args.attr_dim, args.relation_dim).cuda()
    processor_model = Processor([3 * 128 + 1, 2 * 128 + 1, 2 * 128 + 1], use_global=False).cuda()
    decoder_model = Decoder().cuda()

    encoder_model.load_state_dict(torch.load(osp.join(vv['model_dir'], "enc_{}".format(vv['model_name']))))
    processor_model.load_state_dict(torch.load(osp.join(vv['model_dir'], "proc_{}".format(vv['model_name']))))
    decoder_model.load_state_dict(torch.load(osp.join(vv['model_dir'], "dec_{}".format(vv['model_name']))))
    transition_model = [encoder_model, processor_model, decoder_model]

    env_args = copy.deepcopy(env_arg_dict[vv['env_name']])
    env_args['render_mode'] = 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['observation_mode'] = 'key_point'
    env_args['render'] = True
    # env_args['headless'] = False
    env_args['action_repeat'] = vv['action_repeat']
    env_args['picker_radius'] = 0.01
    if vv['env_name'] == 'ClothFlatten':
        env_args['cached_states_path'] = 'cloth_flatten_init_states_small_2.pkl'
        env_args['num_variations'] = 20
    env = SOFTGYM_ENVS[vv['env_name']](**env_args)
    # visual_env = SOFTGYM_ENVS[vv['env_name']](**env_args)

    phase = 'train'
    dataset = ClothDataset(args, phase, env=env)

    particle_radius = 0.00625
    if vv['mode'] == 'low-level':
        cem_policy = CEMPlanner(vv['action_size'], vv['planning_horizon'], vv['optimisation_iters'], 
            vv['candidates'], vv['top_candidates'], transition_model, 
            reward_model,
            np.array(vv['action_low']), np.array(vv['action_high']), device, vv['num_worker'])
    else:
        cem_policy = CEMPickandPlacePlanner(vv['cem_num_pick'], vv['delta_y'], vv['move_distance'], 
            vv['cem_stage_1_step'], vv['cem_stage_2_step'], vv['cem_stage_3_step'],
            transition_model, reward_model, num_worker=vv['num_worker'], env=env)


    initial_states, action_trajs, configs, all_infos = [], [], [], []
    for episode_idx in range(vv['test_episodes']):
        # setup environment
        env.reset()
        # move one picker below the ground, set another picker randomly to a picked point
        prepare_policy(env, vv['mode']) 

        config = env.get_current_config()
        cloth_xdim, cloth_ydim = config['ClothSize']
        config_id = env.current_config_id
        scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

        # prepare environment and do downsample
        downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, args.down_sample_scale)
        scene_params[1] = downsample_x_dim
        scene_params[2] = downsample_y_dim

        # prepare gnn input
        positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        positions = positions.astype(np.float32)[downsample_idx]
        vel_history = np.zeros((len(downsample_idx), args.n_his * 3), dtype=np.float32)
        vel_history[:, :3] = pyflex.get_velocities().reshape(-1, 3)[downsample_idx]
        picker_position = env.action_tool._get_pos()[0]
        picked_points = [-1, -1]
        data = [positions, vel_history, picker_position, env.action_space.sample(), picked_points, scene_params]

        initial_state = env.get_state()
        initial_states.append(initial_state)
        configs.append(config)

        ret = 0
        action_traj = []
        infos = []
        frames = []

        # for debug 
        cem_policy.downsample_idx = downsample_idx

        if vv['mode'] == 'low-level':
            for t in range(env.horizon):
                # use cem to plan an action
                action, picked_points = cem_policy.get_action(args, data, dataset)
                obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=720)

                print("step {} plan done".format(t))

                frames.extend(info['flex_env_recorded_frames'])
                info.pop("flex_env_recorded_frames")

                # update the data used for cem
                positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
                positions = positions.astype(np.float32)[downsample_idx]
                vel = pyflex.get_velocities().reshape(-1, 3)[downsample_idx]
                vel_old = vel_history[:, :-3]
                vel_history[:, :3] = vel
                vel_history[:, 3:] = vel_old
                picker_position = env.action_tool._get_pos()[0]
                data = [positions, vel_history, picker_position, env.action_space.sample(), picked_points, scene_params]

                ret += reward
                infos.append(info)
                action_traj.append(action)

        elif vv['mode'] == 'pick-and-place':
            cem_pick_and_place_steps = vv['cem_stage_1_step'] + vv['cem_stage_2_step'] + vv['cem_stage_3_step']
            
            gt_positions = np.zeros((vv['pick_and_place_num'], cem_pick_and_place_steps, len(downsample_idx), 3))
            gt_shape_positions = np.zeros((vv['pick_and_place_num'], cem_pick_and_place_steps, 2, 3))
            model_pred_particle_poses = np.zeros((vv['pick_and_place_num'], cem_pick_and_place_steps, len(downsample_idx), 3))

            for pick_try_idx in range(vv['pick_and_place_num']):
                action_sequence, model_pred_particle_pos, _ = cem_policy.get_action(args, data, dataset)
                model_pred_particle_poses[pick_try_idx] = model_pred_particle_pos
                print("pick_and_place idx: ", pick_try_idx)   
                gt_vel = np.zeros((cem_pick_and_place_steps, len(downsample_idx), 3))

                for t_idx, ac in enumerate(action_sequence):
                    obs, reward, done, info = env.step(ac, record_continuous_video=True, img_size=720)

                    frames.extend(info['flex_env_recorded_frames'])
                    info.pop("flex_env_recorded_frames")

                    ret += reward
                    infos.append(info)
                    action_traj.append(ac)

                    gt_positions[pick_try_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[downsample_idx, :3]
                    gt_vel[t_idx] = pyflex.get_velocities().reshape(-1, 3)[downsample_idx, :3]
                    shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                    for k in range(2):
                        gt_shape_positions[pick_try_idx][t_idx][k] = shape_pos[k][:3]
                
                # update the data used for cem
                positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
                positions = positions.astype(np.float32)[downsample_idx]
                vel_history = np.zeros((len(downsample_idx), args.n_his * 3), dtype=np.float32)
                for i in range(args.n_his):
                    vel_history[:, i*3:(i+1)*3] = gt_vel[-i]
                picker_position = env.action_tool._get_pos()[0]
                data = [positions, vel_history, picker_position, env.action_space.sample(), picked_points, scene_params]

            for pick_try_idx in range(vv['pick_and_place_num']):
                frames_model = visualize(env, model_pred_particle_poses[pick_try_idx], 
                                gt_shape_positions[pick_try_idx], config_id, downsample_idx)
                frames_gt = visualize(env, gt_positions[pick_try_idx], 
                    gt_shape_positions[pick_try_idx], config_id, downsample_idx)
                combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in zip(frames_gt, frames_model)]
                save_numpy_as_gif(np.array(combined_frames), osp.join(logdir, '{}-{}.gif'.format(
                    episode_idx, pick_try_idx
                )))
        
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

        cem_make_gif([frames], logger.get_dir(), 
            vv['env_name'] + '{}.gif'.format(episode_idx))

        action_trajs.append(action_traj)
        all_infos.append(infos)

    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)

    


    

    

    

    