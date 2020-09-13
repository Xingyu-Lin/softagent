import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
import cv2
from matplotlib import pyplot as plt
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from cloth_manipulation.gemo_utils import *

def render_sawyer_cloth():
    env_kwargs = env_arg_dict['ClothFold']

    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = 1
    env_kwargs['render'] = True
    env_kwargs['headless'] = False
    env_kwargs['camera_width'] = 300
    env_kwargs['camera_height'] = 300
    env_kwargs['render_mode'] = 'both'
    # env_kwargs['action_mode'] = 'sawyer'
    height, width = env_kwargs['camera_height'], env_kwargs['camera_width']

    # env = normalize(SOFTGYM_ENVS['ClothFold'](**env_kwargs))
    env = ClothFoldEnv(**env_kwargs)
    env.reset()
    action_repeat = env_kwargs['action_repeat']
    print('action repeat: ', env_kwargs['action_repeat'])

    pos = pyflex.get_positions().reshape((-1, 4))
    minx = np.min(pos[:, 0])
    maxx = np.max(pos[:, 0])
    minz = np.min(pos[:, 2])
    maxz = np.max(pos[:, 2])

    corner1 = np.array([minx, 0.05, minz])
    corner2 = np.array([minx, 0.05, maxz])

    picker_pos, _ = env.action_tool._get_pos()

    differ1 = corner1 - picker_pos[0]
    differ2 = corner2 - picker_pos[1]

    steps = 10 
    for i in range(steps):
        action = np.zeros((2, 4))
        action[0, :3] = differ1 / steps / env_kwargs['action_repeat']
        action[1, :3] = differ2 / steps / env_kwargs['action_repeat']
        action = action.flatten()

        obs, _, _, _ = env.step(action)

        # rgbd = pyflex.render_sensor()
        # rgbd = np.array(rgbd).reshape(env_kwargs['camera_height'], env_kwargs['camera_width'], 4)
        # rgbd = rgbd[::-1, :, :]
        # rgb = rgbd[:, :, :3]
        # depth = rgbd[:, :, 3]

        # world_coordinates = get_world_coords(rgb, depth, env)
        # particle_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        # observable_particle_indices = get_observable_particle_index(world_coordinates, particle_pos, rgb, depth)
        
        # print(len(np.unique(observable_particle_indices)))
        # phases = np.zeros(pyflex.get_n_particles())
        # phases[observable_particle_indices] = 1
        # pyflex.set_phases(phases)
        # img = env.get_image()
        # cv2.imshow('observable images', img[:, :, ::-1])
        # cv2.waitKey()

    picker_pos, _ = env.action_tool._get_pos()
    steps = 6
    for i in range(steps):
        action = np.zeros((2, 4))
        action[:, -1] = 1
        action[:, 1] = 0.004
        _, _, _, _ = env.step(action)

        # rgbd = pyflex.render_sensor()
        # rgbd = np.array(rgbd).reshape(env_kwargs['camera_height'], env_kwargs['camera_width'], 4)
        # rgbd = rgbd[::-1, :, :]
        # rgb = rgbd[:, :, :3]
        # depth = rgbd[:, :, 3]

        # world_coordinates = get_world_coords(rgb, depth, env)
        # particle_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        # observable_particle_indices = get_observable_particle_index(world_coordinates, particle_pos, rgb, depth)
        
        # # print(observable_particle_indices)
        # print(len(np.unique(observable_particle_indices)))
        # phases = np.zeros(pyflex.get_n_particles())
        # phases[observable_particle_indices] = 1
        # pyflex.set_phases(phases)
        # img = env.get_image()
        # cv2.imshow('observable images', img[:, :, ::-1])
        # cv2.waitKey()

    pos = pyflex.get_positions().reshape((-1, 4))
    minx = np.min(pos[:, 0])
    maxx = np.max(pos[:, 0])
    minz = np.min(pos[:, 2])
    maxz = np.max(pos[:, 2])
    target_corner_1 = np.array([maxx, 0.10, minz])
    target_corner_2 = np.array([maxx, 0.10, maxz])
    picker_pos, _ = env.action_tool._get_pos()

    differ1 = target_corner_1 - picker_pos[0]
    differ2 = target_corner_2 - picker_pos[1]

    steps = 20
    for i in range(steps):
        action = np.ones((2, 4))
        action[0, :3] = differ1 / steps / action_repeat
        action[1, :3] = differ2 / steps / action_repeat
        _, _, _, _ = env.step(action)

        rgbd = pyflex.render_sensor()
        rgbd = np.array(rgbd).reshape(env_kwargs['camera_height'], env_kwargs['camera_width'], 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

     

        if i > 10:
            world_coordinates = get_world_coords(rgb, depth, env)
            particle_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
            observable_particle_indices = get_observable_particle_index(world_coordinates, particle_pos, rgb, depth)
            
            # print(observable_particle_indices)
            print(len(np.unique(observable_particle_indices)))
            phases = np.zeros(pyflex.get_n_particles())
            phases[observable_particle_indices] = 1
            pyflex.set_phases(phases)
            original_img = env.get_image()
            # cv2.imshow('observable images', img[:, :, ::-1])
            # cv2.waitKey()
            env.update_camera('cam_2d', {'pos': np.array([0.2, .1, 0.6]),
                       'angle': np.array([0.35, 0, 0.]),
                       'width': env.camera_width,
                       'height': env.camera_height})

            phases = np.zeros(pyflex.get_n_particles())
            phases[observable_particle_indices] = 1
            pyflex.set_phases(phases)
            img = env.get_image()

            cv2.imshow('observable images', img[:, :, ::-1])
            cv2.imshow('camera view', original_img[:, :, ::-1])
            cv2.waitKey()
            input("wait enter")

    steps = 20 
    for i in range(steps):
        action = np.zeros((2, 4))
        _, reward, _, _ = env.step(action)
        final = np.sort(np.unique(observable_particle_indices))
        for x in final:
            print(x)
            input("wait")
        phases = np.zeros(pyflex.get_n_particles())
        phases[observable_particle_indices] = 1
        pyflex.set_phases(phases)
        img = env.get_image()
        cv2.imshow('observable images', img[:, :, ::-1])
        cv2.waitKey()

        
    # for i in range(1000):
    #     action = env.action_space.sample()
    #     _, _, _, info = env.step(action)
        

        # exit()



if __name__ == '__main__':
    render_sawyer_cloth()