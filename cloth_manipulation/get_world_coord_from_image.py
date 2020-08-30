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
    env_kwargs['camera_width'] = 200
    env_kwargs['camera_height'] = 200
    # env_kwargs['action_mode'] = 'sawyer'

    env = normalize(SOFTGYM_ENVS['ClothFold'](**env_kwargs))
    env.reset()

    for i in range(1000):
        # action = env.action_space.sample()
        # _, _, _, info = env.step(action)
        rgbd = pyflex.render_sensor()

        rgbd = np.array(rgbd).reshape(env_kwargs['camera_height'], env_kwargs['camera_width'], 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        height, width, _ = rgb.shape
        K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees

        # Apply back-projection: K_inv @ pixels * depth
        cam_coords = np.ones((height, width, 4))
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]
        # Loop through each pixel in the image
        for v in range(height):
            for u in range(width):
                # Apply equation in fig 3
                x = (u - u0) * depth[v, u] / fx
                y = (v - v0) * depth[v, u] / fy
                z = depth[v, u]
                cam_coords[v][u][:3] = (x, y, z)

        particle_pos = pyflex.get_positions().reshape((-1, 4))
        print('cloth pixels: ', np.count_nonzero(depth))
        print("cloth particle num: ", pyflex.get_n_particles())

        # debug: print camera coordinates
        # print(cam_coords.shape)
        # cnt = 0
        # for v in range(height):
        #     for u in range(width):
        #         if depth[v][u] > 0:
        #             print("v: {} u: {} cnt: {} cam_coord: {} approximate particle pos: {}".format(
        #                     v, u, cnt, cam_coords[v][u], particle_pos[cnt]))
        #             rgb = rgbd[:, :, :3].copy()
        #             rgb[v][u][0] = 255
        #             rgb[v][u][1] = 0
        #             rgb[v][u][2] = 0
        #             cv2.imshow('rgb', rgb[:, :, ::-1])
        #             cv2.waitKey()
        #             cnt += 1

        # from cam coord to world coord
        cam_x, cam_y, cam_z = env.camera_params['default_camera']['pos'][0], env.camera_params['default_camera']['pos'][1], env.camera_params['default_camera']['pos'][2]
        cam_x_angle, cam_y_angle, cam_z_angle = env.camera_params['default_camera']['angle'][0], env.camera_params['default_camera']['angle'][1], env.camera_params['default_camera']['angle'][2]

        # get rotation matrix: from world to camera
        matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0]) 
        # matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [np.cos(cam_x_angle), 0, np.sin(cam_x_angle)])
        matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
        rotation_matrix = matrix2 @ matrix1
        
        # get translation matrix: from world to camera
        translation_matrix = np.zeros((4, 4))
        translation_matrix[0][0] = 1
        translation_matrix[1][1] = 1
        translation_matrix[2][2] = 1
        translation_matrix[3][3] = 1
        translation_matrix[0][3] = - cam_x
        translation_matrix[1][3] = - cam_y
        translation_matrix[2][3] = - cam_z

        # debug: from world to camera
        cloth_x, cloth_y = env.current_config['ClothSize'][0], env.current_config['ClothSize'][1]
        # cnt = 0
        # for u in range(height):
        #     for v in range(width):
        #         if depth[u][v] > 0:
        #             world_coord = np.ones(4)
        #             world_coord[:3] = particle_pos[cnt][:3]
        #             convert_cam_coord =  rotation_matrix @ translation_matrix @ world_coord
        #             # convert_cam_coord =  translation_matrix  @ matrix2 @ matrix1 @ world_coord
        #             print("u {} v {} \n world coord {} \n convert camera coord {} \n real camera coord {}".format(
        #                 u, v, world_coord, convert_cam_coord, cam_coords[u][v]
        #             ))
        #             cnt += 1
        #             input('wait...')


        # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
        # cam_coords[:, :, 1] *= -1
        # cam_coords[:, :, 2] *= -1
        cam_coords = cam_coords.reshape((-1, 4)).transpose() # 4 x (height x width)
        world_coords = np.linalg.inv(rotation_matrix @ translation_matrix) @ cam_coords # 4 x (height x width)
        world_coords = world_coords.transpose().reshape((height, width, 4))
 
        # roughly check the final world coordinate with the actual coordinate
        # firstu = 0
        # firstv = 0
        # for u in range(height):
        #     for v in range(width):
        #         if depth[u][v]:
        #             if u > firstu: # move to a new line
        #                 firstu = u
        #                 firstv = v

        #             cnt = (u - firstu) * cloth_x + (v - firstv)  
        #             print("u {} v {} cnt{}\nworld_coord\t{}\nparticle coord\t{}\nerror\t{}".format(
        #                 u, v, cnt, world_coords[u][v], particle_pos[cnt], np.linalg.norm( world_coords[u][v] - particle_pos[cnt])))
        #             rgb = rgbd[:, :, :3].copy()
        #             rgb[u][v][0] = 255
        #             rgb[u][v][1] = 0
        #             rgb[u][v][2] = 0
        #             cv2.imshow('rgb', rgb[:, :, ::-1])
        #             cv2.waitKey()
        # exit()

        # perform the matching of pixel particle to real particle
        particle_pos = particle_pos[:, :3]
        for u in range(height):
            for v in range(width):
                if depth[u][v] > 0:
                    estimated_world_coord = world_coords[u][v][:3]
                    distance = np.linalg.norm(estimated_world_coord - particle_pos, axis=1)
                    estimated_particle_idx = np.argmin(distance)
                    print("u {} v {} estimated particle idx {}".format(u, v, estimated_particle_idx))
                    rgb = rgbd[:, :, :3].copy()
                    rgb[u][v][0] = 255
                    rgb[u][v][1] = 0
                    rgb[u][v][2] = 0
                    cv2.imshow('chosen_idx', rgb[:, :, ::-1])
                    cv2.waitKey()
        # exit()
                



if __name__ == '__main__':
    render_sawyer_cloth()