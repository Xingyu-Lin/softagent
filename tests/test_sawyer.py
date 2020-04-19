import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.pass_water import PassWater1DEnv
from softgym.utils.normalized_env import normalize


def render_sawyer_cloth():
    env = ClothFoldEnv(
        observation_mode='cam_rgb',
        action_mode='sawyer',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth',
        cached_states_path='cloth_fold_test.pkl',
        use_cached_states=False,
        save_cache_states=False,
        deterministic=True)
    config = {'ClothPos': [-0.31, -0.6, -0.184], 'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
              'ClothStiff': [0.8, 1, 0.9], 'camera_name': 'default_camera',
              'camera_params': {'default_camera': {'pos': np.array([0.0, 1.4, 0.6]),
                                                   'angle': np.array([0, -60 / 180. * np.pi, 0.]),
                                                   'width': 720,
                                                   'height': 720}}, 'env_idx': 14, 'mass': 0.3}

    camera_params = config['camera_params'][config['camera_name']]
    scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], 1,
                             *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], 0.5])
    pyflex.set_scene(14, scene_params, 0, [0.])
    pyflex.loop()


def render_sawyer_rope():
    pass


def render_sawyer_fluid():
    pass


if __name__ == '__main__':
    particle_radius = 0.00625
    generate_cache = True
    if generate_cache:
        # env = ClothFoldEnv(
        #     observation_mode='cam_rgb',
        #     action_mode='sawyer',
        #     num_picker=2,
        #     render=True,
        #     headless=False,
        #     horizon=75,
        #     action_repeat=8,
        #     render_mode='cloth',
        #     cached_states_path='cloth_fold_test.pkl',
        #     use_cached_states=False,
        #     save_cache_states=False,
        #     deterministic=True)
        # PassWater1DEnvNew(observation_mode='cam_rgb',
        #                   action_mode='direct',
        #                   render=True,
        #                   headless=False,
        #                   horizon=75,
        #                   action_repeat=8,
        #                   render_mode='fluid',
        #                   delta_reward=False,
        #                   deterministic=True,
        #                   num_variations=1,
        #                   cached_states_path='cloth_fold_test.pkl',
        #                   use_cached_states=False,
        #                   save_cache_states=False,
        #                   )
        config = {'ClothPos': [-0.31, -0.6, -0.184], 'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
                  'ClothStiff': [0.8, 1, 0.9], 'camera_name': 'default_camera',
                  'camera_params': {'default_camera': {'pos': np.array([0.0, 1.4, 0.6]),
                                                       'angle': np.array([0, -60 / 180. * np.pi, 0.]),
                                                       'width': 720,
                                                       'height': 720}}, 'env_idx': 14, 'mass': 0.3}

        camera_params = config['camera_params'][config['camera_name']]
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], 1,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], 0.5])
        pyflex.set_scene(14, scene_params, 0, [0.])
        # env.cached_configs, env.cached_init_states = env.generate_env_variation(1, vary_cloth_size=False, config=config, save_to_file=False)
        pyflex.loop()
        exit()
    else:
        env = ClothFoldEnv(
            observation_mode='cam_rgb',
            action_mode='pickerpickplace',
            num_picker=1,
            render=True,
            headless=False,
            horizon=75,
            action_repeat=8,
            render_mode='cloth',
            cached_states_path='cloth_fold_demo_init_states.pkl',
            use_cached_states=True,
            save_cache_states=False,
            deterministic=True,
            picker_radius=0.03,
            particle_radius=particle_radius)
    pickpts, placepts = get_robot_data()

    # env = normalize(env) TODO: No normalization as actions matter here. Be cautious when actually using the pick and place action space
    env.start_record()

    for i in range(pickpts.shape[0]):
        env.reset()
        env.action_tool.update_picker_boundary([-np.inf] * 3, [np.inf] * 3)
        # actions = generate_pick_and_place(0, 0, 0, 0, 0.2, 0.2 + 0.1065

        actions = generate_pick_and_place(pickpts[i, 0], pickpts[i, 1] - 0.06, placepts[i, 0], placepts[i, 1] - 0.06, 0.2, 0.2 + 0.1065)

        for action in actions:
            print('action:', action)
            env.step(action)
        for j in range(100):
            pyflex.step()
            if env.recording:
                env.video_frames.append(env.render(mode='rgb_array'))
        break
    env.end_record(video_path='./experiments/realism/demo.gif')