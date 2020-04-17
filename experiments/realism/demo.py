import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.utils.normalized_env import normalize


def generate_pick_and_place(sx, sy, ex, ey, zl, zh):
    a1 = [sx, zl, sy, 0]
    a2 = [sx, zh, sy, 1]
    a3 = [ex, zh, ey, 1]
    a4 = [ex, zh, ey, 0]
    actions = np.array([a1, a2, a3, a4])
    return actions


def get_robot_data():
    pickpts, placepts = np.load('experiments/realism/pickpts.npy'), np.load('experiments/realism/placepts.npy')
    centerx = 0.525
    centery = -0.02
    pickpts[:, 0] -= centerx
    placepts[:, 0] -= centerx
    pickpts[:, 1] -= centery
    placepts[:, 1] -= centery
    return pickpts, placepts


if __name__ == '__main__':
    generate_cache = False
    if generate_cache:
        env = ClothFoldEnv(
            observation_mode='cam_rgb',
            action_mode='picker',
            num_picker=2,
            render=True,
            headless=False,
            horizon=75,
            action_repeat=8,
            render_mode='cloth',
            cached_states_path='cloth_fold_demo_init_states.pkl',
            use_cached_states=False,
            save_cache_states=False,
            deterministic=True)
        config = {'ClothPos': [-0.35, -0.17, -0.184], 'ClothSize': [115, 60], 'ClothStiff': [0.8, 0.25, 0.25], 'camera_name': 'default_camera',
                  'camera_params': {'default_camera': {'pos': np.array([0.0, 1.4, 0.6]),
                                                       'angle': np.array([0, -60 / 180. * np.pi, 0.]),
                                                       'width': 720,
                                                       'height': 720}}, 'env_idx': 14,
                  'BlockSize': [0.05, 0.05, 0.05], 'BlockPos': [-0.35, 0.2, 0.2]}
        env.cached_configs, env.cached_init_states = env.generate_env_variation(1, vary_cloth_size=False, config=config, save_to_file=True)
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
            particle_radius=0.00625)
    pickpts, placepts = get_robot_data()

    # env = normalize(env) TODO: No normalization as actions matter here. Be cautious when actually using the pick and place action space
    env.start_record()

    for i in range(pickpts.shape[0]):
        env.reset()
        env.action_tool.update_picker_boundary([-np.inf] * 3, [np.inf] * 3)
        if i ==0:
            actions = generate_pick_and_place(-0.25, 0.15, 0.2, -0.1, 0.2, 0.4)
        else:
            actions = generate_pick_and_place(-pickpts[i, 0], -pickpts[i, 1], -placepts[i, 0], -placepts[i, 1], 0.2, 0.2 + 0.1065
                                              )
        for action in actions:
            print('action:', action)
            env.step(action)
        for j in range(100):
            pyflex.step()
            if env.recording:
                env.video_frames.append(env.render(mode='rgb_array'))
        if i == 2:
            break
    env.end_record(video_path='./experiments/realism/demo.gif')
