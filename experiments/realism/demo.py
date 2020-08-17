import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
import cv2, os

def generate_pick_and_place(sx, sy, ex, ey, zl, zh):
    a0 = [sx, zh, sy, 0]
    a1 = [sx, zl, sy, 0]
    a2 = [sx, zh, sy, 1]
    a3 = [ex, zh, ey, 1]
    a4 = [ex, zh, ey, 0]
    actions = np.array([a0, a1, a2, a3, a4])
    return actions

def write_video(frames, title, path=''):
    frames = np.stack(frames, axis=0).astype(np.uint8)[:, :, :,
             ::-1]  # VideoWrite expects H x W x C in BGR

    _, H, W, _ = frames.shape
    save_path = os.path.join(path, '%s.mp4' % title)
    
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 40., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()
    print('saved to: ', save_path)
    

def get_robot_data():
    pickpts, placepts = np.load('experiments/realism/pickpts.npy'), np.load('experiments/realism/placepts.npy')
    centerx = 0.525
    centery = -0.02

    pickpts[:, 0] -= centerx
    placepts[:, 0] -= centerx
    pickpts[:, 1] -= centery
    placepts[:, 1] -= centery
    sim_pickpts = np.zeros_like(pickpts)
    sim_placepts = np.zeros_like(placepts)
    sim_pickpts[:, 0] = pickpts[:, 1]
    sim_placepts[:, 0] = placepts[:, 1]
    sim_pickpts[:, 1] = pickpts[:, 0]
    sim_placepts[:, 1] = placepts[:, 0]
    # Transform the coordinates
    return sim_pickpts, sim_placepts


if __name__ == '__main__':
    particle_radius = 0.00625
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
            save_cached_states=False,
            deterministic=True)
        config = {'ClothPos': [-0.31, -0.2, -0.184], 'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
                  'ClothStiff': [0.8, 1, 0.9], 'camera_name': 'default_camera',
                  'camera_params': {'default_camera': {'pos': np.array([-0.06, 0.9, 0.75]),
                                                       'angle': np.array([0, -45 / 180. * np.pi, 0.]),
                                                       'width': 720,
                                                       'height': 720}}, 'env_idx': 14, 'mass': 0.3}
        env.cached_configs, env.cached_init_states = env.generate_env_variation(1, vary_cloth_size=False, config=config, save_to_file=True)
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
            save_cached_states=False,
            deterministic=True,
            picker_radius=0.03,
            particle_radius=particle_radius)
    pickpts, placepts = get_robot_data()

    # env = normalize(env) TODO: No normalization as actions matter here. Be cautious when actually using the pick and place action space

    for i in range(8, pickpts.shape[0]):
        print(i)
        env.reset()
        env.action_tool.update_picker_boundary([-np.inf] * 3, [np.inf] * 3)
        # actions = generate_pick_and_place(0, 0, 0, 0, 0.2, 0.2 + 0.1065

        actions = generate_pick_and_place(pickpts[i, 0], pickpts[i, 1] - 0.06, placepts[i, 0] , placepts[i, 1]-0.06, 0.2, 0.2 + 0.1065)

        env.step(actions[0])
        env.start_record()
        for j in range(30):
            pyflex.step()
            if env.recording:
                env.video_frames.append(env.render(mode='rgb_array'))
        for action in actions[1:]:
            print('action:', action)
            env.step(action)
        for j in range(5):
            pyflex.step()
            if env.recording:
                env.video_frames.append(env.render(mode='rgb_array'))
        
        # print(len(env.video_frames))
        # print(env.video_frames[0].shape)
        # exit()
        write_video(env.video_frames, 'demo_{}'.format(i), 'data/realism_demo')
        env.end_record(video_path='./experiments/realism/demo_{}.gif'.format(i))
        # break

    # env.end_record(video_path='./experiments/realism/demo.gif')
