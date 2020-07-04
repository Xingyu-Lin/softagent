import pyflex
import numpy as np


def test_pyflex_scene():
    pyflex.init(False, True, 720, 720)
    pyflex.set_scene(14, [20, 1, 20, 2,  # Dimension and num pieces
                          10, 1.,  # invMass, rigid
                          -0.0, 1.2, 1.2,
                          0, -45 / 180. * np.pi, 0.,
                          ], 0)
    phase = pyflex.get_phases()
    pos = pyflex.get_positions()
    stat = {}
    for x in phase:
        if x not in stat:
            stat[x] = 1
        else:
            stat[x] += 1

    for i in range(10000):
        pyflex.step(render=True)


def test_flex_env():
    from softgym.envs.rigid_cloth_fold import RigidClothFoldEnv
    env = RigidClothFoldEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_pieces=2,
        render=True,
        headless=False,
        action_repeat=8,
        use_cached_states=False,
        save_cache_states=True,
        num_variations=2
    )
    env.reset()
    for _ in range(100000):
        # pyflex.step(render=True)
        action = env.action_space.sample()
        action = np.zeros_like(action)
        env.step(action, record_continuous_video=True, img_size=720)


def visualize_cem():
    from softgym.envs.rigid_cloth_fold import RigidClothFoldEnv
    env = RigidClothFoldEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_pieces=2,
        render=True,
        headless=False,
        action_repeat=8,
        use_cached_states=False,
        save_cache_states=False,
        num_variations=1
    )
    env.reset()
    for _ in range(100000):
        # pyflex.step(render=True)
        action = env.action_space.sample()

        action = np.zeros_like(action)
        env.step(action, record_continuous_video=True, img_size=720)

    file_path = osp.join(args.exp_dir, 'cem_traj.pkl')
    with open(file_path, 'rb') as f:
        traj_dict = pickle.load(f)
    initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']


if __name__ == '__main__':
    # test_pyflex_scene()
    # import time
    # time.sleep(10)
    # test_flex_env()
    visualize_cem()
