import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.utils.visualization import save_numpy_as_gif
# from softgym.utils.visualization import save_numpy_to_gif_matplotlib
import torch, torchvision, cv2, time


def test_picker(num_picker=3, save_dir='./videos', script='manual'):
    env = ClothFlattenEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=1,
        num_variations=1000,
        deterministic=False,
        delta_reward=False,
        render_mode='cloth')

    imgs = []
    for _ in range(2):
        env.reset()
        first_pos = pyflex.get_positions()[:3]
        last_pos = pyflex.get_positions()[-4:-1]

        print(first_pos)
        print(last_pos)

        picker_pos, _ = env.action_tool._get_pos()
        print(picker_pos[0])
        print(picker_pos[1])
        diff_first = first_pos - picker_pos[0]
        diff_last = last_pos - picker_pos[1]
        move_step = 20
        for i in range(move_step):
            action = np.zeros((num_picker, 4))
            # env.step(action)
            action[0, :3] = diff_first / move_step
            action[1, :3] = diff_last / move_step
            _, reward, _, _ = env.step(action)
            # print("reward: ", reward)
            img = env.render(mode='rgb_array')
            imgs.append(img)

        cloth_size_x, cloth_size_z = env.get_current_config()['ClothSize'][0], env.get_current_config()['ClothSize'][1]
        first_pos = pyflex.get_positions()[:3]
        first_target = first_pos  # np.array([-cloth_size_x / 0.7 * 0.05, 0.05, 0])
        last_target = np.array([cloth_size_x * 1.7 * 0.05, 0.05, cloth_size_z / 2 * 0.05])

        first_target = np.array([cloth_size_x * 1.3 * 0.05, 0.05, -cloth_size_z / 2 * 0.01])
        last_target = pyflex.get_positions()[-4:-1] + np.array([-0.1, 0, -0.1])
        picker_pos, _ = env.action_tool._get_pos()
        diff_first = first_target - picker_pos[0]
        diff_last = last_target - picker_pos[1]

        print("-" * 50, 'move to target done', '-' * 50)
        print(picker_pos[0])
        print(picker_pos[1])

        time.sleep(1)
        move_step = 100
        for i in range(move_step):
            action = np.ones((num_picker, 4))
            action[0, :3] = diff_first / move_step
            action[1, :3] = diff_last / move_step
            _, reward, _, _ = env.step(action)
            print("reward: ", reward)

            img = env.render(mode='rgb_array')
            imgs.append(img)

        # for i in range(50):
        #     print('step: ', i)
        #     action = np.zeros((num_picker, 4))
        #     if i < 12:
        #         action[:, 1] = -0.01
        #         action[:, 3] = 0
        #     elif i < 30:
        #         action[:, 1] = 0.01
        #         action[:, 3] = 1
        #     elif i < 40:
        #         action[:, 3] = 0
        #     if script == 'random':
        #         action = env.action_space.sample()
        #     env.step(action)
        #     img = env.render(mode='rgb_array')
        #     imgs.append(img)

    # fp_out = './videos/flatten_picker_random_{}_0075'.format(num_picker)
    # save_numpy_as_gif(np.asarray(imgs), fp_out)
    # exit()

    num = 8
    show_imgs = []
    factor = len(imgs) // num
    for i in range(num):
        img = imgs[i * factor].transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs = grid_imgs[:, :, ::-1]
    cv2.imwrite('cloth_flatten.jpg', grid_imgs)


def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    test_picker(num_picker=2, script='manual')
