from rlpyt.envs.dm_control_env import DMControlEnv
import cv2 as cv


def cv_render(img, name='GoalEnvExt', scale=1):
    '''Take an image in ndarray format and show it with opencv. '''
    img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)] / 256.
    h, w = new_img.shape[:2]
    new_img = cv.resize(new_img, (w * scale, h * scale))
    cv.imshow(name, new_img)
    cv.waitKey(20)


if __name__ == '__main__':
    kwargs = {'domain': 'cloth_point', 'task': 'easy', 'max_path_length': 120,
              'pixel_wrapper_kwargs': {'observation_key': 'pixels', 'pixels_only': False,
                                       'render_kwargs': {'width': 64, 'height': 64, 'camera_id': 0}},
              'task_kwargs': {'random_location': True, 'pixels_only': True, 'train_mode': True}}

    env = DMControlEnv(**kwargs)
    o = env.reset()
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
        img = env.render()
        cv_render(img)
