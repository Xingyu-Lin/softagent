from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.utils.normalized_env import normalize
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    env_kwargs = {'observation_mode': 'cam_rgb',
                  'action_mode': 'direct',
                  'render_mode': 'fluid',
                  'deterministic': False,
                  'render': True,
                  'headless': True,
                  'horizon': 75,
                  'camera_name': 'default_camera'}

    env = PourWaterPosControlEnv(**env_kwargs)
    env = normalize(env)
    for i in range(10):
        env.reset()
        for _ in range(10000):
            action = env.action_space.sample()
            action = np.zeros_like(action)
            obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=720)
            img = env.get_image(720, 720)
            plt.imshow(img)
            plt.show()
            # print(obs.shape, reward)
