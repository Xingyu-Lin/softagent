import gym
import softgym
from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.utils.normalized_env import normalize
if __name__ == '__main__':

    env_kwargs = {'observation_mode': 'cam_img',
                  'action_mode': 'direct',
                  'render_mode': 'fluid',
                  'deterministic': True,
                  'render': True,
                  'headless': False,
                  'horizon': 75,
                  'camera_name': 'cam_2d'}

    env = PourWaterPosControlEnv(**env_kwargs)
    env = normalize(env)

    for _ in range(env.horizon):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs.shape, reward)
