import gym
import softgym
if __name__ == '__main__':
    softgym.register_flex_envs()
    env = gym.make('PourWaterPosControl-v0')
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs.shape, reward)
    exit()

