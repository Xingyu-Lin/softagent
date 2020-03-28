from ResRL.envs.box1d import Box1d

if __name__ == '__main__':
    env = Box1d(image_observation=True)
    while (1):
        env.reset()
        while True:
            action = env.action_space.sample()
            obs, _, done, info = env.step(action)
            print(obs[:, env.image_dim // 2, 0])
            env.render()
            # exit()
            if done:
                break
