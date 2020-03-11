from ResRL.envs.box1d import Box1d

if __name__ == '__main__':
    env = Box1d()
    while (1):
        env.reset()
        while True:
            action = env.action_space.sample()
            _, _, done, info = env.step(action)
            env.render()
            if done:
                break
