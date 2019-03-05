import gym
env = gym.make("MontezumaRevenge-v0")


def monte_revenge():
    while True:
        env.reset()
        for t in range(200):
            env.render()
            a = env.action_space.sample()
            _, _, done, _ = env.step(a)
            if done:
                break


monte_revenge()
