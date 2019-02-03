import gym
env = gym.make("Acrobot-v1")


def acrobot():
    while True:
        env.reset()
        for t in range(200):
            env.render()
            a = env.action_space.sample()
            _, _, done, info = env.step(a)
            if done:
                break


acrobot()
