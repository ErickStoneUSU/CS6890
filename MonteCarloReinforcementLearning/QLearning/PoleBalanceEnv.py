import gym
env = gym.make("CartPole-v0")


def cart_pole():
    while True:
        env.reset()
        for t in range(200):
            env.render()
            a = env.action_space.sample()
            _, _, done, _ = env.step(a)
            if done:
                break


cart_pole()
