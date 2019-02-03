import gym
env = gym.make("MountainCar-v0")


def mountain_car():
    while True:
        env.reset()
        for t in range(200):
            env.render()
            a = env.action_space.sample()
            _, _, done, _ = env.step(a)
            if done:
                break


mountain_car()
