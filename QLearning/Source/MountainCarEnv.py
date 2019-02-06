import gym
import numpy as np

import QLearning


def get_state(s, bins):
    s = tuple(np.round(s, bins))
    return s


def get_reward(s, s_n, r, done):
    if done:
        return -1
    return 1/r


r_clo = get_reward
QLearning.q_learning(gym.make("MountainCar-v0"), 25000, .2, 1, -200, 0, 2, 2500, 15, r_clo, get_state)



