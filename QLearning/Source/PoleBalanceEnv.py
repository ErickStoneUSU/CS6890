import gym
import numpy as np
from matplotlib import pyplot as plt
import QLearning


def get_state(s, bins):
    s = tuple(np.round(s, bins))
    return s


def get_reward(s, s_n, r, done):
    if done:
        return 1/r
    return r


# 8 appears to be the best
# 8,9,10 appear to be very very good
success_table, episode_table = QLearning.q_learning(gym.make("CartPole-v0"), 200000, .01, .999, 0, 11, 6, 3000, 200, get_reward, get_state)

plt.plot(success_table, episode_table)
plt.ylabel('Episodes')
plt.xlabel('Values')
plt.draw()
plt.show()
