import gym
import numpy as np
import QLearning
from matplotlib import pyplot as plt


def get_state(s, bins):
    t = np.round(s / 4, 2)
    for i in range(len(s)):
        if s[i] < 0:
            s[i] = 0  # make the negative side worth nothing, so we only have values on one side to create spin
    s = np.abs(np.sum(s[0:3])) / 4  # this gives a value between 0 and 1
    s = np.round(s, bins)  # this should give us 1 to .01
    return s + t[4] + t[5]  # add value for speed


def get_reward(s, s_n, r, done):
    # finished episodes count against forcing progression
    # big rewards for being above the half way mark
    if done and s_n < .5:
        return -10
    elif s_n > .5:
        return 10
    # down is pi/2 this means that up which is desired is 3 pi/2
    # increase if clock wise of previous state
    # [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    # 0 is either 180 or 0
    # one of these is down
    # i think down is 1,0 1,0
    return s_n + r


# env, num_ep, alpha, gamma,
# min_ts, max_ts, bins, train_until,
# success_count, r_clo
success_table, episode_table = QLearning.q_learning(gym.make("Acrobot-v1"), 50000, .1, 1,
                     0, 2000, 1, 6000,
                     15, get_reward, get_state)

plt.plot(success_table, episode_table)
plt.ylabel('Episodes')
plt.xlabel('Values')
plt.draw()
plt.show()