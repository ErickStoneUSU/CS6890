import gym
import numpy as np
import QLearning


def get_state(s, bins):
    t = s
    for i in range(len(s)):
        if s[i] < 0:
            s[i] = 0  # make the negative side worth nothing, so we only have values on one side to create spin
    s = np.abs(np.sum(s[0:3])) / 4  # this gives a value between 0 and 1
    s = (s + (1 - 1/(.0001 + t[4])) + (1 - 1/(.0001 + t[5])))/3
    s = np.round(s, bins)  # this should give us 1 to .01
    return s  # add value for speed


def get_reward(s, s_n, r, done):
    if done:
        return -1
    # down is pi/2 this means that up which is desired is 3 pi/2
    # increase if clock wise of previous state
    # [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    # 0 is either 180 or 0
    # one of these is down
    # i think down is 1,0 1,0
    for i in range(len(s_n)):
        if s_n[i] < 0:
            s_n[i] = s_n[i] / 2  # make the negative side worth less then the positive side to create spin
    s_n = np.abs(np.sum(s_n[0:3])) / 4  # this gives a value between 0 and 1
    s = np.round(s_n, 2)  # this should give us 1 to .01
    return s + r


# env, num_ep, alpha, gamma,
# min_ts, max_ts, bins, train_until,
# success_count, r_clo
QLearning.q_learning(gym.make("Acrobot-v1"), 50000, .1, 1,
                     -250, 0, 1, 3000,
                     15, get_reward, get_state)
