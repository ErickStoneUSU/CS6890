import os
import pickle
from collections import defaultdict
import gym
import numpy as np
import QLearning

env = gym.make("Acrobot-v1")
name = env.unwrapped.spec.id


def dd():
    return np.zeros(env.action_space.n)


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
    if done:
        return -1
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

q_table = defaultdict(dd)
if os.path.exists(name + "save.p"):
    q_table = pickle.load(open(name + "save.p", "rb"))

# QLearning.display(q_table, env, 300000, get_state, 4)
# q_table, env, num_ep, alpha, gamma, bins, train_until, r_clo, s_clo
q_table, success_table, episode_table = QLearning.q_learning(
    q_table, env, 80000, .01, 1, 4, 6000, get_reward, get_state)

pickle.dump(q_table, open(name + "save.p", "wb"))
