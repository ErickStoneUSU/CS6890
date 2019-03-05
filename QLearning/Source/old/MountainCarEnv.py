import os
import pickle
from collections import defaultdict
import gym
import numpy as np
from old import QLearning

env = gym.make("MountainCar-v0")
name = env.unwrapped.spec.id


def dd():
    return np.zeros(env.action_space.n)


def get_state(s, bins):
    s = tuple(np.round(s, bins))
    return s


def get_reward(s, s_n, r, done):
    if done:
        return -1
    return 1/r


q_table = defaultdict(dd)
if os.path.exists(name + "save.p"):
    q_table = pickle.load(open(name + "save.p", "rb"))

QLearning.display(q_table, env, 300000, get_state, 2)
# q_table, env, num_ep, alpha, gamma, bins, train_until, r_clo, s_clo
# q_table, success_table, episode_table = QLearning.q_learning(
#     q_table, env, 20000, .2, 1, 2, 2500, get_reward, get_state)
# pickle.dump(q_table, open(name + "save.p", "wb"))
