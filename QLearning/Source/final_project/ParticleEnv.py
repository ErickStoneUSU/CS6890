import os
import pickle
from collections import defaultdict
import make_env
import numpy as np
import QLearningParticleAgent as QLearning

env = make_env.make_env('simple_world_comm')


def dd():
    return np.zeros(6)


def get_state(s):
    return s


def get_reward(r):
    return r


# initial testing, just use reward given with same method
def load():
    table = [{'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward},
             {'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward},
             {'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward},
             {'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward},
             {'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward},
             {'q_table': defaultdict(dd), 'state_clo': get_state, 'reward_clo': get_reward}]

    if os.path.exists('particle' + "save.p"):
        table = pickle.load(open('particle' + "save.p", "rb"))
    return table


def save(table):
    pickle.dump(table, open('particle' + "save.p", "wb"))


def train():
    # agent_table(q_table,reward_closure,state_closure)
    agent_table = load()
    agent_table, success_table, episode_table = QLearning.q_learning(
        agent_table=agent_table,
        env=env,
        num_ep=20000,
        alpha=.2,
        gamma=1,
        train_until=2500)
    save(agent_table)


def watch():
    agent_table = load()
    QLearning.display(agent_table, env, 300000, get_state, 2)


train()
# watch()
