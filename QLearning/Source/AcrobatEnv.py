import gym
import numpy as np

import QLearning


def get_reward(s, s_n, r, done):
    if done:
        return -1
    # down is pi/2 this means that up which is desired is 3 pi/2
    # increase if clock wise of previous state
    # [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    # 0 is either 180 or 0
    # one of these is down
    # i think down is 1,0 1,0
    reward = 0
    arm1cos = np.arccos(s[0])
    arm1sin = np.arcsin(s[1])
    arm2cos = np.arccos(s[2])
    arm2sin = np.arcsin(s[3])
    velocity1 = s[4]
    velocity2 = s[5]

    reward += np.sum(np.abs(s))
    # give a reward if the two arms spin together
    # if velocity1 * 1.05 < velocity2 < velocity1 * .95:
    #     reward += 5
    # else:
    #     reward -= 3
    # if arm1cos * 1.05 < arm2cos < arm1cos * .95:
    #     reward += 5
    # else:
    #     reward -= 3
    # if arm1sin * 1.05 < arm2sin < arm1sin * .95:
    #     reward += 5
    # else:
    #     reward -= 3
    # if velocity1 < s_n[4]:
    #     reward += 20
    return r


# env, num_ep, alpha, gamma,
# min_ts, max_ts, bins, train_until,
# success_count, r_clo
QLearning.q_learning(gym.make("Acrobot-v1"), 50000, .01, 1,
                     -200, 0, 3, 3000,
                     1, get_reward)
