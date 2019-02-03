# value iteration
import matplotlib.pyplot as plt
import numpy as np
import time

pHeads = 0.4
size = 101


def policy_iteration():
    # initialization
    pi = np.ones((size,), dtype=int)
    v = np.zeros(size)
    v[size-1] = 1.0 #terminal reward
    # repeat until policy converges
    itt = 0
    while True:
        itt += 1
        v = policy_evaluation(pi, v)
        pi, p_stat = policy_improvement(v, pi)

        # this means there was no improvement to the policies
        if p_stat:
            break
    print(itt)
    plt.plot(pi, drawstyle="steps")
    plt.show()
    return [v, pi]


def policy_evaluation(pi, v):
    while True:
        delta = 0
        for s in range(size-1):
            old = v[s]
            # solve the linear equations "Bellman Equation"
            v[s] = round(pHeads * v[s + pi[s]] + (1 - pHeads) * v[s - pi[s]], 6)
            delta = max(delta, np.absolute(old - v[s]))
        if delta < .01:
            return v


def policy_improvement(v, pi):
    # improve the policy at each state
    p_stat = True
    for s in range(1, size-1):
        old_action = pi[s]
        old_value = v[s]
        # Bellman equation, loop through all possible actions maximizing
        for a in range(1,1 + min(s, (size-1-s))):  # all possible bets
            # for this action, two outcomes, heads or tails
            # if heads get the money so next state is s+a
            # if tails lose the money so next state is s-a
            v_new = round(pHeads * v[s + a] + (1 - pHeads) * v[s - a], 6)
            if old_value < v_new:
                pi[s] = a
                old_value = v_new

        # update the value of s with the new value
        if old_action != pi[s]:
            p_stat = False
    return pi, p_stat


start = time.time()
v, piOut = policy_iteration()
end = time.time()
print(end-start)

