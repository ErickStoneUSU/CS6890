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
        for s in range(1, size-1):
            old = v[s]
            # solve the linear equations "Bellman Equation"
            v[s] = pHeads * v[s + pi[s]] + (1 - pHeads) * v[s - pi[s]]
            delta = max(delta, np.absolute(old - v[s]))
        if delta < .01:
            return v


def policy_improvement(v, pi):
    # improve the policy at each state
    p_stat = True
    for s in range(1, size-1):
        old_action = pi[s]

        # Argxmax
        best = -1
        best_action = -1
        for a in range(min(s, (100 - s)), 0, -1):  # all possible bets
            # Note the round! So very similar probabilities look the same
            now = round(pHeads * v[s + a] + (1 - pHeads) * v[s - a], 6)
            if now >= best:
                best = now
                best_action = a

        pi[s] = best_action

        # update the value of s with the new value
        if old_action != pi[s]:
            p_stat = False
    return pi, p_stat


start = time.time()
v, piOut = policy_iteration()
end = time.time()
print(end-start)

