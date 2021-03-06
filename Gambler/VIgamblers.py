# value iteration
import matplotlib.pyplot as plt
import numpy as np
import time

size = 111
v = np.array(size)
pHeads = 0.4

def valueIteration():
    # initialize the v table
    v = np.zeros(size)
    v[100] = 1.0 #terminal reward
    delta = 0
    while True:
        delta = 0
        # loop through all possible states
        for s in range(1,size-1):
            vOld = v[s]
            best = -1
            # Bellman equation, loop through all possible actions maximizing
            for a in range(1,1 + min(s, (size-1-s))): #all possible bets
                # for this action, two outcomes, heads or tails
                # if heads get the money so next state is s+a
                # if tails lose the money so next state is s-a
                best = max(best, pHeads*v[s+a] + (1-pHeads)*v[s-a])
            # update the value of s with the new value
            v[s] = best
            delta = max(delta, abs(vOld - best))
        if delta < 1e-10:
            break
    # compute policy
    pi = np.zeros(size)
    # for each state
    for s in range(1,size-1):
        best = -1
        # try each action and do an argmax
        # note, to get the same graph as in the book, start with the largest
        # bets and use >= so policy will pick the smalest equivalent bet
        for a in range(min(s, (size-1-s)), 0, -1): #all possible bets
            # Note the round! So very similar propabilities look the same
            now = round(pHeads*v[s+a] + (1-pHeads)*v[s-a],6)
            if now >= best:
                best = now
                bestAction = a
        pi[s] = bestAction
    #plt.plot(v, drawstyle="steps")
    plt.plot(pi, drawstyle="steps")
    plt.show()


start = time.time()
valueIteration()
end = time.time()
print(end-start)



