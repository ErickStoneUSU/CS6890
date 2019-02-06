import datetime
import random
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np


# STEPS
# 1. Init Q Table
# 2. Choose an action
# 3. Perform action
# 4. Measure Reward
# 5. Update Q Table
def q_learning(env, num_ep, alpha, gamma, min_ts, max_ts, bins, train_until, success_count, r_clo, s_clo):
    print('Beginning Learning' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # Step 1
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    success_table = []
    episode_table = []
    success = 0

    for m in range(num_ep):
        s = env.reset()
        s = s_clo(s, bins)

        if m % 100 == 0:
            print("|-- Episode {} starting.".format(m + 1) + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        t = 0
        ts = 0
        ri = random.randint(0, 10) < 2

        while True:
            # step 2
            if success < 1 and (m < train_until or ri):
                a = env.action_space.sample()
            else:
                a = np.argmax(q_table[s])

            # step 3
            s_next, reward, done, _ = env.step(a)
            reward = r_clo(s, s_next, reward, done)
            ts += reward
            t += 1
            s_next = tuple(np.round(s_next, bins))

            # step 4 & 5 :: Use Bellman equation from 6.5
            # big alpha if the reward is small
            # little alpha if the reward is big
            temp = q_table[s][a]
            q_table[s][a] = temp + alpha * (reward + gamma * np.max(q_table[s_next]) - temp)

            s = s_next

            if done:
                break

            if success > success_count and m % 20 == 0 and m > 4000:
                env.render()

        if m % 1000 == 0:
            print('Total Value: ' + str(ts))
            print('QTable Length: ' + str(len(q_table)))
            success_table.append(ts)
            episode_table.append(m)

        if min_ts < ts < max_ts and m % 1000 == 0:
            print("success")
            success += 1

    plt.plot(success_table, episode_table)
    plt.ylabel('Episodes')
    plt.xlabel('Values')
    plt.draw()
    plt.show()
