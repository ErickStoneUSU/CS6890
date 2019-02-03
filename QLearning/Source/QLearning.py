import random
from collections import defaultdict

import numpy as np


# STEPS
# 1. Init Q Table
# 2. Choose an action
# 3. Perform action
# 4. Measure Reward
# 5. Update Q Table
def q_learning(env, num_ep, alpha, gamma, min_ts, max_ts, bins, train_until, success_count):
    # Step 1
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    success = 0
    best_ts = 100

    for m in range(num_ep):
        s = tuple(np.round(env.reset(), bins))

        if m % 100 == 0:
            print("|-- Episode {} starting.".format(m + 1))
        t = 0
        ts = 0
        ri = random.randint(0, 10) < 2

        while True:
            # step 2
            if success < 1 and (m < train_until or ri):
                a = env.action_space.sample()
            else:
                # if m > 5000:
                    # env.render()
                a = np.argmax(q_table[s])

            # step 3
            s_next, reward, done, _ = env.step(a)
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

            if success > success_count:
                env.render()

        if ts < best_ts:
            best_ts = ts

        if min_ts < ts < max_ts and m % 100 == 0:
            print("success")
            success += 1
