import datetime
import random
import numpy as np


def display(q_table, env, num_ep, s_clo, bins):
    for m in range(num_ep):
        s = env.reset()
        s = s_clo(s, bins)
        while True:
            a = np.argmax(q_table[s])
            s_next, reward, done, _ = env.step(a)
            s_next = s_clo(s_next, bins)
            s = s_next
            env.render()
            if done:
                break


# STEPS
# 1. Init Q Table
# 2. Choose an action
# 3. Perform action
# 4. Measure Reward
# 5. Update Q Table
def q_learning(q_table, env, num_ep, alpha, gamma, bins, train_until, r_clo, s_clo):
    # Step 1
    print('Beginning Learning' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    success_table = []
    episode_table = []
    success = 0

    for m in range(num_ep):
        s = env.reset()
        s = s_clo(s, bins)
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
            s_next = s_clo(s_next, bins)
            reward = r_clo(s, s_next, reward, done)
            ts += reward
            t += 1

            # step 4 & 5 :: Use Bellman equation from 6.5
            # big alpha if the reward is small
            # little alpha if the reward is big
            temp = q_table[s][a]
            q_table[s][a] = temp + alpha * (reward + gamma * np.max(q_table[s_next]) - temp)

            s = s_next

            if done:
                break

        if m % 1000 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'Episode: ' + str(m))
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'Total Value: ' + str(ts))
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'QTable Length: ' + str(len(q_table)))

    return q_table, success_table, episode_table
