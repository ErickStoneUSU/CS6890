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
# agent_table(q_table,reward_closure,state_closure)
def q_learning(agent_table, env, num_ep, alpha, gamma, train_until):
    # Step 1
    print('Beginning Learning' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for m in range(num_ep):
        s = env.reset()
        while True:
            env.render()
            actions = []
            for i in range(6):
                if m > train_until:
                    # exploratory
                    # position + velocity
                    action = []
                    for j in range(1, len(env.action_space)):
                        action.append(random.randint(0, env.action_space[j].sample()))
                    actions.append(action)
                else:
                    # greedy
                    actions.append(np.argmax(agent_table[i]['q_table'][round(sum(s[i]))]))

            # take a step for all agents in environment
            states, rewards, infos, _ = env.step(actions)

            # evaluate custom goals
            next_states = []
            next_rewards = []
            for i in range(6):
                next_states.append(agent_table[i].state_clo(states[i]))
                next_rewards.append(agent_table[i].reward_clo(rewards[i]))

            # update q_tables
            for i in range(6):
                temp = agent_table[i][next_states[i]][actions[i]]
                agent_table[i][next_states[i]][actions[i]] = temp + alpha * (
                            rewards[i] + gamma * np.max(agent_table[i][states[i]]) - temp)

    return agent_table, None, None
