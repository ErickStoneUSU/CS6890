import gym
import QLearning

# 8 appears to be the best
# 8,9,10 appear to be very very good
QLearning.q_learning(gym.make("CartPole-v0"), 200000, .01, .999, 0, 11, 6, 3000, 200)
