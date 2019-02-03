import gym
import QLearning

QLearning.q_learning(gym.make("Acrobot-v1"), 2000, .1, 1, -100, 100)
