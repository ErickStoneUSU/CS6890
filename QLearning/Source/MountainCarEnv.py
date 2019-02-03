import gym
import QLearning

QLearning.q_learning(gym.make("MountainCar-v0"), 25000, .2, 1, -200, 0, 2, 2500, 15)
