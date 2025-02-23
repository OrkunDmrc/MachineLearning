import gym
import numpy as np
import random

env = gym.make("Taxi-v3",render_mode="ansi")
env.reset()

action_space = env.action_space.n
state_space = env.observation_space.n

qtable = np.zeros((state_space,action_space))

alpha = 0.1 #learning rate
gamma = 0.6 #discount rate
epsilon = 0.1 #epsilon

for i in range(1, 100001):
    state, _= env.reset()
    done = False
    while not done:
        if random.uniform(0,1) < epsilon: #explore
            action = env.action_space.sample()
        else: #exploit
            action = np.argmax(qtable[state])
    new_state, reward, done, info, _ = env.step(action)
    qtable[state,action] = qtable[state,action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state,action])
    state = new_state
print("Training Finished")

#test
total_epoch, total_penalties = 0,0
episodes = 100

for i in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0,0,0
    done = False
    while not done:
        action = np.argmax(qtable[state])
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        if reward == -10:
            penalties += 1
        epochs += 1
    total_epoch += epochs
    total_penalties += penalties
print("Result after {} episodes".format(episodes))
print("Average timesteps per episode:",total_epoch/episodes)
print("Average penalties per episode:",total_penalties/episodes) 
