import gym
import numpy as np
import matplotlib.pyplot as plt

enviroment = gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
enviroment.reset()

nb_states = enviroment.observation_space.n
nb_actions = enviroment.action_space.n
qtable = np.zeros((nb_states,nb_actions))

print("Q-table")
print(qtable)

episodes = 1000
alpha = 0.5 #learning rate
gamma = 0.9 #discount rate

outcomes = []

#training
for _ in range(episodes):
    state, _ = enviroment.reset()
    done = False
    outcomes.append("Failure")
    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = enviroment.action_space.sample()
            """
            left : 0
            down : 1
            right : 2
            up : 3 
            """
        new_state, reward, done, info, _ = enviroment.step(action)
        qtable[state,action] = qtable[state,action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state,action])
        state = new_state

        if reward:
            outcomes[-1] = "Success"

print(qtable)

plt.bar(range(episodes),outcomes)
plt.show()

#test
episodes = 100
nb_success = 0
for _ in range(episodes):
    state, _ = enviroment.reset()
    done = False
    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = enviroment.action_space.sample()
        new_state, reward, done, info, _ = enviroment.step(action)
        state = new_state
        nb_success += reward

print("Success rate:", 100 * nb_success/episodes)