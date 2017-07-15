from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from src.agent import Agent
from src.env import Env


def generate_baseline(nb_users):
    psi = np.array([0.065, 0.145, -0.495])
    z1 = np.random.poisson(35, nb_users)
    z2 = np.random.binomial(1, 0.35, nb_users)
    z3 = np.random.normal(0, 5.29, nb_users)
    z = np.vstack((z1, z2, z3)).T
    b = z.dot(psi)
    return b

np.random.seed(0)
nb_users = 40
state_size = 3
learning_rate = 1e-2
baseline = generate_baseline(nb_users=nb_users)
env = Env(baseline=baseline, p=state_size)
nb_actions = env.action_space.shape[-1]

model = Sequential()
model.add(Dense(10, input_dim=state_size, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
print(model.summary())

agent = Agent(model=model, nb_users=nb_users, nb_actions=nb_actions,
              state_size=state_size, gamma=.99)
agent.warmup(env, nb_steps=50)
history = agent.fit(env, nb_steps=100)

agent2 = Agent(model=model, nb_users=nb_users, nb_actions=nb_actions,
              state_size=state_size, gamma=.99)
agent2.warmup(env, nb_steps=50)
history2 = agent.fit(env, nb_steps=200)

print(np.average(history.reshape(-1, history.shape[-1]), axis=0)[4])
print(np.average(history2.reshape(-1, history2.shape[-1]), axis=0)[4])
print(np.average(history.reshape(-1, history.shape[-1]), axis=0).shape)

with open('data/history', 'w') as f:
    for i in range(len(history)):
        f.write("user #{}: \n".format(i))
        for j in range(len(history[i])):
            f.write(', '.join(map(str, history[i, j, :5])) + '\n')
        f.write("*" * 79 + '\n')
