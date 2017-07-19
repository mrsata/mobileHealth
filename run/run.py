from __future__ import print_function
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--nb-steps', type=int, default=100)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

np.random.seed(0)
nb_steps = args.nb_steps
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
import time
t = time.time()
history = agent.fit(env, nb_steps=nb_steps)
print("time:", time.time() - t)

item_size = history.shape[-1]
for i in range(nb_steps / 20, nb_steps + 1, nb_steps / 20):
    print(i, np.average(history[:, :i, :].reshape(-1, item_size), axis=0)[4])

with open('data/history', 'w') as f:
    for i in range(len(history)):
        f.write("user #{}: \n".format(i))
        for j in range(len(history[i])):
            f.write(', '.join(map(str, history[i, j, :5])) + '\n')
        f.write("*" * 79 + '\n')
