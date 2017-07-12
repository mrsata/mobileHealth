from __future__ import print_function
import numpy as np
from src.agent import Agent
from src.env import Env


def generate_baseline(n=40):
    psi = np.array([0.065, 0.145, -0.495])
    z1 = np.random.poisson(35, n)
    z2 = np.random.binomial(1, 0.35, n)
    z3 = np.random.normal(0, 5.29, n)
    z = np.vstack((z1, z2, z3)).T
    b = z.dot(psi)
    return b

np.random.seed(0)
action_space = [0, 1]
baseline = generate_baseline()
env = Env(baseline, action_space)
# print(env.reset())
agent = Agent(action_space)
wst = agent.warmup(env, nb_steps=50)
# print(wst.shape)

def print_transit(transits):
    for i in range(len(transits)):
        print("user #{}: ".format(i))
        for j in range(len(transits[i])):
            print(', '.join(map(str, transits[i, j])))
        print("*" * 79)

history = agent.fit(env, nb_steps=100, wst=wst)
print_transit(history)
