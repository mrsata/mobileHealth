import numpy as np
from src.agent import Agent
from src.env import Env

np.random.seed(0)
action_space = [0, 1]

def generate_baseline(n=40):
    psi = np.array([0.065, 0.145, -0.495])
    z1 = np.random.poisson(35, n)
    z2 = np.random.binomial(1, 0.35, n)
    z3 = np.random.normal(0, 5.29, n)
    z = np.vstack((z1, z2, z3)).T
    b = z.dot(psi)
    return b

baseline = generate_baseline()
env = Env()
print(env.sigma)

env = Env(4)
print(env.sigma)
