from __future__ import print_function
import argparse
import os.path
import numpy as np

from src.agents.oac import OACAgent
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
parser.add_argument('--nb-steps', type=int, default=2)
parser.add_argument('--test', type=int, default=0)
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

agent = OACAgent(nb_users=nb_users, nb_actions=nb_actions,
                 state_size=state_size, gamma=0, zetaC=.1, zetaA=.1)
history = agent.fit(env, nb_steps=nb_steps)

file_path = "data/oac_{}".format(nb_steps)
if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        for i in range(len(history)):
            f.write("user #{}: \n".format(i))
            for j in range(len(history[i])):
                f.write(", ".join(map(str, history[i, j, :5])) + "\n")
            f.write("*" * 79 + '\n')
    f.close()

if args.test > 0:
    agent.test(env, nb_steps=args.test)
