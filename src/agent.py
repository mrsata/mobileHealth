from __future__ import division
from __future__ import print_function
import timeit

import numpy as np


class Agent(object):
    """Abstract base class for agents

    """

    def __init__(self, nb_users, nb_actions, state_size):
        self.nb_users = nb_users
        self.nb_actions = nb_actions
        self.state_size = state_size
        self.memory = None
        self.training = True

    def warmup(self, env, nb_steps=50):
        print("Start warmup: {} steps".format(nb_steps))
        step = 0
        state = env.reset()
        action = np.random.randint(self.nb_actions, size=state.shape[0])
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis],
                                     s_new))[:, np.newaxis, :]
                self.memory = transit if step == 0 else np.concatenate(
                    (self.memory, transit), axis=1)
                action = np.random.randint(self.nb_actions,
                                           size=state.shape[0])
                state = s_new
                step += 1
        except KeyboardInterrupt:
            print("Warmup interrupted at step:", step)
        return self.memory

    def fit(self, env, nb_steps=100):
        print("Start training: ")
        self.training = True
        start_time = timeit.default_timer()
        step, checkpoint = 0, 0
        state = env.reset()
        action = self.forward(state)
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis],
                                     s_new))[:, np.newaxis, :]
                self.memory = transit if self.memory is None else \
                    np.concatenate((self.memory, transit), axis=1)
                action = self.forward(s_new)
                state = s_new
                self.backward()
                step += 1
                if step == checkpoint + nb_steps // 20:
                    print(step, np.mean(self.memory.reshape(-1,
                                        self.memory.shape[-1]), axis=0)[4])
                    checkpoint = step
        except KeyboardInterrupt:
            print("Training interrupted at step:", step)
        print("duration: {}s".format(timeit.default_timer() - start_time))
        return self.memory

    def test(self, env, nb_steps=1000):
        print("Start testing: ")
        self.training = False
        step, checkpoint = 0, 0
        state = env.reset()
        action = self.forward(state)
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis],
                                     s_new))[:, np.newaxis, :]
                memory = transit if step == 0 else np.concatenate(
                    (memory, transit), axis=1)
                action = self.forward(s_new)
                state = s_new
                step += 1
                if step == checkpoint + nb_steps // 20:
                    print(step, np.mean(memory.reshape(-1, memory.shape[-1]),
                                        axis=0)[4])
                    checkpoint = step
        except KeyboardInterrupt:
            print("Testing interrupted at step:", step)
        return memory

    def forward(self, state):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()
