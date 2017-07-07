import numpy as np

class Env(object):
    """Abstract base class for environments

    An environment take a given action and return a state and a reward.
    """

    def __init__(self, p=3):
        self.p = p
        self.beta = np.array([0.40, 0.25, 0.35, 0.65, 0.10, 0.50, 0.22, 600,
                              0.15, 0.20, 0.32, 0.10, 0.45, 1.50])
        sigma1 = np.array([[1, 0.3, -0.3], [0.3, 1, -0.3], [-0.3, -0.3, 1]])
        self.sigma = np.pad(sigma1, ((0, p), (0, p)), 'constant',
                            constant_values=0)
        for i in range(4, p):
            self.sigma[i, i] += 1

    def step(self, state, action):
        beta = self.beta
        s1 = np.tanh(state[0])
        s2 = np.tanh(state[1])
        s3 = np.tanh()
        if p > 3:
            pass
        return s, r

    def reset(self):
        state = np.random.multivariate_normal(0, self.sigma, p)
        reward = 0
        return state, reward
