import numpy as np


class Env(object):
    """Abstract base class for environments

    An environment that takes a given action and returns a state and a reward.
    Generates inital states and rewards by calling reset() function.
    """

    def __init__(self, baseline, p=3):
        self.baseline = baseline
        self.nb_users = baseline.shape[0]
        self.action_space = np.tile(np.array([0, 1]), (self.nb_users, 1))
        self.p = p
        beta = np.array([0.40, 0.25, 0.35, 0.65, 0.10, 0.50, 0.22, 600, 0.15,
                         0.20, 0.32, 0.10, 0.45])
        sigma_beta = 5e-2
        delta = np.random.multivariate_normal(np.zeros(beta.shape[0], ),
                                              np.identity(beta.shape[0]) *
                                              sigma_beta, self.nb_users)
        self.beta = np.tile(beta, (self.nb_users, 1)) + delta

    def step(self, state, action):
        b, beta = self.baseline, self.beta
        action = self.action_space[range(self.nb_users), action]
        sigma_s, sigma_r = 0.5, 1
        xi = np.random.multivariate_normal(np.zeros(self.p, ),
                                           np.identity(self.p) * sigma_s)
        rho = np.random.normal(0, sigma_r)
        s_new = np.zeros_like(state)
        s_new[:, 0] = np.tanh(beta[:, 0] * state[:, 0] + xi[0])
        s_new[:, 1] = np.tanh(beta[:, 1] * state[:, 1] + beta[:, 2] * action +
                              xi[1])
        s_new[:, 2] = np.tanh(beta[:, 3] * state[:, 2] + beta[:, 4] *
                              state[:, 2] * action + beta[:, 5] * action +
                              xi[2])
        for i in range(3, self.p):
            s_new[i] = np.tanh(beta[:, 6] * state[:, i] + xi[i])
        reward = beta[:, 7] * (b + action * (beta[:, 8] + beta[:, 9] *
                               state[:, 0] + beta[:, 10] * state[:, 1]) +
                               beta[:, 11] * state[:, 0] - beta[:, 12] *
                               state[:, 2] + rho)
        return s_new, reward

    def reset(self):
        N, b, beta = self.nb_users, self.baseline, self.beta
        SIGMA = np.identity(self.p)
        SIGMA_1 = np.array([[1, 0.3, -0.3], [0.3, 1, -0.3], [-0.3, -0.3, 1]])
        SIGMA[:3, :3] = SIGMA_1
        state = np.random.multivariate_normal(np.zeros(self.p), SIGMA, N)
        state = np.maximum(0, state)
        return state
