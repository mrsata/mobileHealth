import numpy as np

class Env(object):
    """Abstract base class for environments

    An environment take a given action and return a state and a reward.
    """

    def __init__(self, baseline, action_space, p=3):
        self.baseline = baseline
        self.action_space = action_space
        self.p = p
        self.beta = np.array([0.40, 0.25, 0.35, 0.65, 0.10, 0.50, 0.22, 600,
                              0.15, 0.20, 0.32, 0.10, 0.45, 1.50])

    def step(self, state, action):
        b, beta = self.baseline, self.beta
        xi = np.zeros_like(beta)
        # np.random.multivariate_normal(np.zeros(self.p), cov)
        rho = np.zeros_like(b)
        # np.random.multivariate_normal(np.zeros(self.p), cov)
        s = np.zeros_like(state)
        s[:, 0] = np.tanh(beta[0] * state[:, 0] + xi[0])
        s[:, 1] = np.tanh(beta[1] * state[:, 1] + beta[2] * action + xi[1])
        s[:, 2] = np.tanh(beta[3] * state[:, 2] + beta[4] * state[:, 2] *
                          action + beta[5] * action + xi[2])
        for i in range(3, self.p):
            s[i] = np.tanh(beta[6] * state[:, i] + xi[i])
        r = beta[7] * (b + action * (beta[8] + beta[9] * s[:, 0] + beta[10] *
            s[:, 1]) + beta[11] * s[:, 0] - beta[12] * s[:, 2] + rho)
        return s, r

    def step_one(self, state, action, user_idx):
        b, beta = self.baseline, self.beta
        xi = np.zeros_like(beta)
        # np.random.multivariate_normal(np.zeros(self.p), cov)
        rho = np.zeros_like(b)
        # np.random.multivariate_normal(np.zeros(self.p), cov)
        s = np.zeros_like(state)
        s[user_idx, 0] = np.tanh(beta[0] * state[user_idx, 0] + xi[0])
        s[user_idx, 1] = np.tanh(beta[1] * state[user_idx, 1] + beta[2] *
                         action + xi[1])
        s[user_idx, 2] = np.tanh(beta[3] * state[user_idx, 2] + beta[4] *
                         state[user_idx, 2] * action + beta[5] * action +
                         xi[2])
        for i in range(3, self.p):
            s[i] = np.tanh(beta[6] * state[user_idx, i] + xi[i])
        r = beta[7] * (b[user_idx] + action * (beta[8] + beta[9] *
            s[user_idx, 0] + beta[10] * s[user_idx, 1]) + beta[11] *
            s[user_idx, 0] - beta[12] * s[user_idx, 2] + rho)
        return s, r

    def reset(self):
        N, b, beta = self.baseline.shape[0], self.baseline, self.beta
        SIGMA = np.identity(self.p)
        SIGMA_1 = np.array([[1, 0.3, -0.3], [0.3, 1, -0.3], [-0.3, -0.3, 1]])
        SIGMA[:3, :3] = SIGMA_1
        rho = np.random.normal(0, 0.01, N)
        state = np.random.multivariate_normal(np.zeros(self.p), SIGMA, N)
        action = np.random.choice(self.action_space, N)
        reward = beta[7] * (b + action * (beta[8] + beta[9] * state[:, 0] +
                            beta[10] * state[:, 1]) + beta[11] * state[:, 0] -
                            beta[12] * state[:, 2] + rho)
        return state, reward
