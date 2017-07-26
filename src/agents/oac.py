from __future__ import division

import numpy as np
from src.agent import Agent


class OACAgent(Agent):
    """Abstract base class for Online Actor Critic agents

    """

    def __init__(self, gamma=0, zetaC=.1, zetaA=.1, **kwargs):
        super(OACAgent, self).__init__(**kwargs)
        self.gamma = gamma
        self.zetaC = zetaC
        self.zetaA = zetaA
        self.theta = np.zeros((self.nb_users, self.state_size + 1))
        self.v = np.zeros((self.nb_users, 2 * self.state_size + 2))

    def forward(self, state):
        policy = 1 / (1 + np.exp((self.theta * self.feaPol(state)).sum(axis=1)))
        action = np.ones((self.nb_users), dtype=np.int8)
        action[np.random.sample(size=self.nb_users) < policy] = 0
        return action

    def backward(self):
        if self.memory.shape[1] < 2: return
        a = self.memory[:, :-1, self.state_size + 1]
        r = self.memory[:, :-1, self.state_size + 2]
        x = self.memory[:, :-1, :self.state_size]
        y = self.memory[:, 1:, :self.state_size]
        xx = self.feaBas(x)
        yy = self.feaBas(y)
        xa = self.feaVal(xx, a)
        x0 = self.feaVal(xx, np.zeros_like(a))
        x1 = self.feaVal(xx, np.ones_like(a))
        xg = self.feaPol(x)
        yg = self.feaPol(y)
        w = self.criticUpdate(self.theta, xa, yy, yg, r, self.gamma,
                              self.zetaC)
        return self.theta, w

    def criticUpdate(self, theta, XXA, YY, YG, rrr, gamma, zetaC):
        XXA = XXA.reshape(-1, XXA.shape[-1])
        NT = XXA.shape[0]
        thetaGo = np.zeros((self.nb_users, self.memory.shape[1] - 1, 1))
        for i in range(self.nb_users):
            thetaGo[i] = YG[i].dot(theta[i][:, np.newaxis])
        expThetaGo = np.exp(thetaGo)
        Pi = expThetaGo / (1 + expThetaGo)
        YYA = self.feaVal(YY, Pi)
        YYA = YYA.reshape(-1, YYA.shape[-1])
        if gamma < 1 and gamma >= 0:
            # the contextual bandit & the discount reward method
            # critic update for vt
            w = np.linalg.solve(
                zetaC * np.identity(XXA.shape[-1]) +
                1 / NT * XXA.T.dot(XXA - gamma * YYA),
                1 / NT * XXA.T.dot(rrr.reshape(-1, 1))
            )
        elif gamma == 1:  # the average reward method
            XXAvg = XXA - np.mean(XXA, axis=0)
            w = np.linalg.solve(
                zetaC * np.identity(XXA.shape[-1]) +
                1 / NT * XXAvg.T.dot(XXA - gamma * YYA),
                1 / NT * XXAvg.T.dot(rrr.reshape(-1, 1))
            )
        return w

    def actorUpdate(self, theta, w, XG, XX0, XX1, zetaA, Lgo):
        pass
        return f, g

    def feaBas(self, state):
        feaBas = state
        return feaBas

    def feaPol(self, state):
        if state.ndim == 2:
            feaPol = np.hstack((np.ones((state.shape[0], 1)), state))
        elif state.ndim == 3:
            feaPol = np.concatenate((
                np.ones((state.shape[0], state.shape[1], 1)),
                state
            ), axis=2)
        return feaPol

    def feaVal(self, feaBas, action):
        if feaBas.ndim == 2:
            feaVal = np.hstack((
                np.ones((feaBas.shape[0], 1)),
                feaBas,
                action[:, np.newaxis],
                feaBas * action
            ))
        elif feaBas.ndim == 3:
            feaVal = np.concatenate((
                np.ones((feaBas.shape[0], feaBas.shape[1], 1)),
                feaBas,
                action.reshape((feaBas.shape[0], feaBas.shape[1], 1)),
                feaBas * action.reshape((feaBas.shape[0], feaBas.shape[1], 1))
            ), axis=2)
        return feaVal
