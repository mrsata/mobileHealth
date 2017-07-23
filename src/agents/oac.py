import numpy as np
from src.agent import Agent


class OACAgent(Agent):
    """Abstract base class for Online Actor Critic agents

    """

    def __init__(self, gamma=0, zeta_c=.1, zeta_a=.1, **kwargs):
        super(OACAgent, self).__init__(**kwargs)
        self.gamma = gamma
        self.zeta_c = zeta_c
        self.zeta_a = zeta_a
        self.theta = None
        self.v = None

    def forward(self, state):
        action = np.random.randint(self.nb_actions, size=self.nb_users)
        return action

    def backward(self):
        pass
        return theta, w

    def criticUpdate():
        pass
        return w

    def actorUpdate():
        pass
        return f, g
