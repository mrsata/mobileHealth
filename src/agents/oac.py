import numpy as np
from src.agent import Agent


class OACAgent(Agent):
    """Abstract base class for Online Actor Critic agents

    """

    def __init__(self, **kwargs):
        super(OACAgent, self).__init__(**kwargs)

    def forward(self, state):
        action = np.random.randint(self.nb_actions, size=self.nb_users)
        return action

    def backward(self):
        pass
