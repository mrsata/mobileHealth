import numpy as np

class Agent(object):
    """Abstract base class for agents

    An agent select an action with a given policy, which is a random policy or
    a learned policy. Default policy of an agent is the random policy.
    """

    def __init__(self, random_policy=True, action_space=None):
        self.random_policy = random_policy
        self.action_space = action_space
        self.step = 0

    def forward(self, state, reward):
        action = None
        if self.random_policy:
            action = np.random.choice(action_space)
        else:
            pass
        return action

    def train(self, env, nb_steps=210):
        state, reward = env.reset()
        action = self.forward(state, reward)
        history = [(state, action, reward)]
        while self.step < nb_steps:
            state, reward = env.step(state, action)
            action = self.forward(state, reward)
            history.append((state, action, reward))
        return history
