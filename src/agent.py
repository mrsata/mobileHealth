import numpy as np


class Agent(object):
    """Abstract base class for agents

    An agent select an action with a given policy, which is a random policy or
    a learned policy. Default policy of an agent is the random policy.
    """

    def __init__(self, action_space=None):
        self.action_space = action_space

    def forward(self, state):
        action = np.zeros(state.shape[0], )
        ### TODO: implement this ###
        
        return action

    def warmup(self, env, nb_steps=50):
        state = env.reset()
        action = np.random.choice(self.action_space, state.shape[0])
        s_new, reward = env.step(state, action)
        history = np.hstack((state,
                             action[:, np.newaxis],
                             reward[:, np.newaxis]))[:, np.newaxis, :]
        action = np.random.choice(self.action_space, state.shape[0])
        state = s_new
        step = 1
        while step < nb_steps:
            s_new, reward = env.step(state, action)
            transit = np.hstack((state,
                                 action[:, np.newaxis],
                                 reward[:, np.newaxis]))[:, np.newaxis, :]
            history = np.concatenate((history, transit), axis=1)
            action = np.random.choice(self.action_space, state.shape[0])
            state = s_new
            step += 1
        return history

    def fit(self, env, nb_steps=100, wst=None):
        state, action, reward, history, step = None, None, None, None, 0
        state = env.reset()
        action = self.forward(state)
        s_new, reward = env.step(state, action)
        transit = np.hstack((state,
                             action[:, np.newaxis],
                             reward[:, np.newaxis]))[:, np.newaxis, :]
        if wst is not None:
            history = wst
            history = np.concatenate((history, transit), axis=1)
        else:
            history = transit
        action = self.forward(s_new)
        state = s_new
        step = 1
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis]))[:, np.newaxis, :]
                history = np.concatenate((history, transit), axis=1)
                action = self.forward(s_new)
                state = s_new
                step += 1
        except KeyboardInterrupt:
            pass
        return history
