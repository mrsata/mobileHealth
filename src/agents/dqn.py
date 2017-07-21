import numpy as np
from src.agent import Agent


class DQNAgent(Agent):
    """Abstract base class for DQN agents

    """

    def __init__(self, eps=1.0, batch_size=32, **kwargs):
        super(DQNAgent, self).__init__(**kwargs)
        self.eps = eps
        self.batch_size = batch_size

    def forward(self, state):
        eps = self.eps if self.training else .05
        q_values = self.model.predict(state, batch_size=self.nb_users)
        action = np.argmax(q_values, axis=1)
        mask = np.random.sample(size=action.shape[0]) < eps
        action[mask] = np.random.randint(self.nb_actions,
                                         size=action[mask].shape[0])
        if self.training and self.eps > .1:
            self.eps = self.eps - 9e-4
        return action

    def backward(self):
        replay = self.memory.reshape(-1, self.memory.shape[-1])
        mask = np.random.randint(replay.shape[0], size=self.batch_size)
        minibatch = replay[mask, :]
        state = minibatch[:, :self.state_size]
        action = minibatch[:, self.state_size].astype(int)
        reward = minibatch[:, self.state_size + 1]
        s_new = minibatch[:, -self.state_size:]
        target = reward + self.gamma * \
            np.amax(self.model.predict(s_new, batch_size=self.batch_size)[0])
        target_f = self.model.predict(state, batch_size=self.batch_size)
        target_f[range(self.batch_size), action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0,
                       batch_size=self.batch_size)
