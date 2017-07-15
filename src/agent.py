import numpy as np


class Agent(object):
    """Abstract base class for agents

    An agent select an action with a given policy, which is a random policy or
    a learned policy. Default policy of an agent is the random policy.
    """

    def __init__(self, model, nb_users, nb_actions, state_size, gamma=.99):
        self.model = model
        self.nb_users = nb_users
        self.nb_actions = nb_actions
        self.state_size = state_size
        self.gamma = gamma
        self.memory = None

    def forward(self, state):
        q_values = []
        for s in state:
            q_values.append(self.model.predict(s[np.newaxis, :]))
        q_values = np.array(q_values).reshape(self.nb_users, self.nb_actions)
        action = np.argmax(q_values, axis=1)
        return action

    def backward(self, batch_size):
        replay = self.memory.reshape(-1, self.memory.shape[-1])
        mask = np.random.randint(replay.shape[0], size=batch_size)
        minibatch = replay[mask, :]
        for batch in minibatch:
            state = batch[:self.state_size]
            action = int(batch[self.state_size])
            reward = batch[self.state_size + 1]
            s_new = batch[-self.state_size:]
            target = reward + self.gamma * \
                np.amax(self.model.predict(s_new[np.newaxis, :])[0])
            target_f = self.model.predict(state[np.newaxis, :])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

    def warmup(self, env, nb_steps=50):
        state = env.reset()
        action = np.random.randint(self.nb_actions, size=state.shape[0])
        s_new, reward = env.step(state, action)
        self.memory = np.hstack((state,
                                action[:, np.newaxis],
                                reward[:, np.newaxis],
                                s_new))[:, np.newaxis, :]
        action = np.random.randint(self.nb_actions, size=state.shape[0])
        state = s_new
        step = 1
        while step < nb_steps:
            s_new, reward = env.step(state, action)
            transit = np.hstack((state,
                                 action[:, np.newaxis],
                                 reward[:, np.newaxis],
                                 s_new))[:, np.newaxis, :]
            self.memory = np.concatenate((self.memory, transit), axis=1)
            action = np.random.randint(self.nb_actions, size=state.shape[0])
            state = s_new
            step += 1
        return self.memory

    def fit(self, env, nb_steps=100):
        state, action, reward, step = None, None, None, 0
        state = env.reset()
        action = self.forward(state)
        s_new, reward = env.step(state, action)
        transit = np.hstack((state,
                             action[:, np.newaxis],
                             reward[:, np.newaxis],
                             s_new))[:, np.newaxis, :]
        if self.memory is None:
            self.memory = transit
        else:
            self.memory = np.concatenate((self.memory, transit), axis=1)
        action = self.forward(s_new)
        state = s_new
        step = 1
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis],
                                     s_new))[:, np.newaxis, :]
                self.memory = np.concatenate((self.memory, transit), axis=1)
                action = self.forward(s_new)
                state = s_new
                step += 1
                self.backward(32)
                print step
        except KeyboardInterrupt:
            pass
        return self.memory
