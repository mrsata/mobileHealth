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
        self.eps = 1
        self.memory = None
        self.training = True

    def forward(self, state):
        eps = self.eps if self.training else .05
        q_values = self.model.predict(state, batch_size=self.nb_users)
        action = np.argmax(q_values, axis=1)
        mask = np.random.sample(size=action.shape[0]) < eps
        action[mask] = np.random.randint(self.nb_actions,
                                         size=action[mask].shape[0])
        return action

    def backward(self, batch_size):
        replay = self.memory.reshape(-1, self.memory.shape[-1])
        mask = np.random.randint(replay.shape[0], size=batch_size)
        minibatch = replay[mask, :]
        state = minibatch[:, :self.state_size]
        action = minibatch[:, self.state_size].astype(int)
        reward = minibatch[:, self.state_size + 1]
        s_new = minibatch[:, -self.state_size:]
        target = reward + self.gamma * \
            np.amax(self.model.predict(s_new, batch_size=batch_size)[0])
        target_f = self.model.predict(state, batch_size=batch_size)
        target_f[range(batch_size), action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0,
                       batch_size=batch_size)

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
        self.training = True
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
        checkpoint = 0
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
                self.eps = self.eps - 9e-4 if self.eps > .1 else self.eps
                if step == checkpoint + nb_steps / 20:
                    print step, np.average(self.memory.reshape(-1,
                                           self.memory.shape[-1]), axis=0)[4]
                    checkpoint = step
        except KeyboardInterrupt:
            pass
        return self.memory

    def test(self, env, nb_steps=1000):
        self.training = False
        state, action, reward, step = None, None, None, 0
        state = env.reset()
        action = self.forward(state)
        s_new, reward = env.step(state, action)
        transit = np.hstack((state,
                             action[:, np.newaxis],
                             reward[:, np.newaxis],
                             s_new))[:, np.newaxis, :]
        memory = transit
        action = self.forward(s_new)
        state = s_new
        step = 1
        checkpoint = 0
        try:
            while step < nb_steps:
                s_new, reward = env.step(state, action)
                transit = np.hstack((state,
                                     action[:, np.newaxis],
                                     reward[:, np.newaxis],
                                     s_new))[:, np.newaxis, :]
                memory = np.concatenate((memory, transit), axis=1)
                action = self.forward(s_new)
                state = s_new
                step += 1
                if step == checkpoint + nb_steps / 20:
                    print step, np.average(memory.reshape(-1,
                                           memory.shape[-1]), axis=0)[4]
                    checkpoint = step
        except KeyboardInterrupt:
            pass
        return memory
