import numpy as np

class BayesianAgent:
    def __init__(self, env, states, actions, gammas, scalingFactor=1):
        self.env = env
        self._gammas = gammas
        self._alpha = 1
        self._beta = 1
        self._scalingFactor = scalingFactor
        self.policy = np.zeros((len(self._gammas), states.n))
        for i, gamma in enumerate(self._gammas):
            self.policy[i] = self._compute_policy(env, gamma)
        self.gamma_belief = 1 - (self._alpha / (self._alpha + self._beta))
        self._qtable_index = None

    def _compute_policy(self, env, gamma):
        qvalues = np.zeros((env.observation_space.n, env.action_space.n))
        transition_probabilities, rewards = self.env.get_model()
        error = 1

        V = np.zeros(qvalues.shape[0])
        while error > 1e-8:
            for action in range(qvalues.shape[1]):
                qvalues[:, action] = rewards[:, action] + gamma * np.dot(
                    transition_probabilities[action, :, :], V
                )
            V_new = np.max(qvalues, axis=1)
            error = np.linalg.norm(V - V_new)
            V = V_new

        return np.argmax(qvalues, axis=1)

    def update_belief(self, timesteps):
        self._alpha += 1
        self._beta += timesteps
        self.gamma_belief = 1 - (self._alpha / (self._alpha + self._beta))
        if self._alpha % self._scalingFactor == 0:
            self._alpha /= self._scalingFactor
            self._beta /= self._scalingFactor

    def select_action(self, state):
        min_dist = np.inf
        for i, gamma in enumerate(self._gammas):
            if np.abs(gamma - self.gamma_belief) < min_dist:
                min_dist = np.abs(gamma - self.gamma_belief)
                self._qtable_index = i
        return self.policy[self._qtable_index][state]
