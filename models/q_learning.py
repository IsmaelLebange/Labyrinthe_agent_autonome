import numpy as np


class QLearningAgent:

    def __init__(
        self,
        state_size=30,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):

        self.state_size = state_size

        self.n_actions = 4

        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = epsilon_decay

        # Q-TABLE
        #
        # Pour cette version, on utilise une représentation
        # discrétisée de l'état.
        #
        # La table sera créée dynamiquement.

        self.q_table = {}

    # CONVERSION DE L'ETAT EN CLE

    def state_to_key(self, state):

        return tuple(
            np.round(
                np.array(state, dtype=np.float32),
                2,
            )
        )

    # OBTENIR LES Q-VALUES

    def get_q_values(self, state):

        key = self.state_to_key(state)

        if key not in self.q_table:

            self.q_table[key] = np.zeros(
                self.n_actions,
                dtype=np.float32,
            )

        return self.q_table[key]

    # CHOISIR UNE ACTION

    def choose_action(self, state):

        q_values = self.get_q_values(state)

        if np.random.random() < self.epsilon:

            return np.random.randint(
                self.n_actions
            )

        return int(
            np.argmax(q_values)
        )

    # MISE A JOUR

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        current_q_values = self.get_q_values(
            state
        )

        next_q_values = self.get_q_values(
            next_state
        )

        current_q = current_q_values[action]

        if done:

            target = reward

        else:

            best_next_q = np.max(
                next_q_values
            )

            target = (
                reward
                + self.discount_factor
                * best_next_q
            )

        current_q_values[action] = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

    # DIMINUER EPSILON

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # SAUVEGARDER

    def save(self, path):

        np.save(
            path,
            self.q_table,
            allow_pickle=True,
        )

    # CHARGER

    def load(self, path):

        data = np.load(
            path,
            allow_pickle=True,
        )

        self.q_table = data.item()