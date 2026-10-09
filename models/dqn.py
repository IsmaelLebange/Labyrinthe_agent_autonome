import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):

    def __init__(
        self,
        input_size=30,
        output_size=4,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, output_size),
        )

    def forward(self, state):

        return self.network(state)


class DQNAgent:

    def __init__(
        self,
        state_size=30,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
    ):

        self.state_size = state_size

        self.n_actions = 4

        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.memory = deque(
            maxlen=memory_size
        )

        # DEVICE

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # RÉSEAU

        self.model = DQNNetwork(
            input_size=self.state_size,
            output_size=self.n_actions,
        ).to(self.device)

        # OPTIMISEUR

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # FONCTION DE PERTE

        self.loss_function = nn.MSELoss()

    # CONVERSION DE L'ÉTAT EN TENSEUR

    def state_to_tensor(self, state):

        state_array = np.array(
            state,
            dtype=np.float32,
        )

        return torch.tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    # CHOIX DE L'ACTION

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.n_actions
            )

        # Exploitation
        state_tensor = (
            self.state_to_tensor(state)
        )

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )

    # MÉMOIRE

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    # APPRENTISSAGE

    def replay(self):

        # Pas assez d'expériences
        if len(self.memory) < self.batch_size:

            return None

        # ÉCHANTILLONNAGE

        batch = random.sample(
            self.memory,
            self.batch_size,
        )

        # ÉTATS

        states = torch.tensor(
            np.array(
                [
                    experience[0]
                    for experience in batch
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ACTIONS

        actions = torch.tensor(
            [
                experience[1]
                for experience in batch
            ],
            dtype=torch.long,
            device=self.device,
        )

        # RÉCOMPENSES

        rewards = torch.tensor(
            [
                experience[2]
                for experience in batch
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # ÉTATS SUIVANTS

        next_states = torch.tensor(
            np.array(
                [
                    experience[3]
                    for experience in batch
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # TERMINAISONS

        dones = torch.tensor(
            [
                experience[4]
                for experience in batch
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Q-VALUES ACTUELLES

        current_q_values = (
            self.model(states)
            .gather(
                1,
                actions.unsqueeze(1),
            )
            .squeeze(1)
        )

        # Q-VALUES SUIVANTES

        with torch.no_grad():

            next_q_values = (
                self.model(next_states)
                .max(
                    dim=1
                )
                .values
            )

            # Q-VALUE CIBLE

            target_q_values = (
                rewards
                + (
                    (1 - dones)
                    * self.discount_factor
                    * next_q_values
                )
            )

        # CALCUL DE LA LOSS

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        # BACKPROPAGATION

        self.optimizer.zero_grad()

        loss.backward()

        # Évite des gradients trop importants
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return loss.item()

    # DIMINUTION D'EPSILON

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # SAUVEGARDE

    def save(self, path):

        torch.save(
            self.model.state_dict(),
            path,
        )

    # CHARGEMENT

    def load(self, path):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )

        self.model.eval()