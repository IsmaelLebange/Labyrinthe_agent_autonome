import numpy as np

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe
from environments.state_encoder import encoder_etat

from models.q_learning import QLearningAgent
from models.dqn import DQNAgent


MAX_STEPS = 200
N_EPISODES = 100

MIN_SIZE = 5
MAX_SIZE = 10


def evaluate_qlearning():

    agent = QLearningAgent(
        state_size=30
    )

    agent.load(
        "results/qlearning/q_table.npy"
    )

    # Exploitation uniquement
    agent.epsilon = 0.0

    results = {}

    for taille in range(
        MIN_SIZE,
        MAX_SIZE + 1
    ):

        rewards = []
        steps_list = []
        successes = []

        for _ in range(N_EPISODES):

            grille = generer_labyrinthe(
                taille
            )

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS
            )

            position = env.reset()

            state = encoder_etat(
                env.grille,
                position
            )

            total_reward = 0
            steps = 0
            success = False

            while steps < MAX_STEPS:

                action = agent.choose_action(
                    state
                )

                next_position, reward, done = env.step(
                    action
                )

                next_state = encoder_etat(
                    env.grille,
                    next_position
                )

                state = next_state

                total_reward += reward
                steps += 1

                row, col = next_position

                if env.grille[row][col] == "G":
                    success = True

                if done:
                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                steps
            )

            successes.append(
                1 if success else 0
            )

        results[taille] = {
            "reward": np.mean(rewards),
            "steps": np.mean(steps_list),
            "success_rate": np.mean(successes) * 100
        }

    return results


def evaluate_dqn():

    agent = DQNAgent(
        state_size=30,
    )

    agent.load(
        "results/dqn/dqn_state_dict.pth"
    )

    # Exploitation uniquement
    agent.epsilon = 0.0

    results = {}

    for taille in range(
        MIN_SIZE,
        MAX_SIZE + 1
    ):

        rewards = []
        steps_list = []
        successes = []

        for _ in range(N_EPISODES):

            grille = generer_labyrinthe(
                taille
            )

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS
            )

            position = env.reset()

            state = encoder_etat(
                env.grille,
                position
            )

            total_reward = 0
            steps = 0
            success = False

            while steps < MAX_STEPS:

                action = agent.choose_action(
                    state
                )

                next_position, reward, done = env.step(
                    action
                )

                next_state = encoder_etat(
                    env.grille,
                    next_position
                )

                state = next_state

                total_reward += reward
                steps += 1

                row, col = next_position

                if env.grille[row][col] == "G":
                    success = True

                if done:
                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                steps
            )

            successes.append(
                1 if success else 0
            )

        results[taille] = {
            "reward": np.mean(rewards),
            "steps": np.mean(steps_list),
            "success_rate": np.mean(successes) * 100
        }

    return results