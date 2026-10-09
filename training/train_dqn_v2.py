import os
import time

import numpy as np

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe
from environments.qlearning_state_encoder_v2 import (
    encoder_etat_qlearning_v2,
)
from models.dqn import DQNAgent


# PARAMÈTRES

EPISODES = 5000

MIN_SIZE = 5
MAX_SIZE = 10

MAX_STEPS = 200

RESULTS_DIR = "results/dqn_v2"

MODEL_PATH = (
    f"{RESULTS_DIR}/model.pth"
)

REWARDS_PATH = (
    f"{RESULTS_DIR}/rewards.npy"
)

STEPS_PATH = (
    f"{RESULTS_DIR}/steps.npy"
)

SUCCESSES_PATH = (
    f"{RESULTS_DIR}/successes.npy"
)


# PRÉPARATION

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


agent = DQNAgent(
    state_size=12,
    learning_rate=0.001,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    memory_size=10000,
    batch_size=64,
)


rewards_history = []
steps_history = []
successes_history = []


start_time = time.time()


print("=" * 70)
print("ENTRAÎNEMENT DU DQN AVEC REPRÉSENTATION V2")
print("=" * 70)

print()
print(f"Nombre d'épisodes : {EPISODES}")
print(
    f"Tailles utilisées : "
    f"{MIN_SIZE}x{MIN_SIZE} à "
    f"{MAX_SIZE}x{MAX_SIZE}"
)
print(
    f"Nombre maximal de mouvements : "
    f"{MAX_STEPS}"
)
print("Taille de l'état : 12")
print()


# ENTRAÎNEMENT

for episode in range(EPISODES):

    # Génération du labyrinthe

    taille = np.random.randint(
        MIN_SIZE,
        MAX_SIZE + 1,
    )

    grille = generer_labyrinthe(
        taille
    )

    env = LabyrintheEnv(
        grille,
        max_steps=MAX_STEPS,
    )

    position = env.reset()

    # Première représentation de l'état

    derniere_action = 4

    etat = encoder_etat_qlearning_v2(
        env.grille,
        position,
        derniere_action,
    )

    total_reward = 0
    steps = 0
    success = 0

    # Épisode

    for step in range(MAX_STEPS):

        # Choix de l'action
        action = agent.choose_action(
            etat
        )

        # Exécution dans l'environnement
        next_position, reward, done = env.step(
            action
        )

        # Encodage du nouvel état
        next_etat = encoder_etat_qlearning_v2(
            env.grille,
            next_position,
            action,
        )

        # Stockage de l'expérience

        agent.remember(
            etat,
            action,
            reward,
            next_etat,
            done,
        )

        # Apprentissage du réseau

        agent.replay()

        # État suivant
        etat = next_etat

        total_reward += reward

        steps += 1

        # Fin de l'épisode

        if done:

            if (
                env.grille[
                    env.position[0]
                ][
                    env.position[1]
                ] == "G"
            ):
                success = 1

            break

    # Diminution de l'exploration

    agent.decay_epsilon()

    # Sauvegarde de l'historique

    rewards_history.append(
        total_reward
    )

    steps_history.append(
        steps
    )

    successes_history.append(
        success
    )

    # Affichage

    if (
        (episode + 1) % 100 == 0
        or episode == 0
    ):

        start = max(
            0,
            len(rewards_history) - 100,
        )

        avg_reward = np.mean(
            rewards_history[start:]
        )

        success_rate = (
            np.mean(
                successes_history[start:]
            )
            * 100
        )

        avg_steps = np.mean(
            steps_history[start:]
        )

        print(
            f"Episode {episode + 1:4d}/{EPISODES} | "
            f"Récompense : {avg_reward:8.2f} | "
            f"Succès : {success_rate:6.2f}% | "
            f"Mouvements : {avg_steps:6.2f} | "
            f"Epsilon : {agent.epsilon:.4f}"
        )


# RÉSULTATS FINAUX

training_time = (
    time.time() - start_time
)

final_avg_reward = np.mean(
    rewards_history[-100:]
)

final_success_rate = (
    np.mean(
        successes_history[-100:]
    )
    * 100
)

final_avg_steps = np.mean(
    steps_history[-100:]
)


# SAUVEGARDE

agent.save(
    MODEL_PATH
)

np.save(
    REWARDS_PATH,
    np.array(
        rewards_history,
        dtype=np.float32,
    ),
)

np.save(
    STEPS_PATH,
    np.array(
        steps_history,
        dtype=np.int32,
    ),
)

np.save(
    SUCCESSES_PATH,
    np.array(
        successes_history,
        dtype=np.int32,
    ),
)


# AFFICHAGE FINAL

print()
print("=" * 70)
print("ENTRAÎNEMENT DQN V2 TERMINÉ")
print("=" * 70)

print(
    f"Récompense moyenne finale : "
    f"{final_avg_reward:.2f}"
)

print(
    f"Taux de réussite final : "
    f"{final_success_rate:.2f}%"
)

print(
    f"Nombre moyen de mouvements : "
    f"{final_avg_steps:.2f}"
)

print(
    f"Temps d'entraînement : "
    f"{training_time:.2f} secondes"
)

print()
print(
    "Modèle sauvegardé dans : "
    f"{MODEL_PATH}"
)

print("=" * 70)