import os
import time
import random

import numpy as np

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe
from environments.state_encoder import encoder_etat

from models.q_learning import QLearningAgent


# PARAMÈTRES

EPISODES = 3000

MIN_SIZE = 5
MAX_SIZE = 10

MAX_STEPS = 200

RESULTS_DIR = "results/qlearning"

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "q_table.npy",
)

REWARDS_PATH = os.path.join(
    RESULTS_DIR,
    "rewards.npy",
)

STEPS_PATH = os.path.join(
    RESULTS_DIR,
    "steps.npy",
)

SUCCESSES_PATH = os.path.join(
    RESULTS_DIR,
    "successes.npy",
)


# PRÉPARATION

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# AGENT Q-LEARNING

agent = QLearningAgent(
    state_size=30,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
)


# HISTORIQUE

rewards_history = []
steps_history = []
successes_history = []

start_time = time.time()


# ENTRAÎNEMENT

print("=" * 60)
print("ENTRAÎNEMENT Q-LEARNING")
print("=" * 60)

print()
print(
    f"Nombre d'épisodes : {EPISODES}"
)

print(
    f"Tailles utilisées : {MIN_SIZE}x{MIN_SIZE} "
    f"à {MAX_SIZE}x{MAX_SIZE}"
)

print(
    f"Nombre maximum de mouvements : {MAX_STEPS}"
)

print()


for episode in range(1, EPISODES + 1):

    # Choisir aléatoirement la taille du labyrinthe

    taille = random.randint(
        MIN_SIZE,
        MAX_SIZE,
    )

    # Générer un nouveau labyrinthe

    grille = generer_labyrinthe(
        taille
    )

    # Créer l'environnement

    env = LabyrintheEnv(
        grille,
        max_steps=MAX_STEPS,
    )

    # État initial

    position = env.reset()

    state = encoder_etat(
        env.grille,
        position,
    )

    total_reward = 0
    steps = 0
    success = False

    # Boucle de l'épisode

    while steps < MAX_STEPS:

        # Choisir une action
        action = agent.choose_action(
            state
        )

        # Exécuter l'action
        next_position, reward, done = env.step(
            action
        )

        # Encoder le nouvel état
        next_state = encoder_etat(
            env.grille,
            next_position,
        )

        # Mettre à jour Q-Learning
        agent.update(
            state,
            action,
            reward,
            next_state,
            done,
        )

        # Mise à jour
        state = next_state

        total_reward += reward

        steps += 1

        # Vérifier si l'objectif est atteint
        row, col = next_position

        if env.grille[row][col] == "G":
            success = True

        # Fin de l'épisode
        if done:
            break

    # Diminution de epsilon

    agent.decay_epsilon()

    # Sauvegarder les statistiques

    rewards_history.append(
        total_reward
    )

    steps_history.append(
        steps
    )

    successes_history.append(
        1 if success else 0
    )

    # Affichage de progression

    if episode == 1 or episode % 100 == 0:

        recent_rewards = rewards_history[-100:]

        recent_successes = successes_history[-100:]

        average_reward = np.mean(
            recent_rewards
        )

        success_rate = (
            np.mean(recent_successes)
            * 100
        )

        print(
            f"Épisode {episode:4d}/{EPISODES} | "
            f"Taille : {taille}x{taille} | "
            f"Récompense moyenne : "
            f"{average_reward:7.2f} | "
            f"Succès : "
            f"{success_rate:6.2f}% | "
            f"Epsilon : "
            f"{agent.epsilon:.4f}"
        )


# SAUVEGARDE

training_time = (
    time.time()
    - start_time
)


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


# RÉSULTATS FINAUX

average_reward = np.mean(
    rewards_history
)

success_rate = (
    np.mean(successes_history)
    * 100
)

average_steps = np.mean(
    steps_history
)


print()
print("=" * 60)
print("ENTRAÎNEMENT TERMINÉ")
print("=" * 60)

print(
    f"Récompense moyenne : "
    f"{average_reward:.2f}"
)

print(
    f"Taux de réussite : "
    f"{success_rate:.2f}%"
)

print(
    f"Nombre moyen de mouvements : "
    f"{average_steps:.2f}"
)

print(
    f"Temps d'entraînement : "
    f"{training_time:.2f} secondes"
)

print()
print(
    f"Modèle sauvegardé : "
    f"{MODEL_PATH}"
)

print(
    f"Récompenses sauvegardées : "
    f"{REWARDS_PATH}"
)

print(
    f"Mouvements sauvegardés : "
    f"{STEPS_PATH}"
)

print(
    f"Succès sauvegardés : "
    f"{SUCCESSES_PATH}"
)

print("=" * 60)