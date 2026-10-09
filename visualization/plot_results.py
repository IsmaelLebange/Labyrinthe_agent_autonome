import os

import numpy as np
import matplotlib.pyplot as plt


# CONFIGURATION

QLEARNING_DIR = "results/qlearning"
DQN_DIR = "results/dqn"
OUTPUT_DIR = "results/graphs"

WINDOW = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


# CHARGEMENT DES RESULTATS

print("=" * 60)
print("CHARGEMENT DES RESULTATS")
print("=" * 60)

q_rewards = np.load(
    os.path.join(QLEARNING_DIR, "rewards.npy")
)

q_steps = np.load(
    os.path.join(QLEARNING_DIR, "steps.npy")
)

q_successes = np.load(
    os.path.join(QLEARNING_DIR, "successes.npy")
)

dqn_rewards = np.load(
    os.path.join(DQN_DIR, "rewards.npy")
)

dqn_steps = np.load(
    os.path.join(DQN_DIR, "steps.npy")
)

dqn_successes = np.load(
    os.path.join(DQN_DIR, "successes.npy")
)

print("✓ Résultats Q-Learning chargés.")
print("✓ Résultats DQN chargés.")


# MOYENNE PAR BLOCS

def moyenne_par_blocs(values, window=100):
    moyennes = []

    for debut in range(0, len(values), window):
        fin = debut + window
        bloc = values[debut:fin]

        if len(bloc) > 0:
            moyennes.append(np.mean(bloc))

    return np.array(moyennes)


q_rewards_avg = moyenne_par_blocs(q_rewards, WINDOW)
dqn_rewards_avg = moyenne_par_blocs(dqn_rewards, WINDOW)

q_steps_avg = moyenne_par_blocs(q_steps, WINDOW)
dqn_steps_avg = moyenne_par_blocs(dqn_steps, WINDOW)

q_success_avg = moyenne_par_blocs(
    q_successes,
    WINDOW
) * 100

dqn_success_avg = moyenne_par_blocs(
    dqn_successes,
    WINDOW
) * 100


episodes_blocks = np.arange(
    WINDOW,
    len(q_rewards) + 1,
    WINDOW
)


# GRAPHIQUE 1
# RECOMPENSES

plt.figure(figsize=(10, 6))

plt.plot(
    episodes_blocks,
    q_rewards_avg,
    marker="o",
    label="Q-Learning"
)

plt.plot(
    episodes_blocks,
    dqn_rewards_avg,
    marker="o",
    label="DQN"
)

plt.title("Évolution de la récompense moyenne")
plt.xlabel("Nombre d'épisodes")
plt.ylabel("Récompense moyenne")

plt.grid(True)
plt.legend()

plt.tight_layout()

reward_path = os.path.join(
    OUTPUT_DIR,
    "rewards_comparison.png"
)

plt.savefig(
    reward_path,
    dpi=300
)

plt.close()

print("✓ rewards_comparison.png")


# GRAPHIQUE 2
# TAUX DE REUSSITE

plt.figure(figsize=(10, 6))

plt.plot(
    episodes_blocks,
    q_success_avg,
    marker="o",
    label="Q-Learning"
)

plt.plot(
    episodes_blocks,
    dqn_success_avg,
    marker="o",
    label="DQN"
)

plt.title("Évolution du taux de réussite")
plt.xlabel("Nombre d'épisodes")
plt.ylabel("Taux de réussite (%)")

plt.ylim(0, 105)

plt.grid(True)
plt.legend()

plt.tight_layout()

success_path = os.path.join(
    OUTPUT_DIR,
    "success_rate_comparison.png"
)

plt.savefig(
    success_path,
    dpi=300
)

plt.close()

print("✓ success_rate_comparison.png")


# GRAPHIQUE 3
# NOMBRE DE MOUVEMENTS

plt.figure(figsize=(10, 6))

plt.plot(
    episodes_blocks,
    q_steps_avg,
    marker="o",
    label="Q-Learning"
)

plt.plot(
    episodes_blocks,
    dqn_steps_avg,
    marker="o",
    label="DQN"
)

plt.title("Évolution du nombre moyen de mouvements")
plt.xlabel("Nombre d'épisodes")
plt.ylabel("Nombre moyen de mouvements")

plt.grid(True)
plt.legend()

plt.tight_layout()

steps_path = os.path.join(
    OUTPUT_DIR,
    "steps_comparison.png"
)

plt.savefig(
    steps_path,
    dpi=300
)

plt.close()

print("✓ steps_comparison.png")


# GRAPHIQUE 4
# TEMPS D'ENTRAINEMENT

q_training_time = 1.02
dqn_training_time = 35.41

models = [
    "Q-Learning",
    "DQN"
]

training_times = [
    q_training_time,
    dqn_training_time
]

plt.figure(figsize=(8, 6))

plt.bar(
    models,
    training_times
)

plt.title("Temps d'entraînement des modèles")
plt.xlabel("Modèle")
plt.ylabel("Temps d'entraînement (secondes)")

plt.grid(axis="y")

plt.tight_layout()

training_time_path = os.path.join(
    OUTPUT_DIR,
    "training_time_comparison.png"
)

plt.savefig(
    training_time_path,
    dpi=300
)

plt.close()

print("✓ training_time_comparison.png")


# AFFICHAGE DES 4 GRAPHIQUES

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)


# Récompenses

axes[0, 0].plot(
    episodes_blocks,
    q_rewards_avg,
    marker="o",
    label="Q-Learning"
)

axes[0, 0].plot(
    episodes_blocks,
    dqn_rewards_avg,
    marker="o",
    label="DQN"
)

axes[0, 0].set_title(
    "Récompense moyenne"
)

axes[0, 0].set_xlabel(
    "Épisodes"
)

axes[0, 0].set_ylabel(
    "Récompense"
)

axes[0, 0].grid(True)
axes[0, 0].legend()


# Taux de réussite

axes[0, 1].plot(
    episodes_blocks,
    q_success_avg,
    marker="o",
    label="Q-Learning"
)

axes[0, 1].plot(
    episodes_blocks,
    dqn_success_avg,
    marker="o",
    label="DQN"
)

axes[0, 1].set_title(
    "Taux de réussite"
)

axes[0, 1].set_xlabel(
    "Épisodes"
)

axes[0, 1].set_ylabel(
    "Réussite (%)"
)

axes[0, 1].set_ylim(0, 105)

axes[0, 1].grid(True)
axes[0, 1].legend()


# Mouvements

axes[1, 0].plot(
    episodes_blocks,
    q_steps_avg,
    marker="o",
    label="Q-Learning"
)

axes[1, 0].plot(
    episodes_blocks,
    dqn_steps_avg,
    marker="o",
    label="DQN"
)

axes[1, 0].set_title(
    "Nombre moyen de mouvements"
)

axes[1, 0].set_xlabel(
    "Épisodes"
)

axes[1, 0].set_ylabel(
    "Mouvements"
)

axes[1, 0].grid(True)
axes[1, 0].legend()


# Temps d'entraînement

axes[1, 1].bar(
    models,
    training_times
)

axes[1, 1].set_title(
    "Temps d'entraînement"
)

axes[1, 1].set_xlabel(
    "Modèle"
)

axes[1, 1].set_ylabel(
    "Secondes"
)

axes[1, 1].grid(axis="y")


# SAUVEGARDE DU TABLEAU COMPLET

plt.suptitle(
    "Comparaison Q-Learning vs DQN",
    fontsize=16
)

plt.tight_layout()

all_graphs_path = os.path.join(
    OUTPUT_DIR,
    "all_results.png"
)

plt.savefig(
    all_graphs_path,
    dpi=300
)

print()
print("=" * 60)
print("GRAPHIQUES GENERES")
print("=" * 60)

print(
    f"✓ {reward_path}"
)

print(
    f"✓ {success_path}"
)

print(
    f"✓ {steps_path}"
)

print(
    f"✓ {training_time_path}"
)

print(
    f"✓ {all_graphs_path}"
)

print("=" * 60)

# Afficher les 4 graphiques ensemble
plt.show()