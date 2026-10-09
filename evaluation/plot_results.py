import os
import numpy as np
import matplotlib.pyplot as plt


# Chargement des résultats

qlearning_rewards = np.load(
    "results/qlearning/rewards.npy"
)

qlearning_steps = np.load(
    "results/qlearning/steps.npy"
)

qlearning_successes = np.load(
    "results/qlearning/successes.npy"
)

dqn_rewards = np.load(
    "results/dqn/rewards.npy"
)

dqn_steps = np.load(
    "results/dqn/steps.npy"
)

dqn_successes = np.load(
    "results/dqn/successes.npy"
)

dqn_losses = np.load(
    "results/dqn/losses.npy"
)


# Création du dossier des graphiques

os.makedirs("results/plots", exist_ok=True)


# 1. Récompenses Q-learning

plt.figure(figsize=(10, 5))

plt.plot(qlearning_rewards)

plt.title("Évolution de la récompense - Q-learning")
plt.xlabel("Épisode")
plt.ylabel("Récompense")

plt.grid(True)

plt.savefig(
    "results/plots/qlearning_rewards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 2. Récompenses DQN

plt.figure(figsize=(10, 5))

plt.plot(dqn_rewards)

plt.title("Évolution de la récompense - DQN")
plt.xlabel("Épisode")
plt.ylabel("Récompense")

plt.grid(True)

plt.savefig(
    "results/plots/dqn_rewards.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 3. Comparaison des récompenses

plt.figure(figsize=(10, 5))

plt.plot(
    qlearning_rewards,
    label="Q-learning"
)

plt.plot(
    dqn_rewards,
    label="DQN"
)

plt.title("Comparaison des récompenses")
plt.xlabel("Épisode")
plt.ylabel("Récompense")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/plots/rewards_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 4. Comparaison du nombre de mouvements

plt.figure(figsize=(10, 5))

plt.plot(
    qlearning_steps,
    label="Q-learning"
)

plt.plot(
    dqn_steps,
    label="DQN"
)

plt.title("Nombre de mouvements par épisode")
plt.xlabel("Épisode")
plt.ylabel("Nombre de mouvements")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/plots/steps_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 5. Taux de réussite

qlearning_success_rate = (
    np.mean(qlearning_successes) * 100
)

dqn_success_rate = (
    np.mean(dqn_successes) * 100
)

models = [
    "Q-learning",
    "DQN"
]

success_rates = [
    qlearning_success_rate,
    dqn_success_rate
]

plt.figure(figsize=(8, 5))

plt.bar(
    models,
    success_rates
)

plt.title("Taux de réussite")
plt.ylabel("Taux de réussite (%)")

plt.ylim(0, 100)

plt.grid(
    axis="y"
)

plt.savefig(
    "results/plots/success_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 6. Perte du DQN

# On retire les valeurs None éventuelles
dqn_losses = np.array(
    [
        loss for loss in dqn_losses
        if loss is not None
    ],
    dtype=float
)

if len(dqn_losses) > 0:

    plt.figure(figsize=(10, 5))

    plt.plot(dqn_losses)

    plt.title("Évolution de la perte - DQN")
    plt.xlabel("Mise à jour")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig(
        "results/plots/dqn_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Résumé

print()
print("=" * 60)
print("GRAPHIQUES GÉNÉRÉS")
print("=" * 60)

print("✓ Q-learning rewards")
print("✓ DQN rewards")
print("✓ Comparaison des récompenses")
print("✓ Comparaison des mouvements")
print("✓ Taux de réussite")
print("✓ Perte du DQN")

print()
print("Les graphiques sont enregistrés dans :")
print("results/plots/")