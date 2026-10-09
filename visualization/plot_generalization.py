import json
import os

import matplotlib.pyplot as plt


# CONFIGURATION

RESULTS_FILE = (
    "results/generalized/evaluation_results.json"
)

OUTPUT_DIR = "results/graphs/generalization"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# CHARGEMENT DES DONNÉES

with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8",
) as fichier:

    results = json.load(fichier)


qlearning = results["qlearning"]
dqn = results["dqn"]


sizes = [
    int(taille)
    for taille in qlearning.keys()
]

labels = [
    f"{taille}x{taille}"
    for taille in sizes
]


q_rewards = [
    qlearning[str(taille)]["reward"]
    for taille in sizes
]

dqn_rewards = [
    dqn[str(taille)]["reward"]
    for taille in sizes
]


q_steps = [
    qlearning[str(taille)]["steps"]
    for taille in sizes
]

dqn_steps = [
    dqn[str(taille)]["steps"]
    for taille in sizes
]


q_success = [
    qlearning[str(taille)]["success_rate"]
    for taille in sizes
]

dqn_success = [
    dqn[str(taille)]["success_rate"]
    for taille in sizes
]


# 1. TAUX DE RÉUSSITE

plt.figure(figsize=(10, 6))

plt.plot(
    labels,
    q_success,
    marker="o",
    linewidth=2,
    label="Q-Learning",
)

plt.plot(
    labels,
    dqn_success,
    marker="o",
    linewidth=2,
    label="DQN",
)

plt.xlabel("Taille du labyrinthe")
plt.ylabel("Taux de réussite (%)")
plt.title(
    "Généralisation : taux de réussite"
)

plt.ylim(0, 105)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "generalization_success_rate.png",
    ),
    dpi=300,
)

plt.close()


# 2. RÉCOMPENSE MOYENNE

plt.figure(figsize=(10, 6))

plt.plot(
    labels,
    q_rewards,
    marker="o",
    linewidth=2,
    label="Q-Learning",
)

plt.plot(
    labels,
    dqn_rewards,
    marker="o",
    linewidth=2,
    label="DQN",
)

plt.xlabel("Taille du labyrinthe")
plt.ylabel("Récompense moyenne")
plt.title(
    "Généralisation : récompense moyenne"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "generalization_rewards.png",
    ),
    dpi=300,
)

plt.close()


# 3. NOMBRE MOYEN DE MOUVEMENTS

plt.figure(figsize=(10, 6))

plt.plot(
    labels,
    q_steps,
    marker="o",
    linewidth=2,
    label="Q-Learning",
)

plt.plot(
    labels,
    dqn_steps,
    marker="o",
    linewidth=2,
    label="DQN",
)

plt.xlabel("Taille du labyrinthe")
plt.ylabel("Nombre moyen de mouvements")
plt.title(
    "Généralisation : nombre moyen de mouvements"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "generalization_steps.png",
    ),
    dpi=300,
)

plt.close()


# 4. COMPARAISON GLOBALE DU TAUX DE RÉUSSITE

q_average_success = sum(q_success) / len(q_success)

dqn_average_success = sum(dqn_success) / len(dqn_success)


plt.figure(figsize=(8, 6))

plt.bar(
    ["Q-Learning", "DQN"],
    [
        q_average_success,
        dqn_average_success,
    ],
)

plt.ylabel("Taux de réussite moyen (%)")

plt.title(
    "Taux de réussite moyen sur les nouveaux labyrinthes"
)

plt.ylim(0, 105)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "generalization_global_success.png",
    ),
    dpi=300,
)

plt.close()


# 5. RÉSUMÉ

print("=" * 70)
print("GRAPHIQUES DE GÉNÉRALISATION")
print("=" * 70)

print()

print(
    f"Taux de réussite moyen Q-Learning : "
    f"{q_average_success:.2f}%"
)

print(
    f"Taux de réussite moyen DQN        : "
    f"{dqn_average_success:.2f}%"
)

print()

print(
    "Graphiques sauvegardés dans :"
)

print(
    f"  {OUTPUT_DIR}"
)

print()

print("=" * 70)
print("TERMINÉ")
print("=" * 70)