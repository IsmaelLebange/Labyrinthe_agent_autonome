import json
import os

from evaluation.evaluate import (
    evaluate_qlearning,
    evaluate_dqn,
)


print("=" * 70)
print("ÉVALUATION DES MODÈLES GÉNÉRALISÉS")
print("=" * 70)


# Q-LEARNING

print()
print("-" * 70)
print("Q-LEARNING")
print("-" * 70)

qlearning_results = evaluate_qlearning()

print()

for taille, result in qlearning_results.items():
    print(
        f"{taille}x{taille} | "
        f"Récompense : {result['reward']:8.2f} | "
        f"Mouvements : {result['steps']:7.2f} | "
        f"Succès : {result['success_rate']:6.2f}%"
    )


# DQN

print()
print("-" * 70)
print("DQN")
print("-" * 70)

dqn_results = evaluate_dqn()

print()

for taille, result in dqn_results.items():
    print(
        f"{taille}x{taille} | "
        f"Récompense : {result['reward']:8.2f} | "
        f"Mouvements : {result['steps']:7.2f} | "
        f"Succès : {result['success_rate']:6.2f}%"
    )


# SAUVEGARDE

results = {
    "qlearning": qlearning_results,
    "dqn": dqn_results,
}

os.makedirs(
    "results/generalized",
    exist_ok=True,
)

with open(
    "results/generalized/evaluation_results.json",
    "w",
    encoding="utf-8",
) as fichier:

    json.dump(
        results,
        fichier,
        indent=4,
    )


print()
print("=" * 70)
print("ÉVALUATION TERMINÉE")
print("=" * 70)

print()
print(
    "Résultats sauvegardés dans : "
    "results/generalized/evaluation_results.json"
)