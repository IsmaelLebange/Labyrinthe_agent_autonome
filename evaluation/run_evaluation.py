import numpy as np

from models.q_learning import QLearningAgent
from models.dqn import DQNAgent

from evaluation.evaluate import evaluate_qlearning, evaluate_dqn
def main():

    print("=" * 60)
    print("ÉVALUATION DES AGENTS")
    print("=" * 60)

    # 1. Création du Q-learning

    qlearning_agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    # Chargement de la Q-table entraînée
    qlearning_agent.load(
        "results/qlearning/q_table.npy"
    )

    print("\n✓ Q-table chargée avec succès.")

    # 2. Création du DQN

    dqn_agent = DQNAgent()

    # Chargement du réseau entraîné
    dqn_agent.load(
        "results/dqn/dqn_state_dict.pth"
    )

    print("✓ Modèle DQN chargé avec succès.")

    # 3. Évaluation du Q-learning

    print("\n" + "-" * 60)
    print("ÉVALUATION Q-LEARNING")
    print("-" * 60)

    qlearning_results = evaluate_qlearning(
        qlearning_agent,
        episodes=100,
        max_steps=200,
    )

    print(
        f"Récompense moyenne : "
        f"{qlearning_results['average_reward']:.2f}"
    )

    print(
        f"Taux de réussite : "
        f"{qlearning_results['success_rate']:.2f}%"
    )

    print(
        f"Nombre moyen de mouvements : "
        f"{qlearning_results['average_steps']:.2f}"
    )

    print(
        f"Temps d'évaluation : "
        f"{qlearning_results['evaluation_time']:.4f} secondes"
    )

    # 4. Évaluation du DQN

    print("\n" + "-" * 60)
    print("ÉVALUATION DQN")
    print("-" * 60)

    dqn_results = evaluate_dqn(
        dqn_agent,
        episodes=100,
        max_steps=200,
    )

    print(
        f"Récompense moyenne : "
        f"{dqn_results['average_reward']:.2f}"
    )

    print(
        f"Taux de réussite : "
        f"{dqn_results['success_rate']:.2f}%"
    )

    print(
        f"Nombre moyen de mouvements : "
        f"{dqn_results['average_steps']:.2f}"
    )

    print(
        f"Temps d'évaluation : "
        f"{dqn_results['evaluation_time']:.4f} secondes"
    )

    # 5. Comparaison finale

    print("\n" + "=" * 60)
    print("COMPARAISON Q-LEARNING VS DQN")
    print("=" * 60)

    print(
        f"{'Métrique':<30}"
        f"{'Q-learning':>15}"
        f"{'DQN':>15}"
    )

    print("-" * 60)

    print(
        f"{'Récompense moyenne':<30}"
        f"{qlearning_results['average_reward']:>15.2f}"
        f"{dqn_results['average_reward']:>15.2f}"
    )

    print(
        f"{'Taux de réussite (%)':<30}"
        f"{qlearning_results['success_rate']:>15.2f}"
        f"{dqn_results['success_rate']:>15.2f}"
    )

    print(
        f"{'Mouvements moyens':<30}"
        f"{qlearning_results['average_steps']:>15.2f}"
        f"{dqn_results['average_steps']:>15.2f}"
    )

    print(
        f"{'Temps évaluation (s)':<30}"
        f"{qlearning_results['evaluation_time']:>15.4f}"
        f"{dqn_results['evaluation_time']:>15.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()