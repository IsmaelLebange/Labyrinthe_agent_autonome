import matplotlib.pyplot as plt


def plot_rewards(rewards, save_path):
    """
    Affiche l'évolution des récompenses
    au cours de l'entraînement.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(rewards)

    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.title("Évolution de la récompense - Q-learning")

    plt.grid(True)

    plt.savefig(save_path)

    plt.close()


def plot_success_rate(successes, save_path, window=100):
    """
    Affiche l'évolution du taux de réussite.

    Le taux est calculé sur une fenêtre glissante.
    """

    success_rates = []

    for i in range(len(successes)):

        start = max(0, i - window + 1)

        rate = sum(
            successes[start:i + 1]
        ) / (i - start + 1)

        success_rates.append(rate * 100)

    plt.figure(figsize=(10, 5))

    plt.plot(success_rates)

    plt.xlabel("Épisode")
    plt.ylabel("Taux de réussite (%)")
    plt.title("Évolution du taux de réussite")

    plt.grid(True)

    plt.savefig(save_path)

    plt.close()