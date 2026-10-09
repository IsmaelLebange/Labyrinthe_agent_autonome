import numpy as np


def calculate_success_rate(successes):
    """
    Calcule le taux de réussite en pourcentage.
    """

    if len(successes) == 0:
        return 0.0

    return np.mean(successes) * 100


def calculate_average_reward(rewards):
    """
    Calcule la récompense moyenne.
    """

    if len(rewards) == 0:
        return 0.0

    return np.mean(rewards)


def calculate_average_steps(steps):
    """
    Calcule le nombre moyen de déplacements.
    """

    if len(steps) == 0:
        return 0.0

    return np.mean(steps)


def calculate_metrics(rewards, steps, successes):
    """
    Calcule toutes les métriques principales.
    """

    return {
        "success_rate": calculate_success_rate(successes),
        "average_reward": calculate_average_reward(rewards),
        "average_steps": calculate_average_steps(steps),
    }