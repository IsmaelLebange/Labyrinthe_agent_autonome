import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "generalized",
    "all_models_evaluation.npy",
)
OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "graphs",
    "four_models_comparison.png",
)

MODELS = {
    "qlearning_original": "Q-Learning original",
    "qlearning_v2": "Q-Learning V2",
    "dqn_original": "DQN original",
    "dqn_v2": "DQN V2",
}
METRICS = [
    ("success", "Taux de réussite (%)"),
    ("reward", "Récompense moyenne"),
    ("steps", "Nombre moyen de mouvements"),
]


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Résultats d'évaluation introuvables : {RESULTS_PATH}. "
            "Lancez evaluation/evaluate_all_models.py d'abord."
        )

    results = np.load(RESULTS_PATH, allow_pickle=True).item()
    sizes = sorted(results["qlearning_original"])
    labels = [f"{size}x{size}" for size in sizes]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]
    markers = ["o", "s", "^", "D"]

    for axis, (metric, ylabel) in zip(axes, METRICS):
        for (model_key, model_name), color, marker in zip(
            MODELS.items(),
            colors,
            markers,
        ):
            values = [
                results[model_key][size][metric]
                for size in sizes
            ]
            axis.plot(
                labels,
                values,
                label=model_name,
                color=color,
                marker=marker,
                linewidth=2,
            )

        axis.set_title(ylabel)
        axis.set_xlabel("Taille du labyrinthe")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

        if metric == "success":
            axis.set_ylim(0, 105)

    figure.suptitle("Comparaison des quatre modèles")
    figure.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Graphique des quatre modèles enregistré dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
