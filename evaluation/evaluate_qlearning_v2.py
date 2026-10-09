import os
import numpy as np

from environnement import LabyrintheEnv
from models.q_learning import QLearningAgent
from environments.maze_generator import generer_labyrinthe
from environments.qlearning_state_encoder_v2 import (
    encoder_etat_qlearning_v2,
)


# PARAMÈTRES

# Même nombre de tests que pour le DQN
TESTS_PAR_TAILLE = 20

# Tailles des labyrinthes
TAILLES = [5, 6, 7, 8, 9, 10]

# Nombre maximal de mouvements
MAX_STEPS = 200

# Même seed de base que pour le DQN
SEED = 42

# Chemin de la Q-table
MODEL_PATH = (
    "results/qlearning_improved_v2/q_table.npy"
)

# Dossier de sauvegarde
OUTPUT_DIR = "results/generalized"

OUTPUT_PATH = (
    f"{OUTPUT_DIR}/qlearning_v2_evaluation.npy"
)


# ÉVALUATION D'UN LABYRINTHE

def evaluer_labyrinthe(
    agent,
    taille,
    seed,
):
    """
    Évalue le Q-Learning sur un seul labyrinthe.

    Le labyrinthe est généré avec exactement
    la même seed que celle utilisée pour
    l'évaluation du DQN V2.
    """

    # Génération du labyrinthe

    grille = generer_labyrinthe(
        taille,
        seed=seed,
    )

    # Création de l'environnement

    env = LabyrintheEnv(
        grille,
        max_steps=MAX_STEPS,
    )

    position = env.reset()

    # État initial

    derniere_action = 4

    total_reward = 0

    steps = 0

    success = False

    # Parcours du labyrinthe

    for _ in range(MAX_STEPS):

        # Encodage de l'état
        state = encoder_etat_qlearning_v2(
            env.grille,
            position,
            derniere_action,
        )

        # Évaluation sans exploration

        agent.epsilon = 0.0

        action = agent.choose_action(
            state
        )

        # Exécution de l'action

        next_position, reward, done = env.step(
            action
        )

        total_reward += reward

        steps += 1

        # Nouvel état

        position = next_position

        derniere_action = action

        # Vérification du succès

        row, col = position

        if grille[row][col] == "G":

            success = True

            break

        # Fin de l'épisode

        if done:
            break

    return {
        "taille": taille,
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "success": success,
    }


# PROGRAMME PRINCIPAL

def main():

    print("=" * 70)
    print("ÉVALUATION DU Q-LEARNING V2")
    print("=" * 70)

    print()

    total_tests = (
        TESTS_PAR_TAILLE
        * len(TAILLES)
    )

    print(
        f"Nombre de labyrinthes : "
        f"{total_tests}"
    )

    print(
        "Tailles utilisées : "
        "5x5 à 10x10"
    )

    print(
        f"Nombre de tests par taille : "
        f"{TESTS_PAR_TAILLE}"
    )

    print(
        f"Nombre maximal de mouvements : "
        f"{MAX_STEPS}"
    )

    print(
        "Taille de l'état : 12"
    )

    print()

    # VÉRIFICATION DE LA Q-TABLE

    if not os.path.exists(MODEL_PATH):

        print(
            "ERREUR : Q-table introuvable."
        )

        print(
            f"Chemin attendu : {MODEL_PATH}"
        )

        return

    # CRÉATION DE L'AGENT

    agent = QLearningAgent(
        state_size=12,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    # CHARGEMENT DE LA Q-TABLE

    agent.load(
        MODEL_PATH
    )

    # Désactivation de l'exploration
    agent.epsilon = 0.0

    print(
        "✓ Q-table V2 chargée avec succès."
    )

    print(
        f"✓ Nombre d'états appris : "
        f"{len(agent.q_table)}"
    )

    print()

    print(
        "✓ Évaluation avec exactement "
        "les mêmes labyrinthes que le DQN V2."
    )

    print()

    # ÉVALUATION

    resultats = []

    compteur = 0

    for taille in TAILLES:

        for numero_test in range(
            TESTS_PAR_TAILLE
        ):

            compteur += 1

            # MÊME SEED QUE LE DQN

            seed = (
                SEED
                + taille * 1000
                + numero_test
            )

            resultat = evaluer_labyrinthe(
                agent,
                taille,
                seed,
            )

            resultats.append(
                resultat
            )

            # Affichage

            succes_texte = (
                "OUI"
                if resultat["success"]
                else "NON"
            )

            print(
                f"Labyrinthe "
                f"{compteur:3d}/{total_tests} | "
                f"Taille : "
                f"{taille}x{taille} | "
                f"Reward : "
                f"{resultat['reward']:7.1f} | "
                f"Mouvements : "
                f"{resultat['steps']:3d} | "
                f"Succès : "
                f"{succes_texte}"
            )

    # RÉSULTATS GLOBAUX

    nombre_succes = sum(
        resultat["success"]
        for resultat in resultats
    )

    taux_reussite = (
        nombre_succes
        / len(resultats)
        * 100
    )

    reward_moyen = np.mean(
        [
            resultat["reward"]
            for resultat in resultats
        ]
    )

    mouvements_moyens = np.mean(
        [
            resultat["steps"]
            for resultat in resultats
        ]
    )

    # AFFICHAGE GLOBAL

    print()

    print("=" * 70)
    print("RÉSULTATS GLOBAUX")
    print("=" * 70)

    print(
        f"Nombre de labyrinthes évalués : "
        f"{len(resultats)}"
    )

    print(
        f"Nombre de succès : "
        f"{nombre_succes}/{len(resultats)}"
    )

    print(
        f"Taux de réussite : "
        f"{taux_reussite:.2f}%"
    )

    print(
        f"Récompense moyenne : "
        f"{reward_moyen:.2f}"
    )

    print(
        f"Nombre moyen de mouvements : "
        f"{mouvements_moyens:.2f}"
    )

    # RÉSULTATS PAR TAILLE

    print()

    print("=" * 70)
    print("RÉSULTATS PAR TAILLE DE LABYRINTHE")
    print("=" * 70)

    for taille in TAILLES:

        resultats_taille = [
            resultat
            for resultat in resultats
            if resultat["taille"] == taille
        ]

        nombre_succes_taille = sum(
            resultat["success"]
            for resultat in resultats_taille
        )

        taux_reussite_taille = (
            nombre_succes_taille
            / len(resultats_taille)
            * 100
        )

        reward_moyen_taille = np.mean(
            [
                resultat["reward"]
                for resultat in resultats_taille
            ]
        )

        mouvements_moyens_taille = np.mean(
            [
                resultat["steps"]
                for resultat in resultats_taille
            ]
        )

        print(
            f"{taille}x{taille} | "
            f"Tests : "
            f"{len(resultats_taille):2d} | "
            f"Réussite : "
            f"{taux_reussite_taille:6.2f}% | "
            f"Reward moyen : "
            f"{reward_moyen_taille:7.2f} | "
            f"Mouvements moyens : "
            f"{mouvements_moyens_taille:6.2f}"
        )

    # SAUVEGARDE

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    np.save(
        OUTPUT_PATH,
        np.array(
            resultats,
            dtype=object,
        ),
        allow_pickle=True,
    )

    print()

    print(
        f"✓ Résultats sauvegardés dans : "
        f"{OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)


# LANCEMENT

if __name__ == "__main__":
    main()