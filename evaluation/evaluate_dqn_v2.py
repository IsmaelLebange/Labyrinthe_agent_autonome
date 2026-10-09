
import os
import random

import numpy as np
import torch

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe
from environments.qlearning_state_encoder_v2 import (
    encoder_etat_qlearning_v2,
)
from models.dqn import DQNNetwork


# PARAMÈTRES

MODEL_PATH = "results/dqn_v2/model.pth"

# 20 labyrinthes pour chaque taille
TESTS_PAR_TAILLE = 20

# Tailles testées
MIN_SIZE = 5
MAX_SIZE = 10

# Nombre maximal de mouvements
MAX_STEPS = 200

# Taille de l'état
STATE_SIZE = 12

# Nombre d'actions
ACTION_SIZE = 4

# Graine pour rendre l'évaluation reproductible
SEED = 42


# PRÉPARATION

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# CHARGEMENT DU MODÈLE

print("=" * 70)
print("ÉVALUATION DU DQN V2")
print("=" * 70)

print()

print(
    f"Nombre de labyrinthes : "
    f"{TESTS_PAR_TAILLE * (MAX_SIZE - MIN_SIZE + 1)}"
)

print(
    f"Tailles utilisées : "
    f"{MIN_SIZE}x{MIN_SIZE} à "
    f"{MAX_SIZE}x{MAX_SIZE}"
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
    f"Taille de l'état : "
    f"{STATE_SIZE}"
)

print()


# CRÉATION DU RÉSEAU

model = DQNNetwork(
    input_size=STATE_SIZE,
    output_size=ACTION_SIZE,
)


# CHARGEMENT DES POIDS

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
    )
)

model.eval()


print(
    "✓ Modèle DQN V2 chargé avec succès."
)

print()


# FONCTION DE CHOIX DE L'ACTION

def choisir_action_deterministe(
    model,
    etat,
):
    """
    Choisit l'action ayant la plus grande
    valeur Q.

    Pendant l'évaluation, aucune exploration
    aléatoire n'est utilisée.
    """

    etat_array = np.array(
        etat,
        dtype=np.float32,
    )

    etat_tensor = torch.tensor(
        etat_array,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model(
            etat_tensor
        )

    action = torch.argmax(
        q_values,
        dim=1,
    ).item()

    return action


# ÉVALUATION D'UN LABYRINTHE

def evaluer_labyrinthe(
    model,
    taille,
    seed,
):
    """
    Évalue le DQN sur un seul labyrinthe.
    """

    # Génération reproductible

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

    etat = encoder_etat_qlearning_v2(
        env.grille,
        position,
        derniere_action,
    )

    total_reward = 0

    steps = 0

    success = False

    # Épisode

    for _ in range(MAX_STEPS):

        # Choix déterministe
        action = choisir_action_deterministe(
            model,
            etat,
        )

        # Exécution
        next_position, reward, done = env.step(
            action
        )

        # Nouvel état
        next_etat = encoder_etat_qlearning_v2(
            env.grille,
            next_position,
            action,
        )

        # Mise à jour
        etat = next_etat

        total_reward += reward

        steps += 1

        # Vérification du succès

        if (
            env.grille[
                env.position[0]
            ][
                env.position[1]
            ] == "G"
        ):

            success = True

        # Fin de l'épisode

        if done:
            break

    return {
        "taille": taille,
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "seed": seed,
    }


# ÉVALUATION

resultats = []


compteur = 0

total_tests = (
    TESTS_PAR_TAILLE
    * (MAX_SIZE - MIN_SIZE + 1)
)


for taille in range(
    MIN_SIZE,
    MAX_SIZE + 1,
):

    for numero_test in range(
        TESTS_PAR_TAILLE
    ):

        compteur += 1

        # Seed unique pour chaque labyrinthe

        seed = (
            SEED
            + taille * 1000
            + numero_test
        )

        resultat = evaluer_labyrinthe(
            model,
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


# AFFICHAGE DES RÉSULTATS GLOBAUX

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


for taille in range(
    MIN_SIZE,
    MAX_SIZE + 1,
):

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


# FIN

print()

print("=" * 70)
print("ÉVALUATION TERMINÉE")
print("=" * 70)