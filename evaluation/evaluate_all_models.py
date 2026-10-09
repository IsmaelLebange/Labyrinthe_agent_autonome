import os
import numpy as np

from environnement import LabyrintheEnv

from models.q_learning import QLearningAgent
from models.dqn import DQNAgent, DQNNetwork

from environments.maze_generator import generer_labyrinthe

from environments.qlearning_state_encoder_v2 import (
    encoder_etat_qlearning_v2,
)

from environments.state_encoder import encoder_etat


# PARAMÈTRES

EPISODES_PAR_TAILLE = 100

MAX_STEPS = 200

TAILLES = [5, 6, 7, 8, 9, 10]

SEED_BASE = 50000


# CHEMINS DES MODÈLES

QLEARNING_PATH = (
    "results/qlearning/q_table.npy"
)

QLEARNING_V2_PATH = (
    "results/qlearning_improved_v2/q_table.npy"
)

DQN_PATH = (
    "results/dqn/dqn_state_dict.pth"
)

DQN_V2_PATH = (
    "results/dqn_v2/model.pth"
)


# DOSSIER DE SORTIE

OUTPUT_DIR = "results/generalized"


# GÉNÉRATION DES LABYRINTHES DE TEST

def generer_labyrinthes_test():

    """
    Génère les 600 labyrinthes utilisés
    pour l'évaluation des quatre modèles.

    Chaque taille possède 100 labyrinthes.

    Les quatre modèles utiliseront exactement
    les mêmes labyrinthes.
    """

    labyrinthes = {}

    print()
    print("=" * 70)
    print("GÉNÉRATION DES LABYRINTHES DE TEST")
    print("=" * 70)

    for taille in TAILLES:

        labyrinthes[taille] = []

        print(
            f"Génération des labyrinthes {taille}x{taille}..."
        )

        for episode in range(
            EPISODES_PAR_TAILLE
        ):

            # Seed fixe
            seed = (
                SEED_BASE
                + taille * 1000
                + episode
            )

            grille = generer_labyrinthe(
                taille,
                seed=seed,
            )

            labyrinthes[taille].append(
                grille
            )

    nombre_total = (
        EPISODES_PAR_TAILLE
        * len(TAILLES)
    )

    print()
    print(
        f"✓ {nombre_total} labyrinthes générés."
    )

    print(
        "✓ Les quatre modèles utiliseront "
        "exactement les mêmes labyrinthes."
    )

    return labyrinthes

# ÉVALUATION Q-LEARNING ORIGINAL

def evaluer_qlearning(
    agent,
    labyrinthes,
):

    resultats = {}

    agent.epsilon = 0.0

    for taille in TAILLES:

        rewards = []
        steps_list = []
        successes = []

        for grille in labyrinthes[taille]:

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS,
            )

            position = env.reset()

            total_reward = 0

            success = False

            for step in range(MAX_STEPS):

                # Q-Learning original :
                # l'état est directement la position.
                state = position

                action = agent.choose_action(
                    state
                )

                (
                    next_position,
                    reward,
                    done,
                ) = env.step(action)

                total_reward += reward

                position = next_position

                row, col = position

                if grille[row][col] == "G":

                    success = True

                    break

                if done:

                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                step + 1
            )

            successes.append(
                1 if success else 0
            )

        resultats[taille] = {

            "reward": float(
                np.mean(rewards)
            ),

            "steps": float(
                np.mean(steps_list)
            ),

            "success": float(
                np.mean(successes)
                * 100
            ),
        }

    return resultats


# ÉVALUATION Q-LEARNING V2

def evaluer_qlearning_v2(
    agent,
    labyrinthes,
):

    resultats = {}

    agent.epsilon = 0.0

    for taille in TAILLES:

        rewards = []
        steps_list = []
        successes = []

        for grille in labyrinthes[taille]:

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS,
            )

            position = env.reset()

            derniere_action = 4

            total_reward = 0

            success = False

            for step in range(MAX_STEPS):

                state = (
                    encoder_etat_qlearning_v2(
                        grille,
                        position,
                        derniere_action,
                    )
                )

                action = agent.choose_action(
                    state
                )

                (
                    next_position,
                    reward,
                    done,
                ) = env.step(action)

                total_reward += reward

                position = next_position

                derniere_action = action

                row, col = position

                if grille[row][col] == "G":

                    success = True

                    break

                if done:

                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                step + 1
            )

            successes.append(
                1 if success else 0
            )

        resultats[taille] = {

            "reward": float(
                np.mean(rewards)
            ),

            "steps": float(
                np.mean(steps_list)
            ),

            "success": float(
                np.mean(successes)
                * 100
            ),
        }

    return resultats

# ÉVALUATION DQN ORIGINAL

def evaluer_dqn(
    agent,
    labyrinthes,
):

    resultats = {}

    agent.epsilon = 0.0

    for taille in TAILLES:

        rewards = []
        steps_list = []
        successes = []

        for grille in labyrinthes[taille]:

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS,
            )

            position = env.reset()

            total_reward = 0

            success = False

            for step in range(MAX_STEPS):

                # DQN original :
                # utilise l'ancien encodeur.
                state = encoder_etat(
                    grille,
                    position,
                )

                action = agent.choose_action(
                    state
                )

                (
                    next_position,
                    reward,
                    done,
                ) = env.step(action)

                total_reward += reward

                position = next_position

                row, col = position

                if grille[row][col] == "G":

                    success = True

                    break

                if done:

                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                step + 1
            )

            successes.append(
                1 if success else 0
            )

        resultats[taille] = {

            "reward": float(
                np.mean(rewards)
            ),

            "steps": float(
                np.mean(steps_list)
            ),

            "success": float(
                np.mean(successes)
                * 100
            ),
        }

    return resultats


# ÉVALUATION DQN V2

def evaluer_dqn_v2(
    agent,
    labyrinthes,
):

    resultats = {}

    agent.epsilon = 0.0

    for taille in TAILLES:

        rewards = []
        steps_list = []
        successes = []

        for grille in labyrinthes[taille]:

            env = LabyrintheEnv(
                grille,
                max_steps=MAX_STEPS,
            )

            position = env.reset()

            derniere_action = 4

            total_reward = 0

            success = False

            for step in range(MAX_STEPS):

                # DQN V2 utilise exactement
                # le même encodeur que Q-Learning V2.
                state = (
                    encoder_etat_qlearning_v2(
                        grille,
                        position,
                        derniere_action,
                    )
                )

                action = agent.choose_action(
                    state
                )

                (
                    next_position,
                    reward,
                    done,
                ) = env.step(action)

                total_reward += reward

                position = next_position

                derniere_action = action

                row, col = position

                if grille[row][col] == "G":

                    success = True

                    break

                if done:

                    break

            rewards.append(
                total_reward
            )

            steps_list.append(
                step + 1
            )

            successes.append(
                1 if success else 0
            )

        resultats[taille] = {

            "reward": float(
                np.mean(rewards)
            ),

            "steps": float(
                np.mean(steps_list)
            ),

            "success": float(
                np.mean(successes)
                * 100
            ),
        }

    return resultats

# AFFICHAGE DES RÉSULTATS

def afficher_resultats(
    nom,
    resultats,
):

    print()
    print("=" * 70)
    print(nom)
    print("=" * 70)

    print(
        f"{'Taille':<12}"
        f"{'Récompense':<18}"
        f"{'Mouvements':<18}"
        f"{'Réussite':<12}"
    )

    print("-" * 70)

    for taille in TAILLES:

        resultat = resultats[taille]

        print(
            f"{taille}x{taille:<8}"
            f"{resultat['reward']:<18.2f}"
            f"{resultat['steps']:<18.2f}"
            f"{resultat['success']:.2f}%"
        )


# CALCUL DES MOYENNES GLOBALES

def calculer_moyennes(
    resultats,
):

    rewards = [
        resultats[taille]["reward"]
        for taille in TAILLES
    ]

    steps = [
        resultats[taille]["steps"]
        for taille in TAILLES
    ]

    successes = [
        resultats[taille]["success"]
        for taille in TAILLES
    ]

    return {

        "reward": float(
            np.mean(rewards)
        ),

        "steps": float(
            np.mean(steps)
        ),

        "success": float(
            np.mean(successes)
        ),
    }


# VÉRIFICATION DES FICHIERS

def verifier_modeles():

    chemins = {

        "Q-Learning original":
            QLEARNING_PATH,

        "Q-Learning V2":
            QLEARNING_V2_PATH,

        "DQN original":
            DQN_PATH,

        "DQN V2":
            DQN_V2_PATH,
    }

    print()
    print("=" * 70)
    print("VÉRIFICATION DES MODÈLES")
    print("=" * 70)

    tous_presents = True

    for nom, chemin in chemins.items():

        if os.path.exists(chemin):

            print(
                f"✓ {nom:<25} : {chemin}"
            )

        else:

            print(
                f"✗ {nom:<25} : INTROUVABLE"
            )

            tous_presents = False

    if not tous_presents:

        print()
        print(
            "ERREUR : un ou plusieurs modèles "
            "sont introuvables."
        )

        return False

    print()
    print(
        "✓ Les quatre modèles sont disponibles."
    )

    return True


# PROGRAMME PRINCIPAL

def main():

    print()
    print("=" * 70)
    print("ÉVALUATION FINALE DES 4 MODÈLES")
    print("=" * 70)

    print()
    print("Modèles évalués :")

    print(
        "1. Q-Learning original"
    )

    print(
        "2. Q-Learning V2"
    )

    print(
        "3. DQN original"
    )

    print(
        "4. DQN V2"
    )

    print()
    print(
        f"Tests par taille : "
        f"{EPISODES_PAR_TAILLE}"
    )

    print(
        f"Tailles : "
        f"{TAILLES}"
    )

    print(
        f"Nombre total de labyrinthes : "
        f"{EPISODES_PAR_TAILLE * len(TAILLES)}"
    )

    print(
        f"Maximum de mouvements : "
        f"{MAX_STEPS}"
    )


    # VÉRIFICATION DES MODÈLES

    if not verifier_modeles():

        return

    # GÉNÉRATION DES LABYRINTHES

    labyrinthes = (
        generer_labyrinthes_test()
    )

   # Q-LEARNING ORIGINAL

    print()
    print("=" * 70)
    print("CHARGEMENT DU Q-LEARNING ORIGINAL")
    print("=" * 70)

    qlearning = QLearningAgent(

        state_size=30,

        learning_rate=0.1,

        discount_factor=0.95,

        epsilon=0.0,

        epsilon_min=0.01,

        epsilon_decay=0.995,
    )

    qlearning.load(
        QLEARNING_PATH
    )

    qlearning.epsilon = 0.0

    print(
        f"✓ Q-table originale chargée."
    )

    print(
        f"✓ Nombre d'états : "
        f"{len(qlearning.q_table)}"
    )


    # Q-LEARNING V2

    print()
    print("=" * 70)
    print("CHARGEMENT DU Q-LEARNING V2")
    print("=" * 70)

    qlearning_v2 = QLearningAgent(

        state_size=12,

        learning_rate=0.1,

        discount_factor=0.95,

        epsilon=0.0,

        epsilon_min=0.01,

        epsilon_decay=0.995,
    )

    qlearning_v2.load(
        QLEARNING_V2_PATH
    )

    qlearning_v2.epsilon = 0.0

    print(
        "✓ Q-table V2 chargée."
    )

    print(
        f"✓ Nombre d'états : "
        f"{len(qlearning_v2.q_table)}"
    )


    # DQN ORIGINAL

    print()
    print("=" * 70)
    print("CHARGEMENT DU DQN ORIGINAL")
    print("=" * 70)

    dqn = DQNAgent(

        state_size=30,

        epsilon=0.0,
    )

    dqn.load(
        DQN_PATH
    )

    dqn.epsilon = 0.0

    print(
        "✓ DQN original chargé."
    )


    # DQN V2

    print()
    print("=" * 70)
    print("CHARGEMENT DU DQN V2")
    print("=" * 70)

    dqn_v2 = DQNAgent(

        state_size=12,

        epsilon=0.0,
    )

    dqn_v2.load(
        DQN_V2_PATH
    )

    dqn_v2.epsilon = 0.0

    print(
        "✓ DQN V2 chargé."
    )


   # ÉVALUATION Q-LEARNING ORIGINAL

    print()
    print("=" * 70)
    print("ÉVALUATION DU Q-LEARNING ORIGINAL")
    print("=" * 70)

    resultats_qlearning = (
        evaluer_qlearning(
            qlearning,
            labyrinthes,
        )
    )

    print(
        "✓ Évaluation terminée."
    )


    # ÉVALUATION Q-LEARNING V2

    print()
    print("=" * 70)
    print("ÉVALUATION DU Q-LEARNING V2")
    print("=" * 70)

    resultats_qlearning_v2 = (
        evaluer_qlearning_v2(
            qlearning_v2,
            labyrinthes,
        )
    )

    print(
        "✓ Évaluation terminée."
    )

    # ÉVALUATION DQN ORIGINAL

    print()
    print("=" * 70)
    print("ÉVALUATION DU DQN ORIGINAL")
    print("=" * 70)

    resultats_dqn = (
        evaluer_dqn(
            dqn,
            labyrinthes,
        )
    )

    print(
        "✓ Évaluation terminée."
    )


     # ÉVALUATION DQN V2

    print()
    print("=" * 70)
    print("ÉVALUATION DU DQN V2")
    print("=" * 70)

    resultats_dqn_v2 = (
        evaluer_dqn_v2(
            dqn_v2,
            labyrinthes,
        )
    )

    print(
        "✓ Évaluation terminée."
    )
 # AFFICHAGE PAR MODÈLE

    afficher_resultats(
        "Q-LEARNING ORIGINAL",
        resultats_qlearning,
    )

    afficher_resultats(
        "Q-LEARNING V2",
        resultats_qlearning_v2,
    )

    afficher_resultats(
        "DQN ORIGINAL",
        resultats_dqn,
    )

    afficher_resultats(
        "DQN V2",
        resultats_dqn_v2,
    )

    # MOYENNES GLOBALES

    moyenne_qlearning = (
        calculer_moyennes(
            resultats_qlearning
        )
    )

    moyenne_qlearning_v2 = (
        calculer_moyennes(
            resultats_qlearning_v2
        )
    )

    moyenne_dqn = (
        calculer_moyennes(
            resultats_dqn
        )
    )

    moyenne_dqn_v2 = (
        calculer_moyennes(
            resultats_dqn_v2
        )
    )


    # COMPARAISON GLOBALE

    print()
    print("=" * 70)
    print("COMPARAISON GLOBALE DES 4 MODÈLES")
    print("=" * 70)

    print(
        f"{'Modèle':<25}"
        f"{'Récompense':<18}"
        f"{'Mouvements':<18}"
        f"{'Réussite':<12}"
    )

    print("-" * 70)

    print(
        f"{'Q-Learning original':<25}"
        f"{moyenne_qlearning['reward']:<18.2f}"
        f"{moyenne_qlearning['steps']:<18.2f}"
        f"{moyenne_qlearning['success']:.2f}%"
    )

    print(
        f"{'Q-Learning V2':<25}"
        f"{moyenne_qlearning_v2['reward']:<18.2f}"
        f"{moyenne_qlearning_v2['steps']:<18.2f}"
        f"{moyenne_qlearning_v2['success']:.2f}%"
    )

    print(
        f"{'DQN original':<25}"
        f"{moyenne_dqn['reward']:<18.2f}"
        f"{moyenne_dqn['steps']:<18.2f}"
        f"{moyenne_dqn['success']:.2f}%"
    )

    print(
        f"{'DQN V2':<25}"
        f"{moyenne_dqn_v2['reward']:<18.2f}"
        f"{moyenne_dqn_v2['steps']:<18.2f}"
        f"{moyenne_dqn_v2['success']:.2f}%"
    )

    print("-" * 70)


     # CLASSEMENT

    modeles = {

        "Q-Learning original":
            moyenne_qlearning,

        "Q-Learning V2":
            moyenne_qlearning_v2,

        "DQN original":
            moyenne_dqn,

        "DQN V2":
            moyenne_dqn_v2,
    }


    # Classement selon le taux de réussite
    classement_success = sorted(
        modeles.items(),
        key=lambda item: item[1]["success"],
        reverse=True,
    )


    print()
    print("=" * 70)
    print("CLASSEMENT SELON LE TAUX DE RÉUSSITE")
    print("=" * 70)

    for rang, (
        nom,
        resultat,
    ) in enumerate(
        classement_success,
        start=1,
    ):

        print(
            f"{rang}. {nom:<25} "
            f"{resultat['success']:.2f}%"
        )


     # SAUVEGARDE

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    resultats_complets = {

        "qlearning_original":
            resultats_qlearning,

        "qlearning_v2":
            resultats_qlearning_v2,

        "dqn_original":
            resultats_dqn,

        "dqn_v2":
            resultats_dqn_v2,

        "global": {

            "qlearning_original":
                moyenne_qlearning,

            "qlearning_v2":
                moyenne_qlearning_v2,

            "dqn_original":
                moyenne_dqn,

            "dqn_v2":
                moyenne_dqn_v2,
        },
    }


    output_path = (
        f"{OUTPUT_DIR}/"
        f"all_models_evaluation.npy"
    )


    np.save(
        output_path,
        resultats_complets,
        allow_pickle=True,
    )


    # FIN

    print()
    print("=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)

    print()
    print(
        "✓ Résultats sauvegardés dans :"
    )

    print(
        f"  {output_path}"
    )

    print()
    print(
        "✓ 4 modèles évalués."
    )

    print(
        "✓ 600 labyrinthes utilisés "
        "pour chaque modèle."
    )

    print(
        "✓ Les quatre modèles ont utilisé "
        "les mêmes labyrinthes."
    )

    print()


# POINT D'ENTRÉE

if __name__ == "__main__":

    main()
