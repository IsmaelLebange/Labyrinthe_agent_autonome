import time

from environnement import LabyrintheEnv, labyrinthe
from models.q_learning import QLearningAgent
from models.dqn import DQNAgent


# CHEMINS DES MODÈLES

Q_TABLE_PATH = "results/qlearning/q_table.npy"
DQN_MODEL_PATH = "results/dqn/dqn_state_dict.pth"


# AFFICHAGE DU LABYRINTHE

def afficher_labyrinthe(grille, position, chemin=None):
    """
    Affiche le labyrinthe dans le terminal.

    A = position actuelle de l'agent
    * = cases déjà parcourues
    """

    if chemin is None:
        chemin = []

    # Nettoyer le terminal
    print("\033[H\033[J", end="")

    for i in range(len(grille)):
        ligne = ""

        for j in range(len(grille[0])):
            position_actuelle = (i, j)

            if position_actuelle == position:
                ligne += "A "

            elif position_actuelle in chemin:
                ligne += "* "

            else:
                ligne += grille[i][j] + " "

        print(ligne)

    print()
    print("A = agent")
    print("* = parcours de l'agent")


# EXECUTION D'UN AGENT

def executer_agent(
    env,
    choisir_action,
    nom_agent,
    vitesse=0.2,
    max_steps=200,
):
    """
    Exécute un agent entraîné dans l'environnement
    et affiche son parcours étape par étape.
    """

    print()
    print("=" * 50)
    print(f"VISUALISATION : {nom_agent}")
    print("=" * 50)

    # Réinitialiser l'environnement
    etat = env.reset()

    chemin = [etat]
    total_reward = 0

    # Affichage initial
    afficher_labyrinthe(
        env.grille,
        env.position,
        chemin,
    )

    print(f"Position : {etat}")
    print("Étape : 0")
    print("Récompense : 0")
    print("Récompense totale : 0")

    time.sleep(vitesse)

    # Exécution de l'agent
    for _ in range(max_steps):

        # L'agent choisit une action
        action = choisir_action(etat)

        # Exécution de l'action
        nouvel_etat, reward, done = env.step(action)

        # Mise à jour
        total_reward += reward
        chemin.append(nouvel_etat)

        # Affichage
        afficher_labyrinthe(
            env.grille,
            env.position,
            chemin,
        )

        print(f"Agent : {nom_agent}")
        print(f"Position : {nouvel_etat}")
        print(f"Étape : {env.steps}")
        print(f"Action : {action}")
        print(f"Récompense : {reward}")
        print(f"Récompense totale : {total_reward}")

        time.sleep(vitesse)

        etat = nouvel_etat

        # Si l'environnement indique que l'épisode est terminé
        if done:
            break

    # RESULTAT FINAL

    print()
    print("=" * 50)

    x, y = env.position

    if env.grille[x][y] == "G":
        print(f"{nom_agent} : OBJECTIF ATTEINT !")
    else:
        print(f"{nom_agent} : ÉCHEC")

    print(f"Nombre d'étapes : {env.steps}")
    print(f"Récompense totale : {total_reward}")

    print("=" * 50)

    return chemin, total_reward


# PROGRAMME PRINCIPAL

if __name__ == "__main__":

    print("=" * 60)
    print("VISUALISATION DES AGENTS ENTRAÎNÉS")
    print("=" * 60)

    # CHARGEMENT DU Q-LEARNING

    print()
    print("Chargement du modèle Q-Learning...")

    qlearning_agent = QLearningAgent(
        rows=len(labyrinthe),
        cols=len(labyrinthe[0]),
        epsilon=0.0,
    )

    qlearning_agent.load(Q_TABLE_PATH)

    # Désactiver l'exploration
    qlearning_agent.epsilon = 0.0

    print("✓ Q-table chargée avec succès.")

    # CHARGEMENT DU DQN

    print()
    print("Chargement du modèle DQN...")

    dqn_agent = DQNAgent(
        epsilon=0.0,
    )

    dqn_agent.load(DQN_MODEL_PATH)

    # Désactiver l'exploration
    dqn_agent.epsilon = 0.0

    print("✓ Modèle DQN chargé avec succès.")

    time.sleep(2)

    # VISUALISATION Q-LEARNING

    env_qlearning = LabyrintheEnv(
        labyrinthe,
        max_steps=200,
    )

    executer_agent(
        env_qlearning,
        qlearning_agent.choose_action,
        "Q-LEARNING",
        vitesse=0.2,
        max_steps=200,
    )

    # Pause entre les deux agents
    print()
    print("Passage au DQN...")
    time.sleep(3)

    # VISUALISATION DQN

    env_dqn = LabyrintheEnv(
        labyrinthe,
        max_steps=200,
    )

    executer_agent(
        env_dqn,
        dqn_agent.choose_action,
        "DQN",
        vitesse=0.2,
        max_steps=200,
    )

    # FIN

    print()
    print("=" * 60)
    print("VISUALISATION TERMINÉE")
    print("=" * 60)