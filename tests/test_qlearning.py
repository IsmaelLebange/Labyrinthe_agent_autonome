from environnement import LabyrintheEnv, labyrinthe
from models.q_learning import QLearningAgent


def main():
    # Création de l'environnement de Josué
    env = LabyrintheEnv(labyrinthe)

    # Création de l'agent
    agent = QLearningAgent(
        rows=len(env.grille),
        cols=len(env.grille[0]),
    )

    # État initial
    state = env.reset()

    print("État initial :", state)

    # Conversion état -> indice
    state_index = agent.state_to_index(state)

    print("Indice de l'état :", state_index)

    # Choix d'une action
    action = agent.choose_action(state)

    print("Action choisie :", action)

    # Exécution dans l'environnement
    next_state, reward, done = env.step(action)

    print("Nouvel état :", next_state)
    print("Récompense :", reward)
    print("Terminé :", done)

    # Q avant mise à jour
    old_q = agent.q_table[
        state_index,
        action
    ]

    print("Q avant mise à jour :", old_q)

    # Apprentissage
    agent.update(
        state,
        action,
        reward,
        next_state,
        done,
    )

    # Q après mise à jour
    new_q = agent.q_table[
        state_index,
        action
    ]

    print("Q après mise à jour :", new_q)

    # Epsilon
    print(
        "Epsilon avant :",
        agent.epsilon
    )

    agent.decay_epsilon()

    print(
        "Epsilon après :",
        agent.epsilon
    )


if __name__ == "__main__":
    main()