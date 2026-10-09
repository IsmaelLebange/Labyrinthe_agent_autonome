from models.dqn import DQNAgent


def main():
    agent = DQNAgent()

    state = (0, 0)

    print("===== TEST DQN =====")
    print("État initial :", state)

    action = agent.choose_action(state)

    print("Action choisie :", action)
    print("Epsilon :", agent.epsilon)
    print("Taille de la mémoire :", len(agent.memory))
    print("Device utilisé :", agent.device)

    next_state = (0, 1)
    reward = -1
    done = False

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done,
    )

    print("Mémoire après ajout :", len(agent.memory))

    loss = agent.replay()

    print("Loss :", loss)
    print("===== TEST TERMINÉ =====")


if __name__ == "__main__":
    main()