from environnement import LabyrintheEnv, labyrinthe


def main():
    env = LabyrintheEnv(labyrinthe)

    state = env.reset()

    print("État initial :", state)

    actions_test = [3, 3, 1, 2, 1, 1, 3, 3, 1, 3]

    for action in actions_test:
        next_state, reward, done = env.step(action)

        print(
            "État :", next_state,
            "| Récompense :", reward,
            "| Terminé :", done
        )

        if done:
            break


if __name__ == "__main__":
    main()