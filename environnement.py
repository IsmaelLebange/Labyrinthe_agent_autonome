class LabyrintheEnv:
    def __init__(self, grille, max_steps=200):
        self.grille = grille
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.position = self.find_start()
        self.steps = 0
        return self.position

    def step(self, action):
        self.steps += 1
        dx, dy = actions[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Mouvement invalide
        if not self.is_valid(new_x, new_y):

            # Si le maximum de mouvements est atteint
            if self.steps >= self.max_steps:
                return self.position, -20, True

            # Mouvement invalide normal
            return self.position, -10, False

        # Mouvement valide
        self.position = (new_x, new_y)

        # Objectif atteint
        if self.grille[new_x][new_y] == "G":
            return self.position, 100, True

        # Nombre maximal de mouvements atteint
        if self.steps >= self.max_steps:
            return self.position, -20, True

        # Mouvement valide normal
        return self.position, -1, False

    def is_valid(self, x, y):
        return (
            0 <= x < len(self.grille)
            and 0 <= y < len(self.grille[0])
            and self.grille[x][y] != "#"
        )

    def find_start(self):
        for i, row in enumerate(self.grille):
            for j, val in enumerate(row):
                if val == "S":
                    return (i, j)

        raise ValueError("La position de départ 'S' est introuvable.")


# Grille du labyrinthe
labyrinthe = [
    ["S", ".", ".", "#", "."],
    ["#", ".", ".", "#", "."],
    [".", ".", "#", ".", "."],
    [".", ".", ".", ".", "."],
    ["#", "#", ".", ".", "G"]
]


# Actions
# 0 = haut
# 1 = bas
# 2 = gauche
# 3 = droite
actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


# Test de l'environnement
if __name__ == "__main__":
    env = LabyrintheEnv(labyrinthe)

    etat = env.reset()

    print("Départ:", etat)

    # Trajet permettant d'atteindre l'objectif
    trajet = [3, 3, 1, 2, 1, 1, 3, 3, 1, 3]

    for action in trajet:
        etat, reward, done = env.step(action)

        print(
            "État:", etat,
            "| Récompense:", reward,
            "| Terminé:", done
        )

        if done:
            break
