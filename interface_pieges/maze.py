"""Labyrinthe à branches et pièges, isolé de l'environnement original."""

import random
from collections import deque


MIN_SIZE = 5
MAX_SIZE = 10
ACTIONS = {
    0: (-1, 0),  # haut
    1: (1, 0),   # bas
    2: (0, -1),  # gauche
    3: (0, 1),   # droite
}


def _chemin(grille, depart, objectif):
    file = deque([depart])
    parents = {depart: None}
    while file:
        position = file.popleft()
        if position == objectif:
            break
        row, col = position
        for dr, dc in ACTIONS.values():
            voisin = (row + dr, col + dc)
            r, c = voisin
            if (
                0 <= r < len(grille)
                and 0 <= c < len(grille[0])
                and grille[r][c] != "#"
                and voisin not in parents
            ):
                parents[voisin] = position
                file.append(voisin)

    if objectif not in parents:
        return set()
    resultat = set()
    position = objectif
    while position is not None:
        resultat.add(position)
        position = parents[position]
    return resultat


def generer_labyrinthe(taille, seed=None, trap_ratio=0.03, loop_ratio=0.08):
    """Génère un labyrinthe 5x5 à 10x10 avec des impasses, boucles et pièges."""
    if not isinstance(taille, int) or not MIN_SIZE <= taille <= MAX_SIZE:
        raise ValueError("La taille doit être comprise entre 5 et 10.")
    if not 0 <= trap_ratio <= 0.25 or not 0 <= loop_ratio <= 0.5:
        raise ValueError("Les proportions de pièges ou de boucles sont invalides.")

    rng = random.Random(seed)
    cellules = list(range(1, taille - 1, 2))
    depart = (cellules[0], cellules[0])
    objectif = (cellules[-1], cellules[-1])
    grille = [["#"] * taille for _ in range(taille)]
    grille[depart[0]][depart[1]] = "."

    # Creuser un arbre DFS aléatoire : il produit des détours et des impasses.
    visitees = {depart}
    pile = [depart]
    while pile:
        row, col = pile[-1]
        voisins = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            voisin = (row + dr, col + dc)
            if voisin[0] in cellules and voisin[1] in cellules and voisin not in visitees:
                voisins.append((voisin, dr, dc))
        if not voisins:
            pile.pop()
            continue
        (nr, nc), dr, dc = rng.choice(voisins)
        grille[row + dr // 2][col + dc // 2] = "."
        grille[nr][nc] = "."
        visitees.add((nr, nc))
        pile.append((nr, nc))

    # Ouvrir quelques murs entre cellules pour créer des itinéraires alternatifs.
    murs = []
    for row in cellules:
        for col in cellules:
            if col + 2 in cellules and grille[row][col + 1] == "#":
                murs.append((row, col + 1))
            if row + 2 in cellules and grille[row + 1][col] == "#":
                murs.append((row + 1, col))
    if murs and loop_ratio > 0:
        nombre_boucles = min(len(murs), max(1, round(len(murs) * loop_ratio)))
        for row, col in rng.sample(murs, nombre_boucles):
            grille[row][col] = "."

    grille[depart[0]][depart[1]] = "S"
    grille[objectif[0]][objectif[1]] = "G"

    # Placer les pièges hors du plus court chemin sûr.
    chemin_sur = _chemin(grille, depart, objectif)
    candidats = [
        (row, col)
        for row in range(taille)
        for col in range(taille)
        if grille[row][col] == "." and (row, col) not in chemin_sur
    ]
    if candidats and trap_ratio > 0:
        nombre_pieges = min(len(candidats), max(1, round(len(candidats) * trap_ratio)))
        for row, col in rng.sample(candidats, nombre_pieges):
            grille[row][col] = "T"
    return grille


class LabyrintheEnv:
    """Règles locales du jeu ; une case piège renvoie au départ."""

    def __init__(self, grille, max_steps=200):
        self.grille = grille
        self.max_steps = max_steps
        self.reset()

    def find_start(self):
        for row, ligne in enumerate(self.grille):
            for col, case in enumerate(ligne):
                if case == "S":
                    return row, col
        raise ValueError("Le labyrinthe ne contient pas de départ S.")

    def reset(self):
        self.position = self.find_start()
        self.steps = 0
        return self.position

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError("L'action doit être comprise entre 0 et 3.")
        self.steps += 1
        dr, dc = ACTIONS[action]
        nr = self.position[0] + dr
        nc = self.position[1] + dc

        if not (0 <= nr < len(self.grille) and 0 <= nc < len(self.grille[0])) or self.grille[nr][nc] == "#":
            if self.steps >= self.max_steps:
                return self.position, -20, True
            return self.position, -10, False

        self.position = (nr, nc)
        case = self.grille[nr][nc]
        if case == "G":
            return self.position, 100, True
        if case == "T":
            self.position = self.find_start()
            return self.position, -25, self.steps >= self.max_steps
        if self.steps >= self.max_steps:
            return self.position, -20, True
        return self.position, -1, False
