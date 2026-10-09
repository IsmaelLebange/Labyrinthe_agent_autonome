import random


def generer_labyrinthe(taille, seed=None):
    """
    Génère un labyrinthe carré avec :
    S = départ
    G = objectif
    # = mur
    . = passage

    La génération garantit qu'il existe
    un chemin entre S et G.
    """

    if taille < 5:
        raise ValueError(
            "La taille minimale du labyrinthe est 5."
        )

    if seed is not None:
        random.seed(seed)

    # Pour le générateur DFS, on travaille avec
    # une grille interne de cellules.
    lignes = taille
    colonnes = taille

    grille = [
        ["#"] * colonnes
        for _ in range(lignes)
    ]

    # Départ
    depart = (0, 0)

    # Objectif
    objectif = (taille - 1, taille - 1)

    # On ouvre le départ
    grille[depart[0]][depart[1]] = "."

    # Génération d'un chemin garanti

    position = depart

    while position != objectif:

        row, col = position

        mouvements = []

        # Aller vers le bas
        if row < taille - 1:
            mouvements.append(
                (row + 1, col)
            )

        # Aller vers la droite
        if col < taille - 1:
            mouvements.append(
                (row, col + 1)
            )

        # Ajouter parfois des mouvements
        # dans d'autres directions pour varier
        if row > 0:
            mouvements.append(
                (row - 1, col)
            )

        if col > 0:
            mouvements.append(
                (row, col - 1)
            )

        # On privilégie les mouvements
        # qui rapprochent de l'objectif.
        mouvements_valides = []

        for new_row, new_col in mouvements:

            distance_actuelle = (
                abs(row - objectif[0])
                + abs(col - objectif[1])
            )

            nouvelle_distance = (
                abs(new_row - objectif[0])
                + abs(new_col - objectif[1])
            )

            if nouvelle_distance <= distance_actuelle:
                mouvements_valides.append(
                    (new_row, new_col)
                )

        if mouvements_valides:

            position = random.choice(
                mouvements_valides
            )

        else:

            position = random.choice(
                mouvements
            )

        grille[position[0]][position[1]] = "."

    # Ajouter quelques passages supplémentaires

    nombre_passages = max(
        1,
        taille * taille // 5
    )

    for _ in range(nombre_passages):

        row = random.randint(
            0,
            taille - 1
        )

        col = random.randint(
            0,
            taille - 1
        )

        grille[row][col] = "."

    # Départ et objectif

    grille[0][0] = "S"

    grille[taille - 1][taille - 1] = "G"

    return grille


def afficher_labyrinthe(grille):

    for ligne in grille:
        print(" ".join(ligne))