def encoder_etat_qlearning(grille, position):
    """
    Encode l'état pour le Q-Learning amélioré.

    L'état contient :
    - les 4 directions immédiates :
        haut, bas, gauche, droite
    - la direction verticale vers l'objectif
    - la direction horizontale vers l'objectif

    Chaque direction est représentée par :
        0 = mur / hors limite
        1 = passage

    Direction vers l'objectif :
        -1 = objectif vers le haut / gauche
         0 = même ligne / même colonne
         1 = objectif vers le bas / droite

    Total : 6 valeurs.
    """

    lignes = len(grille)
    colonnes = len(grille[0])

    row, col = position

    # 1. Vérifier les quatre directions autour de l'agent

    directions = [
        (-1, 0),  # haut
        (1, 0),   # bas
        (0, -1),  # gauche
        (0, 1),   # droite
    ]

    observation = []

    for delta_row, delta_col in directions:

        nouvelle_row = row + delta_row
        nouvelle_col = col + delta_col

        if (
            nouvelle_row < 0
            or nouvelle_row >= lignes
            or nouvelle_col < 0
            or nouvelle_col >= colonnes
        ):
            observation.append(0)

        elif grille[nouvelle_row][nouvelle_col] == "#":
            observation.append(0)

        else:
            observation.append(1)

     # 2. Chercher la position de l'objectif

    objectif_row = None
    objectif_col = None

    for i in range(lignes):
        for j in range(colonnes):

            if grille[i][j] == "G":
                objectif_row = i
                objectif_col = j
                break

        if objectif_row is not None:
            break

    # 3. Direction verticale vers l'objectif

    if objectif_row > row:
        direction_verticale = 1

    elif objectif_row < row:
        direction_verticale = -1

    else:
        direction_verticale = 0

    # 4. Direction horizontale vers l'objectif

    if objectif_col > col:
        direction_horizontale = 1

    elif objectif_col < col:
        direction_horizontale = -1

    else:
        direction_horizontale = 0

    # 5. Construire l'état final

    etat = (
        observation[0],  # haut
        observation[1],  # bas
        observation[2],  # gauche
        observation[3],  # droite
        direction_verticale,
        direction_horizontale,
    )

    return etat