def encoder_etat(grille, position, taille_observation=5):
    """
    Transforme le labyrinthe et la position de l'agent
    en un état de taille fixe.

    État composé de :
    - observation locale 5x5 : 25 valeurs
    - direction vers l'objectif : 2 valeurs
    - position normalisée de l'agent : 2 valeurs
    - taille normalisée du labyrinthe : 1 valeur

    Total : 30 valeurs.
    """

    if taille_observation % 2 == 0:
        raise ValueError(
            "La taille de l'observation doit être impaire."
        )

    lignes = len(grille)
    colonnes = len(grille[0])

    row, col = position

    rayon = taille_observation // 2

    etat = []

    # 1. OBSERVATION LOCALE 5x5

    for i in range(
        row - rayon,
        row + rayon + 1
    ):

        for j in range(
            col - rayon,
            col + rayon + 1
        ):

            if (
                i < 0
                or i >= lignes
                or j < 0
                or j >= colonnes
            ):
                valeur = -1

            else:

                case = grille[i][j]

                if case == "#":
                    valeur = -1

                elif case == "G":
                    valeur = 2

                elif case == "S":
                    valeur = 1

                else:
                    valeur = 0

            etat.append(valeur)

    # 2. RECHERCHE DE L'OBJECTIF

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

     # 3. DIRECTION VERS L'OBJECTIF

    if lignes > 1:

        distance_row = (
            objectif_row - row
        ) / (lignes - 1)

    else:

        distance_row = 0.0

    if colonnes > 1:

        distance_col = (
            objectif_col - col
        ) / (colonnes - 1)

    else:

        distance_col = 0.0

    etat.append(
        distance_row
    )

    etat.append(
        distance_col
    )
    # 4. POSITION NORMALISÉE DE L'AGENT

    if lignes > 1:

        position_row = (
            row / (lignes - 1)
        )

    else:

        position_row = 0.0

    if colonnes > 1:

        position_col = (
            col / (colonnes - 1)
        )

    else:

        position_col = 0.0

    etat.append(
        position_row
    )

    etat.append(
        position_col
    )

    # 5. TAILLE NORMALISÉE DU LABYRINTHE

    taille_normalisee = (
        (lignes - 5) / 5
    )

    etat.append(
        taille_normalisee
    )

    return tuple(etat)


def encoder_etat_depuis_env(env):
    """
    Encode directement l'état actuel
    d'un LabyrintheEnv.
    """

    return encoder_etat(
        env.grille,
        env.position,
    )