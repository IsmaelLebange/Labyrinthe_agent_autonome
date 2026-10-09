def encoder_etat_qlearning_v2(
    grille,
    position,
    derniere_action=4,
):
    """
    Encode l'état pour le Q-Learning amélioré V2.

    L'état contient :

    - une observation locale 3x3 autour de l'agent : 9 valeurs
    - la direction verticale vers l'objectif : 1 valeur
    - la direction horizontale vers l'objectif : 1 valeur
    - la dernière action effectuée : 1 valeur

    Total : 12 valeurs.

    Valeurs de l'observation :
        0 = mur / hors limite
        1 = passage
        2 = départ
        3 = objectif
        4 = position actuelle

    Direction verticale :
        -1 = objectif vers le haut
         0 = même ligne
         1 = objectif vers le bas

    Direction horizontale :
        -1 = objectif vers la gauche
         0 = même colonne
         1 = objectif vers la droite

    Dernière action :
        0 = haut
        1 = bas
        2 = gauche
        3 = droite
        4 = aucune action précédente
    """

    lignes = len(grille)
    colonnes = len(grille[0])

    row, col = position

    # 1. Observation locale 3x3

    observation = []

    for i in range(row - 1, row + 2):

        for j in range(col - 1, col + 2):

            if (
                i < 0
                or i >= lignes
                or j < 0
                or j >= colonnes
            ):
                valeur = 0

            elif i == row and j == col:
                valeur = 4

            else:

                case = grille[i][j]

                if case == "#":
                    valeur = 0

                elif case == "S":
                    valeur = 2

                elif case == "G":
                    valeur = 3

                else:
                    valeur = 1

            observation.append(valeur)

    # 2. Chercher l'objectif

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

     # 3. Direction verticale

    if objectif_row > row:
        direction_verticale = 1

    elif objectif_row < row:
        direction_verticale = -1

    else:
        direction_verticale = 0

   # 4. Direction horizontale

    if objectif_col > col:
        direction_horizontale = 1

    elif objectif_col < col:
        direction_horizontale = -1

    else:
        direction_horizontale = 0

    # 5. Construire l'état

    etat = tuple(
        observation
        + [
            direction_verticale,
            direction_horizontale,
            derniere_action,
        ]
    )

    return etat