"""Variante de l'interface API dont le labyrinthe ne contient aucun mur."""

import tkinter as tk

from interface.app_api import LabyrintheApiApp


class LabyrintheApiSansMursApp(LabyrintheApiApp):
    TITRE = "Labyrinthe ouvert — agent connecté à l'API"
    LEGENDE = "S = Départ     G = Objectif\nToutes les cases sont ouvertes     ● = Agent"

    @staticmethod
    def generer_grille(taille, seed=None):
        grille = [["." for _ in range(taille)] for _ in range(taille)]
        grille[0][0] = "S"
        grille[-1][-1] = "G"
        return grille


def main():
    root = tk.Tk()
    LabyrintheApiSansMursApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
