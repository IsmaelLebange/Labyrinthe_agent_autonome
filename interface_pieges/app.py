"""Interface Tkinter des labyrinthes à pièges ; l'agent est servi par l'API."""

import os
import random
import tkinter as tk
from tkinter import messagebox, ttk

import httpx

from interface_pieges.maze import LabyrintheEnv, generer_labyrinthe


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_STEPS = 200
MODELES = {
    "Q-Learning V2": {"model_type": "qlearning", "version": "v2"},
    "DQN V2": {"model_type": "dqn", "version": "v2"},
}


class LabyrintheApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Labyrinthe avec pièges")
        self.root.geometry("1100x750")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#f2f2f2")

        self.taille = 5
        self.grille = generer_labyrinthe(self.taille)
        self.env = None
        self.position = self.position_depart(self.grille)
        self.agent_selectionne = None
        self.apres_id = None
        self.etapes = 0
        self.reward_total = 0
        self.derniere_action = 4
        self.termine = False

        self._construire_interface()
        self.dessiner_labyrinthe()

    @staticmethod
    def position_depart(grille):
        for row, ligne in enumerate(grille):
            for col, case in enumerate(ligne):
                if case == "S":
                    return row, col
        raise ValueError("Le labyrinthe ne contient pas de départ S.")

    def _construire_interface(self):
        principal = tk.Frame(self.root, bg="#f2f2f2")
        principal.pack(fill="both", expand=True, padx=18, pady=18)
        principal.grid_columnconfigure(0, weight=1)
        principal.grid_rowconfigure(0, weight=1)

        cadre_labyrinthe = tk.LabelFrame(principal, text="Labyrinthe", padx=10, pady=10, bg="white")
        cadre_labyrinthe.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.canvas = tk.Canvas(cadre_labyrinthe, width=520, height=520, bg="white")
        self.canvas.pack(expand=True)

        panneau = tk.Frame(principal, width=390, bg="#f2f2f2")
        panneau.grid(row=0, column=1, sticky="ns")
        panneau.grid_propagate(False)
        tk.Label(panneau, text="Contrôle de l'agent", font=("Arial", 16, "bold"), bg="#f2f2f2").pack(pady=(0, 12))

        modeles = tk.LabelFrame(panneau, text="Agent", padx=8, pady=8, bg="white")
        modeles.pack(fill="x", pady=(0, 10))
        self.combo_modele = ttk.Combobox(modeles, values=list(MODELES), state="readonly")
        self.combo_modele.set("DQN V2")
        self.combo_modele.pack(fill="x", pady=4)
        tk.Button(modeles, text="Charger l'agent", command=self.charger_agent).pack(fill="x", pady=4)

        tailles = tk.LabelFrame(panneau, text="Taille", padx=8, pady=8, bg="white")
        tailles.pack(fill="x", pady=(0, 10))
        self.combo_taille = ttk.Combobox(
            tailles,
            values=[f"{taille}x{taille}" for taille in range(5, 11)],
            state="readonly",
        )
        self.combo_taille.set("5x5")
        self.combo_taille.pack(fill="x", pady=4)

        boutons = tk.Frame(panneau, bg="#f2f2f2")
        boutons.pack(fill="x", pady=(0, 10))
        tk.Button(boutons, text="Nouveau labyrinthe", command=self.nouveau_labyrinthe).pack(side="left", expand=True, fill="x", padx=(0, 3))
        tk.Button(boutons, text="Démarrer", command=self.demarrer).pack(side="left", expand=True, fill="x", padx=3)
        tk.Button(boutons, text="Réinitialiser", command=self.reinitialiser).pack(side="left", expand=True, fill="x", padx=(3, 0))

        resultats = tk.LabelFrame(panneau, text="Résultats", padx=12, pady=8, bg="white")
        resultats.pack(fill="x", pady=(0, 10))
        self.label_agent = self._ligne_resultat(resultats, 0, "Agent :", "Aucun")
        self.label_taille = self._ligne_resultat(resultats, 1, "Taille :", "5x5")
        self.label_etapes = self._ligne_resultat(resultats, 2, "Étapes :", "0 / 200")
        self.label_reward = self._ligne_resultat(resultats, 3, "Récompense :", "0")
        self.label_statut = self._ligne_resultat(resultats, 4, "Statut :", "En attente")

        legende = tk.LabelFrame(panneau, text="Légende", padx=8, pady=8, bg="white")
        legende.pack(fill="x")
        tk.Label(legende, text="S départ   G objectif   T piège", bg="white", anchor="w").pack(fill="x")
        tk.Label(legende, text="Un piège renvoie au départ (-25)", bg="white", anchor="w").pack(fill="x")
        tk.Label(legende, text="Case noire : mur   ● : agent", bg="white", anchor="w").pack(fill="x")

    @staticmethod
    def _ligne_resultat(parent, row, title, value):
        tk.Label(parent, text=title, font=("Arial", 10, "bold"), bg="white").grid(row=row, column=0, sticky="w", pady=3)
        label = tk.Label(parent, text=value, bg="white", anchor="w")
        label.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        return label

    def charger_agent(self):
        nom = self.combo_modele.get()
        try:
            response = httpx.post(
                f"{API_BASE_URL}/load_model",
                json=MODELES[nom],
                timeout=15,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            detail = str(error)
            try:
                detail = response.json().get("detail", detail)
            except (NameError, ValueError):
                pass
            messagebox.showerror("Chargement impossible", detail)
            return
        self.agent_selectionne = nom
        self.label_agent.config(text=nom)
        self.label_statut.config(text="Agent chargé")

    def nouveau_labyrinthe(self):
        self.taille = int(self.combo_taille.get().split("x")[0])
        self.grille = generer_labyrinthe(self.taille, seed=random.randrange(1_000_001))
        self.label_taille.config(text=f"{self.taille}x{self.taille}")
        self.reinitialiser()

    def reinitialiser(self):
        self._annuler_action()
        self.position = self.position_depart(self.grille)
        self.etapes = 0
        self.reward_total = 0
        self.derniere_action = 4
        self.termine = False
        self.label_etapes.config(text=f"0 / {MAX_STEPS}")
        self.label_reward.config(text="0")
        self.label_statut.config(text="En attente")
        self.dessiner_labyrinthe()

    def demarrer(self):
        if not self.agent_selectionne:
            messagebox.showwarning("Agent requis", "Charge un agent avant de démarrer.")
            return
        self.reinitialiser()
        self.env = LabyrintheEnv(self.grille, max_steps=MAX_STEPS)
        self.position = self.env.reset()
        self.label_statut.config(text="En cours")
        self.apres_id = self.root.after(400, self.executer_etape)

    def executer_etape(self):
        self.apres_id = None
        if self.termine:
            return
        try:
            response = httpx.post(
                f"{API_BASE_URL}/predict_maze",
                json={
                    "grille": self.grille,
                    "position": list(self.position),
                    "derniere_action": self.derniere_action,
                },
                timeout=5,
            )
            response.raise_for_status()
            prediction = response.json()
            if prediction.get("source") != "model":
                raise RuntimeError("L'API n'a pas de modèle chargé.")
            action = int(prediction["action"])
            self.position, reward, done = self.env.step(action)
        except (httpx.HTTPError, KeyError, RuntimeError, ValueError) as error:
            self.termine = True
            messagebox.showerror("Erreur", str(error))
            return

        self.derniere_action = action
        self.etapes += 1
        self.reward_total += reward
        self.label_etapes.config(text=f"{self.etapes} / {MAX_STEPS}")
        self.label_reward.config(text=str(self.reward_total))
        self.dessiner_labyrinthe()

        if done:
            self.termine = True
            self.label_statut.config(text="Objectif atteint" if reward == 100 else "Échec")
            return
        self.apres_id = self.root.after(300, self.executer_etape)

    def _annuler_action(self):
        if self.apres_id is not None:
            self.root.after_cancel(self.apres_id)
            self.apres_id = None

    def dessiner_labyrinthe(self):
        self.canvas.delete("all")
        lignes, colonnes = len(self.grille), len(self.grille[0])
        cote = min(500 / colonnes, 500 / lignes)
        ox, oy = (500 - cote * colonnes) / 2, (500 - cote * lignes) / 2
        styles = {
            "#": ("#333333", ""),
            "S": ("#d9ead3", "S"),
            "G": ("#f4cccc", "G"),
            "T": ("#e06666", "T"),
            ".": ("#ffffff", ""),
        }
        for row, ligne in enumerate(self.grille):
            for col, case in enumerate(ligne):
                x1, y1 = ox + col * cote, oy + row * cote
                couleur, texte = styles.get(case, ("#ffffff", ""))
                self.canvas.create_rectangle(x1, y1, x1 + cote, y1 + cote, fill=couleur, outline="#cccccc")
                if texte:
                    self.canvas.create_text(x1 + cote / 2, y1 + cote / 2, text=texte, fill="white" if case == "T" else "black", font=("Arial", max(7, int(cote * 0.4)), "bold"))

        row, col = self.position
        cx, cy = ox + (col + 0.5) * cote, oy + (row + 0.5) * cote
        rayon = cote * 0.24
        self.canvas.create_oval(cx - rayon, cy - rayon, cx + rayon, cy + rayon, fill="#4a86e8", outline="#1c4587", width=2)


def main():
    root = tk.Tk()
    LabyrintheApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
