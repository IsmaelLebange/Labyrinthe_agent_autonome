import tkinter as tk
from tkinter import ttk, messagebox
import os
import random

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe
from environments.state_encoder import encoder_etat
from environments.qlearning_state_encoder_v2 import encoder_etat_qlearning_v2
from models.q_learning import QLearningAgent
from models.dqn import DQNAgent


class LabyrintheApp:
    TAILLES_DISPONIBLES = ["5x5", "6x6", "7x7", "8x8", "9x9", "10x10"]
    MODELES = ["Q-Learning", "Q-Learning V2", "DQN", "DQN V2"]

    def __init__(self, root):
        self.root = root
        self.root.title("Résolution d'un labyrinthe avec agent autonome")
        self.root.geometry("1100x750")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#f2f2f2")

        self.taille = 5
        self.grille = generer_labyrinthe(self.taille)

        self.env = None
        self.agent_selectionne = None
        self.agent = None
        self.apres_id = None

        self.position = (0, 0)
        self.etapes = 0
        self.reward_total = 0
        self.termine = False
        self.derniere_action = 4

        self.modeles = {nom: None for nom in self.MODELES}
        self.boutons_agents = {}

        self.charger_modeles()
        self.configurer_interface_principale()
        self.dessiner_labyrinthe()

    def configurer_interface_principale(self):
        frame_principal = tk.Frame(self.root, bg="#f2f2f2")
        frame_principal.pack(fill="both", expand=True, padx=20, pady=20)

        frame_principal.grid_columnconfigure(0, weight=1)
        frame_principal.grid_columnconfigure(1, weight=0)
        frame_principal.grid_rowconfigure(0, weight=1)

        frame_labyrinthe = tk.LabelFrame(
            frame_principal,
            text="Labyrinthe",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=10,
            pady=10,
        )
        frame_labyrinthe.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

        self.canvas = tk.Canvas(
            frame_labyrinthe,
            width=520,
            height=520,
            bg="white",
            highlightthickness=1,
            highlightbackground="#aaaaaa",
        )
        self.canvas.pack(expand=True)

        frame_controle = tk.Frame(frame_principal, width=440, bg="#f2f2f2")
        frame_controle.grid(row=0, column=1, sticky="ns")
        frame_controle.grid_propagate(False)

        tk.Label(
            frame_controle,
            text="Contrôle de l'agent",
            font=("Arial", 16, "bold"),
            bg="#f2f2f2",
        ).pack(pady=(0, 15))

        self.creer_section_agents(frame_controle)
        self.creer_section_taille(frame_controle)
        self.creer_section_controles(frame_controle)
        self.creer_section_resultats(frame_controle)
        self.creer_section_legende(frame_controle)

    def creer_section_agents(self, conteneur):
        frame_agents = tk.LabelFrame(
            conteneur,
            text="Choisir l'agent",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            padx=10,
            pady=10,
        )
        frame_agents.pack(fill="x", pady=(0, 12))
        frame_agents.grid_columnconfigure(0, weight=1)
        frame_agents.grid_columnconfigure(1, weight=1)

        for index, nom in enumerate(self.MODELES):
            row, col = divmod(index, 2)
            bouton = tk.Button(
                frame_agents,
                text=nom,
                font=("Arial", 10, "bold"),
                width=18,
                height=2,
                command=lambda nom=nom: self.selectionner_agent(nom),
            )
            bouton.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            self.boutons_agents[nom] = bouton

    def creer_section_taille(self, conteneur):
        frame_taille = tk.LabelFrame(
            conteneur,
            text="Taille du labyrinthe",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            padx=10,
            pady=10,
        )
        frame_taille.pack(fill="x", pady=(0, 12))

        self.combo_taille = ttk.Combobox(
            frame_taille,
            values=self.TAILLES_DISPONIBLES,
            state="readonly",
            font=("Arial", 11),
        )
        self.combo_taille.set("5x5")
        self.combo_taille.pack(fill="x", padx=5, pady=5)

    def creer_section_controles(self, conteneur):
        frame_boutons = tk.Frame(conteneur, bg="#f2f2f2")
        frame_boutons.pack(fill="x", pady=(0, 12))
        for index in range(3):
            frame_boutons.grid_columnconfigure(index, weight=1)

        actions = [
            ("Nouveau labyrinthe", self.nouveau_labyrinthe),
            ("Démarrer", self.demarrer),
            ("Réinitialiser", self.reinitialiser),
        ]

        for index, (texte, commande) in enumerate(actions):
            tk.Button(
                frame_boutons,
                text=texte,
                font=("Arial", 10, "bold"),
                command=commande,
            ).grid(row=0, column=index, padx=3, sticky="ew")

    def creer_section_resultats(self, conteneur):
        frame_infos = tk.LabelFrame(
            conteneur,
            text="Résultats",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            padx=15,
            pady=10,
        )
        frame_infos.pack(fill="x", pady=(0, 12))

        labels = [
            ("Agent :", "Aucun", "label_agent"),
            ("Taille :", "5x5", "label_taille_resultat"),
            ("Étapes :", "0", "label_etapes"),
            ("Récompense :", "0", "label_reward"),
            ("Statut :", "En attente", "label_statut"),
        ]

        for index, (libelle, valeur, attr) in enumerate(labels):
            tk.Label(
                frame_infos,
                text=libelle,
                font=("Arial", 10, "bold"),
                bg="#ffffff",
            ).grid(row=index, column=0, sticky="w", pady=3)

            widget = tk.Label(
                frame_infos,
                text=valeur,
                font=("Arial", 10),
                bg="#ffffff",
                anchor="w",
            )
            widget.grid(row=index, column=1, sticky="w", padx=(10, 0), pady=3)
            setattr(self, attr, widget)

    def creer_section_legende(self, conteneur):
        frame_legende = tk.LabelFrame(
            conteneur,
            text="Légende",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            padx=10,
            pady=8,
        )
        frame_legende.pack(fill="x")

        tk.Label(frame_legende, text="S = Départ     G = Objectif", font=("Arial", 10), bg="#ffffff").pack(anchor="w")
        tk.Label(frame_legende, text="case noire = Mur        ● = Agent", font=("Arial", 10), bg="#ffffff").pack(anchor="w")

    def _annuler_prochaine_action(self):
        if self.apres_id is not None:
            self.root.after_cancel(self.apres_id)
            self.apres_id = None

    # CHARGEMENT DES MODELES

    def charger_modeles(self):
        print("\nChargement des modèles...")

        # Q-LEARNING ORIGINAL

        chemin_qlearning = "results/qlearning/q_table.npy"

        if os.path.exists(chemin_qlearning):
            try:
                agent = QLearningAgent(state_size=30)
                agent.load(chemin_qlearning)
                agent.epsilon = 0.0
                self.modeles["Q-Learning"] = agent
                print("✓ Q-Learning chargé.")
            except Exception as e:
                print("✗ Erreur Q-Learning :", e)
        else:
            print("✗ Fichier Q-Learning introuvable.")

         # Q-LEARNING V2

        chemin_qlearning_v2 = (
            "results/qlearning_improved_v2/q_table.npy"
        )

        if os.path.exists(chemin_qlearning_v2):
            try:
                agent = QLearningAgent(state_size=12)
                agent.load(chemin_qlearning_v2)
                agent.epsilon = 0.0
                self.modeles["Q-Learning V2"] = agent
                print("✓ Q-Learning V2 chargé.")
            except Exception as e:
                print("✗ Erreur Q-Learning V2 :", e)
        else:
            print("✗ Fichier Q-Learning V2 introuvable.")

        # DQN ORIGINAL

        chemin_dqn = "results/dqn/dqn_state_dict.pth"

        if os.path.exists(chemin_dqn):
            try:
                agent = DQNAgent(state_size=30)
                agent.load(chemin_dqn)
                agent.epsilon = 0.0
                self.modeles["DQN"] = agent
                print("✓ DQN chargé.")
            except Exception as e:
                print("✗ Erreur DQN :", e)
        else:
            print("✗ Fichier DQN introuvable.")
        # DQN V2

        chemin_dqn_v2 = "results/dqn_v2/model.pth"

        if os.path.exists(chemin_dqn_v2):
            try:
                agent = DQNAgent(state_size=12)
                agent.load(chemin_dqn_v2)
                agent.epsilon = 0.0
                self.modeles["DQN V2"] = agent
                print("✓ DQN V2 chargé.")
            except Exception as e:
                print("✗ Erreur DQN V2 :", e)
        else:
            print("✗ Fichier DQN V2 introuvable.")
        # SELECTION AGENT

    def selectionner_agent(self, nom):
        if self.modeles[nom] is None:
            messagebox.showerror(
                "Modèle indisponible",
                f"Le modèle {nom} n'a pas pu être chargé."
            )
            return

        self.agent_selectionne = nom
        self.agent = self.modeles[nom]

        self.label_agent.config(
            text=nom
        )

        # Mise à jour visuelle des boutons
        for nom_agent, bouton in self.boutons_agents.items():
            bouton.config(
                relief="raised"
            )

        self.boutons_agents[nom].config(
            relief="sunken"
        )

        self.label_statut.config(
            text="Agent sélectionné"
        )

    # NOUVEAU LABYRINTHE

    def nouveau_labyrinthe(self):
        valeur = self.combo_taille.get()

        if not valeur:
            valeur = "5x5"

        self.taille = int(
            valeur.split("x")[0]
        )

        self.grille = generer_labyrinthe(
            self.taille,
            seed=random.randint(0, 1000000)
        )

        self.reinitialiser()

        self.label_taille_resultat.config(
            text=valeur
        )

        self.dessiner_labyrinthe()
    # REINITIALISATION

    def reinitialiser(self):
        self._annuler_prochaine_action()
        self.position = (0, 0)
        self.etapes = 0
        self.reward_total = 0
        self.termine = False
        self.derniere_action = 4

        self.label_etapes.config(text="0")
        self.label_reward.config(text="0")
        self.label_statut.config(text="En attente")

        self.dessiner_labyrinthe()

    # OBTENIR L'ETAT

    def obtenir_etat(self):
        if self.agent_selectionne in [
            "Q-Learning V2",
            "DQN V2"
        ]:
            return encoder_etat_qlearning_v2(
                self.grille,
                self.position,
                self.derniere_action
            )

        return encoder_etat(
            self.grille,
            self.position
        )

    # DEMARRER

    def demarrer(self):
        if self.agent is None:
            messagebox.showwarning(
                "Agent non sélectionné",
                "Veuillez sélectionner un agent."
            )
            return

        self._annuler_prochaine_action()
        self.reinitialiser()

        # Création de l'environnement avec le labyrinthe actuel
        self.env = LabyrintheEnv(self.grille)

        # On force la position de départ
        self.position = self.env.reset()

        self.label_statut.config(
            text="En cours..."
        )

        self.apres_id = self.root.after(
            500,
            self.executer_etape
        )

    # EXECUTION D'UNE ETAPE

    def executer_etape(self):
        self.apres_id = None

        if self.termine:
            return

        if self.etapes >= 200:
            self.termine = True
            self.label_statut.config(
                text="Échec - 200 mouvements"
            )
            return

        # ETAT

        etat = self.obtenir_etat()
        # ACTION

        try:
            action = self.agent.choose_action(etat)
        except Exception as e:
            self.termine = True
            messagebox.showerror(
                "Erreur",
                f"Erreur lors du choix de l'action :\n{e}"
            )
            return

        # ENVIRONNEMENT

        try:
            prochain_etat, reward, done = self.env.step(action)
        except Exception as e:
            self.termine = True
            messagebox.showerror(
                "Erreur",
                f"Erreur dans l'environnement :\n{e}"
            )
            return

        self.position = prochain_etat
        self.etapes += 1
        self.reward_total += reward

        # Dernière action pour les modèles V2
        self.derniere_action = action

        # MISE A JOUR AFFICHAGE

        self.label_etapes.config(
            text=str(self.etapes)
        )

        self.label_reward.config(
            text=str(self.reward_total)
        )

        self.dessiner_labyrinthe()

        # FIN

        if done:
            self.termine = True

            if reward == 100:
                self.label_statut.config(
                    text="✓ Objectif atteint"
                )
            else:
                self.label_statut.config(
                    text="✗ Échec"
                )

            return

        if self.etapes >= 200:
            self.termine = True

            self.label_statut.config(
                text="✗ Échec - 200 mouvements"
            )

            return

        # PROCHAINE ETAPE

        self.apres_id = self.root.after(
            300,
            self.executer_etape
        )

    # DESSIN DU LABYRINTHE

    def dessiner_labyrinthe(self):
        self.canvas.delete("all")

        lignes = len(self.grille)
        colonnes = len(self.grille[0])

        # Taille disponible du canvas
        largeur = 500
        hauteur = 500

        taille_case = min(
            largeur / colonnes,
            hauteur / lignes
        )

        offset_x = (
            largeur - taille_case * colonnes
        ) / 2

        offset_y = (
            hauteur - taille_case * lignes
        ) / 2

        for i in range(lignes):
            for j in range(colonnes):

                x1 = offset_x + j * taille_case
                y1 = offset_y + i * taille_case
                x2 = x1 + taille_case
                y2 = y1 + taille_case

                case = self.grille[i][j]

                # MUR

                if case == "#":
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#333333",
                        outline="#ffffff"
                    )

                # DEPART

                elif case == "S":
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#d9ead3",
                        outline="#ffffff"
                    )

                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text="S",
                        font=("Arial", 16, "bold")
                    )

                # OBJECTIF

                elif case == "G":
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#f4cccc",
                        outline="#ffffff"
                    )

                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text="G",
                        font=("Arial", 16, "bold")
                    )

                # PASSAGE

                else:
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#ffffff",
                        outline="#cccccc"
                    )

        # DESSIN DE L'AGENT

        row, col = self.position

        x_center = (
            offset_x
            + col * taille_case
            + taille_case / 2
        )

        y_center = (
            offset_y
            + row * taille_case
            + taille_case / 2
        )

        rayon = taille_case * 0.25

        self.canvas.create_oval(
            x_center - rayon,
            y_center - rayon,
            x_center + rayon,
            y_center + rayon,
            fill="#4a86e8",
            outline="#1c4587",
            width=2
        )
# PROGRAMME PRINCIPAL

if __name__ == "__main__":
    root = tk.Tk()

    app = LabyrintheApp(root)

    root.mainloop()