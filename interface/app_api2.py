"""Interface Tkinter qui demande les actions à l'API publique du labyrinthe."""

import json
import os
import random
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from environnement import LabyrintheEnv
from environments.maze_generator import generer_labyrinthe as generer_labyrinthe_base


API_URL_PAR_DEFAUT = "https://labyrinthe-agent-autonome.onrender.com"
API_URL = os.getenv("MAZE_API_URL", API_URL_PAR_DEFAUT).rstrip("/")
MODELES_API = {"Q-Learning V2": "qlearning", "DQN V2": "dqn"}


def generer_labyrinthe(taille, seed=None):
    """Génère un labyrinthe avec environ la moitié des murs d'origine."""
    grille = generer_labyrinthe_base(taille, seed=seed)
    murs = [
        (ligne, colonne)
        for ligne, cases in enumerate(grille)
        for colonne, case in enumerate(cases)
        if case == "#"
    ]

    # Ouvrir la moitié des murs tout en gardant le départ et l'objectif.
    for ligne, colonne in random.sample(murs, len(murs) // 2):
        grille[ligne][colonne] = "."

    return grille


class LabyrintheApiApp:
    TAILLES_DISPONIBLES = [f"{n}x{n}" for n in range(5, 11)]
    TITRE = "Labyrinthe — interface connectée à l'API"
    LEGENDE = "S = Départ     G = Objectif\nCase noire = mur     ● = Agent"
    generer_grille = staticmethod(generer_labyrinthe)

    def __init__(self, root):
        self.root = root
        self.root.title(self.TITRE)
        self.root.geometry("1100x750")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#f2f2f2")

        self.taille = 5
        self.grille = self.generer_grille(self.taille)
        self.env = None
        self.position = (0, 0)
        self.etapes = 0
        self.reward_total = 0
        self.derniere_action = 4
        self.modele = None
        self.en_attente = False
        self.termine = False
        self.apres_id = None
        self.boutons_modeles = {}

        self._construire_interface()
        self.dessiner_labyrinthe()
        self._verifier_api()

    def _construire_interface(self):
        principal = tk.Frame(self.root, bg="#f2f2f2")
        principal.pack(fill="both", expand=True, padx=20, pady=20)
        principal.grid_columnconfigure(0, weight=1)
        principal.grid_rowconfigure(0, weight=1)

        cadre_labyrinthe = tk.LabelFrame(
            principal, text="Labyrinthe", font=("Arial", 12, "bold"),
            bg="white", padx=10, pady=10,
        )
        cadre_labyrinthe.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        self.canvas = tk.Canvas(
            cadre_labyrinthe, width=520, height=520, bg="white",
            highlightthickness=1, highlightbackground="#aaaaaa",
        )
        self.canvas.pack(expand=True)

        controle = tk.Frame(principal, width=440, bg="#f2f2f2")
        controle.grid(row=0, column=1, sticky="ns")
        controle.grid_propagate(False)
        tk.Label(
            controle, text="Contrôle de l'agent", font=("Arial", 16, "bold"),
            bg="#f2f2f2",
        ).pack(pady=(0, 12))

        modeles = tk.LabelFrame(controle, text="Modèle servi par l'API", bg="white", padx=8, pady=8)
        modeles.pack(fill="x", pady=(0, 10))
        for nom, cle in MODELES_API.items():
            bouton = tk.Button(
                modeles, text=nom, font=("Arial", 10, "bold"),
                command=lambda cle=cle, nom=nom: self.selectionner_modele(cle, nom),
            )
            bouton.pack(side="left", expand=True, fill="x", padx=4, pady=4)
            self.boutons_modeles[cle] = bouton

        cadre_url = tk.LabelFrame(controle, text="Adresse de l'API", bg="white", padx=8, pady=8)
        cadre_url.pack(fill="x", pady=(0, 10))
        self.url_var = tk.StringVar(value=API_URL)
        tk.Entry(cadre_url, textvariable=self.url_var).pack(fill="x", padx=4, pady=4)
        tk.Button(cadre_url, text="Vérifier la connexion", command=self._verifier_api).pack(anchor="e", padx=4)

        cadre_taille = tk.LabelFrame(controle, text="Taille du labyrinthe", bg="white", padx=8, pady=8)
        cadre_taille.pack(fill="x", pady=(0, 10))
        self.combo_taille = ttk.Combobox(cadre_taille, values=self.TAILLES_DISPONIBLES, state="readonly")
        self.combo_taille.set("5x5")
        self.combo_taille.pack(fill="x", padx=4, pady=4)

        actions = tk.Frame(controle, bg="#f2f2f2")
        actions.pack(fill="x", pady=(0, 10))
        for index, (texte, commande) in enumerate((
            ("Nouveau labyrinthe", self.nouveau_labyrinthe),
            ("Démarrer", self.demarrer),
            ("Réinitialiser", self.reinitialiser),
        )):
            tk.Button(actions, text=texte, command=commande).grid(row=0, column=index, padx=2, sticky="ew")
            actions.grid_columnconfigure(index, weight=1)

        infos = tk.LabelFrame(controle, text="Résultats", bg="white", padx=12, pady=8)
        infos.pack(fill="x", pady=(0, 10))
        self.label_agent = self._ligne_info(infos, 0, "Modèle :", "Aucun")
        self.label_taille = self._ligne_info(infos, 1, "Taille :", "5x5")
        self.label_etapes = self._ligne_info(infos, 2, "Étapes :", "0")
        self.label_reward = self._ligne_info(infos, 3, "Récompense :", "0")
        self.label_source = self._ligne_info(infos, 4, "Réponse API :", "—")
        self.label_statut = self._ligne_info(infos, 5, "Statut :", "En attente")

        legende = tk.LabelFrame(controle, text="Légende", bg="white", padx=8, pady=8)
        legende.pack(fill="x")
        tk.Label(legende, text=self.LEGENDE, bg="white", justify="left").pack(anchor="w")

    @staticmethod
    def _ligne_info(parent, row, title, value):
        tk.Label(parent, text=title, font=("Arial", 10, "bold"), bg="white").grid(row=row, column=0, sticky="w", pady=3)
        label = tk.Label(parent, text=value, font=("Arial", 10), bg="white", anchor="w", wraplength=245)
        label.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        return label

    def _url_api(self):
        url = self.url_var.get().strip().rstrip("/")
        if not url:
            raise ValueError("Saisis l'adresse de l'API.")
        return url

    def _requete_api(self, chemin, payload=None, timeout=20):
        url = self._url_api() + chemin
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"L'API a répondu HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Connexion à l'API impossible : {exc}") from exc

    def _verifier_api(self):
        self.label_statut.config(text="Vérification de l'API…")

        def worker():
            try:
                # L'API Render peut avoir besoin d'une minute pour sortir de veille.
                result = self._requete_api("/health", timeout=120)
                if result.get("status") != "ok":
                    status = "Réponse API inattendue"
                elif not result.get("model_loaded", False):
                    status = "API joignable, aucun modèle chargé"
                else:
                    status = "API joignable, modèle chargé"
                self.root.after(0, lambda: self.label_statut.config(text=status))
            except Exception as exc:
                self.root.after(0, lambda err=str(exc): self.label_statut.config(text=err))

        threading.Thread(target=worker, daemon=True).start()

    def selectionner_modele(self, cle, nom):
        self.modele = cle
        self.label_agent.config(text=nom)
        for modele, bouton in self.boutons_modeles.items():
            bouton.config(relief="sunken" if modele == cle else "raised")
        self.label_statut.config(text="Modèle sélectionné")

    def nouveau_labyrinthe(self):
        valeur = self.combo_taille.get() or "5x5"
        self.taille = int(valeur.split("x")[0])
        self.grille = self.generer_grille(self.taille, seed=random.randint(0, 1_000_000))
        self.reinitialiser()
        self.label_taille.config(text=valeur)

    def reinitialiser(self):
        if self.apres_id is not None:
            self.root.after_cancel(self.apres_id)
            self.apres_id = None
        self.en_attente = False
        self.termine = False
        self.position = (0, 0)
        self.etapes = 0
        self.reward_total = 0
        self.derniere_action = 4
        self.label_etapes.config(text="0")
        self.label_reward.config(text="0")
        self.label_source.config(text="—")
        self.label_statut.config(text="En attente")
        self.dessiner_labyrinthe()

    def demarrer(self):
        if self.modele is None:
            messagebox.showwarning("Choisir un modèle", "Sélectionne Q-Learning V2 ou DQN V2.")
            return
        self.reinitialiser()
        self.env = LabyrintheEnv(self.grille, max_steps=200)
        self.position = self.env.reset()
        self.label_statut.config(text="Demande d'action à l'API…")
        self.apres_id = self.root.after(250, self.executer_etape)

    def executer_etape(self):
        self.apres_id = None
        if self.termine or self.en_attente:
            return
        if self.etapes >= 200:
            self.termine = True
            self.label_statut.config(text="Échec — limite de 200 mouvements")
            return

        self.en_attente = True
        payload = {
            "model_type": self.modele,
            "grille": self.grille,
            "position": list(self.position),
            "derniere_action": self.derniere_action,
        }

        def worker():
            try:
                # Render peut mettre son service en veille : laisse le temps au réveil.
                result = self._requete_api("/predict_maze", payload, timeout=120)
                self.root.after(0, lambda: self._appliquer_action(result))
            except Exception as exc:
                self.root.after(0, lambda err=str(exc): self._erreur_api(err))

        threading.Thread(target=worker, daemon=True).start()

    def _erreur_api(self, erreur):
        self.en_attente = False
        self.termine = True
        self.label_statut.config(text="Erreur API")
        messagebox.showerror("Appel à l'API impossible", erreur)

    def _appliquer_action(self, resultat):
        self.en_attente = False
        try:
            action = int(resultat["action"])
            if action not in range(4):
                raise ValueError("L'action reçue doit être entre 0 et 3.")
            position, reward, done = self.env.step(action)
        except (KeyError, TypeError, ValueError) as exc:
            self._erreur_api(f"Réponse invalide de l'API : {exc}")
            return

        self.position = position
        self.derniere_action = action
        self.etapes += 1
        self.reward_total += reward
        self.label_etapes.config(text=str(self.etapes))
        self.label_reward.config(text=str(self.reward_total))
        self.label_source.config(text=str(resultat.get("source", "inconnue")))
        self.dessiner_labyrinthe()

        if done:
            self.termine = True
            self.label_statut.config(text="✓ Objectif atteint" if reward == 100 else "✗ Échec")
            return
        self.label_statut.config(text="En cours — action décidée par l'API")
        self.apres_id = self.root.after(300, self.executer_etape)

    def dessiner_labyrinthe(self):
        self.canvas.delete("all")
        lignes, colonnes = len(self.grille), len(self.grille[0])
        largeur = hauteur = 500
        case_size = min(largeur / colonnes, hauteur / lignes)
        offset_x = (largeur - case_size * colonnes) / 2
        offset_y = (hauteur - case_size * lignes) / 2
        for i, ligne in enumerate(self.grille):
            for j, case in enumerate(ligne):
                x1, y1 = offset_x + j * case_size, offset_y + i * case_size
                x2, y2 = x1 + case_size, y1 + case_size
                couleur = "#333333" if case == "#" else "#d9ead3" if case == "S" else "#f4cccc" if case == "G" else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="#cccccc")
                if case in ("S", "G"):
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=case, font=("Arial", max(10, int(case_size * 0.35)), "bold"))

        row, col = self.position
        cx = offset_x + col * case_size + case_size / 2
        cy = offset_y + row * case_size + case_size / 2
        radius = case_size * 0.25
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#4a86e8", outline="#1c4587", width=2)


def main():
    root = tk.Tk()
    LabyrintheApiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
