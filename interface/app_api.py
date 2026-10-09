"""Copie de l'interface de Marie-France, avec les décisions prises par l'API."""

import json
import os
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from interface.app import LabyrintheApp


API_URL = os.getenv(
    "MAZE_API_URL",
    "https://labyrinthe-agent-autonome.onrender.com",
).rstrip("/")


class LabyrintheAppAPI(LabyrintheApp):
    """Réutilise l'affichage original et délègue les actions à /predict_maze."""

    # L'API actuellement déployée ne sert que les modèles V2 (états de 12 valeurs).
    MODELES = ["Q-Learning V2", "DQN V2"]
    TYPES_API = {"Q-Learning V2": "qlearning", "DQN V2": "dqn"}

    def charger_modeles(self):
        # Le chargement se fait sur le serveur API, jamais dans cette interface.
        pass

    def selectionner_agent(self, nom):
        self.agent_selectionne = nom
        # Valeur sentinelle : demarrer() de l'interface originale vérifie agent is None.
        self.agent = nom
        self.label_agent.config(text=nom)

        for nom_agent, bouton in self.boutons_agents.items():
            bouton.config(relief="sunken" if nom_agent == nom else "raised")

        self.label_statut.config(text="Agent sélectionné")

    def executer_etape(self):
        self.apres_id = None
        if self.termine or getattr(self, "en_attente", False):
            return

        if self.etapes >= 200:
            self.termine = True
            self.label_statut.config(text="Échec - 200 mouvements")
            return

        self.en_attente = True
        payload = {
            "model_type": self.TYPES_API[self.agent_selectionne],
            "grille": self.grille,
            "position": list(self.position),
            "derniere_action": self.derniere_action,
        }

        def demander_action():
            try:
                request = Request(
                    f"{API_URL}/predict_maze",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                # Une instance Render gratuite peut mettre un moment à sortir de veille.
                with urlopen(request, timeout=120) as response:
                    resultat = json.loads(response.read().decode("utf-8"))
                self.root.after(0, lambda: self._appliquer_action_api(resultat))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                erreur = f"L'API a répondu HTTP {exc.code}: {detail}"
                self.root.after(0, lambda err=erreur: self._erreur_api(err))
            except (URLError, TimeoutError, OSError) as exc:
                self.root.after(
                    0,
                    lambda err=f"Connexion à l'API impossible : {exc}": self._erreur_api(err),
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.root.after(
                    0,
                    lambda err=f"Réponse invalide de l'API : {exc}": self._erreur_api(err),
                )

        threading.Thread(target=demander_action, daemon=True).start()

    def _erreur_api(self, erreur):
        self.en_attente = False
        self.termine = True
        self.label_statut.config(text="Erreur API")
        from tkinter import messagebox

        messagebox.showerror("Erreur API", erreur)

    def _appliquer_action_api(self, resultat):
        self.en_attente = False
        try:
            action = int(resultat["action"])
            if action not in range(4):
                raise ValueError("L'action doit être comprise entre 0 et 3.")
            prochain_etat, reward, done = self.env.step(action)
        except (KeyError, TypeError, ValueError) as exc:
            self._erreur_api(f"Réponse invalide de l'API : {exc}")
            return

        self.position = prochain_etat
        self.etapes += 1
        self.reward_total += reward
        self.derniere_action = action

        self.label_etapes.config(text=str(self.etapes))
        self.label_reward.config(text=str(self.reward_total))
        self.dessiner_labyrinthe()

        source = resultat.get("source", "inconnue")
        indication = " (mock)" if source == "mock" else ""
        if done:
            self.termine = True
            statut = "✓ Objectif atteint" if reward == 100 else "✗ Échec"
            self.label_statut.config(text=statut + indication)
            return

        if self.etapes >= 200:
            self.termine = True
            self.label_statut.config(text="✗ Échec - 200 mouvements" + indication)
            return

        self.label_statut.config(text="En cours..." + indication)
        self.apres_id = self.root.after(300, self.executer_etape)


def main():
    import tkinter as tk

    root = tk.Tk()
    LabyrintheAppAPI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
