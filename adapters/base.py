"""
ModelAdapter : interface commune pour brancher n'importe quel agent
(Q-learning tabulaire ou DQN) derrière la même API FastAPI.

Contrat fixé par le document de cadrage du projet :
    - état  = vecteur V2 de 12 valeurs (observation locale, direction objectif,
      dernière action)
    - action = entier dans {0,1,2,3}    -> 0=haut, 1=bas, 2=gauche, 3=droite
"""
from abc import ABC, abstractmethod
from typing import Sequence


class ModelAdapter(ABC):
    """Interface que tout adapter de modèle doit respecter."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Charge le modèle depuis un fichier sur disque."""
        raise NotImplementedError

    @abstractmethod
    def act(self, state: Sequence[float]) -> int:
        """Retourne l'action (0..3) à jouer pour l'état donné."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True si un modèle est actuellement chargé en mémoire."""
        raise NotImplementedError
