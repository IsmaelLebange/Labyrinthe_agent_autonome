"""
QTableAdapter : Q-table de Marie-France (models/q_learning.py).

Format réel : un dict {clé: array de 4 valeurs}, où la clé est le tuple de
l'état arrondi à 2 décimales (calcul identique à QLearningAgent.state_to_key),
sauvegardé avec np.save(path, dict, allow_pickle=True).
"""
import pickle
from typing import Sequence

import numpy as np

from .base import ModelAdapter


class QTableAdapter(ModelAdapter):
    def __init__(self) -> None:
        self._q = None
        self._loaded = False

    def load(self, path: str) -> None:
        if path.endswith(".npy"):
            data = np.load(path, allow_pickle=True)
            # np.save d'un dict donne un tableau 0-d d'objets : .item() rend le dict
            self._q = data.item() if data.ndim == 0 else data
        elif path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as f:
                self._q = pickle.load(f)
        else:
            raise ValueError(f"Format non supporté: {path} (attendu .npy ou .pkl)")

        if not isinstance(self._q, dict):
            raise ValueError("Q-table attendue sous forme de dict {état: valeurs}")
        self._loaded = True

    @staticmethod
    def _key(state: Sequence[float]) -> tuple:
        # même calcul que state_to_key de l'agent, sinon les clés ne correspondent pas
        return tuple(np.round(np.array(state, dtype=np.float32), 2))

    def act(self, state: Sequence[float]) -> int:
        if not self._loaded:
            raise RuntimeError("Aucune Q-table chargée. Appeler load() d'abord.")
        values = self._q.get(self._key(state))
        if values is None:
            return 0  # état jamais vu à l'entraînement : l'agent d'origine aurait aussi pris l'action 0
        return int(np.argmax(values))

    @property
    def is_loaded(self) -> bool:
        return self._loaded