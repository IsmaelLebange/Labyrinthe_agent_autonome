"""
DQNAdapter : réseau DQN de Marie-France (models/dqn.py).

Le réseau reprend exactement ses noms de couches (self.network), donc les clés du
state_dict correspondent. La taille d'entrée et la taille cachée sont lues dans
les poids eux-mêmes : pas besoin de les connaître à l'avance.
"""
from typing import Sequence

import torch
import torch.nn as nn

from .base import ModelAdapter


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int = 30, hidden_size: int = 128, output_size: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAdapter(ModelAdapter):
    def __init__(self) -> None:
        self._model = None
        self._input_size = None
        self._loaded = False

    def load(self, path: str) -> None:
        if not path.endswith((".pth", ".pt")):
            raise ValueError(f"Format non supporté: {path} (attendu .pth ou .pt)")

        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        hidden_size, input_size = state_dict["network.0.weight"].shape
        output_size = state_dict["network.4.weight"].shape[0]

        model = DQNNetwork(input_size, hidden_size, output_size)
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        self._input_size = input_size
        self._loaded = True

    def act(self, state: Sequence[float]) -> int:
        if not self._loaded:
            raise RuntimeError("Aucun modèle DQN chargé. Appeler load() d'abord.")

        values = [float(v) for v in state]
        if len(values) != self._input_size:
            raise ValueError(f"État de taille {len(values)} reçu, {self._input_size} attendu")

        x = torch.tensor([values], dtype=torch.float32)
        with torch.no_grad():
            q_values = self._model(x)
        return int(torch.argmax(q_values, dim=1).item())

    @property
    def is_loaded(self) -> bool:
        return self._loaded