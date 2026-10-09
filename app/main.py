"""API d'inférence pour les modèles Q-Learning V2 et DQN V2."""

import logging
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adapters import DQNAdapter, ModelAdapter, QTableAdapter
from environments.qlearning_state_encoder_v2 import encoder_etat_qlearning_v2

logger = logging.getLogger(__name__)

ModelType = Literal["qlearning", "dqn"]

MODEL_REGISTRY: Dict[str, str] = {
    "dqn": os.getenv("DQN_MODEL_PATH", "results/dqn_v2/model.pth"),
    "qlearning": os.getenv(
        "QLEARNING_MODEL_PATH",
        "results/qlearning_improved_v2/q_table.npy",
    ),
}

app = FastAPI(title="Labyrinthe RL - API d'inférence")
_adapters: Dict[str, ModelAdapter] = {}


class LoadModelRequest(BaseModel):
    model_type: ModelType
    path: Optional[str] = None


class PredictRequest(BaseModel):
    model_type: ModelType = "dqn"
    state: List[float]


class PredictMazeRequest(BaseModel):
    model_type: ModelType = "dqn"
    grille: List[List[str]]
    position: Tuple[int, int]
    derniere_action: int = 4


class PredictResponse(BaseModel):
    action: int
    action_label: str
    source: Literal["model", "mock"]


ACTION_LABELS = {0: "haut", 1: "bas", 2: "gauche", 3: "droite"}


def _new_adapter(model_type: str) -> ModelAdapter:
    if model_type == "qlearning":
        return QTableAdapter()
    return DQNAdapter()


def _load_adapter(model_type: str, path: str) -> ModelAdapter:
    adapter = _new_adapter(model_type)
    adapter.load(path)
    _adapters[model_type] = adapter
    return adapter


@app.on_event("startup")
def load_configured_models() -> None:
    """Charge les fichiers présents au démarrage; les absents restent visibles via /models."""
    for model_type, path in MODEL_REGISTRY.items():
        if not os.path.isfile(path):
            logger.warning("Modèle %s introuvable : %s", model_type, path)
            continue
        try:
            _load_adapter(model_type, path)
            logger.info("Modèle %s chargé depuis %s", model_type, path)
        except Exception:
            logger.exception("Échec du chargement du modèle %s depuis %s", model_type, path)


@app.get("/health")
def health():
    loaded = {
        model_type: adapter.is_loaded
        for model_type, adapter in _adapters.items()
    }
    return {
        "status": "ok",
        "model_loaded": any(loaded.values()),
        "loaded_models": loaded,
    }


@app.get("/models")
def list_models():
    return {
        model_type: {
            "path": path,
            "available": os.path.isfile(path),
            "loaded": model_type in _adapters and _adapters[model_type].is_loaded,
        }
        for model_type, path in MODEL_REGISTRY.items()
    }


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    path = req.path or MODEL_REGISTRY[req.model_type]
    try:
        _load_adapter(req.model_type, path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Échec du chargement : {exc}") from exc

    return {"status": "loaded", "model_type": req.model_type, "path": path}


def _decide(model_type: str, state: List[float]) -> PredictResponse:
    if len(state) != 12:
        raise HTTPException(
            status_code=422,
            detail=f"État V2 invalide : 12 valeurs attendues, {len(state)} reçues.",
        )

    adapter = _adapters.get(model_type)
    if adapter is None or not adapter.is_loaded:
        action = random.randint(0, 3)
        source = "mock"
    else:
        try:
            action = adapter.act(state)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        source = "model"

    return PredictResponse(
        action=action,
        action_label=ACTION_LABELS[action],
        source=source,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Prédit à partir d'un état V2 déjà encodé (12 valeurs)."""
    return _decide(req.model_type, req.state)


@app.post("/predict_maze", response_model=PredictResponse)
def predict_maze(req: PredictMazeRequest):
    """Encode la grille et prédit l'action pour le modèle demandé."""
    try:
        state = encoder_etat_qlearning_v2(
            req.grille,
            req.position,
            req.derniere_action,
        )
    except (ValueError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Grille/position invalide : {exc}") from exc

    return _decide(req.model_type, list(state))
