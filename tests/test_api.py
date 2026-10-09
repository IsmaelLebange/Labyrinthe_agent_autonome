"""Tests des routes d'inférence avec le format d'état V2."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as api

client = TestClient(api.app)

MAZE = [
    ["S", ".", "."],
    ["#", "#", "."],
    [".", ".", "G"],
]


@pytest.fixture(autouse=True)
def clear_loaded_models():
    previous = api._adapters.copy()
    api._adapters.clear()
    yield
    api._adapters.clear()
    api._adapters.update(previous)


def test_health_reports_no_loaded_models():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["loaded_models"] == {}


def test_predict_accepts_v2_state_and_marks_mock_source():
    response = client.post(
        "/predict",
        json={"model_type": "dqn", "state": [0] * 12},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in (0, 1, 2, 3)
    assert body["source"] == "mock"


def test_predict_rejects_non_v2_state_length():
    response = client.post(
        "/predict",
        json={"model_type": "dqn", "state": [0, 1]},
    )
    assert response.status_code == 422


def test_predict_maze_encodes_state_and_returns_action():
    response = client.post(
        "/predict_maze",
        json={
            "model_type": "qlearning",
            "grille": MAZE,
            "position": [0, 0],
            "derniere_action": 4,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in (0, 1, 2, 3)
    assert body["source"] == "mock"


def test_load_model_missing_file_returns_400():
    response = client.post(
        "/load_model",
        json={"model_type": "qlearning", "path": "/chemin/inexistant.npy"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("model_type", "path"),
    [
        ("dqn", "results/dqn_v2/model.pth"),
        ("qlearning", "results/qlearning_improved_v2/q_table.npy"),
    ],
)
def test_v2_model_loads_and_predicts_when_artifact_exists(model_type, path):
    if not Path(path).is_file():
        pytest.skip(f"Artefact local absent : {path}")

    loaded = client.post(
        "/load_model",
        json={"model_type": model_type, "path": path},
    )
    assert loaded.status_code == 200, loaded.text

    predicted = client.post(
        "/predict_maze",
        json={
            "model_type": model_type,
            "grille": MAZE,
            "position": [0, 0],
            "derniere_action": 4,
        },
    )
    assert predicted.status_code == 200, predicted.text
    assert predicted.json()["source"] == "model"
