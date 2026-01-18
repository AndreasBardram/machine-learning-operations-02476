import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project import api as api_module  # noqa: E402


class DummyPredictor:
    def __init__(self) -> None:
        self.model_id = "dummy-model"

    def predict(self, texts: list[str]) -> list[api_module.Prediction]:
        return [api_module.Prediction(label="Food & Dining", label_id=3, confidence=0.9) for _ in texts]


def test_health(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: DummyPredictor())
    with TestClient(api_module.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_predict_single_text(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: DummyPredictor())
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={"text": "STARBUCKS"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["model"] == "dummy-model"
        assert payload["predictions"] == [{"label": "Food & Dining", "label_id": 3, "confidence": 0.9}]


def test_predict_batch(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: DummyPredictor())
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={"texts": ["STARBUCKS", "UBER"]})
        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload["predictions"]) == 2


def test_predict_requires_input(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: DummyPredictor())
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={})
        assert resp.status_code == 422
