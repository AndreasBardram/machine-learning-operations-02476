import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project import api as api_module  # noqa: E402


class _DummyTokenizer:
    def __call__(self, texts, **kwargs):  # noqa: ARG002
        batch_size = len(texts)
        return {
            "input_ids": torch.zeros((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
        }


class _DummyModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummyModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.to_device = None
        self.was_eval = False

    def __call__(self, **kwargs):  # noqa: ARG002
        return _DummyModelOutput(self._logits)

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.was_eval = True
        return self


def test_predict_request_validation():
    with pytest.raises(ValidationError):
        api_module.PredictRequest()
    with pytest.raises(ValidationError):
        api_module.PredictRequest(text="   ")

    req = api_module.PredictRequest(texts=[" A "])
    assert req.texts == [" A "]


def test_transformer_predictor_predict_label_mapping():
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    predictor = api_module.TransformerPredictor(
        model=_DummyModel(logits),
        tokenizer=_DummyTokenizer(),
        labels=["neg", "pos"],
        max_length=8,
        device=torch.device("cpu"),
        model_id="x",
    )
    preds = predictor.predict(["a", "b"])
    assert [p.label for p in preds] == ["pos", "neg"]
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)


def test_parse_labels_from_env(monkeypatch):
    monkeypatch.delenv("LABELS", raising=False)
    assert api_module._parse_labels_from_env() is None

    monkeypatch.setenv("LABELS", " a, b ,,  ")
    assert api_module._parse_labels_from_env() == ["a", "b"]


def test_find_latest_ckpt(tmp_path):
    assert api_module._find_latest_ckpt(tmp_path / "missing") is None

    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_text("a")
    b.write_text("b")
    a.touch()
    b.touch()
    assert api_module._find_latest_ckpt(tmp_path).name in {"a.ckpt", "b.ckpt"}


def test_create_predictor_hf_path(monkeypatch):
    dummy_model = _DummyModel(torch.zeros((1, 2)))

    monkeypatch.delenv("MODEL_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("MODEL_NAME_OR_PATH", "some-model")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MAX_LENGTH", "12")

    monkeypatch.setattr(api_module, "_find_latest_ckpt", lambda _: None)
    monkeypatch.setattr(
        api_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda name: dummy_model if name == "some-model" else None,
    )
    monkeypatch.setattr(api_module.AutoTokenizer, "from_pretrained", lambda _name: _DummyTokenizer())

    predictor = api_module.create_predictor()
    assert predictor.model_id == "some-model"
    assert predictor.max_length == 12
    assert dummy_model.was_eval is True
    assert str(dummy_model.to_device) == "cpu"


def test_create_predictor_ckpt_path(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("x")

    class _DummyLightning:
        def __init__(self) -> None:
            self.model = _DummyModel(torch.zeros((1, 2)))
            self.hparams = SimpleNamespace(model_name_or_path="tok")

    class _DummyTTM:
        @staticmethod
        def load_from_checkpoint(_path):
            return _DummyLightning()

    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("MODEL_NAME_OR_PATH", raising=False)

    monkeypatch.setattr(api_module, "TransformerTransactionModel", _DummyTTM)
    monkeypatch.setattr(api_module.AutoTokenizer, "from_pretrained", lambda _name: _DummyTokenizer())

    predictor = api_module.create_predictor()
    assert predictor.model_id.startswith("lightning_ckpt:")


def test_predict_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: SimpleNamespace(model_id="x", predict=lambda _: []))
    with TestClient(api_module.app) as client:
        del client.app.state.predictor
        resp = client.post("/predict", json={"texts": ["hi"]})
        assert resp.status_code == 503


def test_predict_422_when_all_texts_blank(monkeypatch):
    monkeypatch.setattr(api_module, "create_predictor", lambda: SimpleNamespace(model_id="x", predict=lambda _: []))
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={"texts": ["  ", ""]})
        assert resp.status_code == 422
