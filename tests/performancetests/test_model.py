import os
import time
from pathlib import Path

import pytest
import torch
import wandb

from ml_ops_project.model import TransactionModel
from ml_ops_project.model_transformer import TransformerTransactionModel

MODEL_ENV = "MODEL_NAME"
MAX_SECONDS_ENV = "MODEL_PERF_MAX_SECONDS"
DEFAULT_MAX_SECONDS = 10.0


def _get_model_name() -> str:
    model_name = os.getenv(MODEL_ENV, "").strip()
    if not model_name:
        pytest.skip("MODEL_NAME not set; skipping staged model performance test.")
    return model_name


def _download_artifact(model_name: str) -> Path:
    api = wandb.Api()
    artifact = api.artifact(model_name)
    return Path(artifact.download())


def _find_checkpoint(artifact_dir: Path) -> Path:
    for pattern in ("*.ckpt", "*.pt", "*.pth"):
        matches = sorted(artifact_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint files found under {artifact_dir}")


def _try_load_lightning_checkpoint(checkpoint_path: Path):
    for model_cls in (TransactionModel, TransformerTransactionModel):
        try:
            model = model_cls.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
            model.eval()
            return model
        except Exception:
            continue
    return None


def _load_model(checkpoint_path: Path):
    model = _try_load_lightning_checkpoint(checkpoint_path)
    if model is not None:
        return model

    try:
        model = torch.jit.load(str(checkpoint_path))
        model.eval()
        return model
    except Exception:
        pass

    try:
        obj = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", obj)
        model = TransactionModel()
        try:
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as exc:
            raise RuntimeError(f"Checkpoint {checkpoint_path} did not match TransactionModel: {exc}") from exc

    raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}")


def _build_inputs(model) -> tuple[tuple, dict]:
    if hasattr(model, "input_dim"):
        batch = torch.randn(1, int(model.input_dim))
        return (batch,), {}

    vocab_size = 30522
    if hasattr(model, "model") and hasattr(model.model, "config"):
        vocab_size = int(getattr(model.model.config, "vocab_size", vocab_size))

    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    return (), {"input_ids": input_ids, "attention_mask": attention_mask}


def test_staged_model_performance():
    model_name = _get_model_name()
    artifact_dir = _download_artifact(model_name)
    checkpoint_path = _find_checkpoint(artifact_dir)
    model = _load_model(checkpoint_path)

    args, kwargs = _build_inputs(model)

    with torch.no_grad():
        for _ in range(5):
            model(*args, **kwargs)

        start = time.perf_counter()
        for _ in range(100):
            model(*args, **kwargs)
        elapsed = time.perf_counter() - start

    max_seconds = float(os.getenv(MAX_SECONDS_ENV, DEFAULT_MAX_SECONDS))
    assert elapsed < max_seconds, f"100 predictions took {elapsed:.3f}s (limit {max_seconds:.3f}s)"
