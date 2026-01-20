import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from ml_ops_project.model_transformer import TransformerTransactionModel
except Exception:  # pragma: no cover
    TransformerTransactionModel = None


class PredictRequest(BaseModel):
    text: str | None = Field(default=None, description="Single transaction description.")
    texts: list[str] | None = Field(default=None, description="Batch of transaction descriptions.")

    @model_validator(mode="after")
    def _validate_input(self):
        if (self.text is None or self.text.strip() == "") and (not self.texts):
            raise ValueError("Provide either `text` or `texts`.")
        return self


class Prediction(BaseModel):
    label: str | None = None
    label_id: int
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str


class Predictor(Protocol):
    model_id: str

    def predict(self, texts: list[str]) -> list[Prediction]:
        ...


@dataclass(frozen=True)
class DummyPredictor:
    labels: list[str] | None
    model_id: str = "dummy"

    def predict(self, texts: list[str]) -> list[Prediction]:
        label = self.labels[0] if self.labels else None
        return [Prediction(label=label, label_id=0, confidence=1.0) for _ in texts]


@dataclass(frozen=True)
class TransformerPredictor:
    model: Any
    tokenizer: Any
    labels: list[str] | None
    max_length: int
    device: torch.device
    model_id: str

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[Prediction]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confs, pred_ids = torch.max(probs, dim=-1)

        results: list[Prediction] = []
        for label_id, confidence in zip(pred_ids.tolist(), confs.tolist(), strict=True):
            label = self.labels[label_id] if self.labels and 0 <= label_id < len(self.labels) else None
            results.append(Prediction(label=label, label_id=label_id, confidence=float(confidence)))
        return results


def _parse_labels_from_env() -> list[str] | None:
    raw = os.getenv("LABELS")
    if not raw:
        return None
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels or None


def _find_latest_ckpt(dir_path: Path) -> Path | None:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    ckpts = sorted(dir_path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def create_predictor() -> Predictor:
    use_dummy = os.getenv("USE_DUMMY_PREDICTOR", "").strip().lower() in {"1", "true", "yes"}
    if use_dummy:
        return DummyPredictor(labels=_parse_labels_from_env())

    device_str = os.getenv("DEVICE", "cpu")
    device = torch.device(device_str)

    max_length = int(os.getenv("MAX_LENGTH", "64"))
    labels = _parse_labels_from_env()

    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
    else:
        ckpt = _find_latest_ckpt(Path(os.getenv("MODEL_CHECKPOINT_DIR", "models/checkpoints_transformer")))

    if ckpt and ckpt.exists():
        if TransformerTransactionModel is None:
            raise RuntimeError("TransformerTransactionModel could not be imported; cannot load Lightning checkpoint.")
        lightning_model = TransformerTransactionModel.load_from_checkpoint(str(ckpt))
        hf_model = lightning_model.model
        model_id = f"lightning_ckpt:{ckpt.as_posix()}"
        tokenizer_name = str(lightning_model.hparams.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "distilbert-base-uncased")
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_id = model_name_or_path

    hf_model.to(device)
    hf_model.eval()

    return TransformerPredictor(
        model=hf_model,
        tokenizer=tokenizer,
        labels=labels,
        max_length=max_length,
        device=device,
        model_id=model_id,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.predictor = create_predictor()
    yield


app = FastAPI(title="ml_ops_project inference API", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor: Predictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    texts = [req.text] if req.text is not None else (req.texts or [])
    texts = [t.strip() for t in texts if t is not None and t.strip() != ""]
    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty texts provided.")

    preds = predictor.predict(texts)
    return PredictResponse(predictions=preds, model=predictor.model_id)
