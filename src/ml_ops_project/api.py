import os
import json

import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

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


def _labels_from_model_config(model: Any) -> list[str] | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    id2label = getattr(config, "id2label", None)
    if not id2label:
        return None
    if isinstance(id2label, dict):
        try:
            items = sorted(id2label.items(), key=lambda kv: int(kv[0]))
        except Exception:
            items = sorted(id2label.items(), key=lambda kv: str(kv[0]))
        labels = [str(label) for _, label in items]
        return labels or None
    if isinstance(id2label, list):
        labels = [str(label) for label in id2label]
        return labels or None
    return None


def _labels_from_lightning_hparams(lightning_model: Any) -> list[str] | None:
    hparams = getattr(lightning_model, "hparams", None)
    if hparams is None:
        return None
    labels = hparams.get("labels") if isinstance(hparams, dict) else getattr(hparams, "labels", None)
    return labels if labels else None


def _find_latest_ckpt(dir_path: Path) -> Path | None:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    ckpts = sorted(dir_path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def create_predictor() -> Predictor:
    use_dummy = os.getenv("USE_DUMMY_PREDICTOR", "").strip().lower() in {"1", "true", "yes"}
    if use_dummy:
        labels = _parse_labels_from_env() or ["DUMMY"]
        return DummyPredictor(labels=labels)

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
        if labels is None:
            labels = _labels_from_lightning_hparams(lightning_model)
    else:
        model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "distilbert-base-uncased")
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_id = model_name_or_path

    if labels is None:
        labels = _labels_from_model_config(hf_model)

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


def save_data_to_bucket(texts: list[str], predictions: list[dict], model_id: str) -> None:
    from google.cloud import storage

    bucket_name = os.getenv("DATA_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("DATA_BUCKET_NAME environment variable is not set.")

    timestamp = datetime.now(tz=timezone.utc)

    data = {
        "timestamp": timestamp.isoformat(),
        "model_id": model_id,
        "texts": texts,
        "predictions": predictions,
    }

    print("data", data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"predictions/predictions_{timestamp}.json")
    blob.upload_from_string(json.dumps(data))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """
    Make predictions for transaction descriptions.
    And save the results to a data bucket for later analysis.

    Args:
        req (PredictRequest): The prediction request containing text(s).
    Returns:
        PredictResponse: The prediction results.

    """
    predictor: Predictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    texts = [req.text] if req.text is not None else (req.texts or [])
    texts = [t.strip() for t in texts if t is not None and t.strip() != ""]
    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty texts provided.")

    preds = predictor.predict(texts)

    # Save data to bucket in the background
    pred_dicts = [pred.model_dump() for pred in preds]
    background_tasks.add_task(save_data_to_bucket, texts, pred_dicts, predictor.model_id)

    return PredictResponse(predictions=preds, model=predictor.model_id)


def download_files(n: int = 5):
    from google.cloud import storage

    # load current data from bucket
    bucket_name = os.getenv("DATA_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("DATA_BUCKET_NAME environment variable is not set.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # data is stored in folder "predictions/" as prediction_{timestamp}.json
    blobs = bucket.list_blobs(prefix="predictions/")
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    for blob in latest_blobs:
        blob.download_to_filename(blob.name)


def load_latest_files(directory: Path, n: int = 5):
    """
    Load latest n files from the from the directory
    """
    # download latest n files from the bucket
    download_files(n=n)

    # get all prediction files in the predictions/ folder
    files = directory.glob("predictions/prediction_*.json")

    # sort files by when they were created
    files = sorted(files, key=os.path.getmtime)

    # load latest n files
    latest_files = files[-n:]

    data = {"texts": [], "target": []}
    for file in latest_files:
        with open(file, "r") as f:
            data = json.load(f)
            data["texts"].extend(data.get("texts", []))
            data["target"].extend(data.get("predictions", []))
    df = pd.DataFrame(data)
    return df


def generate_drift_report(reference_data: pd.DataFrame, prediction_data: pd.DataFrame) -> str:
    from evidently import Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from evidently.report import Report

    data_definition = DataDefinition(
        text_columns=["texts"],
        categorical_columns=["target"],
    )

    reference_dataset = Dataset(data=reference_data, data_definition=data_definition)
    prediction_dataset = Dataset(data=prediction_data, data_definition=data_definition)

    preset = DataDriftPreset()
    report = Report(metrics=[preset])
    report.run(reference_data=reference_dataset, current_data=prediction_dataset)

    # generate HTML report
    html_report = report.as_html()
    return html_report

@app.get("/")
async def data_drift_report():
    """
    Endpoint for data drift monitoring.
    """
    from evidently import Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from google.cloud import storage

    # load reference data (training data)
    reference_data = ""

    # load latest data from the bucket
    prediction_data = load_latest_files(".", n=5)



    




    




