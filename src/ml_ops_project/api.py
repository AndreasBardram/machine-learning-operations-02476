import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import anyio
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary, make_asgi_app
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

try:
    from ml_ops_project.model_transformer import TransformerTransactionModel
except Exception:  # pragma: no cover
    TransformerTransactionModel = None

METRICS_REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter("api_requests_total", "Total requests to /predict.", registry=METRICS_REGISTRY)
ERRORS_TOTAL = Counter("api_errors_total", "Total errors in /predict.", registry=METRICS_REGISTRY)
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency for /predict.", registry=METRICS_REGISTRY)
INPUT_LENGTH = Summary("api_input_length_chars", "Input size for /predict.", registry=METRICS_REGISTRY)


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


def _find_latest_ckpt_in_dirs(dir_paths: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for dir_path in dir_paths:
        ckpt = _find_latest_ckpt(dir_path)
        if ckpt is not None:
            candidates.append(ckpt)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
        checkpoint_dir = os.getenv("MODEL_CHECKPOINT_DIR")
        if checkpoint_dir:
            ckpt = _find_latest_ckpt(Path(checkpoint_dir))
        else:
            ckpt = _find_latest_ckpt_in_dirs(
                [
                    Path("models/checkpoints_transformer"),
                    Path("models/checkpoints_transformer_subset"),
                ]
            )

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

    # Load training data
    # We prefer the processed subset that the model was likely trained on,
    # but fallback to raw data if needed.
    processed_path = Path("data/processed/transactiq_processed_text_subset_5000")
    raw_path = Path("data/raw/transactiq_enriched_hf")

    if processed_path.exists():
        ds = load_from_disk(str(processed_path))
    else:
        if not raw_path.exists():
            ds = load_dataset("sreesharvesh/transactiq-enriched")
            ds.save_to_disk(str(raw_path))
        ds = load_from_disk(str(raw_path))

    if hasattr(ds, "keys") and "train" in ds:
        app.state.training_data = ds["train"].to_pandas()
    else:
        app.state.training_data = ds.to_pandas()

    yield


app = FastAPI(title="ml_ops_project inference API", version="0.1.0", lifespan=_lifespan)
app.mount("/metrics", make_asgi_app(registry=METRICS_REGISTRY))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def save_data_to_bucket(texts: list[str], predictions: list[dict], model_id: str) -> None:
    from google.cloud import storage

    bucket_name = os.getenv("DATA_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("DATA_BUCKET_NAME environment variable is not set.")

    timestamp = datetime.now(tz=UTC)

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
def predict(req: PredictRequest) -> PredictResponse:
    REQUESTS_TOTAL.inc()
    start_time = time.perf_counter()
    predictor: Predictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    texts = [req.text] if req.text is not None else (req.texts or [])
    texts = [t.strip() for t in texts if t is not None and t.strip() != ""]
    if not texts:
        ERRORS_TOTAL.inc()
        raise HTTPException(status_code=422, detail="No non-empty texts provided.")


    

    total_chars = sum(len(t) for t in texts)
    INPUT_LENGTH.observe(total_chars)

    try:
        preds = predictor.predict(texts)
        # Save data to bucket in the background
        pred_dicts = [pred.model_dump() for pred in preds]
        background_tasks.add_task(save_data_to_bucket, texts, pred_dicts, predictor.model_id)
    except Exception:
        ERRORS_TOTAL.inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)
    return PredictResponse(predictions=preds, model=predictor.model_id)


def download_files(n: int = 5):
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed or configured. Skipping download.")
        return

    bucket_name = os.getenv("DATA_BUCKET_NAME")
    if not bucket_name:
        print("DATA_BUCKET_NAME environment variable is not set. Skipping download.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # list_blobs returns an iterator, convert to list to sort
        blobs = list(bucket.list_blobs(prefix="predictions/"))
        blobs.sort(key=lambda x: x.updated, reverse=True)
        latest_blobs = blobs[:n]

        for blob in latest_blobs:
            local_path = Path(blob.name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
    except Exception as e:
        print(f"Failed to download files: {e}")


def load_latest_files(directory: Path, n: int = 5) -> pd.DataFrame:
    """
    Load latest n files from the from the directory
    """
    # download latest n files from the bucket
    download_files(n=n)

    p = directory / "predictions"
    if not p.exists():
        return pd.DataFrame(columns=["transaction_description", "category"])

    # get all prediction files in the predictions/ folder
    files = list(p.glob("predictions_*.json"))

    # sort files by when they were created
    files = sorted(files, key=os.path.getmtime)

    # load latest n files
    latest_files = files[-n:]

    all_texts = []
    all_preds = []
    for file in latest_files:
        with open(file) as f:
            content = json.load(f)
            texts = content.get("texts", [])
            preds = content.get("predictions", [])  # list of dicts

            if len(texts) == len(preds):
                all_texts.extend(texts)
                # Extract 'label' from the prediction dictionary
                all_preds.extend([pred.get("label") for pred in preds])

    return pd.DataFrame({"transaction_description": all_texts, "category": all_preds})


def generate_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> str:
    # Ensure comparsion on common columns
    common_cols = [c for c in reference_data.columns if c in current_data.columns]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data[common_cols], current_data=current_data[common_cols])
    snapshot.save_html("data_drift_report.html")


@app.get("/report", response_class=HTMLResponse)
async def data_drift_report():
    """
    Endpoint for data drift monitoring.
    """
    training_data = getattr(app.state, "training_data", None)
    if training_data is None:
        return HTMLResponse("<h1>Training data not available</h1>", status_code=503)

    prediction_data = load_latest_files(Path("."), n=5)

    if prediction_data.empty:
        return HTMLResponse("<h1>No prediction data found</h1>", status_code=404)

    generate_drift_report(training_data, prediction_data)

    async with await anyio.open_file("data_drift_report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
