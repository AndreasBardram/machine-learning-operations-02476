# AGENTS

## Project overview
Receipt line-item classifier with preprocessing, training (baseline + transformer), and inference APIs (FastAPI + ONNX). Uses `uv` for dependency management and `invoke` tasks for common workflows.

## Environment setup
- Python: 3.11 (see `pyproject.toml`).
- Install deps: `uv sync` (or `uv sync --group dev` for dev tools).
- Data/models are tracked via DVC (`data.dvc`, `models.dvc`); if needed: `dvc pull`.

## Common commands (Invoke)
- Preprocess baseline data: `uv run invoke preprocess-data` (add `--subset` for 10%).
- Preprocess transformer data: `uv run invoke preprocess-data-transformer`.
- Train baseline: `uv run invoke train --epochs 10` (add `--subset` or `--experiment NAME`).
- Train transformer: `uv run invoke train-transformer --epochs 10`.
- Run FastAPI: `uv run uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port 8000 --reload`.
- Run ONNX API: `uv run invoke run-onnx-api`.
- Streamlit UI: `uv run streamlit run src/ml_ops_project/streamlit_app.py`.

## Tests
- Unit tests with coverage: `uv run invoke test`.
- Integration tests require a running API and `MYENDPOINT`:
  - `USE_DUMMY_PREDICTOR=1 uv run uvicorn src.ml_ops_project.api:app --host 127.0.0.1 --port 8000`
  - `MYENDPOINT=http://127.0.0.1:8000 uv run pytest -q tests/integrationtests`

## Linting and formatting
- Lint: `uv run invoke lint` (add `--fix` to autofix).
- Format: `uv run ruff format .` (or `uv run ruff format . --check`).

## Model loading for API
First match wins:
- `MODEL_CHECKPOINT_PATH=/path/to/checkpoint.ckpt`
- `MODEL_CHECKPOINT_DIR=models/checkpoints_transformer` (newest `*.ckpt`)
- `MODEL_NAME_OR_PATH=distilbert-base-uncased`

Optional: `LABELS`, `MAX_LENGTH`, `DEVICE`.
