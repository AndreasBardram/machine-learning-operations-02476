# Receipt line-item classifier (MLOps)

End-to-end ML system for classifying short transaction descriptions (receipt line-items / bank statement text) into spending categories. It includes data preprocessing, baseline + transformer training, evaluation, ONNX export, FastAPI/ONNX inference APIs, Streamlit UI, load tests, CI, and deployment docs.

## Environment
- Python 3.11 (see `pyproject.toml`).
- Dependency manager: `uv`.

```bash
uv sync
# dev tools
uv sync --group dev
```

If you need tracked data/models, pull them with DVC:
```bash
dvc pull
```

## Common tasks (Invoke)
All tasks are defined in `tasks.py` and should be run via `uv`:

```bash
# Data
uv run invoke preprocess-data
uv run invoke preprocess-data --subset
uv run invoke preprocess-data-transformer

# Train
uv run invoke train --epochs 10
uv run invoke train --epochs 10 --subset
uv run invoke train --epochs 10 --experiment baseline_full
uv run invoke train-transformer --epochs 10
uv run invoke train-transformer --epochs 10 --experiment transformer_full

# Tests
uv run invoke test

# Lint
uv run invoke lint
uv run invoke lint --fix

# Docs (MkDocs)
uv run invoke build-docs
uv run invoke serve-docs

# ONNX API
uv run invoke run-onnx-api
uv run invoke test-onnx-api
```

## Data and artifacts (DVC)
- Tracked via `data.dvc` and `models.dvc`.
- Local data lives under `data/` and models under `models/`.
- Use `dvc pull` to fetch artifacts when missing.

## Training and evaluation
- Baseline model: TF-IDF + linear classifier.
- Transformer model: fine-tuned DistilBERT via Hydra configs.
- Entry points:
  - `src/ml_ops_project/train.py`
  - `src/ml_ops_project/train_transformer.py`
  - `src/ml_ops_project/evaluate.py`

Hydra configs live in `configs/`, including experiments under `configs/experiment/` and subsets in `configs/*_subset*.yaml`.

## Inference APIs
### FastAPI (PyTorch)
```bash
uv run uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `POST /predict` with `{"text": "STARBUCKS"}` or `{"texts": ["STARBUCKS", "UBER"]}`

Model loading (first match wins):
- `MODEL_CHECKPOINT_PATH=/path/to/checkpoint.ckpt`
- `MODEL_CHECKPOINT_DIR=models/checkpoints_transformer`
- `MODEL_NAME_OR_PATH=distilbert-base-uncased`

Optional:
- `LABELS="Food & Dining,Transportation,..."`
- `MAX_LENGTH=64`
- `DEVICE=cpu`

### ONNX FastAPI
```bash
uv run invoke run-onnx-api
```

ONNX utilities:
- Export model: `src/ml_ops_project/convert_model_to_onnx.py`
- Run local ONNX inference: `src/ml_ops_project/run_onnx_model.py`
- Visualize ONNX graph: `src/ml_ops_project/view_onnx_graph.py`

## Streamlit UI
```bash
uv run streamlit run src/ml_ops_project/streamlit_app.py
```
By default it targets `http://127.0.0.1:8000`. Start the FastAPI server first or update the sidebar URL.

## Tests
- Unit tests with coverage:
  ```bash
  uv run invoke test
  ```

- Integration tests (requires a running API and `MYENDPOINT`):
  ```bash
  USE_DUMMY_PREDICTOR=1 uv run uvicorn src.ml_ops_project.api:app --host 127.0.0.1 --port 8000
  MYENDPOINT=http://127.0.0.1:8000 uv run pytest -q tests/integrationtests
  ```

- ONNX API tests (requires ONNX API running):
  ```bash
  uv run invoke run-onnx-api
  uv run invoke test-onnx-api
  ```

## Load testing
- Locust entrypoint: `locustfile.py`.
- Example results and CSVs are stored under `docs/load_tests/`.

## Linting and formatting
```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
```

## Docker
- Monolithic image (API/train/eval/preprocess/onnx-api): `Dockerfile`.
- Component images: `dockerfiles/*.dockerfile`.
- Entrypoint: `docker/entrypoint.sh`.

```bash
# Single image (API, train, eval, preprocess, onnx-api)
docker build -t ml-ops-app .

# Run API (default)
docker run --rm -p 8000:8000 ml-ops-app

# Train/eval (mount data/models)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" ml-ops-app train trainer.max_epochs=1
```

## Docs and deployment
- MkDocs config: `docs/mkdocs.yaml`, sources in `docs/source/`.
- Deployment guide: `docs/deploy_gcp_cloud_run.md` (Cloud Run via GitHub Actions + WIF).

## Project structure
```txt
.
├── AGENTS.md                     # Local agent instructions
├── Dockerfile                    # Monolithic container (multi-mode entrypoint)
├── Makefile                      # Format/check + coverage update helper
├── config.yaml                   # Root config entry
├── configs/                      # Hydra configs and experiments
│   ├── default.yaml
│   ├── subset.yaml
│   ├── transformer_default.yaml
│   ├── transformer_default_subset.yaml
│   ├── transformer_default_subset_2.yaml
│   ├── sweep.yaml
│   └── experiment/
│       ├── baseline_full.yaml
│       ├── baseline_subset.yaml
│       └── transformer_full.yaml
├── data.dvc                      # DVC data tracking
├── models.dvc                    # DVC model tracking
├── docker/                       # Container entrypoint
│   └── entrypoint.sh
├── dockerfiles/                  # Component Dockerfiles
│   ├── api.dockerfile
│   ├── eval.dockerfile
│   ├── test.dockerfile
│   ├── train.dockerfile
│   └── train_transformer.dockerfile
├── docs/                         # MkDocs docs + load test results
│   ├── README.md
│   ├── deploy_gcp_cloud_run.md
│   ├── mkdocs.yaml
│   ├── source/
│   │   └── index.md
│   └── load_tests/
│       ├── locust_baseline_stats.csv
│       ├── locust_baseline_stats_history.csv
│       ├── locust_baseline_failures.csv
│       ├── locust_baseline_exceptions.csv
│       ├── locust_batch8_stats.csv
│       ├── locust_batch8_stats_history.csv
│       ├── locust_batch8_failures.csv
│       └── locust_batch8_exceptions.csv
├── notebooks/                    # Exploratory notebooks
│   └── compare_data.ipynb
├── reports/                      # Reports and figures
├── src/ml_ops_project/           # Core Python package
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── data_transformer.py
│   ├── model.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── streamlit_app.py
│   ├── onnx_fastapi.py
│   ├── convert_model_to_onnx.py
│   ├── run_onnx_model.py
│   ├── view_onnx_graph.py
│   └── test_onnx_api.py
├── tests/                        # Unit + integration tests
│   ├── integrationtests/
│   │   ├── __init__.py
│   │   └── test_apis.py
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_api_more.py
│   ├── test_data.py
│   ├── test_data_transformer.py
│   ├── test_data_transformer_prepare_data.py
│   ├── test_model.py
│   ├── test_model_transformer.py
│   └── test_train_scripts.py
├── locustfile.py                 # Load testing
├── tasks.py                      # Invoke tasks
├── pyproject.toml                # Project metadata and deps
├── uv.lock                       # Locked dependencies
├── dvc-credentials.json          # CI/service account metadata
├── report_dynamo_export.sarif    # Static analysis report
└── ToDo                          # Project checklist/notes
```

Coverage report 21-01-2026

Name                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
src/ml_ops_project/__init__.py                0      0   100%
src/ml_ops_project/api.py                   146     21    86%   47, 56-57, 104-117, 123, 138-139, 155
src/ml_ops_project/data.py                  106      5    95%   55-56, 138-139, 143
src/ml_ops_project/data_transformer.py       85      3    96%   18-20
src/ml_ops_project/model.py                  54      8    85%   84-94
src/ml_ops_project/model_transformer.py      57     12    79%   57-61, 68-72, 79-83
src/ml_ops_project/train.py                  44      1    98%   62
src/ml_ops_project/train_transformer.py      45      1    98%   63
-----------------------------------------------------------------------
TOTAL                                       537     51    91%
