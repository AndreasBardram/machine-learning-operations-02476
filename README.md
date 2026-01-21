## Project: Receipt line-item classifier
DTU course: 02476, machine learning operations

## Commands to run project:
### Running data processing modules:
- invoke preprocess-data
- invoke preprocess-data --subset  (if you want 10% subset instead)
- invoke preprocess-data-transformer

### DVC tracked artifacts
- Data and trained models are tracked via DVC (`data.dvc`, `models.dvc`).
- Use `dvc pull` to fetch them if they are missing locally.

### Running training modules: (requires data modules to be run first)
- invoke train
- invoke train --epochs 10 --subset (if you want to train for 10 epoch and on subset data)
- invoke train-transformer

## Inference API (FastAPI)
Run the API locally:
- `uv run uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port 8000 --reload`

Endpoints:
- `GET /health`
- `POST /predict` with `{"text": "STARBUCKS"}` or `{"texts": ["STARBUCKS", "UBER"]}`

## Streamlit Frontend
Run the Streamlit UI:
- `uv run streamlit run src/ml_ops_project/streamlit_app.py`

By default it targets `http://127.0.0.1:8000`. Start the FastAPI server first or update the sidebar URL.

## API Integration Tests
The integration tests call a running API over HTTP (via `httpx`) and use `MYENDPOINT` to know where it is deployed.
If `MYENDPOINT` is not set, these tests are skipped.

## Unit Tests
Run the full unit test suite with coverage:
- `uv run invoke test`

Run locally with a lightweight dummy predictor (fast startup, no model downloads):
- `USE_DUMMY_PREDICTOR=1 uv run uvicorn src.ml_ops_project.api:app --host 127.0.0.1 --port 8000`
- `MYENDPOINT=http://127.0.0.1:8000 uv run pytest -q tests/integrationtests`

Run against a deployed API:
- `MYENDPOINT=https://<your-service-url> uv run pytest -q tests/integrationtests`

Model loading (first match wins):
- `MODEL_CHECKPOINT_PATH=/path/to/checkpoint.ckpt` (Lightning checkpoint from `models/checkpoints_transformer/`)
- `MODEL_CHECKPOINT_DIR=models/checkpoints_transformer` (defaults to this; newest `*.ckpt` is used)
- `MODEL_NAME_OR_PATH=distilbert-base-uncased` (Hugging Face model id/path fallback)

Optional:
- `LABELS="Food & Dining,Transportation,..."` to map `label_id` → `label`
- `MAX_LENGTH=64` and `DEVICE=cpu`

## Inference ONNX API
Run the ONNX api locally
- invoke run-onnx-api

test the API (required api to be running locally)
- invoke test-onnx-api

## Docker Images
Build Docker images for different components:

```bash
# Single image (API, train, eval, preprocess, onnx-api)
docker build -t ml-ops-app .

# Run API (default)
docker run --rm -p 8000:8000 ml-ops-app

# General usage:
#   docker run --rm [ports] [env] [volumes] ml-ops-app <mode> [args...]
# Modes: api, onnx-api, preprocess, preprocess-transformer, train, train-transformer, eval
# Example: mount data/models for train/eval
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" ml-ops-app train trainer.max_epochs=1

# Run training (mount data/models as needed)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" ml-ops-app train trainer.max_epochs=1

# Run eval
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" ml-ops-app eval

# Environment variables for API and training
# - PORT=8000
# - MODEL_CHECKPOINT_PATH=/app/models/checkpoints_transformer/your.ckpt
# - MODEL_CHECKPOINT_DIR=/app/models/checkpoints_transformer
# - MODEL_NAME_OR_PATH=distilbert-base-uncased
# - LABELS="Food & Dining,Transportation,..."
# - MAX_LENGTH=64
# - DEVICE=cpu

# API image - runs FastAPI inference server
docker build -f dockerfiles/api.dockerfile -t ml-ops-api .

# Train baseline model
docker build -f dockerfiles/train.dockerfile -t ml-ops-train .

# Train transformer model
docker build -f dockerfiles/train_transformer.dockerfile -t ml-ops-train-transformer .

# Evaluate model
docker build -f dockerfiles/eval.dockerfile -t ml-ops-eval .

# Run tests in Docker
docker build -f dockerfiles/test.dockerfile -t ml-ops-tests .
docker run --rm ml-ops-tests
```

All images use `uv` for Python dependency management with Python 3.12.

## Code Coverage:
Coverage: 93.0% (run `uv run invoke test`)

### Goal
Build an end-to-end ML system that classifies short transaction descriptions (receipt line-items / bank statement text) into spending categories (e.g., Food & Dining, Transportation, Utilities, Shopping). Input is a single text string, output is a predicted category + confidence. The project emphasizes a reproducible MLOps pipeline: data download + preprocessing, training + evaluation, experiment tracking, packaging and deployment of an inference API, and automated testing/CI.

### Dataset
We will use a public labeled transaction dataset from Hugging Face. Minimum required columns:
- `transaction_description` (string): short free-text description
- `category` (string): target label (one of the predefined categories)

If available, we may also use: `amount`, `currency`, `country`, `timestamp` / derived time features.

The full dataset is large (millions of rows), so we will start with a small subset to iterate quickly. Plan:
- Subsample to 50k–200k rows for development (stratified by `category`)
- Train/val/test split (e.g., 80/10/10)
- Basic text cleaning (trim, normalize whitespace, optional lowercasing)
- Handle rare categories by merging into `Other` or filtering if needed

Link to dataset: https://huggingface.co/datasets/sreesharvesh/transactiq-enriched

### Expected models
1. **Baseline:** TF-IDF + linear classifier (Logistic Regression / Linear SVM)
2. **Neural:** Fine-tuned Transformer (e.g., DistilBERT) for multi-class classification

We will compare models using accuracy and macro-F1 and deploy the best-performing/most robust model behind a simple API that returns category + confidence.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
