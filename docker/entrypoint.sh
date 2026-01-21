#!/bin/sh
set -eu

mode="${1:-api}"
shift || true

pull_models_if_needed() {
  if [ "${DVC_PULL_ON_STARTUP:-1}" = "0" ]; then
    return 0
  fi
  if [ -d models ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
    return 0
  fi
  echo "Models missing; pulling via DVC..."
  DVC_NO_SCM=1 uv run dvc pull models.dvc
}

case "$mode" in
  api)
    pull_models_if_needed
    exec uv run uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  streamlit)
    exec uv run streamlit run src/ml_ops_project/streamlit_app.py \
      --server.address 0.0.0.0 \
      --server.port "${PORT:-8501}" \
      --server.headless true \
      --browser.gatherUsageStats false \
      "$@"
    ;;
  onnx-api)
    pull_models_if_needed
    exec uv run src/ml_ops_project/onnx_fastapi.py "$@"
    ;;
  preprocess)
    exec uv run src/ml_ops_project/data.py "$@"
    ;;
  preprocess-transformer)
    exec uv run src/ml_ops_project/data_transformer.py "$@"
    ;;
  train)
    exec uv run src/ml_ops_project/train.py "$@"
    ;;
  train-transformer)
    exec uv run src/ml_ops_project/train_transformer.py "$@"
    ;;
  eval)
    exec uv run src/ml_ops_project/eval.py "$@"
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    echo "Valid modes: api, streamlit, onnx-api, preprocess, preprocess-transformer, train, train-transformer, eval" >&2
    exit 1
    ;;
esac
