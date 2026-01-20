#!/bin/sh
set -eu

mode="${1:-api}"
shift || true

case "$mode" in
  api)
    exec uv run uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  onnx-api)
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
    echo "Valid modes: api, onnx-api, preprocess, preprocess-transformer, train, train-transformer, eval" >&2
    exit 1
    ;;
esac
