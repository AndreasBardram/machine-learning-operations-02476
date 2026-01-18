FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

COPY src src/

RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "uvicorn", "src.ml_ops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
