FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./
COPY .dvc/ .dvc/
COPY data.dvc models.dvc ./
RUN printf "[core]\n    no_scm = true\n" > .dvc/config.local
RUN uv sync --locked --no-cache --no-install-project

COPY src/ src/
COPY configs/ configs/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p data models
RUN uv sync --locked --no-cache

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
