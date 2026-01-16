import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ml_ops_project"
PYTHON_VERSION = "3.11"


@task
def preprocess_data(ctx: Context, subset: bool = False) -> None:
    """Preprocess data."""
    subset_flag = "--subset" if subset else "--no-subset"
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py {subset_flag}", echo=True, pty=not WINDOWS)


@task
def preprocess_data_transformer(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data_transformer.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, epochs: int = 10, subset: bool = False, experiment: str | None = None) -> None:
    """Train model."""
    if experiment:
        experiment_arg = f"experiment={experiment} "
    elif subset:
        experiment_arg = "experiment=baseline_subset "
    else:
        experiment_arg = ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train.py {experiment_arg}trainer.max_epochs={epochs}", echo=True, pty=not WINDOWS
    )


@task
def train_transformer(ctx: Context, epochs: int = 10, experiment: str | None = None) -> None:
    """Train the transformer model using Hydra configuration."""
    experiment_arg = f"experiment={experiment} " if experiment else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train_transformer.py {experiment_arg}trainer.max_epochs={epochs}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# ruff commands
@task
def lint(ctx: Context, fix: bool = False):
    """Run ruff linter"""
    fix_flag = "--fix" if fix else ""
    ctx.run(f"uv run ruff check . {fix_flag}", echo=True, pty=not WINDOWS)


def format(ctx: Context, check: bool = False):
    """Run ruff formatting"""
    check_flag = "--check" if check else ""
    ctx.run(f"uv run ruff format . {check_flag}", echo=True, pty=not WINDOWS)
