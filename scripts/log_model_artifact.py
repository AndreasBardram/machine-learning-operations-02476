from pathlib import Path

import os
import typer
import torch
import wandb

from ml_ops_project.model import TransactionModel


def log_model_artifact(
    artifact_name: str = "transaction-classifier",
    checkpoint_path: str = "",
    output_dir: str = "models/registry_dummy",
) -> None:
    """Log a model artifact to W&B (uses a dummy checkpoint if none provided)."""
    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not api_key or not entity or not project:
        typer.echo("Missing WANDB_API_KEY, WANDB_ENTITY, or WANDB_PROJECT in the environment.")
        raise typer.Exit(code=1)

    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            typer.echo(f"Checkpoint not found: {checkpoint}")
            raise typer.Exit(code=1)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = output_path / f"{artifact_name}.pth"
        model = TransactionModel()
        torch.save(model.state_dict(), checkpoint)
        typer.echo(f"Wrote dummy checkpoint to {checkpoint}")

    run = wandb.init(entity=entity, project=project, job_type="artifact-log")
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(checkpoint))
    logged = run.log_artifact(artifact)
    logged.wait()
    run.finish()

    typer.echo(f"Logged artifact: {entity}/{project}/{logged.name}")


if __name__ == "__main__":
    typer.run(log_model_artifact)
