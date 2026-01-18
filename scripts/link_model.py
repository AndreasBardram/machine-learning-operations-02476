import os
from typing import List

import typer
import wandb


def link_model(artifact_path: str, aliases: List[str] = ["staging"]) -> None:
    """Link a W&B artifact into the model registry with the given aliases."""
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        raise typer.Exit(code=1)

    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not api_key or not entity or not project:
        typer.echo("Missing WANDB_API_KEY, WANDB_ENTITY, or WANDB_PROJECT in the environment.")
        raise typer.Exit(code=1)

    api = wandb.Api(api_key=api_key, overrides={"entity": entity, "project": project})
    _, _, artifact_name_version = artifact_path.split("/", maxsplit=2)
    artifact_name, _ = artifact_name_version.split(":", maxsplit=1)

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=f"{entity}/model-registry/{artifact_name}", aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
