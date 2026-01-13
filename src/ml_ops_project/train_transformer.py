import torch
from lightning.pytorch.cli import LightningCLI

from ml_ops_project.data_transformer import TextDataModule
from ml_ops_project.model_transformer import TransformerTransactionModel

torch.set_float32_matmul_precision("high")


def cli_main():
    cli = LightningCLI(TransformerTransactionModel, TextDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
