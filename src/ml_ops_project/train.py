import torch
from lightning.pytorch.cli import LightningCLI

from ml_ops_project.data import TransactionDataModule
from ml_ops_project.model import TransactionModel

torch.set_float32_matmul_precision("high")


def cli_main():
    cli = LightningCLI(TransactionModel, TransactionDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
