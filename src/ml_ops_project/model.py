import lightning as pl
import torch
import torch.nn.functional as f
from torch import nn


class TransactionModel(pl.LightningModule):
    """A PyTorch Lightning model for classifying transaction categories."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Hidden layer 1
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Hidden layer 2
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output layer
            nn.Linear(hidden_dim, output_dim),
        )

        # Loss function for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch):
        x, y = batch["features"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch["features"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch["features"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    # Example usage based on our data stats
    input_dim = 32
    output_dim = 10
    model = TransactionModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)

    # Create dummy input batch (batch_size=4)
    x = torch.rand(4, input_dim)
    print(f"Input shape: {x.shape}")

    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Probabilities: {f.softmax(logits, dim=1)}")
