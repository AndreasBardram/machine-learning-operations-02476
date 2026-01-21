import lightning as pl
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification


class TransformerTransactionModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 10,
        learning_rate: float = 2e-5,  # noqa: ARG002
        weight_decay: float = 0.01,  # noqa: ARG002
        freeze_backbone: bool = False,
        labels: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)

        # Ensure model is in train mode (AutoModel can sometimes init in eval mode)
        self.model.train()

        if labels:
            id2label = dict(enumerate(labels))
            label2id = {label: i for i, label in enumerate(labels)}
            self.model.config.id2label = id2label
            self.model.config.label2id = label2id

        if freeze_backbone:
            print("Freezing backbone model parameters...")
            # For DistilBertForSequenceClassification, the body is in .distilbert
            # For BertForSequenceClassification, the body is in .bert
            # AutoModelForSequenceClassification creates one of them.
            if hasattr(self.model, "base_model"):
                for param in self.model.base_model.parameters():
                    param.requires_grad = False
            else:
                # Fallback: try to guess or freeze everything except classifier
                print(
                    "Warning: Could not identify base_model, attempting to freeze "
                    "named parameters not containing 'classifier' or 'head'."
                )
                for name, param in self.model.named_parameters():
                    if "classifier" not in name and "head" not in name and "pre_classifier" not in name:
                        param.requires_grad = False

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", loss, prog_bar=True)

            preds = torch.argmax(outputs.logits, dim=1)
            acc = (preds == batch["labels"]).float().mean()
            self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        if getattr(self, "_trainer", None) is not None:
            self.log("val_loss", loss, prog_bar=True)

            preds = torch.argmax(outputs.logits, dim=1)
            acc = (preds == batch["labels"]).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        if getattr(self, "_trainer", None) is not None:
            self.log("test_loss", loss, prog_bar=True)

            preds = torch.argmax(outputs.logits, dim=1)
            acc = (preds == batch["labels"]).float().mean()
            self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # Using a scheduler is very important for transformers
        # We need to estimate total steps or use a simpler scheduler.
        # For simplicity in this project, we can just use the optimizer or ReduceLROnPlateau
        return optimizer
