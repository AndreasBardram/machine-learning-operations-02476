from pathlib import Path

import lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Constants for data paths - reusing the ones from data.py or defining new ones
DATASET_ID = "sreesharvesh/transactiq-enriched"
SAVED_NAME = "transactiq_enriched_hf"
PROCESSED_NAME_TEXT = "transactiq_processed_text"


def load_label_list(data_root: Path) -> list[str]:
    raw_path = data_root / "raw" / SAVED_NAME
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(DATASET_ID)
        ds.save_to_disk(raw_path)
    ds = load_from_disk(str(raw_path))
    full_ds_for_labels = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    return sorted({c.strip() for c in full_ds_for_labels.unique("category") if c})


class TextDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = None

    def load(self):
        self.data = load_from_disk(str(self.data_path))
        self.data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        item = self.data[idx]
        return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"], "labels": item["labels"]}


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        data_path: str = "data",
        batch_size: int = 32,
        max_length: int = 64,
        num_workers: int = 4,
        limit_samples: int | None = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.data_root = Path(data_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.limit_samples = limit_samples

        suffix = f"_subset_{limit_samples}" if limit_samples else ""
        self.processed_path = self.data_root / "processed" / (PROCESSED_NAME_TEXT + suffix)

    def prepare_data(self):
        # 1. Download/Load Raw Data (Same as original data.py)
        raw_path = self.data_root / "raw" / SAVED_NAME
        if not raw_path.exists():
            print(f"Downloading raw data to {raw_path}...")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            ds = load_dataset(DATASET_ID)
            ds.save_to_disk(raw_path)

        # 2. Tokenize and Save if not already done
        if not self.processed_path.exists():
            print(f"Processing text data to {self.processed_path}...")
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)

            ds = load_from_disk(str(raw_path))
            # Handle DatasetDict
            train_ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds

            if self.limit_samples:
                print(f"Subsetting data to {self.limit_samples} samples...")
                train_ds = train_ds.shuffle(seed=42).select(range(min(len(train_ds), self.limit_samples)))

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            # Need to encode labels as integers
            # We must derive this from the full dataset (or a known list) to ensure consistency across subsets
            # But for now, using the current (possibly subsetted) ds unique values might be risky if subset
            # misses a class.
            # Ideally we should use the full dataset to get unique categories, or hardcode them if known.
            # Since we loaded the full raw ds above (ds), let's use that for label mapping
            unique_categories = load_label_list(self.data_root)
            label2id = {label: i for i, label in enumerate(unique_categories)}

            print(f"Tokenizing data with {self.model_name_or_path}...")

            def preprocess_function(examples, tokenizer, label2id):
                # Tokenize text
                tokenized = tokenizer(
                    examples["transaction_description"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
                # Map labels to IDs
                tokenized["labels"] = [label2id[c] for c in examples["category"]]
                return tokenized

            processed_ds = train_ds.map(
                preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}
            )

            # Split
            # We do the split once and save separate datasets to disk to ensure consistency
            # or we can verify split seed. For simplicity, saving the whole processed ds
            processed_ds.save_to_disk(self.processed_path)

    def setup(self, stage: str | None = None):  # noqa: ARG002
        from torch.utils.data import random_split

        full_dataset = TextDataset(self.processed_path)
        full_dataset.load()

        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )
