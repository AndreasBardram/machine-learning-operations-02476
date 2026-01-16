from pathlib import Path

import lightning as pl
import numpy as np
import torch
import typer
from datasets import load_dataset, load_from_disk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

DATASET_ID = "sreesharvesh/transactiq-enriched"
SAVED_NAME = "transactiq_enriched_hf"
PROCESSED_NAME = "transactiq_processed"


class MyDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.data = None

    def load(self) -> None:
        obj = load_from_disk(str(self.data_path))
        if hasattr(obj, "keys") and "train" in obj:
            self.data = obj["train"]
        else:
            self.data = obj
        # Set format for efficient tensor access
        self.data.set_format("torch")

    def __len__(self) -> int:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return len(self.data)

    def __getitem__(self, index: int):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        return self.data[index]

    def preprocess(self, output_folder: Path, subset: bool = False) -> None:
        raw_path = output_folder / "raw" / SAVED_NAME
        processed_path = output_folder / "processed" / (f"{PROCESSED_NAME}_subset" if subset else PROCESSED_NAME)

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        ds = None
        if not raw_path.exists():
            print(f"Downloading raw data to {raw_path}...")
            ds = load_dataset(DATASET_ID)
            ds.save_to_disk(raw_path)
            ds = load_from_disk(str(raw_path))
        else:
            print(f"Loading raw data from {raw_path}...")
            ds = load_from_disk(str(raw_path))

        # Handle DatasetDict vs Dataset
        train_ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds

        print("Preprocessing data...")

        # 1. Inspect and Create Mappings for Scikit-Learn
        # Get unique values to initialize encoders
        unique_categories = sorted(train_ds.unique("category"))
        unique_countries = sorted(train_ds.unique("country"))
        unique_currencies = sorted(train_ds.unique("currency"))
        unique_days = sorted(train_ds.unique("day_of_week"))
        unique_months = sorted(train_ds.unique("month"))

        print(f"Categories: {len(unique_categories)}")
        print(f"Countries: {len(unique_countries)}")
        print(f"Currencies: {len(unique_currencies)}")

        # Initialize Encoders
        le = LabelEncoder()
        le.fit(unique_categories)

        # We specify categories to ensure consistent order and dimension
        ohe = OneHotEncoder(
            categories=[unique_countries, unique_currencies, unique_days, unique_months],
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )

        # Dummy fit to initialize internal state (sklearn requires fit before transform)
        # We construct a single row with the first value of each category list
        dummy_data = np.array([[unique_countries[0], unique_currencies[0], unique_days[0], unique_months[0]]])
        ohe.fit(dummy_data)

        if subset:
            n = int(0.1 * len(train_ds))
            print(f"Subsetting data to {n} samples (10%)...")
            train_ds = train_ds.shuffle(seed=42).select(range(n))

        def process_batch(batch):
            # Numeric Features
            # Ensure they are 2D arrays (N, 1)
            log_amounts = np.array(batch["log_amount"], dtype=np.float32).reshape(-1, 1)
            is_weekends = np.array(batch["is_weekend"], dtype=np.float32).reshape(-1, 1)
            years = (np.array(batch["year"], dtype=np.float32) - 2020.0).reshape(-1, 1)

            # Categorical Features for OHE
            c = np.array(batch["country"]).reshape(-1, 1)
            curr = np.array(batch["currency"]).reshape(-1, 1)
            dow = np.array(batch["day_of_week"]).reshape(-1, 1)
            m = np.array(batch["month"]).reshape(-1, 1)

            # Stack categorical columns horizontally
            cat_data = np.hstack([c, curr, dow, m])

            # Transform
            encoded_cats = ohe.transform(cat_data)

            # Combine all features
            features = np.hstack([log_amounts, is_weekends, years, encoded_cats])

            # Labels
            labels = le.transform(batch["category"])

            return {"features": features, "labels": labels}

        # Apply transformation with batching for efficiency
        processed_ds = train_ds.map(process_batch, batched=True, batch_size=10000, remove_columns=train_ds.column_names)

        # Verify
        print(f"Processed shape: {len(processed_ds)} rows")
        print(f"Feature dimension: {len(processed_ds[0]['features'])}")

        # Save
        processed_ds.save_to_disk(processed_path)
        print(f"Saved processed data to {processed_path}")


def preprocess(output_folder: Path = Path("data"), subset: bool = False) -> None:
    # We use MyDataset as a runner
    dataset = MyDataset(output_folder)
    dataset.preprocess(output_folder, subset=subset)


if __name__ == "__main__":
    typer.run(preprocess)


class TransactionDataModule(pl.LightningDataModule):
    def __init__(
        self, data_path: str = "data/processed/transactiq_processed", batch_size: int = 64, num_workers: int = 4
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):  # noqa: ARG002
        print(f"Loading data from {self.data_path}...")
        full_dataset = MyDataset(self.data_path)
        full_dataset.load()

        # Split 80/10/10 Simple
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )
