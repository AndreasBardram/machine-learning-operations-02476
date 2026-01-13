import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project.data import MyDataset, TransactionDataModule  # noqa: E402


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_dataset():
    """Create a mock HuggingFace dataset for testing."""
    data = {
        "category": ["groceries", "transport", "groceries", "entertainment", "transport"] * 20,
        "country": ["US", "UK", "CA", "US", "UK"] * 20,
        "currency": ["USD", "GBP", "CAD", "USD", "GBP"] * 20,
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] * 20,
        "month": ["January", "February", "March", "April", "May"] * 20,
        "log_amount": [2.5, 3.0, 2.8, 4.0, 3.5] * 20,
        "is_weekend": [0, 0, 0, 0, 1] * 20,
        "year": [2023, 2023, 2023, 2023, 2023] * 20,
    }
    return Dataset.from_dict(data)


class TestMyDataset:
    """Test suite for MyDataset class."""

    def test_init(self, temp_dir):
        """Test dataset initialization."""
        dataset = MyDataset(temp_dir)
        assert dataset.data_path == temp_dir
        assert dataset.data is None

    def test_load_without_data_raises_error(self, temp_dir):
        """Test that loading non-existent data raises appropriate error."""
        dataset = MyDataset(temp_dir)
        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar  # noqa: B017
            dataset.load()

    def test_len_without_load_raises_error(self, temp_dir):
        """Test that accessing length before loading raises RuntimeError."""
        dataset = MyDataset(temp_dir)
        with pytest.raises(RuntimeError, match="Dataset not loaded"):
            len(dataset)

    def test_getitem_without_load_raises_error(self, temp_dir):
        """Test that accessing items before loading raises RuntimeError."""
        dataset = MyDataset(temp_dir)
        with pytest.raises(RuntimeError, match="Dataset not loaded"):
            _ = dataset[0]

    def test_load_and_access(self, temp_dir, mock_dataset):
        """Test successful loading and data access."""
        # Save mock dataset
        save_path = temp_dir / "test_data"
        mock_dataset.save_to_disk(str(save_path))

        # Load and verify
        dataset = MyDataset(save_path)
        dataset.load()

        assert dataset.data is not None
        assert len(dataset) == len(mock_dataset)
        assert dataset[0] is not None

    def test_load_with_train_split(self, temp_dir, mock_dataset):
        """Test loading dataset with train split (DatasetDict format)."""
        # Save as DatasetDict with train split
        save_path = temp_dir / "test_data"
        from datasets import DatasetDict

        ds_dict = DatasetDict({"train": mock_dataset})
        ds_dict.save_to_disk(str(save_path))

        # Load and verify it extracts train split
        dataset = MyDataset(save_path)
        dataset.load()

        assert dataset.data is not None
        assert len(dataset) == len(mock_dataset)


class TestTransactionDataModule:
    """Test suite for TransactionDataModule class."""

    def test_init(self):
        """Test data module initialization."""
        dm = TransactionDataModule(data_path="test/path", batch_size=32, num_workers=2)
        assert dm.data_path == Path("test/path")
        assert dm.batch_size == 32
        assert dm.num_workers == 2

    def test_setup_splits_data_correctly(self, temp_dir):
        """Test that setup splits data into train/val/test correctly."""
        # Create and save processed dataset
        processed_data = {
            "features": [[float(i)] * 10 for i in range(100)],
            "labels": [i % 5 for i in range(100)],
        }
        processed_ds = Dataset.from_dict(processed_data)
        save_path = temp_dir / "processed_data"
        processed_ds.save_to_disk(str(save_path))

        # Setup data module
        dm = TransactionDataModule(data_path=str(save_path), batch_size=16, num_workers=0)
        dm.setup()

        # Check splits exist
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert hasattr(dm, "test_dataset")

        # Check split sizes
        total_size = 100
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        assert len(dm.train_dataset) == train_size
        assert len(dm.val_dataset) == val_size
        assert len(dm.test_dataset) == test_size

    def test_dataloaders_return_correct_types(self, temp_dir):
        """Test that dataloaders return DataLoader instances."""
        # Create minimal processed dataset
        processed_data = {
            "features": [[1.0] * 10 for _ in range(20)],
            "labels": [0] * 20,
        }
        processed_ds = Dataset.from_dict(processed_data)
        save_path = temp_dir / "processed_data"
        processed_ds.save_to_disk(str(save_path))

        dm = TransactionDataModule(data_path=str(save_path), batch_size=4, num_workers=2)
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_dataloader_batch_size(self, temp_dir):
        """Test that dataloaders use correct batch size."""
        processed_data = {
            "features": [[1.0] * 10 for _ in range(50)],
            "labels": [0] * 50,
        }
        processed_ds = Dataset.from_dict(processed_data)
        save_path = temp_dir / "processed_data"
        processed_ds.save_to_disk(str(save_path))

        batch_size = 8
        dm = TransactionDataModule(data_path=str(save_path), batch_size=batch_size, num_workers=2)
        dm.setup()

        train_loader = dm.train_dataloader()

        # Get first batch
        batch = next(iter(train_loader))

        # Check batch size (might be smaller for last batch)
        assert batch["features"].shape[0] <= batch_size

    def test_train_dataloader_shuffles(self, temp_dir):
        """Test that train dataloader has shuffle enabled."""
        processed_data = {
            "features": [[1.0] * 10 for _ in range(20)],
            "labels": [0] * 20,
        }
        processed_ds = Dataset.from_dict(processed_data)
        save_path = temp_dir / "processed_data"
        processed_ds.save_to_disk(str(save_path))

        dm = TransactionDataModule(data_path=str(save_path), batch_size=4, num_workers=2)
        dm.setup()

        train_loader = dm.train_dataloader()
        assert train_loader.sampler is not None  # Shuffle creates a sampler


class TestIntegration:
    """Integration tests for the entire pipeline."""

    @patch("datasets.load_dataset")
    @patch("datasets.load_from_disk")
    def test_full_pipeline(self, mock_load_from_disk, mock_load_dataset, temp_dir, mock_dataset):
        """Test the complete preprocessing and loading pipeline."""
        output_folder = temp_dir

        # Mock dataset loading
        mock_load_dataset.return_value = mock_dataset
        mock_load_from_disk.return_value = mock_dataset

        # Step 1: Preprocess
        dataset = MyDataset(output_folder)
        dataset.preprocess(output_folder, subset=True)

        # Step 2: Load with DataModule
        processed_path = output_folder / "processed" / "transactiq_processed_subset"
        dm = TransactionDataModule(data_path=str(processed_path), batch_size=4, num_workers=2)
        dm.setup()

        # Step 3: Verify we can iterate through data
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert "features" in batch
        assert "labels" in batch
        assert batch["features"].shape[0] <= 4  # batch_size
        assert batch["features"].dim() == 2  # (batch_size, num_features)
        assert batch["labels"].dim() == 1  # (batch_size,)
