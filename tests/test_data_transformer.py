import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project.data_transformer import TextDataModule, TextDataset  # noqa: E402


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_text_dataset():
    """Create a mock HuggingFace dataset with text data for testing."""
    data = {
        "transaction_description": [
            "Grocery shopping at Walmart",
            "Uber ride to downtown",
            "Netflix subscription payment",
            "Coffee at Starbucks",
            "Gas station fill-up",
        ]
        * 20,
        "category": ["groceries", "transport", "entertainment", "food", "transport"] * 20,
        "country": ["US", "UK", "CA", "US", "UK"] * 20,
        "currency": ["USD", "GBP", "CAD", "USD", "GBP"] * 20,
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_tokenized_dataset():
    """Create a mock tokenized dataset."""
    data = {
        "input_ids": [[101, 2023, 2003, 1037, 3231, 102] + [0] * 58 for _ in range(100)],
        "attention_mask": [[1, 1, 1, 1, 1, 1] + [0] * 58 for _ in range(100)],
        "labels": [i % 5 for i in range(100)],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    def tokenize_side_effect(texts, max_length=None):
        return {
            "input_ids": [[101, 2023, 2003, 1037, 3231, 102] + [0] * (max_length - 6) for _ in texts],
            "attention_mask": [[1, 1, 1, 1, 1, 1] + [0] * (max_length - 6) for _ in texts],
        }

    tokenizer.side_effect = tokenize_side_effect
    return tokenizer


class TestTextDataset:
    """Test suite for TextDataset class."""

    def test_init(self, temp_dir):
        """Test dataset initialization."""
        dataset = TextDataset(temp_dir)
        assert dataset.data_path == temp_dir
        assert dataset.data is None

    def test_load_without_data_raises_error(self, temp_dir):
        """Test that loading non-existent data raises appropriate error."""
        dataset = TextDataset(temp_dir)
        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar  # noqa: B017
            dataset.load()

    def test_len_without_load_raises_error(self, temp_dir):
        """Test that accessing length before loading raises RuntimeError."""
        dataset = TextDataset(temp_dir)
        with pytest.raises(RuntimeError, match="Dataset not loaded"):
            len(dataset)

    def test_getitem_without_load_raises_error(self, temp_dir):
        """Test that accessing items before loading raises RuntimeError."""
        dataset = TextDataset(temp_dir)
        with pytest.raises(RuntimeError, match="Dataset not loaded"):
            _ = dataset[0]

    def test_load_and_access(self, temp_dir, mock_tokenized_dataset):
        """Test successful loading and data access."""
        # Save mock tokenized dataset
        save_path = temp_dir / "test_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        # Load and verify
        dataset = TextDataset(save_path)
        dataset.load()

        assert dataset.data is not None
        assert len(dataset) == len(mock_tokenized_dataset)

    def test_getitem_returns_correct_format(self, temp_dir, mock_tokenized_dataset):
        """Test that __getitem__ returns correct dictionary format."""
        save_path = temp_dir / "test_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dataset = TextDataset(save_path)
        dataset.load()

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_getitem_shapes(self, temp_dir, mock_tokenized_dataset):
        """Test that retrieved items have correct shapes."""
        save_path = temp_dir / "test_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dataset = TextDataset(save_path)
        dataset.load()

        item = dataset[0]

        # Check shapes
        assert len(item["input_ids"]) == 64  # max_length default
        assert len(item["attention_mask"]) == 64
        assert item["labels"].dim() == 0  # scalar label


class TestTextDataModule:
    """Test suite for TextDataModule class."""

    def test_init_default_params(self):
        """Test data module initialization with default parameters."""
        dm = TextDataModule()
        assert dm.model_name_or_path == "distilbert-base-uncased"
        assert dm.batch_size == 32
        assert dm.max_length == 64
        assert dm.num_workers == 4
        assert dm.limit_samples is None

    def test_init_custom_params(self, temp_dir):
        """Test data module initialization with custom parameters."""
        dm = TextDataModule(
            model_name_or_path="bert-base-uncased",
            data_path=str(temp_dir),
            batch_size=16,
            max_length=128,
            num_workers=2,
            limit_samples=100,
        )
        assert dm.model_name_or_path == "bert-base-uncased"
        assert dm.data_root == temp_dir
        assert dm.batch_size == 16
        assert dm.max_length == 128
        assert dm.num_workers == 2
        assert dm.limit_samples == 100

    def test_processed_path_with_limit(self, temp_dir):
        """Test that processed path includes limit_samples in filename."""
        dm_with_limit = TextDataModule(data_path=str(temp_dir), limit_samples=500)
        dm_without_limit = TextDataModule(data_path=str(temp_dir))

        assert "subset_500" in str(dm_with_limit.processed_path)
        assert "subset" not in str(dm_without_limit.processed_path)

    def test_setup_splits_data_correctly(self, temp_dir, mock_tokenized_dataset):
        """Test that setup splits data into train/val/test correctly."""
        # Save processed dataset
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        # Setup data module with the existing processed path
        dm = TextDataModule(data_path=str(temp_dir), batch_size=16, num_workers=0)
        dm.processed_path = save_path
        dm.setup()

        # Check splits exist
        assert hasattr(dm, "train_ds")
        assert hasattr(dm, "val_ds")
        assert hasattr(dm, "test_ds")

        # Check split sizes
        total_size = 100
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        assert len(dm.train_ds) == train_size
        assert len(dm.val_ds) == val_size
        assert len(dm.test_ds) == test_size

    def test_dataloaders_return_correct_types(self, temp_dir, mock_tokenized_dataset):
        """Test that dataloaders return DataLoader instances."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dm = TextDataModule(data_path=str(temp_dir), batch_size=4, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_dataloader_batch_size(self, temp_dir, mock_tokenized_dataset):
        """Test that dataloaders use correct batch size."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        batch_size = 8
        dm = TextDataModule(data_path=str(temp_dir), batch_size=batch_size, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        train_loader = dm.train_dataloader()

        # Get first batch
        batch = next(iter(train_loader))

        # Check batch size (might be smaller for last batch)
        assert batch["input_ids"].shape[0] <= batch_size
        assert batch["attention_mask"].shape[0] <= batch_size
        assert batch["labels"].shape[0] <= batch_size

    def test_train_dataloader_shuffles(self, temp_dir, mock_tokenized_dataset):
        """Test that train dataloader has shuffle enabled."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dm = TextDataModule(data_path=str(temp_dir), batch_size=4, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        train_loader = dm.train_dataloader()
        assert train_loader.sampler is not None  # Shuffle creates a sampler

    def test_batch_contains_correct_keys(self, temp_dir, mock_tokenized_dataset):
        """Test that batches contain the expected keys."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dm = TextDataModule(data_path=str(temp_dir), batch_size=4, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_batch_tensor_shapes(self, temp_dir, mock_tokenized_dataset):
        """Test that batch tensors have correct shapes."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        max_length = 64
        batch_size = 4
        dm = TextDataModule(data_path=str(temp_dir), batch_size=batch_size, max_length=max_length, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check shapes
        assert batch["input_ids"].dim() == 2
        assert batch["attention_mask"].dim() == 2
        assert batch["labels"].dim() == 1

        # Check dimensions
        assert batch["input_ids"].shape[1] == max_length
        assert batch["attention_mask"].shape[1] == max_length


class TestIntegration:
    """Integration tests for the text processing pipeline."""

    def test_datamodule_consistency_across_calls(self, temp_dir, mock_tokenized_dataset):
        """Test that multiple calls to dataloaders return consistent data."""
        save_path = temp_dir / "processed_data"
        mock_tokenized_dataset.save_to_disk(str(save_path))

        dm = TextDataModule(data_path=str(temp_dir), batch_size=16, num_workers=2)
        dm.processed_path = save_path
        dm.setup()

        # Get dataloaders multiple times
        train_loader1 = dm.train_dataloader()
        train_loader2 = dm.train_dataloader()

        # Both should work
        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))

        assert batch1["input_ids"].shape == batch2["input_ids"].shape
        assert batch1["attention_mask"].shape == batch2["attention_mask"].shape
        assert batch1["labels"].shape == batch2["labels"].shape
