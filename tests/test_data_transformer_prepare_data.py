import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project import data_transformer as dt  # noqa: E402


class _DummyTokenizer:
    def __call__(self, texts, **kwargs):
        max_length = kwargs.get("max_length", 8)
        return {
            "input_ids": [[1] * max_length for _ in texts],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def _dataset(num_rows: int) -> Dataset:
    return Dataset.from_dict(
        {
            "transaction_description": [f"t{i}" for i in range(num_rows)],
            "category": ["a" if i % 2 == 0 else "b" for i in range(num_rows)],
        }
    )


def test_prepare_data_downloads_and_processes(monkeypatch, tmp_path):
    ds = DatasetDict({"train": _dataset(10)})

    monkeypatch.setattr(dt, "load_dataset", lambda _: ds)
    monkeypatch.setattr(dt.AutoTokenizer, "from_pretrained", lambda _: _DummyTokenizer())

    dm = dt.TextDataModule(data_path=str(tmp_path), num_workers=0, max_length=8)
    dm.prepare_data()

    raw_path = tmp_path / "raw" / dt.SAVED_NAME
    assert raw_path.exists()
    assert dm.processed_path.exists()


def test_prepare_data_limit_samples(monkeypatch, tmp_path):
    ds = DatasetDict({"train": _dataset(50)})

    monkeypatch.setattr(dt, "load_dataset", lambda _: ds)
    monkeypatch.setattr(dt.AutoTokenizer, "from_pretrained", lambda _: _DummyTokenizer())

    dm = dt.TextDataModule(data_path=str(tmp_path), num_workers=0, max_length=8, limit_samples=7)
    dm.prepare_data()

    processed = dt.load_from_disk(str(dm.processed_path))
    assert len(processed) == 7
