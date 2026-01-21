import sys
import types
from pathlib import Path
from types import SimpleNamespace

import lightning as l
import pytest
from omegaconf import OmegaConf

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project import train as train_module  # noqa: E402
from src.ml_ops_project import train_transformer as train_transformer_module  # noqa: E402


class _DummyCallback:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _DummyLogger:
    def __init__(self, name: str = "x") -> None:
        self.name = name


@pytest.fixture
def dummy_import_module(monkeypatch):
    mod = types.ModuleType("dummy_import_module")
    mod.DummyCallback = _DummyCallback
    mod.DummyLogger = _DummyLogger
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


@pytest.mark.parametrize("module", [train_module, train_transformer_module])
def test_instantiate_class_passthroughs(module):
    assert module.instantiate_class(None) is None
    assert module.instantiate_class(123) == 123
    assert module.instantiate_class({"not": "a class config"}) == {"not": "a class config"}


@pytest.mark.parametrize("module", [train_module, train_transformer_module])
def test_instantiate_class_creates_instance(module, dummy_import_module):
    cfg = {"class_path": f"{dummy_import_module.__name__}.DummyCallback", "init_args": {"value": 7}}
    instance = module.instantiate_class(cfg)
    assert isinstance(instance, _DummyCallback)
    assert instance.value == 7


@pytest.mark.parametrize(
    ("module", "model_attr", "datamodule_attr"),
    [
        (train_module, "TransactionModel", "TransactionDataModule"),
        (train_transformer_module, "TransformerTransactionModel", "TextDataModule"),
    ],
)
def test_main_builds_trainer_and_calls_fit(monkeypatch, dummy_import_module, module, model_attr, datamodule_attr):
    created = {}

    def _seed_everything(seed):
        created["seed"] = seed

    class _DummyModel:
        def __init__(self, **kwargs) -> None:
            created["model_kwargs"] = kwargs

    class _DummyDataModule:
        def __init__(self, **kwargs) -> None:
            created["data_kwargs"] = kwargs
            self.data_root = Path("/tmp/data")

    class _DummyTrainer:
        def __init__(self, **kwargs) -> None:
            created["trainer_kwargs"] = kwargs

        def fit(self, model, datamodule):
            created["fit_called_with"] = (model, datamodule)

    monkeypatch.setattr(l, "seed_everything", _seed_everything)
    monkeypatch.setattr(l, "Trainer", _DummyTrainer)
    monkeypatch.setattr(module, model_attr, _DummyModel)
    monkeypatch.setattr(module, datamodule_attr, _DummyDataModule)
    if module is train_transformer_module:
        monkeypatch.setattr(module, "load_label_list", lambda data_root: ["label-a", "label-b"])
    monkeypatch.setattr(
        module.hydra.core.hydra_config.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir="/tmp/out"))),
    )

    cfg = OmegaConf.create(
        {
            "seed_everything": 123,
            "model": {"hidden_size": 1},
            "data": {"batch_size": 2},
            "trainer": {
                "max_epochs": 1,
                "callbacks": [
                    {"class_path": f"{dummy_import_module.__name__}.DummyCallback", "init_args": {"value": 5}}
                ],
                "logger": {"class_path": f"{dummy_import_module.__name__}.DummyLogger", "init_args": {"name": "log"}},
            },
        }
    )

    module.main.__wrapped__(cfg)

    assert created["seed"] == 123
    if module is train_transformer_module:
        assert created["model_kwargs"] == {"hidden_size": 1, "labels": ["label-a", "label-b"]}
    else:
        assert created["model_kwargs"] == {"hidden_size": 1}
    assert created["data_kwargs"] == {"batch_size": 2}
    assert created["trainer_kwargs"]["max_epochs"] == 1
    assert isinstance(created["trainer_kwargs"]["callbacks"][0], _DummyCallback)
    assert isinstance(created["trainer_kwargs"]["logger"], _DummyLogger)
    assert isinstance(created["fit_called_with"][0], _DummyModel)
    assert isinstance(created["fit_called_with"][1], _DummyDataModule)
