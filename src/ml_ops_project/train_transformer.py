import logging
from importlib import import_module

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from ml_ops_project.data_transformer import TextDataModule
from ml_ops_project.model_transformer import TransformerTransactionModel

torch.set_float32_matmul_precision("high")
log = logging.getLogger(__name__)


def instantiate_class(config):
    if config is None:
        return None
    if not isinstance(config, dict) or "class_path" not in config:
        return config
    
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    cls = getattr(module, class_name)
    
    init_args = config.get("init_args", {})
    return cls(**init_args)


@hydra.main(version_base=None, config_path="../../configs", config_name="transformer_default")
def main(cfg: DictConfig) -> None:
    log.info(f"Starting transformer training with config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    seed = cfg.get("seed_everything", 42)
    L.seed_everything(seed)
    log.info(f"Seed set to: {seed}")
    
    model = TransformerTransactionModel(**cfg.model)
    datamodule = TextDataModule(**cfg.data)
    log.info("Transformer model and DataModule loaded")
    
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    if "callbacks" in trainer_cfg:
        callbacks = [instantiate_class(cb) for cb in trainer_cfg["callbacks"]]
        trainer_cfg["callbacks"] = callbacks
        log.info(f"Loaded {len(callbacks)} callbacks")
    
    if "logger" in trainer_cfg:
        trainer_cfg["logger"] = instantiate_class(trainer_cfg["logger"])
        log.info("Logger loaded")
    
    trainer = L.Trainer(**trainer_cfg)
    log.info("Trainer created")
    
    trainer.fit(model, datamodule)
    log.info("Training complete")


if __name__ == "__main__":
    main()
