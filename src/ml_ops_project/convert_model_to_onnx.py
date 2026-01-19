import hydra
import torch
from omegaconf import DictConfig

from ml_ops_project.model import TransactionModel


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def export_to_onnx(cfg: DictConfig) -> None:
    """Export the trained TransactionModel to ONNX format."""

    model = TransactionModel(**cfg.model)
    model.eval()

    # Create dummy input with correct shape
    # Shape should be (batch_size, input_dim)
    input_dim = cfg.model.get("input_dim", 32)
    dummy_input = torch.randn(1, input_dim)

    print(f"Exporting model with input shape: {dummy_input.shape}")
    print(f"Model config: {cfg.model}")

    output_path = "models/transaction_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully exported to: {output_path}")


if __name__ == "__main__":
    export_to_onnx()
