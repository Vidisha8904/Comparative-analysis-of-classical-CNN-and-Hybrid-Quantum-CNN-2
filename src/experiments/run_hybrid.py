"""CLI entrypoint: python -m src.experiments.run_hybrid --config configs/base/hybrid_subset.yaml"""

import argparse
import os

import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.hybrid_cnn import HybridCNN
from src.training.engine import run_training
from src.training.metrics import count_parameters
from src.utils.logger import MetricsLogger
from src.utils.plotting import plot_training_curves
from src.utils.seed import set_seed


def main():
    """Load a config file and train the hybrid quantum CNN it describes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["training_seed"])

    train_loader, test_loader = get_dataloaders(config)

    model_cfg = config["model"]
    model = HybridCNN(
        num_classes=model_cfg["num_classes"],
        n_qubits=model_cfg["n_qubits"],
        n_qlayers=model_cfg["n_qlayers"],
        device=model_cfg.get("device", "default.qubit"),
    )
    print(f"Trainable parameters: {count_parameters(model)}")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger = MetricsLogger(training_seed=config["training_seed"], data_seed=config["data_seed"])
    run_training(model, train_loader, test_loader, config, logger)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    csv_path = os.path.join(output_dir, "metrics.csv")
    logger.save(csv_path)
    plot_training_curves(csv_path, os.path.join(output_dir, "training_curves.png"))


if __name__ == "__main__":
    main()
