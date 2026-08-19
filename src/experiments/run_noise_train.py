"""CLI entrypoint: python -m src.experiments.run_noise_train --config configs/noise/hybrid_noise_train_10.yaml

Option 2 noise-robustness experiment: identical training logic to run_hybrid.py
-- a normal full training run, just with model.noise_prob set in the config so
the quantum layer trains under simulated depolarizing noise from the start.
This script exists mainly for clear naming/output separation under
results/noise/, rather than for any new training behavior.
"""

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
        noise_prob=model_cfg.get("noise_prob", 0.0),
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
