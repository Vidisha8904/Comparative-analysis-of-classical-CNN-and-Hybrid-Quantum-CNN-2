"""CLI entrypoint: python -m src.experiments.run_classical --config configs/base/classical_subset.yaml"""

import argparse
import os

import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.classical_cnn import ClassicalCNN
from src.training.engine import run_training
from src.training.metrics import count_parameters
from src.utils.logger import MetricsLogger
from src.utils.plotting import plot_training_curves
from src.utils.seed import set_seed


def run_single(config, output_dir):
    """Train one ClassicalCNN from a config dict and save its outputs.

    Also reused by run_multiseed.py (Experiment 5) and run_capacity.py
    (Experiment 6), so every classical training run goes through this one
    function instead of duplicating the training/saving logic.

    `fc_hidden`/`adaptive_pool_size` default to the original architecture used
    everywhere except Experiment 6 -- base and multiseed configs never set
    them, while Experiment 6's capacity configs set both to hit specific
    parameter-count targets.
    """
    set_seed(config["training_seed"])

    train_loader, test_loader = get_dataloaders(config)

    model_cfg = config["model"]
    model = ClassicalCNN(
        num_classes=model_cfg["num_classes"],
        fc_hidden=model_cfg.get("fc_hidden", 64),
        adaptive_pool_size=model_cfg.get("adaptive_pool_size"),
    )
    print(f"[{output_dir}] trainable parameters: {count_parameters(model)}")

    os.makedirs(output_dir, exist_ok=True)

    logger = MetricsLogger(training_seed=config["training_seed"], data_seed=config["data_seed"])
    run_training(model, train_loader, test_loader, config, logger)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    csv_path = os.path.join(output_dir, "metrics.csv")
    logger.save(csv_path)
    plot_training_curves(csv_path, os.path.join(output_dir, "training_curves.png"))


def main():
    """Load a config file and train the classical CNN it describes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_single(config, config["output_dir"])


if __name__ == "__main__":
    main()
