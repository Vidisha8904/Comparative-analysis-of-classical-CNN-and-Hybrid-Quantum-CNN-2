"""CLI entrypoint: python -m src.experiments.run_noise_eval --config configs/noise/hybrid_noise_eval.yaml

Noise-robustness experiment, evaluation-only variant: no retraining involved.
Loads the already-trained hybrid_subset checkpoint once, then re-evaluates it
at each noise level in config['noise']['levels'] by rebuilding the model with
that noise_prob and loading the same trained weights back in. Noise only
changes how the circuit is executed, not the shape of its trainable
parameters, so the same state dict transfers cleanly across every noise level.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.dataset import get_dataloaders
from src.models.hybrid_cnn import HybridCNN
from src.training.engine import evaluate
from src.training.metrics import compute_metrics
from src.utils.plotting import plot_noise_sweep
from src.utils.seed import set_seed


def main():
    """Evaluate a trained hybrid model at every noise level in the config, and
    plot accuracy/F1 vs. noise level."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["training_seed"])

    _, test_loader = get_dataloaders(config)

    model_cfg = config["model"]
    noise_cfg = config["noise"]
    checkpoint_path = noise_cfg["base_model_checkpoint"]
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    rows = []
    for noise_prob in noise_cfg["levels"]:
        model = HybridCNN(
            num_classes=model_cfg["num_classes"],
            n_qubits=model_cfg["n_qubits"],
            n_qlayers=model_cfg["n_qlayers"],
            noise_prob=noise_prob,
        )
        model.load_state_dict(state_dict)
        model.to(device)

        test_loss, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        metrics_dict = compute_metrics(y_true, y_pred)

        print(
            f"noise_level={noise_prob:.2f} "
            f"accuracy={metrics_dict['accuracy']:.4f} "
            f"f1_macro={metrics_dict['f1_macro']:.4f} "
            f"test_loss={test_loss:.4f}"
        )

        rows.append({
            "noise_level": noise_prob,
            "test_loss": test_loss,
            "training_seed": config["training_seed"],
            "data_seed": config["data_seed"],
            **metrics_dict,
        })

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    os.makedirs("figures", exist_ok=True)
    plot_noise_sweep(csv_path, "figures/noise_eval_sweep.png")


if __name__ == "__main__":
    main()
