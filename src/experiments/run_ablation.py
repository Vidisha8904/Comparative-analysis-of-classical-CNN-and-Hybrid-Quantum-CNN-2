"""Phase 3 ablation: one-factor-at-a-time sweeps over the hybrid model's quantum
circuit qubit count (Sweep A) and depth (Sweep B).

CLI: python -m src.experiments.run_ablation --sweep qubits
     python -m src.experiments.run_ablation --sweep depth
     python -m src.experiments.run_ablation --sweep all

Reuses HybridCNN and run_training unchanged -- this script's only job is to
loop over the configs/ablation/*.yaml grid, keep outputs organized under
results/ablation/, and build the cross-run summary CSV and comparison
figures. The n_qubits=4/n_qlayers=2 point is the existing Phase 1
hybrid_subset run in both sweeps, so it is copied from
results/base/hybrid_subset/ rather than retrained.
"""

import argparse
import os
import shutil

import pandas as pd
import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.hybrid_cnn import HybridCNN
from src.training.engine import run_training
from src.training.metrics import count_parameters
from src.utils.logger import MetricsLogger
from src.utils.plotting import plot_ablation_depth, plot_ablation_qubits, plot_training_curves
from src.utils.seed import set_seed

SWEEPS = {
    "qubits": [f"configs/ablation/qubits_{n}.yaml" for n in (2, 4, 6, 8)],
    "depth": [f"configs/ablation/depth_{n}.yaml" for n in (1, 2, 3, 4)],
}

# Configs whose grid point is identical to the Phase 1 baseline (n_qubits=4,
# n_qlayers=2) -- copy the existing run instead of retraining it.
REUSE_FROM = {
    "configs/ablation/qubits_4.yaml": "results/base/hybrid_subset",
    "configs/ablation/depth_2.yaml": "results/base/hybrid_subset",
}

SUMMARY_PATH = "results/ablation/metrics_summary.csv"


def run_single(config, output_dir):
    """Train one HybridCNN from a config dict and save its outputs -- identical
    logic to run_hybrid.py, just parameterized over output_dir."""
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
    print(f"[{output_dir}] trainable parameters: {count_parameters(model)}")

    os.makedirs(output_dir, exist_ok=True)

    logger = MetricsLogger(training_seed=config["training_seed"], data_seed=config["data_seed"])
    run_training(model, train_loader, test_loader, config, logger)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    csv_path = os.path.join(output_dir, "metrics.csv")
    logger.save(csv_path)
    plot_training_curves(csv_path, os.path.join(output_dir, "training_curves.png"))


def reuse_results(source_dir, output_dir):
    """Copy an existing run's outputs instead of retraining an identical grid point."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in ("metrics.csv", "model.pt", "training_curves.png"):
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    with open(os.path.join(output_dir, "REUSED_FROM.txt"), "w") as f:
        f.write(f"{source_dir}\n")
    print(f"[{output_dir}] reused from {source_dir} (identical to Phase 1 baseline)")


def collect_summary_row(config, output_dir, sweep_name):
    """Build one metrics_summary.csv row from a run's final-epoch metrics."""
    metrics_df = pd.read_csv(os.path.join(output_dir, "metrics.csv"))
    final = metrics_df.iloc[-1]
    model_cfg = config["model"]
    return {
        "sweep": sweep_name,
        "n_qubits": model_cfg["n_qubits"],
        "n_qlayers": model_cfg["n_qlayers"],
        "accuracy": final["accuracy"],
        "f1_macro": final["f1_macro"],
        "precision_macro": final["precision_macro"],
        "recall_macro": final["recall_macro"],
        "parameter_count": final["parameter_count"],
        "epoch_time_sec": final["epoch_time_sec"],
        "training_seed": final["training_seed"],
        "data_seed": final["data_seed"],
        "output_dir": output_dir,
    }


def update_summary_csv(new_rows):
    """Merge new_rows into results/ablation/metrics_summary.csv, replacing any
    existing row for the same output_dir so reruns don't duplicate."""
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(SUMMARY_PATH):
        existing = pd.read_csv(SUMMARY_PATH)
        existing = existing[~existing["output_dir"].isin(new_df["output_dir"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["sweep", "n_qubits", "n_qlayers"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    combined.to_csv(SUMMARY_PATH, index=False)
    return combined


def run_sweep(sweep_name):
    print(f"=== Ablation Sweep: {sweep_name} ===")
    rows = []
    for config_path in SWEEPS[sweep_name]:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = config["output_dir"]

        if config_path in REUSE_FROM:
            reuse_results(REUSE_FROM[config_path], output_dir)
        else:
            print(f"--- Training {config_path} -> {output_dir} ---")
            run_single(config, output_dir)

        rows.append(collect_summary_row(config, output_dir, sweep_name))

    update_summary_csv(rows)
    os.makedirs("figures", exist_ok=True)

    if sweep_name == "qubits":
        plot_ablation_qubits(SUMMARY_PATH, "figures/ablation_qubits.png")
        print("Saved figures/ablation_qubits.png")
    else:
        _, plateaued_or_declined = plot_ablation_depth(SUMMARY_PATH, "figures/ablation_depth.png")
        print("Saved figures/ablation_depth.png")
        if plateaued_or_declined:
            print(
                "Note: peak accuracy occurred below the maximum depth tested -- "
                "possible plateau/decline at higher layer counts (consistent with "
                "the known barren-plateau / vanishing-gradient effect in deep "
                "variational circuits). Worth flagging explicitly in the writeup."
            )

    summary = pd.read_csv(SUMMARY_PATH)
    print(summary[summary["sweep"] == sweep_name].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["qubits", "depth", "all"], required=True)
    args = parser.parse_args()

    if args.sweep in ("qubits", "all"):
        run_sweep("qubits")
    if args.sweep in ("depth", "all"):
        run_sweep("depth")


if __name__ == "__main__":
    main()
