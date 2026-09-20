"""Experiment 3: one-factor-at-a-time sweeps over the hybrid model's quantum
circuit qubit count (Sweep A) and depth (Sweep B).

CLI: python -m src.experiments.run_ablation --sweep qubits
     python -m src.experiments.run_ablation --sweep depth
     python -m src.experiments.run_ablation --sweep all
     python -m src.experiments.run_ablation --sweep qubits --dataset-regime full10_2500

Reuses HybridCNN and run_training unchanged -- this script's only job is to
loop over a config grid, keep outputs organized, and build the cross-run
summary CSV and comparison figures.

Dataset regimes
---------------
`--dataset-regime` picks *which config directory* the same sweep machinery
loads. Nothing else about the run differs between regimes, which is exactly
what makes the two directly comparable.

  subset3 (default)
      configs/ablation/ -- the original 3-class grid: digits 0/1/2, 300 train
      / 100 test per class, batch 32. Every configuration landed in a
      97.7-99.7% band with no trend against qubit count or depth, and the
      later gradient-variance diagnostic (Experiment 7) confirmed accuracy
      doesn't track gradient health here either (Pearson r = +0.12 across the
      qubit sweep despite a 15-fold variance drop). That points to a ceiling
      effect: the task is too easy, and a 300-image test set too coarse, for
      architectural differences to actually show up.

  full10_2500
      configs/ablation_full10/ -- all 10 classes at 2500 train / 500 test per
      class (25,000 / 5,000), batch 64, 15 epochs. A middle ground between
      subset3 and Experiment 1's hybrid_full (60k/10k, ~304s/epoch), which was
      never ablated because the full matrix wasn't tractable. The goal is to
      see whether the flat/zigzagging trend survives real 10-class decision
      boundaries and a much larger test set, or whether it was just masked by
      the subset's ceiling all along.

Shared grid points are trained once and copied, never retrained. Under subset3
the n_qubits=4/n_qlayers=2 point is Experiment 1's existing hybrid_subset run;
under full10_2500 no such prior run exists yet, so Sweep A trains it fresh as
qubits_4 and Sweep B copies that result. If the copy source is missing --
e.g. the depth sweep gets run on its own before the qubit sweep -- the config
is just trained normally instead, so either invocation order works.
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

QUBIT_POINTS = (2, 4, 6, 8)
DEPTH_POINTS = (1, 2, 3, 4)


def _sweeps(config_dir):
    """Both sweeps' config paths for a regime -- an identical OFAT grid either way."""
    return {
        "qubits": [f"{config_dir}/qubits_{n}.yaml" for n in QUBIT_POINTS],
        "depth": [f"{config_dir}/depth_{n}.yaml" for n in DEPTH_POINTS],
    }


# Each regime differs only in which configs are loaded and where results land.
# `reuse_from` maps a config whose grid point duplicates an already-trained run
# to that run's directory, so the shared n_qubits=4/n_qlayers=2 point is never
# trained twice.
REGIMES = {
    "subset3": {
        "config_dir": "configs/ablation",
        "results_dir": "results/ablation",
        "figure_prefix": "ablation",
        "label": "3-class subset (300/100 per class)",
        "reuse_from": {
            "configs/ablation/qubits_4.yaml": "results/base/hybrid_subset",
            "configs/ablation/depth_2.yaml": "results/base/hybrid_subset",
        },
    },
    "full10_2500": {
        "config_dir": "configs/ablation_full10",
        "results_dir": "results/ablation_full10",
        "figure_prefix": "ablation_full10",
        "label": "10-class, 2500/500 per class",
        # No prior 10-class 4-qubit run exists, so Sweep A trains this point and
        # Sweep B copies it. reuse_results() falls back to training when the
        # source is absent, so running the depth sweep alone still works.
        "reuse_from": {
            "configs/ablation_full10/depth_2.yaml": "results/ablation_full10/qubits_4",
        },
    },
}

DEFAULT_REGIME = "subset3"


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
    """Copy an existing run's outputs instead of retraining an identical grid point.

    Returns False when `source_dir` holds no metrics.csv -- the caller then trains
    the config normally. That happens under full10_2500 when the depth sweep runs
    before the qubit sweep that produces the shared 4-qubit/2-layer point.
    """
    if not os.path.exists(os.path.join(source_dir, "metrics.csv")):
        return False

    os.makedirs(output_dir, exist_ok=True)
    for fname in ("metrics.csv", "model.pt", "training_curves.png"):
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    with open(os.path.join(output_dir, "REUSED_FROM.txt"), "w") as f:
        f.write(f"{source_dir}\n")
    print(f"[{output_dir}] reused from {source_dir} (identical grid point, not retrained)")
    return True


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


def update_summary_csv(new_rows, summary_path):
    """Merge new_rows into the regime's metrics_summary.csv, replacing any
    existing row for the same output_dir so reruns don't duplicate."""
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["output_dir"].isin(new_df["output_dir"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["sweep", "n_qubits", "n_qlayers"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    combined.to_csv(summary_path, index=False)
    return combined


def run_sweep(sweep_name, regime=DEFAULT_REGIME):
    """Train (or reuse) every config in one sweep, then update that regime's
    summary CSV and figure."""
    regime_cfg = REGIMES[regime]
    summary_path = os.path.join(regime_cfg["results_dir"], "metrics_summary.csv")
    reuse_from = regime_cfg["reuse_from"]

    print(f"=== Ablation Sweep: {sweep_name} | regime: {regime} ({regime_cfg['label']}) ===")
    rows = []
    for config_path in _sweeps(regime_cfg["config_dir"])[sweep_name]:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = config["output_dir"]

        reused = config_path in reuse_from and reuse_results(reuse_from[config_path], output_dir)
        if not reused:
            print(f"--- Training {config_path} -> {output_dir} ---")
            run_single(config, output_dir)

        rows.append(collect_summary_row(config, output_dir, sweep_name))

    update_summary_csv(rows, summary_path)
    os.makedirs("figures", exist_ok=True)

    prefix = regime_cfg["figure_prefix"]
    # On its own line, not appended in brackets: the regime labels are long
    # enough that a single-line title overflows the 6x4 figure and gets clipped.
    suffix = f"\n{regime_cfg['label']}"

    if sweep_name == "qubits":
        figure_path = f"figures/{prefix}_qubits.png"
        plot_ablation_qubits(
            summary_path, figure_path,
            title=f"Ablation Sweep A: accuracy & parameter count vs qubit count{suffix}",
        )
        print(f"Saved {figure_path}")
    else:
        figure_path = f"figures/{prefix}_depth.png"
        _, plateaued_or_declined = plot_ablation_depth(
            summary_path, figure_path,
            title=f"Ablation Sweep B: accuracy vs circuit depth{suffix}",
        )
        print(f"Saved {figure_path}")
        if plateaued_or_declined:
            print(
                "Note: peak accuracy occurred below the maximum depth tested -- "
                "possible plateau/decline at higher layer counts (consistent with "
                "the known barren-plateau / vanishing-gradient effect in deep "
                "variational circuits). Worth flagging explicitly in the writeup."
            )

    summary = pd.read_csv(summary_path)
    print(summary[summary["sweep"] == sweep_name].to_string(index=False))


def main():
    """Run the requested sweep (or both) over the chosen dataset regime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["qubits", "depth", "all"], required=True)
    parser.add_argument(
        "--dataset-regime", choices=sorted(REGIMES), default=DEFAULT_REGIME,
        help="which config grid to sweep; see the module docstring "
             f"(default: {DEFAULT_REGIME}, Experiment 3's 3-class subset)",
    )
    args = parser.parse_args()

    if args.sweep in ("qubits", "all"):
        run_sweep("qubits", regime=args.dataset_regime)
    if args.sweep in ("depth", "all"):
        run_sweep("depth", regime=args.dataset_regime)


if __name__ == "__main__":
    main()
