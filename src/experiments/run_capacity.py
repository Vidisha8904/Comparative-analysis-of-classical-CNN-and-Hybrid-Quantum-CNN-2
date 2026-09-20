"""Experiment 6: parameter-matched capacity comparison.

Experiment 1's parameter-efficiency finding -- hybrid_full matches
classical_full's accuracy with far fewer parameters -- was never tested under
a controlled, equal-budget comparison. This script trains two new classical
CNNs on full MNIST -- classical_full_matched (~3,410 params, matching
hybrid_full exactly) and classical_full_mid (~10,000 params, filling in the
shape of the curve in between) -- then combines them with the existing (not
retrained) classical_full and hybrid_full results from Experiment 1 into a
single accuracy-vs-parameter-count comparison. See section 14 of the build spec.

Reuses run_classical.py's run_single training logic unchanged; this script's
only job is to loop over configs/capacity/*.yaml, verify each model's actual
parameter count lands within ~1% of its documented target_parameters before
training, organize output under results/capacity/, and fold in the two
reused Experiment 1 points to build results/capacity/summary.csv and
figures/capacity_accuracy_vs_params.png.

CLI: python -m src.experiments.run_capacity
"""

import os

import pandas as pd
import yaml

from src.experiments.run_classical import run_single
from src.models.classical_cnn import ClassicalCNN
from src.training.metrics import count_parameters
from src.utils.plotting import plot_accuracy_vs_params

NEW_CONFIGS = [
    "configs/capacity/classical_full_matched.yaml",
    "configs/capacity/classical_full_mid.yaml",
]

# The two Experiment 1 points this experiment reuses rather than retrains.
REUSED = {
    "classical_full": "results/base/classical_full/metrics.csv",
    "hybrid_full": "results/base/hybrid_full/metrics.csv",
}

SUMMARY_PATH = "results/capacity/summary.csv"
PARAM_COUNT_TOLERANCE = 0.01  # 1%


def final_epoch_row(metrics_csv_path):
    """Return the last row (final epoch) of a run's metrics CSV."""
    df = pd.read_csv(metrics_csv_path)
    return df.iloc[-1]


def check_parameter_count(config):
    """Build the model just to verify its actual parameter count is within
    ~1% of the config's documented target before spending any training time."""
    model_cfg = config["model"]
    model = ClassicalCNN(
        num_classes=model_cfg["num_classes"],
        fc_hidden=model_cfg.get("fc_hidden", 64),
        adaptive_pool_size=model_cfg.get("adaptive_pool_size"),
    )
    actual = count_parameters(model)
    target = model_cfg["target_parameters"]
    rel_error = abs(actual - target) / target
    print(f"  target={target}  actual={actual}  rel_error={rel_error:.2%}")
    if rel_error > PARAM_COUNT_TOLERANCE:
        raise ValueError(
            f"Parameter count {actual} is {rel_error:.2%} off target {target}, "
            f"exceeding the {PARAM_COUNT_TOLERANCE:.0%} tolerance -- adjust fc_hidden."
        )
    return actual


def run_new_configs():
    """Train both capacity-matched classical CNNs and collect their final metrics."""
    rows = []
    for i, config_path in enumerate(NEW_CONFIGS, start=1):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = config["output_dir"]
        config_name = os.path.basename(output_dir)

        print(f"=== [{i}/{len(NEW_CONFIGS)}] {config_name} ({config_path}) ===")
        check_parameter_count(config)

        print(f"--- Training {config_name} -> {output_dir} (12 epochs on full MNIST) ---")
        run_single(config, output_dir)

        final = final_epoch_row(os.path.join(output_dir, "metrics.csv"))
        rows.append({
            "config": config_name,
            "parameters": int(final["parameter_count"]),
            "accuracy": float(final["accuracy"]),
            "f1_macro": float(final["f1_macro"]),
            "precision": float(final["precision_macro"]),
            "recall": float(final["recall_macro"]),
            "time_per_epoch": float(final["epoch_time_sec"]),
        })
        print(
            f"[{i}/{len(NEW_CONFIGS)}] {config_name} done: "
            f"accuracy={final['accuracy']:.4f} f1_macro={final['f1_macro']:.4f}"
        )
    return rows


def collect_reused_rows():
    """Pull final-epoch metrics for the two Experiment 1 points without retraining them."""
    rows = []
    for config_name, metrics_csv_path in REUSED.items():
        final = final_epoch_row(metrics_csv_path)
        rows.append({
            "config": config_name,
            "parameters": int(final["parameter_count"]),
            "accuracy": float(final["accuracy"]),
            "f1_macro": float(final["f1_macro"]),
            "precision": float(final["precision_macro"]),
            "recall": float(final["recall_macro"]),
            "time_per_epoch": float(final["epoch_time_sec"]),
        })
        print(f"Reused {config_name} from {metrics_csv_path} (not retrained)")
    return rows


def main():
    """Train the capacity-matched configs, fold in the reused Experiment 1
    points, and write the combined summary CSV and comparison figure."""
    new_rows = run_new_configs()
    reused_rows = collect_reused_rows()

    summary_df = pd.DataFrame(new_rows + reused_rows).sort_values("parameters").reset_index(drop=True)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved {SUMMARY_PATH}")
    print(summary_df.to_string(index=False))

    os.makedirs("figures", exist_ok=True)
    plot_accuracy_vs_params(
        SUMMARY_PATH, "figures/capacity_accuracy_vs_params.png", highlight_config="hybrid_full"
    )
    print("Saved figures/capacity_accuracy_vs_params.png")


if __name__ == "__main__":
    main()
