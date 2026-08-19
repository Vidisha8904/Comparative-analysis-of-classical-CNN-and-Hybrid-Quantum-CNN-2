"""Phase 4 + Phase 5 multi-seed variance checks.

Phase 4 re-runs the Phase 3 ablation sweep extremes (qubits_2, qubits_8,
depth_1, depth_2) across training_seed in {43, 44, 45, 46} -- data_seed stays
fixed at 42 throughout, so only weight initialization and batch shuffling
vary between seeds, never which images were subsampled. Phase 5 applies the
same treatment to the Phase 1 base model comparison (classical_subset,
hybrid_subset), since that headline result was also single-seed only. In
both phases, the training_seed=42 point already exists from the earlier
phase (results/ablation/<group>/ or results/base/<group>/) and is reused
rather than retrained. Reports mean +/- sample std of accuracy and F1 across
all 5 seeds per config.

CLI: python -m src.experiments.run_multiseed --config-group qubits_2
     python -m src.experiments.run_multiseed --config-group hybrid_subset
     python -m src.experiments.run_multiseed --config-group all
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from src.experiments.run_ablation import run_single as run_single_hybrid
from src.experiments.run_classical import run_single as run_single_classical

# The four Phase 3 sweep extremes plus the two Phase 1 base models under
# study, and where each one's existing training_seed=42 result lives.
GROUPS = {
    "qubits_2": {"model_type": "hybrid", "n_qubits": 2, "n_qlayers": 2, "parent_results": "results/ablation/qubits_2"},
    "qubits_8": {"model_type": "hybrid", "n_qubits": 8, "n_qlayers": 2, "parent_results": "results/ablation/qubits_8"},
    "depth_1": {"model_type": "hybrid", "n_qubits": 4, "n_qlayers": 1, "parent_results": "results/ablation/depth_1"},
    "depth_2": {"model_type": "hybrid", "n_qubits": 4, "n_qlayers": 2, "parent_results": "results/ablation/depth_2"},
    "classical_subset": {"model_type": "classical", "n_qubits": "n/a", "n_qlayers": "n/a", "parent_results": "results/base/classical_subset"},
    "hybrid_subset": {"model_type": "hybrid", "n_qubits": 4, "n_qlayers": 2, "parent_results": "results/base/hybrid_subset"},
}


def train_config(config, output_dir, model_type):
    """Dispatch to the right model's training logic based on the group's model_type."""
    if model_type == "classical":
        run_single_classical(config, output_dir)
    else:
        run_single_hybrid(config, output_dir)

NEW_SEEDS = [43, 44, 45, 46]
SUMMARY_PATH = "results/multiseed/summary.csv"
PER_SEED_PATH = "results/multiseed/per_seed_metrics.csv"


def final_epoch_metrics(metrics_csv_path):
    df = pd.read_csv(metrics_csv_path)
    final = df.iloc[-1]
    return float(final["accuracy"]), float(final["f1_macro"])


def _update_csv(path, new_df, key_col, key_value):
    """Merge new_df into the CSV at path, replacing any existing rows for
    key_value so reruns don't duplicate."""
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing[key_col] != key_value]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(list(new_df.columns)[:2]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def update_summary_csv(row):
    """Merge one config's row into results/multiseed/summary.csv."""
    return _update_csv(SUMMARY_PATH, pd.DataFrame([row]), "config", row["config"])


def update_per_seed_csv(rows):
    """Merge one config's per-seed rows into results/multiseed/per_seed_metrics.csv --
    the flat (config, seed) -> accuracy/f1 table the individual summary.csv row is
    aggregated from, kept around so outlier seeds aren't hidden behind a mean."""
    return _update_csv(PER_SEED_PATH, pd.DataFrame(rows), "config", rows[0]["config"])


def run_group(group_name):
    spec = GROUPS[group_name]
    print(
        f"=== Multiseed group: {group_name} "
        f"(n_qubits={spec['n_qubits']}, n_qlayers={spec['n_qlayers']}) ==="
    )

    # training_seed=42 already exists from the earlier phase -- reuse, don't retrain.
    parent_csv = os.path.join(spec["parent_results"], "metrics.csv")
    acc42, f1_42 = final_epoch_metrics(parent_csv)
    accuracies = [acc42]
    f1s = [f1_42]
    per_seed_rows = [{
        "config": group_name,
        "training_seed": 42,
        "accuracy": acc42,
        "f1_macro": f1_42,
        "reused": True,
        "output_dir": spec["parent_results"],
    }]
    print(f"  seed 42 (reused from {parent_csv}): accuracy={acc42:.4f} f1_macro={f1_42:.4f}")

    for seed in NEW_SEEDS:
        config_path = f"configs/multiseed/{group_name}_seed{seed}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = config["output_dir"]

        print(f"--- Training {config_path} -> {output_dir} ---")
        train_config(config, output_dir, spec["model_type"])

        acc, f1 = final_epoch_metrics(os.path.join(output_dir, "metrics.csv"))
        accuracies.append(acc)
        f1s.append(f1)
        per_seed_rows.append({
            "config": group_name,
            "training_seed": seed,
            "accuracy": acc,
            "f1_macro": f1,
            "reused": False,
            "output_dir": output_dir,
        })
        print(f"  seed {seed}: accuracy={acc:.4f} f1_macro={f1:.4f}")

    update_per_seed_csv(per_seed_rows)

    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    row = {
        "config": group_name,
        "n_qubits": spec["n_qubits"],
        "n_qlayers": spec["n_qlayers"],
        "mean_accuracy": accuracies.mean(),
        "std_accuracy": accuracies.std(ddof=1),
        "mean_f1": f1s.mean(),
        "std_f1": f1s.std(ddof=1),
        "seeds_used": "42,43,44,45,46",
    }
    update_summary_csv(row)
    print(
        f"{group_name}: accuracy = {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}, "
        f"f1 = {row['mean_f1']:.4f} +/- {row['std_f1']:.4f} (n=5 seeds)"
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-group", choices=list(GROUPS) + ["all"], required=True)
    args = parser.parse_args()

    groups = list(GROUPS) if args.config_group == "all" else [args.config_group]
    for group_name in groups:
        run_group(group_name)


if __name__ == "__main__":
    main()
