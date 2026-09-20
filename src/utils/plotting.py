"""Plotting helpers for training curves and cross-run comparisons."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(csv_path, save_path):
    """Plot loss and accuracy vs epoch from a single run's metrics CSV."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["test_loss"], label="test_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["accuracy"], label="accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_noise_sweep(csv_path, save_path):
    """Plot accuracy (and F1) vs noise level from an eval-sweep metrics CSV.

    Expects a `noise_level` column plus `accuracy` and `f1_macro`, as written
    by run_noise_eval.py -- one row per level in `noise.levels`.
    """
    df = pd.read_csv(csv_path).sort_values("noise_level")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["noise_level"], df["accuracy"], marker="o", label="accuracy")
    ax.plot(df["noise_level"], df["f1_macro"], marker="o", label="f1_macro")
    ax.set_xlabel("Depolarizing noise probability")
    ax.set_ylabel("Score")
    ax.set_title("Hybrid model: accuracy vs noise level")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_ablation_qubits(summary_csv, save_path, title=None):
    """Plot final-epoch accuracy and parameter count vs n_qubits (Experiment 3, Sweep A).

    Expects results/ablation/metrics_summary.csv with a `sweep` column; only
    rows where sweep == "qubits" are used. Dual axes since more qubits also
    means more trainable parameters -- the interesting question is whether
    the accuracy gain (if any) justifies the parameter cost.

    `title` overrides the default, e.g. to name the dataset regime being swept
    when the same plot is produced for more than one of them.
    """
    df = pd.read_csv(summary_csv)
    df = df[df["sweep"] == "qubits"].sort_values("n_qubits")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["n_qubits"], df["accuracy"], marker="o", color="tab:blue", label="accuracy")
    ax1.set_xlabel("Number of qubits")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(df["n_qubits"])

    ax2 = ax1.twinx()
    ax2.plot(
        df["n_qubits"], df["parameter_count"], marker="s", color="tab:orange",
        label="parameter count",
    )
    ax2.set_ylabel("Trainable parameters", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(title or "Ablation Sweep A: accuracy & parameter count vs qubit count")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_ablation_depth(summary_csv, save_path, title=None):
    """Plot final-epoch accuracy vs n_qlayers (Experiment 3, Sweep B).

    Expects results/ablation/metrics_summary.csv with a `sweep` column; only
    rows where sweep == "depth" are used. A plateau or decline at higher
    layer counts is a known, citable phenomenon in variational quantum
    circuits (barren plateaus / vanishing gradients), not a bug -- callers
    should check the return value to decide whether to flag it.

    `title` overrides the default, e.g. to name the dataset regime being swept
    when the same plot is produced for more than one of them.
    """
    df = pd.read_csv(summary_csv)
    df = df[df["sweep"] == "depth"].sort_values("n_qlayers")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["n_qlayers"], df["accuracy"], marker="o", color="tab:blue")
    ax.set_xlabel("Number of entangling layers (depth)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title or "Ablation Sweep B: accuracy vs circuit depth")
    ax.set_xticks(df["n_qlayers"])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    peak_idx = df["accuracy"].idxmax()
    plateaued_or_declined = df["n_qlayers"].loc[peak_idx] < df["n_qlayers"].max()
    return fig, plateaued_or_declined


def plot_multiseed_variance(summary_csv, per_seed_csv, save_path, configs=None, title=None):
    """Plot mean +/- std accuracy per config as an error-bar chart, with each
    individual seed's accuracy overlaid as a scatter point.

    The overlay matters: a config's std can be large because every seed is
    moderately spread out, or because one seed collapsed while the rest
    agree closely -- those are different findings, and a bare error bar
    can't distinguish them. Expects summary_csv from run_multiseed.py's
    update_summary_csv (columns: config, mean_accuracy, std_accuracy) and
    per_seed_csv from update_per_seed_csv (columns: config, training_seed,
    accuracy).

    Both CSVs are shared across every multiseed group (Experiment 4 and
    Experiment 5 alike), so `configs` restricts the plot to a chosen subset --
    e.g. just ["classical_subset", "hybrid_subset"] -- in the given order,
    instead of plotting every row accumulated so far. `title` overrides the
    default title for callers plotting a different subset.
    """
    summary_df = pd.read_csv(summary_csv)
    if configs is not None:
        summary_df = summary_df[summary_df["config"].isin(configs)].copy()
        summary_df["config"] = pd.Categorical(summary_df["config"], categories=configs, ordered=True)
        summary_df = summary_df.sort_values("config").reset_index(drop=True)
    else:
        summary_df = summary_df.sort_values("config").reset_index(drop=True)
    per_seed_df = pd.read_csv(per_seed_csv)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_positions = range(len(summary_df))

    ax.errorbar(
        x_positions, summary_df["mean_accuracy"], yerr=summary_df["std_accuracy"],
        fmt="o", markersize=8, capsize=6, color="tab:blue", label="mean +/- std (n=5 seeds)",
        zorder=3,
    )

    for x, config in zip(x_positions, summary_df["config"]):
        seed_accs = per_seed_df.loc[per_seed_df["config"] == config, "accuracy"]
        jitter = (pd.Series(range(len(seed_accs))) - (len(seed_accs) - 1) / 2) * 0.03
        ax.scatter(
            [x] * len(seed_accs) + jitter, seed_accs,
            color="tab:orange", alpha=0.8, s=30, zorder=2,
            label="individual seed" if x == 0 else None,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(summary_df["config"])
    ax.set_ylabel("Accuracy")
    ax.set_title(title or "Experiment 4: accuracy variance across training_seed (data_seed fixed at 42)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_accuracy_vs_params(summary_csv, save_path, highlight_config=None):
    """Plot accuracy vs parameter count (Experiment 6 capacity comparison).

    Expects results/capacity/summary.csv with columns `config`, `parameters`,
    `accuracy`. All rows except `highlight_config` are treated as the classical
    capacity curve (sorted by parameter count and connected with a line, log-scale
    x-axis since parameter counts here span an order of magnitude). `highlight_config`
    (e.g. "hybrid_full") is plotted as a distinct marker instead of being folded into
    the curve, so it's visually obvious whether it sits above, on, or below the
    classical accuracy-per-parameter frontier at its own parameter count.
    """
    df = pd.read_csv(summary_csv)

    classical_df = df[df["config"] != highlight_config].sort_values("parameters")
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        classical_df["parameters"], classical_df["accuracy"],
        marker="o", color="tab:blue", label="classical (parameter-matched)", zorder=2,
    )
    for _, row in classical_df.iterrows():
        ax.annotate(
            row["config"], (row["parameters"], row["accuracy"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )

    if highlight_config is not None and (df["config"] == highlight_config).any():
        highlight_row = df[df["config"] == highlight_config].iloc[0]
        ax.scatter(
            [highlight_row["parameters"]], [highlight_row["accuracy"]],
            marker="*", s=220, color="tab:red", label=highlight_config, zorder=3,
        )
        ax.annotate(
            highlight_config, (highlight_row["parameters"], highlight_row["accuracy"]),
            textcoords="offset points", xytext=(8, -12), fontsize=8, color="tab:red",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment 6: accuracy vs parameter count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_comparison(csv_paths, metric, save_path, title=None):
    """Overlay a single metric across multiple runs.

    Args:
        csv_paths: dict mapping a run label (e.g. "classical_subset", "hybrid")
            to the path of that run's metrics CSV.
        metric: column name to plot (e.g. "accuracy", "f1_macro", "train_loss").
        save_path: where to save the resulting figure.
        title: optional override for the default f"{metric} comparison" title,
            e.g. to name the group being compared.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        ax.plot(df["epoch"], df[metric], label=label, marker="o")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


# Below this, a sampled gradient variance is exactly zero to machine precision
# rather than merely small: the observable and the differentiated weight are
# structurally decoupled by the ansatz (see run_barren_plateau_check.py). Such
# points cannot be drawn on a log axis or included in a decay fit -- 1e-33 would
# stretch the axis over thirty meaningless decades -- so they are separated out
# and marked, never silently dropped.
GRADIENT_ZERO_TOL = 1e-20


def _plot_gradient_variance(summary_csv, save_path, sweep, x_column, x_label, title,
                            trained_range=None):
    """Shared implementation for the two Experiment 7 barren-plateau diagnostic plots.

    Plots Var(dC/dtheta) against `x_column` on a log-scale y-axis, one line per
    cost_type (global vs local). The log scale is the whole point: a barren
    plateau means the variance decays exponentially in the swept quantity, which
    shows up as a *straight line* here. A flat line rules the effect out over the
    range plotted.

    The legend reports the least-squares slope of log10(variance) vs the swept
    quantity, so "is this a straight decline or flat noise?" can be read off a
    number rather than eyeballed -- a slope near 0 is flat, a clearly negative
    slope is exponential decay of the form Var ~ 10^(slope * x).

    `trained_range` optionally shades the sub-range this project actually trains
    models over, so diagnostic-only points sampled beyond it (which no trained
    model in this dissertation uses) stay visually distinguishable.
    """
    df = pd.read_csv(summary_csv)
    df = df[df["sweep"] == sweep].sort_values(x_column)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"global": "tab:blue", "local": "tab:orange"}

    nonzero = df[df["gradient_variance"] > GRADIENT_ZERO_TOL]
    floor = nonzero["gradient_variance"].min() / 5 if not nonzero.empty else 1e-12

    for cost_type, group in df.groupby("cost_type"):
        group = group.sort_values(x_column)
        drawable = group[group["gradient_variance"] > GRADIENT_ZERO_TOL]
        degenerate = group[group["gradient_variance"] <= GRADIENT_ZERO_TOL]

        label = f"{cost_type} cost"
        if len(drawable) >= 2:
            slope = np.polyfit(drawable[x_column], np.log10(drawable["gradient_variance"]), 1)[0]
            label = f"{label} (log10 slope = {slope:+.3f}/step)"
        ax.plot(
            drawable[x_column], drawable["gradient_variance"],
            marker="o", color=colors.get(cost_type), label=label,
        )

        if not degenerate.empty:
            # Pinned below the smallest real measurement, hollow, so it reads as
            # "off the scale / identically zero" rather than "very small".
            ax.scatter(
                degenerate[x_column], [floor] * len(degenerate),
                marker="v", s=70, facecolors="none", edgecolors=colors.get(cost_type),
                zorder=4,
                label=f"{cost_type}: gradient identically zero (structural)",
            )

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 2)

    if trained_range is not None:
        ax.axvspan(
            trained_range[0], trained_range[1], color="tab:green", alpha=0.07, zorder=0,
        )
        ax.axvline(trained_range[1], color="tab:green", linestyle=":", linewidth=1)
        ax.text(
            trained_range[1], ax.get_ylim()[1], " diagnostic-only range ->",
            fontsize=7, color="tab:green", va="top",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Var($\partial C / \partial \theta$)  [log scale]")
    ax.set_title(title)
    ax.set_xticks(sorted(df[x_column].unique()))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_gradient_variance_vs_qubits(summary_csv, save_path, trained_range=(2, 8)):
    """Experiment 7, Sweep A: gradient variance vs qubit count, global and local cost."""
    return _plot_gradient_variance(
        summary_csv, save_path,
        sweep="qubits",
        x_column="n_qubits",
        x_label="Number of qubits",
        title="Barren-plateau check: gradient variance vs qubit count (depth fixed at 2)",
        trained_range=trained_range,
    )


def plot_gradient_variance_vs_depth(summary_csv, save_path):
    """Experiment 7, Sweep B: gradient variance vs circuit depth, global and local cost."""
    return _plot_gradient_variance(
        summary_csv, save_path,
        sweep="depth",
        x_column="n_qlayers",
        x_label="Number of entangling layers (depth)",
        title="Barren-plateau check: gradient variance vs circuit depth (width fixed at 4)",
    )
