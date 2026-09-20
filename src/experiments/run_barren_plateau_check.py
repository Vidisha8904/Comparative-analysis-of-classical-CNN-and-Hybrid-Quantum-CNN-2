"""Experiment 7: barren-plateau diagnostic. Measures gradient variance directly,
rather than inferring the presence or absence of a plateau from accuracy alone.

CLI: python -m src.experiments.run_barren_plateau_check --sweep qubits
     python -m src.experiments.run_barren_plateau_check --sweep depth
     python -m src.experiments.run_barren_plateau_check --sweep all
     python -m src.experiments.run_barren_plateau_check --sweep qubits --extended

Why this exists
---------------
Experiment 3 swept n_qubits and n_qlayers and found no monotonic accuracy trend;
Experiment 4 found the *shallow* configuration (depth_1) to be the unstable one,
which is the opposite of what a vanishing-gradient barren plateau predicts.
Accuracy alone cannot separate "no room left to improve" (ceiling effect on an
easy 3-class subset) from "gradients have vanished", because both look like a
flat accuracy curve. This experiment measures the thing itself.

The diagnostic is the standard one from McClean et al. (2018), "Barren plateaus
in quantum neural network training landscapes": fix one circuit parameter of
interest theta, sample all *other* parameters uniformly from [0, 2pi), and
estimate Var(dC/dtheta) across those samples. Under a barren plateau the
variance decays exponentially in the number of qubits (and, for sufficiently
expressive ansaetze, in depth), so a log-scale plot of the variance shows a
straight-line decline. A flat line rules the effect out over the range plotted.

The circuit under test
----------------------
The circuit is not reimplemented here. `make_quantum_layer` is imported from
src/models/quantum_layer.py and its QNode is differentiated directly, so what is
measured is literally the circuit HybridCNN trains: qml.AngleEmbedding followed
by qml.BasicEntanglerLayers on the same device backend, with the same
(n_layers, n_qubits) weight shape. Only the surrounding classical layers are
absent, which is what makes the measurement interpretable -- the question is
whether the *quantum* landscape is flat.

Diagnostic range vs trainable range
-----------------------------------
The qubit sweep can optionally be extended past 8 (--extended adds 10, 12, 14).
This range deliberately does not have to match Experiment 3's trainable-model
range: no training happens here, only gradient sampling at random weights, so
each extra qubit costs seconds rather than a full 15-epoch run. The whole point
of a barren-plateau check is to observe the trend outside the window that
happened to be convenient for training-time ablations -- a decay that only
becomes visible at 12+ qubits is still the correct answer to "does this ansatz
have a barren plateau?", even though it says nothing about the models actually
trained in Experiments 1-4.

Cost functions
--------------
Two cost functions are measured, on the *same* weight draws (paired, so any
difference between them is a real difference and not sampling noise):

  global: C = sum_i <Z_i> over all n_qubits measured wires. This is the cost the
          project's model actually implies -- HybridCNN feeds all n_qubits
          expectation values into `output_head`, a Linear layer, so every
          measured wire contributes to the training signal.
  local:  C = <Z_0>, a single wire only. Restricting to a local observable is
          one of the standard barren-plateau mitigations (Cerezo et al. 2021),
          so measuring both answers directly whether switching HybridCNN to a
          local readout would buy anything.

Caveat worth stating in the writeup: this "global" cost is a *sum of
single-qubit observables*, because that is what the model uses, not an n-body
product observable such as Z^(x)n. Cerezo et al.'s sharp global-vs-local
separation is stated for the latter. So if the global and local curves here
behave alike, that is evidence about *this model's* readout, and is not
evidence against the global-cost result in general.

Gradients
---------
On `default.qubit` with the torch interface, PennyLane's default diff_method
resolves to backpropagation through the simulator. That is analytically the same
derivative the parameter-shift rule computes -- parameter-shift is exact, not a
finite-difference approximation -- and it is markedly cheaper here, which is why
it is used. `tests/test_barren_plateau.py` asserts the two agree numerically, so
the choice is a performance detail rather than a change of method.

Reproducibility
---------------
Weight draws are seeded through src/utils/seed.py via the `sampling_seed` config
field. It is named separately from the data_seed/training_seed pair used
elsewhere because neither applies: no data is loaded and no model is trained.
Every configuration in a sweep re-seeds from the same value, so configurations
share a common random-number stream -- differences between grid points come from
the circuit, not from the draw.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.quantum_layer import make_quantum_layer
from src.utils.plotting import (
    GRADIENT_ZERO_TOL,
    plot_gradient_variance_vs_depth,
    plot_gradient_variance_vs_qubits,
)
from src.utils.seed import set_seed

SWEEPS = {
    "qubits": [f"configs/barren_plateau/qubits_{n}.yaml" for n in (2, 4, 6, 8)],
    "depth": [f"configs/barren_plateau/depth_{n}.yaml" for n in (1, 2, 3, 4)],
}

# Diagnostic-only grid points beyond the range any trained model in this project
# uses. Appended to the qubit sweep by --extended; see the module docstring.
EXTENDED_SWEEPS = {
    "qubits": [f"configs/barren_plateau/qubits_{n}.yaml" for n in (10, 12, 14)],
    "depth": [],
}

COST_TYPES = ("global", "local")

SUMMARY_PATH = "results/barren_plateau/summary.csv"

# metrics.csv column order, fixed so reruns and hand-inspection stay comparable.
METRIC_COLUMNS = [
    "n_qubits", "n_qlayers", "cost_type", "param_index",
    "gradient_variance", "gradient_mean", "n_samples", "seed",
]


def build_reference_input(n_qubits, reference_input):
    """Return the fixed embedding angles held constant across all N draws.

    "zeros" leaves the register in |0...0> after AngleEmbedding, so all
    randomness in the sampled gradient comes from the circuit weights -- the
    McClean et al. setup. "random" instead draws one representative embedded
    sample uniformly from [-pi, pi] (the range HybridCNN's tanh * pi bridge
    produces) and then holds *that* fixed, which checks the result is not an
    artifact of starting from a computational basis state.
    """
    if reference_input == "zeros":
        return torch.zeros(n_qubits, dtype=torch.float64)
    if reference_input == "random":
        return (torch.rand(n_qubits, dtype=torch.float64) * 2 - 1) * torch.pi
    raise ValueError(f"unknown reference_input: {reference_input!r}")


def sample_gradients(n_qubits, n_qlayers, n_samples, sampling_seed, param_index=0,
                     device="default.qubit", reference_input="zeros"):
    """Sample dC/dtheta at `n_samples` random weight initialisations.

    theta is the weight at flat index `param_index` of the (n_qlayers, n_qubits)
    weight array -- index 0 is the first rotation weight. Following McClean et
    al., theta is not held at a fixed value: the estimator is the variance of the
    partial derivative over the ensemble of random circuits, so every weight
    including theta is redrawn each sample, and `param_index` selects *which*
    derivative is recorded rather than which weight is frozen.

    Both cost functions are differentiated from the same forward pass at the same
    weights, making the global/local comparison paired.

    Returns {cost_type: np.ndarray of shape (n_samples,)}.
    """
    set_seed(sampling_seed)

    qnode = make_quantum_layer(n_qubits, n_qlayers, device=device).qnode
    inputs = build_reference_input(n_qubits, reference_input)

    grads = {cost_type: np.empty(n_samples) for cost_type in COST_TYPES}

    for i in range(n_samples):
        weights = torch.rand(n_qlayers, n_qubits, dtype=torch.float64) * 2 * torch.pi
        weights.requires_grad_(True)

        expvals = torch.stack(list(qnode(inputs, weights)))

        costs = {
            "global": expvals.sum(),   # all measured wires -- what HybridCNN reads out
            "local": expvals[0],       # single wire only -- the standard mitigation
        }
        for cost_type, cost in costs.items():
            (grad,) = torch.autograd.grad(cost, weights, retain_graph=True)
            grads[cost_type][i] = grad.flatten()[param_index].item()

    return grads


def measure_config(config):
    """Run the diagnostic for one config dict and return one row per cost type."""
    model_cfg = config["model"]
    sampling_cfg = config["sampling"]

    n_qubits = model_cfg["n_qubits"]
    n_qlayers = model_cfg["n_qlayers"]
    n_samples = sampling_cfg["n_samples"]
    param_index = sampling_cfg.get("param_index", 0)
    sampling_seed = config["sampling_seed"]

    grads = sample_gradients(
        n_qubits=n_qubits,
        n_qlayers=n_qlayers,
        n_samples=n_samples,
        sampling_seed=sampling_seed,
        param_index=param_index,
        device=model_cfg.get("device", "default.qubit"),
        reference_input=sampling_cfg.get("reference_input", "zeros"),
    )

    rows = []
    for cost_type in COST_TYPES:
        samples = grads[cost_type]
        rows.append({
            "n_qubits": n_qubits,
            "n_qlayers": n_qlayers,
            "cost_type": cost_type,
            "param_index": param_index,
            "gradient_variance": float(np.var(samples)),
            "gradient_mean": float(np.mean(samples)),
            "n_samples": n_samples,
            "seed": sampling_seed,
        })
        print(
            f"  n_qubits={n_qubits} n_qlayers={n_qlayers} cost={cost_type:<6} "
            f"Var(dC/dtheta)={rows[-1]['gradient_variance']:.6e} "
            f"mean={rows[-1]['gradient_mean']:+.6e}"
        )
    return rows


def update_summary_csv(new_rows):
    """Merge new_rows into results/barren_plateau/summary.csv, replacing any
    existing row for the same (sweep, n_qubits, n_qlayers, cost_type) so reruns
    -- including a rerun that adds --extended points -- don't duplicate."""
    key = ["sweep", "n_qubits", "n_qlayers", "cost_type"]
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(SUMMARY_PATH):
        existing = pd.read_csv(SUMMARY_PATH)
        merged = existing.merge(new_df[key], on=key, how="left", indicator=True)
        existing = existing[merged["_merge"].values == "left_only"]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(key).reset_index(drop=True)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    combined.to_csv(SUMMARY_PATH, index=False)
    return combined


def report_trend(df, x_column, label):
    """Print the least-squares slope of log10(variance) per cost type.

    This is the number the deliverable question turns on: a slope near zero is a
    flat landscape over the range tested (no barren plateau), a clearly negative
    slope is exponential decay. Also prints the local/global variance ratio at
    each grid point, which is the direct evidence on whether moving HybridCNN to
    a local readout would help.

    Grid points whose variance is exactly zero to machine precision are excluded
    from the fit and reported separately. They are not small gradients on a
    plateau -- they are an exact structural decoupling between the observable and
    the differentiated weight (the CNOT ring conjugates Z_0 into an operator with
    no support on the wire theta rotates, so the derivative vanishes
    identically). Folding a 1e-33 into a log-linear fit would produce a slope
    describing a numerical artifact rather than a decay rate.
    """
    print(f"\n--- Trend summary ({label}) ---")
    for cost_type in COST_TYPES:
        group = df[df["cost_type"] == cost_type].sort_values(x_column)
        fittable = group[group["gradient_variance"] > GRADIENT_ZERO_TOL]
        degenerate = group[group["gradient_variance"] <= GRADIENT_ZERO_TOL]

        if len(fittable) >= 2:
            slope = np.polyfit(fittable[x_column], np.log10(fittable["gradient_variance"]), 1)[0]
            decade_span = np.log10(
                fittable["gradient_variance"].max() / fittable["gradient_variance"].min()
            )
            print(
                f"{cost_type:<6}: log10(Var) slope = {slope:+.4f} per unit {x_column}; "
                f"total spread = {decade_span:.2f} decades over the range"
                + (" (excluding structurally-zero points)" if not degenerate.empty else "")
            )
        else:
            print(f"{cost_type:<6}: too few non-degenerate points to fit a trend")

        if not degenerate.empty:
            points = ", ".join(f"{x_column}={v}" for v in degenerate[x_column])
            print(
                f"        NOTE: gradient identically zero (< {GRADIENT_ZERO_TOL:g}) at {points} "
                f"-- the {cost_type} observable is structurally decoupled from "
                f"weights[0][{degenerate['param_index'].iloc[0]}] there, "
                "not flattened by a plateau."
            )

    pivot = df.pivot_table(index=x_column, columns="cost_type", values="gradient_variance")
    if {"global", "local"}.issubset(pivot.columns):
        print("local/global variance ratio (>1 would favour a local readout):")
        for x_value, row in pivot.iterrows():
            flag = "  <- structurally zero" if row["local"] <= GRADIENT_ZERO_TOL else ""
            print(f"  {x_column}={x_value}: {row['local'] / row['global']:.3f}x{flag}")


def run_sweep(sweep_name, extended=False):
    """Measure gradient variance across every grid point in one sweep, save
    metrics.csv + the summary CSV, plot the result, and print the trend."""
    print(f"=== Barren-plateau check: {sweep_name} sweep ===")
    config_paths = list(SWEEPS[sweep_name])
    if extended:
        config_paths += EXTENDED_SWEEPS[sweep_name]

    rows = []
    output_dir = None
    for config_path in config_paths:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = config["output_dir"]
        print(f"--- {config_path} ---")
        rows.extend(measure_config(config))

    metrics_df = pd.DataFrame(rows)[METRIC_COLUMNS]
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    update_summary_csv([{"sweep": sweep_name, **row} for row in rows])
    print(f"Saved {SUMMARY_PATH}")

    os.makedirs("figures", exist_ok=True)
    if sweep_name == "qubits":
        figure_path = "figures/gradient_variance_vs_qubits.png"
        plot_gradient_variance_vs_qubits(SUMMARY_PATH, figure_path)
        report_trend(metrics_df, "n_qubits", "gradient variance vs qubit count")
    else:
        figure_path = "figures/gradient_variance_vs_depth.png"
        plot_gradient_variance_vs_depth(SUMMARY_PATH, figure_path)
        report_trend(metrics_df, "n_qlayers", "gradient variance vs circuit depth")
    print(f"Saved {figure_path}\n")

    print(metrics_df.to_string(index=False))


def main():
    """Run the requested sweep (or both), optionally extended past the
    trainable range -- see the module docstring for what that means."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["qubits", "depth", "all"], required=True)
    parser.add_argument(
        "--extended", action="store_true",
        help="add diagnostic-only grid points beyond the trainable range "
             "(qubit sweep: 10, 12, 14) -- see the module docstring",
    )
    args = parser.parse_args()

    if args.sweep in ("qubits", "all"):
        run_sweep("qubits", extended=args.extended)
    if args.sweep in ("depth", "all"):
        run_sweep("depth", extended=args.extended)


if __name__ == "__main__":
    main()
