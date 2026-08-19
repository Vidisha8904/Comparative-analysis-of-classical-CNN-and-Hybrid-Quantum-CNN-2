# QCNN Dissertation: Classical CNN vs. Hybrid Quantum CNN

Direct comparison of two image classifiers trained on identical MNIST splits:

- **Classical CNN** — standard conv/pool/dense architecture, with an optional adaptive-pool + configurable FC width for parameter-budget control (Experiment 6).
- **Hybrid quantum CNN** — classical conv layers feeding a trainable variational quantum circuit (PennyLane, angle embedding + entangling rotations), followed by a classical output layer.

Every experiment shares data (`data_seed`), and where relevant training initialization (`training_seed`), across the classical/hybrid pair being compared, so results isolate the effect under test rather than differences in data or seed. All experiments are evaluated on the same metrics: accuracy, macro F1, precision, recall, training time, and trainable parameter count.

## Project experiments

1. **Base models** — classical CNN vs. hybrid QCNN, on an identical subset (digits 0/1/2, 300/100 per class) and on full MNIST (10 classes, 60k/10k).
2. **Noise robustness** (subset only) — depolarizing-noise eval sweep on a trained model, plus independent noise-aware training runs at fixed noise levels.
3. **Qubit / depth ablation** (subset only) — one-factor-at-a-time sweeps over `n_qubits` and circuit depth (`n_qlayers`), single seed.
4. **Multi-seed variance check** (subset only) — re-ran Experiment 3's sweep extremes across 5 seeds each, to separate genuine architectural effects from random-init noise.
5. **Multi-seed base model comparison** (subset only) — applied the same 5-seed treatment to the Experiment 1 headline result (`classical_subset` vs `hybrid_subset`).
6. **Parameter-matched capacity comparison** (full dataset) — trains classical CNNs at parameter budgets matched to `hybrid_full`, to test whether Experiment 1's "hybrid needs fewer parameters" finding holds under a controlled, equal-budget comparison.

Findings from Experiments 3–6 are summarized in [Key results](#key-results) below and covered in full in the corresponding notebook (see [Notebooks](#notebooks)).

## Tech stack

- Python 3.12+ (developed/tested on 3.13)
- PyTorch + torchvision — classical CNN, training loop, data loading
- PennyLane (`default.qubit` / `lightning.qubit` for ideal runs, `default.mixed` for noisy runs) — quantum circuit simulation
- scikit-learn — F1 / precision / recall / confusion matrix
- matplotlib — plots
- pandas — metrics logging
- PyYAML — config files

## Folder structure

```
qcnn-dissertation/
├── configs/
│   ├── base/                 # Experiment 1: classical + hybrid, subset and full MNIST
│   ├── noise/                # Experiment 2: noise-robustness eval sweep + fixed-level training
│   ├── ablation/              # Experiment 3: qubit count / depth sweeps
│   ├── multiseed/             # Experiments 4 + 5: seed sweep configs (ablation extremes + base models)
│   └── capacity/               # Experiment 6: parameter-matched classical configs
├── data/                     # MNIST cache (gitignored, auto-downloaded)
├── src/
│   ├── data/                 # dataset loading, class filtering, subsampling
│   ├── models/               # ClassicalCNN, QuantumLayer (base + noisy), HybridCNN
│   ├── training/             # train/eval loop, metrics computation
│   ├── utils/                # seeding, CSV logging, plotting
│   └── experiments/          # CLI entrypoints, one per experiment (see Running below)
├── notebooks/                # one exploratory notebook per experiment (see Notebooks below)
├── results/
│   ├── base/                 # Experiment 1 checkpoints, metrics CSVs, plots
│   ├── noise/                # Experiment 2 eval-sweep and fixed-noise-level training outputs
│   ├── ablation/              # Experiment 3 outputs + metrics_summary.csv
│   ├── multiseed/             # Experiments 4 + 5 outputs + shared summary.csv / per_seed_metrics.csv
│   └── capacity/               # Experiment 6 outputs + summary.csv
├── figures/                  # comparison plots, confusion matrices (gitignored)
└── tests/                    # smoke tests for dataset/model/quantum-layer/training-helper behavior
```

Model code stays flat (`src/models/`) rather than being split per experiment — noise is a single optional `noise_prob` parameter on `quantum_layer.py`, seed behavior is two config fields (`data_seed`, `training_seed`), and capacity behavior is two optional constructor args (`fc_hidden`, `adaptive_pool_size`) on `classical_cnn.py`. One shared model implementation serves every experiment; it's the configs and results that are organized by experiment.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

To run the notebooks with this environment, also install a kernel once:

```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name qcnn-venv --display-name "Python 3 (qcnn-dissertation)"
```

## Running

```bash
# Experiment 1: base models
python -m src.experiments.run_classical --config configs/base/classical_subset.yaml
python -m src.experiments.run_classical --config configs/base/classical_full.yaml
python -m src.experiments.run_hybrid --config configs/base/hybrid_subset.yaml
python -m src.experiments.run_hybrid --config configs/base/hybrid_full.yaml

# Experiment 2: noise robustness (subset only)
python -m src.experiments.run_noise_eval --config configs/noise/hybrid_noise_eval.yaml
python -m src.experiments.run_noise_train --config configs/noise/hybrid_noise_train_10.yaml

# Experiment 3: qubit / depth ablation (subset only)
python -m src.experiments.run_ablation --sweep qubits
python -m src.experiments.run_ablation --sweep depth

# Experiments 4 + 5: multi-seed variance (subset only)
python -m src.experiments.run_multiseed --config-group qubits_2      # Experiment 4 groups: qubits_2, qubits_8, depth_1, depth_2
python -m src.experiments.run_multiseed --config-group hybrid_subset # Experiment 5 groups: classical_subset, hybrid_subset

# Experiment 6: parameter-matched capacity comparison (full dataset)
python -m src.experiments.run_capacity
```

Each writes a model checkpoint, `metrics.csv`, and a training-curve plot to the config's `output_dir`; the multi-run scripts (`run_ablation`, `run_multiseed`, `run_capacity`) additionally write a cross-run `summary.csv` and comparison figures.

### Tests

```bash
pytest tests/
```

Smoke tests cover dataset filtering/subsampling counts, quantum-layer output shape and differentiability (both `noise_prob=0.0` and `>0.0`), forward-pass shapes for both models, training-helper checkpoint save/load, and — since Experiment 6 — that `ClassicalCNN`'s `adaptive_pool_size`/`fc_hidden` refactor reproduces Experiment 1's exact parameter counts at its defaults.

## Notebooks

One exploratory notebook per experiment, each loading its experiment's configs/results directly and regenerating its figures on **Run All**:

| Notebook | Experiment | Covers |
|---|---|---|
| [base_model_comparison.ipynb](notebooks/base_model_comparison.ipynb) | 1 | classical vs. hybrid, subset and full MNIST |
| [noise_robustness.ipynb](notebooks/noise_robustness.ipynb) | 2 | eval-sweep vs. noise-aware training |
| [qubit_depth_fixed_seed_ablation.ipynb](notebooks/qubit_depth_fixed_seed_ablation.ipynb) | 3 | qubit-count and depth sweeps, single seed |
| [depth_sweep_training_curves.ipynb](notebooks/depth_sweep_training_curves.ipynb) | 3 | focused overlay of the depth sweep's training curves |
| [qubit_depth_multiseed_ablation.ipynb](notebooks/qubit_depth_multiseed_ablation.ipynb) | 4 | 5-seed variance check on the Experiment 3 sweep extremes |
| [base_model_seed_variance.ipynb](notebooks/base_model_seed_variance.ipynb) | 5 | 5-seed variance check on `classical_subset` vs `hybrid_subset` |
| [parameter_matched_capacity.ipynb](notebooks/parameter_matched_capacity.ipynb) | 6 | accuracy vs. parameter count, classical vs. `hybrid_full` |

## Configuration

All subset configs (`configs/base/*_subset.yaml`, `configs/noise/`, `configs/ablation/`, `configs/multiseed/`) share an identical `dataset` block — `classes`, `samples_per_class_train`, `samples_per_class_test` — and the same `data_seed: 42`. That identity is what makes every classical-vs-hybrid or seed-vs-seed comparison valid; treat any divergence as a bug.

Two separate seed fields, not one: `data_seed` (fixed at 42 everywhere) controls which images `subsample_per_class` picks; `training_seed` controls model weight initialization and batch shuffling only, and is the field that varies across Experiment 4/5's multi-seed sweeps. This split is what makes a multi-seed variance study valid — without it, "changing the seed" would confound two different questions into one number.

`model.n_qubits` / `model.n_qlayers` are hybrid-only. `model.fc_hidden` / `model.adaptive_pool_size` are classical-only, introduced in Experiment 6 to control parameter count; they default to the unmodified Experiment 1–5 architecture when unset.

## Design notes

- The quantum circuit uses `qml.BasicEntanglerLayers` (not `StronglyEntanglingLayers`) to keep the quantum parameter count small — this keeps `default.qubit` simulation fast for ablation sweeps and keeps the "hybrid uses far fewer parameters" comparison clean. See `src/models/quantum_layer.py`.
- The classical-to-quantum bridge (`src/models/hybrid_cnn.py`) pools to a fixed size, projects to exactly `n_qubits` values, then applies `tanh * π` before angle embedding — unbounded inputs would wrap around the Bloch sphere and produce meaningless rotations.
- The device runs in analytic mode (`shots=None`, the default) rather than sampling finite shots, so training reflects the model's learning capacity rather than shot noise — appropriate for a simulation study, not a hardware deployment.
- Every run is seeded (`src/utils/seed.py`) and both `data_seed` and `training_seed` are logged into each run's `metrics.csv` (`src/utils/logger.py`).
- `ClassicalCNN`'s `adaptive_pool_size` (Experiment 6) inserts an `AdaptiveAvgPool2d` before the flatten only when set; left at its default `None`, the architecture is byte-for-byte identical to Experiments 1–5 — verified in `tests/test_models.py`.

## Key results

**Experiment 1 — base models.** Single seed, subset: classical 98.00% vs. hybrid 97.67% (51,683 vs. 2,299 params). Full MNIST: classical 98.67% (52,138 params) vs. hybrid 97.07% (3,410 params).

**Experiment 3 — ablation (single seed, subset).** Non-monotonic on both sweeps — e.g. `depth_1` (99.33%) outscored the project baseline `depth_2` (97.67%), and `qubits_6` (99.67%) outscored `qubits_8` (98.00%) — raising a single-seed statistical caveat.

**Experiment 4 — multi-seed variance check (5 seeds, subset).** Qubit-sweep ranges overlapped heavily across seeds (Experiment 3's ranking was noise, not a real effect). Depth sweep did not: `depth_1` showed a genuine, bimodal ~40% class-collapse risk (90.5% ± 14.0%, individual seeds spanning 66.7%–99.3%), while `depth_2` — the project baseline — never collapsed across 5 seeds (98.9% ± 0.76%).

**Experiment 5 — multi-seed base model comparison (5 seeds, subset).** `classical_subset` 99.07% ± 0.60%; `hybrid_subset` 98.93% ± 0.76%. Ranges overlap almost completely and neither model showed Experiment 4's bimodal failure pattern — Experiment 1's single-seed comparison was representative, and the hybrid model is not meaningfully less reliably trainable than the classical model at this configuration.

**Experiment 6 — parameter-matched capacity comparison (single seed, full MNIST).**

| Config | Parameters | Accuracy |
|---|---:|---:|
| `hybrid_full` | 3,410 | 97.07% |
| `classical_full_matched` | 3,433 | 97.03% |
| `classical_full_mid` | 10,033 | 97.48% |
| `classical_full` | 52,138 | 98.67% |

At matched parameter count, a classical CNN performs statistically indistinguishably from `hybrid_full` (97.03% vs. 97.07%) — undercutting Experiment 1's "hybrid uses far fewer parameters" framing as a quantum-specific advantage. The more defensible reading: full MNIST simply doesn't require many parameters regardless of architecture, and this is the controlled check the unmatched Experiment 1 comparison was missing (see `parameter_matched_capacity.ipynb` for the full discussion, including the ~15–19x training-time gap between the two architectures at matched accuracy).

Numbers above are point-in-time snapshots — see each experiment's `results/*/summary.csv` or `metrics.csv` for current values, and re-run the corresponding notebook to regenerate.
