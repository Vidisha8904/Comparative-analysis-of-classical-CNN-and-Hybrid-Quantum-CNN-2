"""Smoke tests for the Experiment 7 gradient-variance diagnostic.

Wiring checks only -- these assert the measurement runs and produces sane
numbers at a tiny grid point, not that the variance takes any particular value.
The one substantive check is that backpropagation and the parameter-shift rule
agree, which is what licenses run_barren_plateau_check.py using the cheaper of
the two.
"""

import numpy as np
import pennylane as qml
import torch

from src.experiments.run_barren_plateau_check import (
    COST_TYPES,
    build_reference_input,
    measure_config,
    sample_gradients,
)
from src.models.quantum_layer import make_quantum_layer


def test_sample_gradients_returns_finite_samples():
    grads = sample_gradients(n_qubits=2, n_qlayers=1, n_samples=8, sampling_seed=42)

    assert set(grads) == set(COST_TYPES)
    for cost_type in COST_TYPES:
        assert grads[cost_type].shape == (8,)
        assert np.all(np.isfinite(grads[cost_type]))


def test_variance_is_finite_and_non_negative():
    grads = sample_gradients(n_qubits=2, n_qlayers=1, n_samples=8, sampling_seed=42)

    for cost_type in COST_TYPES:
        variance = np.var(grads[cost_type])
        assert np.isfinite(variance)
        assert variance >= 0.0


def test_sampling_seed_is_reproducible():
    kwargs = dict(n_qubits=2, n_qlayers=1, n_samples=8, sampling_seed=42)

    first = sample_gradients(**kwargs)
    second = sample_gradients(**kwargs)

    for cost_type in COST_TYPES:
        assert np.array_equal(first[cost_type], second[cost_type])


def test_measure_config_produces_one_row_per_cost_type():
    config = {
        "sampling_seed": 42,
        "model": {"n_qubits": 2, "n_qlayers": 1, "device": "default.qubit"},
        "sampling": {"n_samples": 8, "param_index": 0, "reference_input": "zeros"},
    }

    rows = measure_config(config)

    assert len(rows) == len(COST_TYPES)
    assert {row["cost_type"] for row in rows} == set(COST_TYPES)
    for row in rows:
        assert row["n_qubits"] == 2
        assert row["n_qlayers"] == 1
        assert row["n_samples"] == 8
        assert row["seed"] == 42
        assert np.isfinite(row["gradient_variance"])
        assert row["gradient_variance"] >= 0.0
        assert np.isfinite(row["gradient_mean"])


def test_zero_reference_input_leaves_register_in_computational_basis():
    inputs = build_reference_input(4, "zeros")

    assert torch.equal(inputs, torch.zeros(4, dtype=torch.float64))


def test_backprop_gradient_matches_parameter_shift():
    """The diagnostic differentiates via backprop for speed. Parameter-shift is
    exact rather than approximate, so the two must agree to numerical precision
    -- if they ever diverge, the speed shortcut is no longer sound."""
    n_qubits, n_qlayers = 2, 1
    inputs = build_reference_input(n_qubits, "zeros")
    weights = torch.rand(n_qlayers, n_qubits, dtype=torch.float64) * 2 * torch.pi

    backprop_node = make_quantum_layer(n_qubits, n_qlayers).qnode
    w = weights.clone().requires_grad_(True)
    cost = torch.stack(list(backprop_node(inputs, w))).sum()
    (backprop_grad,) = torch.autograd.grad(cost, w)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def shift_circuit(weights_):
        qml.AngleEmbedding(inputs.numpy(), wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights_, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    shift_grad = qml.jacobian(lambda w_: qml.numpy.sum(qml.numpy.stack(shift_circuit(w_))))(
        qml.numpy.array(weights.numpy(), requires_grad=True)
    )

    np.testing.assert_allclose(backprop_grad.numpy(), shift_grad, atol=1e-8)
