"""Smoke test confirming the quantum layer produces correctly-shaped output."""

import torch

from src.models.quantum_layer import QuantumLayer


def test_quantum_layer_output_shape():
    n_qubits = 4
    n_layers = 2
    batch_size = 5

    layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
    dummy_input = torch.rand(batch_size, n_qubits) * torch.pi  # bounded angles

    output = layer(dummy_input)

    assert output.shape == (batch_size, n_qubits)


def test_quantum_layer_is_differentiable():
    n_qubits = 4
    n_layers = 1
    layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

    dummy_input = torch.rand(3, n_qubits, requires_grad=True) * torch.pi
    output = layer(dummy_input)
    loss = output.sum()
    loss.backward()

    for param in layer.parameters():
        assert param.grad is not None


def test_quantum_layer_noisy_output_shape():
    n_qubits = 4
    n_layers = 2
    batch_size = 5

    layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers, noise_prob=0.1)
    dummy_input = torch.rand(batch_size, n_qubits) * torch.pi

    output = layer(dummy_input)

    assert output.shape == (batch_size, n_qubits)


def test_quantum_layer_noisy_is_differentiable():
    n_qubits = 4
    n_layers = 1
    layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers, noise_prob=0.2)

    dummy_input = torch.rand(3, n_qubits, requires_grad=True) * torch.pi
    output = layer(dummy_input)
    loss = output.sum()
    loss.backward()

    for param in layer.parameters():
        assert param.grad is not None


def test_quantum_layer_noise_prob_zero_matches_noise_free_device():
    """noise_prob=0.0 must stay on the fast state-vector simulator, not default.mixed."""
    layer = QuantumLayer(n_qubits=4, n_layers=1, noise_prob=0.0)
    assert layer.q_layer.qnode.device.name == "default.qubit"


def test_quantum_layer_noise_prob_above_zero_uses_mixed_device():
    layer = QuantumLayer(n_qubits=4, n_layers=1, noise_prob=0.1)
    assert layer.q_layer.qnode.device.name == "default.mixed"
