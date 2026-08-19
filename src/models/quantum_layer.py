"""Trainable variational quantum circuit, exposed as a standard PyTorch layer.

Gradients: the QNode is wrapped in `qml.qnn.TorchLayer`, which registers the
circuit's weights as a PyTorch parameter and computes gradients through
PennyLane's parameter-shift rule automatically. No manual gradient code (no
explicit backward pass, no manual shift-and-evaluate loop) is needed --
calling `.backward()` on a loss that depends on this layer's output simply
works the same way it does for any other nn.Module. This holds for both the
noise-free path used by Phase 1's base models and the noisy path used by
Phase 2's noise-robustness experiments below -- it's the same class either way.

Entangling layer choice: this uses `qml.BasicEntanglerLayers` (one trainable
RX rotation per qubit per layer, plus a fixed ring of CNOTs) rather than
`qml.StronglyEntanglingLayers` (three trainable rotations per qubit per
layer). BasicEntanglerLayers keeps the quantum parameter count small and
roughly linear in n_qubits * n_layers, which keeps ablation sweeps and the
parameter-count comparison manageable.

Noise: `noise_prob` (default 0.0) switches the circuit to a depolarizing-noise
model. At 0.0 nothing changes -- same `default.qubit`/`lightning.qubit` state-vector
device as Phase 1. Above 0.0, the device becomes `default.mixed` (required for
simulating noise channels -- it tracks a density matrix instead of a state
vector), and a `qml.DepolarizingChannel(noise_prob)` is applied to every qubit
once after the entangling layers, rather than after each individual rotation
gate inside them -- BasicEntanglerLayers applies its rotations as a single
template op, so per-gate insertion would mean hand-unrolling it. One channel
per qubit per circuit call is a standard simplification for this kind of
robustness sweep and keeps the noisy and noise-free circuits structurally
comparable.
"""

import pennylane as qml
import torch.nn as nn


def make_quantum_layer(n_qubits, n_layers, device="default.qubit", noise_prob=0.0):
    """Build a qml.qnn.TorchLayer implementing angle-embedded, entangled rotations.

    When noise_prob > 0.0, runs on `default.mixed` with a depolarizing channel
    on every qubit after the entangling layers; otherwise behaves exactly as
    the noise-free Phase 1 circuit on `device`.
    """
    dev_name = "default.mixed" if noise_prob > 0.0 else device
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        if noise_prob > 0.0:
            for wire in range(n_qubits):
                qml.DepolarizingChannel(noise_prob, wires=wire)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumLayer(nn.Module):
    """Thin nn.Module wrapper so the circuit can be constructed like any other layer."""

    def __init__(self, n_qubits, n_layers, device="default.qubit", noise_prob=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.noise_prob = noise_prob
        self.q_layer = make_quantum_layer(
            n_qubits, n_layers, device=device, noise_prob=noise_prob
        )

    def forward(self, x):
        return self.q_layer(x)
