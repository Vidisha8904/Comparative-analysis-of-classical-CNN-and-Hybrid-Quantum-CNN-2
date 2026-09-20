"""Trainable variational quantum circuit, exposed as a standard PyTorch layer.

Gradients
---------
The QNode is wrapped in `qml.qnn.TorchLayer`, which registers the circuit's
weights as a normal PyTorch parameter and differentiates the circuit through
PennyLane's parameter-shift rule automatically. There's no manual backward
pass or shift-and-evaluate loop to maintain -- calling `.backward()` on a loss
that depends on this layer works exactly like it would for any other
`nn.Module`. That holds for both the noise-free circuit used by the base
models and the noisy variant used in the noise-robustness experiments; it's
the same class either way, just a different `noise_prob`.

Why BasicEntanglerLayers
-------------------------
The ansatz is `qml.BasicEntanglerLayers` (one trainable RX rotation per qubit
per layer, plus a fixed ring of CNOTs) rather than the richer
`qml.StronglyEntanglingLayers` (three trainable rotations per qubit per
layer). Keeping the rotation count to one per qubit keeps the quantum
parameter count small and roughly linear in `n_qubits * n_layers`, which is
what keeps the ablation sweeps and the parameter-count comparisons tractable.

Noise
-----
`noise_prob` (default 0.0) switches the circuit into a depolarizing-noise
model. At 0.0 nothing changes -- it's the same `default.qubit`/`lightning.qubit`
state-vector device used everywhere else. Above 0.0, the device switches to
`default.mixed` (needed to simulate noise channels, since it tracks a density
matrix instead of a state vector), and a `qml.DepolarizingChannel(noise_prob)`
is applied to every qubit once, after the entangling layers -- not after each
individual rotation inside them. `BasicEntanglerLayers` applies its rotations
as a single template op, so per-gate noise insertion would mean hand-unrolling
it. One channel per qubit per circuit call is a standard simplification for
this kind of robustness sweep, and keeps the noisy and noise-free circuits
structurally comparable.
"""

import pennylane as qml
import torch.nn as nn


def make_quantum_layer(n_qubits, n_layers, device="default.qubit", noise_prob=0.0):
    """Build a qml.qnn.TorchLayer implementing angle-embedded, entangled rotations.

    When noise_prob > 0.0, runs on `default.mixed` with a depolarizing channel
    on every qubit after the entangling layers; otherwise behaves exactly as
    the noise-free circuit on `device`.
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
