"""Hybrid classical-quantum CNN: classical conv front-end -> quantum layer -> classical head."""

import torch
import torch.nn as nn

from src.models.quantum_layer import QuantumLayer


class HybridCNN(nn.Module):
    """Classical conv layers feed a trainable variational quantum circuit,
    whose measured qubits are then read out by a classical linear layer.

    The two classical halves are ordinary CNN building blocks; `QuantumLayer`
    in the middle is what makes this "hybrid" rather than just a CNN with an
    unusual bottleneck. See `quantum_layer.py` for what happens inside it.
    """

    def __init__(
        self, num_classes=3, n_qubits=4, n_qlayers=2, device="default.qubit", noise_prob=0.0
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.device = device
        self.noise_prob = noise_prob

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Fixes the spatial size regardless of input resolution, so the
            # bridge below can always assume 16*4*4 features coming in.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Classical-to-quantum bridge: collapses the conv features to exactly
        # n_qubits values, one per qubit's rotation angle.
        self.to_quantum = nn.Linear(16 * 4 * 4, n_qubits)

        self.quantum = QuantumLayer(n_qubits, n_qlayers, device=device, noise_prob=noise_prob)

        self.output_head = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.to_quantum(x)

        # AngleEmbedding uses these values directly as rotation angles, so an
        # unbounded linear output has to be squashed into a bounded range first.
        # Left alone, a large value would wrap around the Bloch sphere several
        # times before landing, turning the rotation into effective noise.
        # tanh -> [-1, 1], scaled to [-pi, pi], keeps every angle on one clean turn.
        x = torch.tanh(x) * torch.pi

        x = self.quantum(x)
        return self.output_head(x)  # raw logits -- softmax happens inside the loss
