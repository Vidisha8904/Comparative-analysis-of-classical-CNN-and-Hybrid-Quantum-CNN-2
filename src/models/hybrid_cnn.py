"""Hybrid classical-quantum CNN: classical conv front-end -> quantum layer -> classical head."""

import torch
import torch.nn as nn

from src.models.quantum_layer import QuantumLayer


class HybridCNN(nn.Module):
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
            # Force a fixed small spatial size regardless of input resolution or
            # conv config upstream, so the bridge below always sees 16*4*4 features.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Classical-to-quantum bridge: maps to exactly n_qubits values.
        self.to_quantum = nn.Linear(16 * 4 * 4, n_qubits)

        self.quantum = QuantumLayer(n_qubits, n_qlayers, device=device, noise_prob=noise_prob)

        self.output_head = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.to_quantum(x)

        # AngleEmbedding rotates by these values directly, so unbounded linear
        # output must be squashed into a bounded range first -- otherwise large
        # magnitudes wrap around the Bloch sphere many times and the rotation
        # angle becomes effectively meaningless noise. tanh -> [-1, 1], then
        # scaled to [-pi, pi], keeps every angle on a single, learnable turn.
        x = torch.tanh(x) * torch.pi

        x = self.quantum(x)
        return self.output_head(x)  # raw logits; loss function applies softmax
