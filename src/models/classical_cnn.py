"""Baseline classical CNN, used as the accuracy/parameter-count reference point
for the hybrid quantum CNN throughout this project.
"""

import torch.nn as nn


class ClassicalCNN(nn.Module):
    """Small conv/pool/dense classifier for MNIST-sized (1x28x28) images.

    `adaptive_pool_size` and `fc_hidden` default to the original architecture
    used everywhere except Experiment 6 (7x7 flatten into a 64-wide FC layer).
    Experiment 6 sets `adaptive_pool_size` to shrink the flattened feature map,
    with `fc_hidden` then giving fine-grained control over the final parameter
    count -- that's what lets a classical model be built at, say, exactly
    hybrid_full's parameter budget for a fair comparison.
    """

    def __init__(self, num_classes=3, fc_hidden=64, adaptive_pool_size=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        if adaptive_pool_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))
            flat_dim = 16 * adaptive_pool_size * adaptive_pool_size
        else:
            self.pool = nn.Identity()
            flat_dim = 16 * 7 * 7

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # raw logits -- softmax happens inside the loss
