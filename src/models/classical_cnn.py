"""Baseline classical CNN for comparison against the hybrid quantum CNN."""

import torch.nn as nn


class ClassicalCNN(nn.Module):
    def __init__(self, num_classes=3, fc_hidden=64, adaptive_pool_size=None):
        """adaptive_pool_size and fc_hidden default to the exact Phase 1-5 architecture
        (7x7 flatten, 64-wide FC). Phase 6's capacity configs set adaptive_pool_size to
        shrink the flattened feature dimension, giving fc_hidden fine-grained control
        over total parameter count for the parameter-matched classical/hybrid comparison."""
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
        return self.classifier(x)  # raw logits; loss function applies softmax
