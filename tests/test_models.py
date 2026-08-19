"""Smoke tests confirming both models run a forward pass without shape errors."""

import torch

from src.models.classical_cnn import ClassicalCNN
from src.models.hybrid_cnn import HybridCNN


def test_classical_cnn_forward_shape():
    model = ClassicalCNN(num_classes=3)
    x = torch.randn(4, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (4, 3)


def test_hybrid_cnn_forward_shape():
    model = HybridCNN(num_classes=3, n_qubits=4, n_qlayers=2)
    x = torch.randn(4, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (4, 3)


def test_hybrid_cnn_forward_is_differentiable():
    model = HybridCNN(num_classes=3, n_qubits=4, n_qlayers=1)
    x = torch.randn(2, 1, 28, 28)

    logits = model(x)
    loss = logits.sum()
    loss.backward()

    grad_found = any(p.grad is not None for p in model.parameters())
    assert grad_found


def test_classical_cnn_default_params_match_phase1():
    """adaptive_pool_size=None must reproduce Phase 1's exact parameter counts --
    the same reproducibility check used for the data_seed/training_seed split in
    Phase 4, now applied to the Phase 6 capacity refactor."""
    assert sum(p.numel() for p in ClassicalCNN(num_classes=10).parameters()) == 52138
    assert sum(p.numel() for p in ClassicalCNN(num_classes=3).parameters()) == 51683


def test_classical_cnn_adaptive_pool_shrinks_params():
    model = ClassicalCNN(num_classes=10, fc_hidden=29, adaptive_pool_size=2)
    x = torch.randn(4, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (4, 10)
    assert sum(p.numel() for p in model.parameters()) == 3433
