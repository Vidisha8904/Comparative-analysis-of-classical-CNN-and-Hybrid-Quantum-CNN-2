"""Smoke tests for dataset filtering and subsampling."""

import torch
from torch.utils.data import Subset

from src.data.dataset import filter_classes, subsample_per_class


class FakeMNIST:
    """Minimal stand-in for torchvision.datasets.MNIST, avoids network access in tests."""

    def __init__(self, targets, num_features=28 * 28):
        self.targets = torch.as_tensor(targets)
        self.data = torch.randn(len(targets), num_features)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


def make_fake_dataset(counts):
    """counts: dict[label -> count]. Builds a dataset with that many samples per label."""
    targets = []
    for label, count in counts.items():
        targets.extend([label] * count)
    return FakeMNIST(targets)


def test_filter_classes_keeps_only_requested_labels():
    dataset = make_fake_dataset({0: 5, 1: 5, 2: 5, 3: 5})
    filtered = filter_classes(dataset, [0, 1, 2])

    assert isinstance(filtered, Subset)
    assert len(filtered) == 15
    labels = {int(dataset.targets[i]) for i in filtered.indices}
    assert labels == {0, 1, 2}


def test_subsample_per_class_returns_exact_counts():
    dataset = make_fake_dataset({0: 50, 1: 50, 2: 50})
    filtered = filter_classes(dataset, [0, 1, 2])
    subsampled = subsample_per_class(filtered, n_per_class=10, seed=42)

    assert len(subsampled) == 30
    labels = [int(dataset.targets[i]) for i in subsampled.indices]
    for label in [0, 1, 2]:
        assert labels.count(label) == 10


def test_subsample_per_class_is_reproducible_with_same_seed():
    dataset = make_fake_dataset({0: 50, 1: 50, 2: 50})
    filtered = filter_classes(dataset, [0, 1, 2])

    subsample_a = subsample_per_class(filtered, n_per_class=10, seed=42)
    subsample_b = subsample_per_class(filtered, n_per_class=10, seed=42)

    assert sorted(subsample_a.indices) == sorted(subsample_b.indices)


def test_subsample_per_class_raises_if_not_enough_samples():
    dataset = make_fake_dataset({0: 5, 1: 5, 2: 5})
    filtered = filter_classes(dataset, [0, 1, 2])

    try:
        subsample_per_class(filtered, n_per_class=10, seed=42)
        assert False, "expected ValueError"
    except ValueError:
        pass
