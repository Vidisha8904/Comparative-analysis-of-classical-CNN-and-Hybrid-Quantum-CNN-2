"""MNIST loading, class filtering, and per-class subsampling.

`get_dataloaders` is the single entrypoint every experiment script uses to
build its train/test split. Because a classical/hybrid pair being compared
always shares the same `dataset` block (classes, samples-per-class, data_seed),
calling this with either config yields an identical split -- that identity is
what makes the comparison fair in the first place.

Note this only ever reads `config["data_seed"]`, never `config["training_seed"]`.
That split matters: `data_seed` decides which images end up in the subset,
while `training_seed` (used elsewhere, via `set_seed()` in the experiment
scripts) only affects weight initialization and batch shuffling. Keeping them
separate is what makes the multi-seed variance experiments valid -- otherwise
"changing the seed" would confound two different questions at once.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_mnist(root, download=True):
    """Return (train_dataset, test_dataset) with ToTensor + normalization applied."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


def filter_classes(dataset, classes):
    """Return a Subset of `dataset` containing only samples whose label is in `classes`."""
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)
    mask = torch.isin(targets, torch.as_tensor(classes))
    indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


def subsample_per_class(dataset, n_per_class, seed):
    """Return a Subset with exactly `n_per_class` samples per class, reproducibly seeded."""
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    base_indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))

    targets = base.targets
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    rng = np.random.default_rng(seed)
    by_class = {}
    for idx in base_indices:
        label = int(targets[idx])
        by_class.setdefault(label, []).append(idx)

    selected = []
    for label in sorted(by_class):
        available = by_class[label]
        if len(available) < n_per_class:
            raise ValueError(
                f"Class {label} has only {len(available)} samples, "
                f"fewer than the requested {n_per_class}."
            )
        chosen = rng.choice(available, size=n_per_class, replace=False)
        selected.extend(int(i) for i in chosen)

    return Subset(base, selected)


def get_dataloaders(config):
    """Build (train_loader, test_loader) from a config dict's `dataset`/`training` blocks."""
    ds_cfg = config["dataset"]
    train_cfg = config["training"]

    train_raw, test_raw = load_mnist(root=ds_cfg["data_root"], download=True)

    train_filtered = filter_classes(train_raw, ds_cfg["classes"])
    test_filtered = filter_classes(test_raw, ds_cfg["classes"])

    if ds_cfg.get("full", False):
        train_loader = DataLoader(train_filtered, batch_size=train_cfg["batch_size"], shuffle=True)
        test_loader = DataLoader(test_filtered, batch_size=train_cfg["batch_size"], shuffle=False)
        return train_loader, test_loader

    train_subset = subsample_per_class(
        train_filtered, ds_cfg["samples_per_class_train"], config["data_seed"]
    )
    test_subset = subsample_per_class(
        test_filtered, ds_cfg["samples_per_class_test"], config["data_seed"]
    )

    train_loader = DataLoader(
        train_subset, batch_size=train_cfg["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=train_cfg["batch_size"], shuffle=False
    )
    return train_loader, test_loader
