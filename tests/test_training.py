"""Regression tests for training helpers such as checkpoint save/load."""

import torch
import torch.nn as nn

from src.training.engine import load_checkpoint, save_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_round_trip(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(model, optimizer, epoch=2, batch_idx=5, seed=7, path=checkpoint_path)
    checkpoint = load_checkpoint(path=checkpoint_path)

    assert checkpoint["epoch"] == 2
    assert checkpoint["batch_idx"] == 5
    assert checkpoint["seed"] == 7
    assert checkpoint["model_state_dict"]["fc.weight"].shape == (2, 4)
