"""Per-epoch metrics accumulation and CSV export, shared by both experiment runs."""

import os

import pandas as pd


class MetricsLogger:
    def __init__(self, training_seed=None, data_seed=None):
        self.training_seed = training_seed
        self.data_seed = data_seed
        self.rows = []

    def log_epoch(self, epoch, train_loss, test_loss, metrics_dict, epoch_time):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_time_sec": epoch_time,
            "training_seed": self.training_seed,
            "data_seed": self.data_seed,
        }
        row.update(metrics_dict)
        self.rows.append(row)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        return df
