"""Shared MLP used by all three experiments."""

from torch import nn


def build_model(config, n_features):
    hidden_1, hidden_2 = config["hidden_sizes"]
    return nn.Sequential(
        nn.Linear(n_features, hidden_1),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Linear(hidden_2, 3),
    )
