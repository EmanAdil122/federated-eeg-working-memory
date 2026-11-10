"""Simulate sample-weighted FedAvg over the training participants."""

from copy import deepcopy

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .eeg_model import build_model


def train_federated(experiment, selected_config):
    seed, device = experiment.seed, experiment.device
    feature_columns = experiment.feature_columns
    X_train = experiment.features(experiment.train)
    y_train = experiment.targets(experiment.train)
    y_val = experiment.targets(experiment.validation)
    val_inputs = torch.from_numpy(experiment.features(experiment.validation)).to(device)
    val_targets = torch.from_numpy(y_val).to(device)
    train = experiment.train
    results_dir = experiment.results_dir

    torch.manual_seed(seed)

    federated_config = {
        **selected_config,
        "rounds": 30,
        "local_epochs": 1,
    }

    # Reuse the training-only scaler from the centralized experiment.
    # This is a federated simulation within one process.
    client_loaders = {}
    client_sizes = {}

    for index, (participant, positions) in enumerate(
        train.groupby("participant_id", sort=True).indices.items()
    ):
        dataset = TensorDataset(
            torch.from_numpy(X_train[positions]),
            torch.from_numpy(y_train[positions]),
        )
        client_loaders[participant] = DataLoader(
            dataset,
            batch_size=federated_config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + index + 1),
        )
        client_sizes[participant] = len(dataset)

    assert len(client_loaders) == 16
    total_trials = sum(client_sizes.values())
    assert total_trials == len(y_train)

    # Start from fresh weights so the federated model is trained independently.
    global_model = build_model(federated_config, len(feature_columns)).to(device)
    criterion = nn.CrossEntropyLoss()
    federated_history = []
    best_federated_f1 = -1.0

    for round_number in range(1, federated_config["rounds"] + 1):
        global_weights = deepcopy(global_model.state_dict())
        averaged_weights = {
            name: torch.zeros_like(weight) for name, weight in global_weights.items()
        }
        total_local_loss = 0.0

        for participant, loader in client_loaders.items():
            client_model = build_model(federated_config, len(feature_columns)).to(device)
            client_model.load_state_dict(global_weights)
            client_model.train()

            optimizer = torch.optim.Adam(
                client_model.parameters(), lr=federated_config["learning_rate"]
            )

            for _ in range(federated_config["local_epochs"]):
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(client_model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                    total_local_loss += loss.item() * len(targets)

            share = client_sizes[participant] / total_trials
            for name, weight in client_model.state_dict().items():
                averaged_weights[name].add_(weight, alpha=share)

        global_model.load_state_dict(averaged_weights)
        global_model.eval()
        with torch.no_grad():
            logits = global_model(val_inputs)
            validation_loss = criterion(logits, val_targets).item()
            predictions = logits.argmax(dim=1).cpu().numpy()

        score = f1_score(y_val, predictions, average="macro", zero_division=0)
        federated_history.append({
            "round": round_number,
            "mean_local_training_loss": total_local_loss / (
                total_trials * federated_config["local_epochs"]
            ),
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy_score(y_val, predictions),
            "validation_macro_f1": score,
        })

        if score > best_federated_f1:
            best_federated_f1 = score
            best_round = round_number
            best_federated_state = {
                name: weight.cpu().clone()
                for name, weight in global_model.state_dict().items()
            }

        if round_number % 5 == 0:
            print(f"Round {round_number}: validation macro-F1 = {score:.4f}")

    global_model.load_state_dict(best_federated_state)
    federated_history_df = pd.DataFrame(federated_history)
    federated_history_df.to_csv(results_dir / "federated_history.csv", index=False)

    torch.save({
        "model_state_dict": best_federated_state,
        "config": federated_config,
        **experiment.checkpoint_metadata(),
        "best_round": best_round,
        "validation_macro_f1": best_federated_f1,
    }, results_dir / "federated_selected.pt")

    print(federated_history_df.loc[federated_history_df["round"].eq(best_round)].round(4).to_string(index=False))
    print("Saved federated model and history to:", results_dir)

    return federated_config, best_federated_state
