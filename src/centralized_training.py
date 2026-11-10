"""Train and select a centralized model using validation macro-F1."""

from copy import deepcopy

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .eeg_model import build_model


def train_centralized(experiment):
    seed, device = experiment.seed, experiment.device
    feature_columns = experiment.feature_columns
    X_train = experiment.features(experiment.train)
    y_train = experiment.targets(experiment.train)
    y_val = experiment.targets(experiment.validation)
    val_inputs = torch.from_numpy(experiment.features(experiment.validation)).to(device)
    val_targets = torch.from_numpy(y_val).to(device)
    results_dir = experiment.results_dir

    baseline_config = {
        "hidden_sizes": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 30,
    }
    experiments = {
        "baseline": baseline_config,
        "smaller_network": {**baseline_config, "hidden_sizes": [32, 16]},
        "more_dropout": {**baseline_config, "dropout": 0.4},
        "lower_learning_rate": {**baseline_config, "learning_rate": 0.0005},
    }


    def train_configuration(settings):
        torch.manual_seed(seed)
        candidate = build_model(settings, len(feature_columns)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            candidate.parameters(), lr=settings["learning_rate"]
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=settings["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        records = []
        best_score = -1.0

        for epoch in range(1, settings["epochs"] + 1):
            candidate.train()
            total_loss = 0.0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(candidate(inputs), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)

            candidate.eval()
            with torch.no_grad():
                logits = candidate(val_inputs)
                validation_loss = criterion(logits, val_targets).item()
                predictions = logits.argmax(dim=1).cpu().numpy()

            score = f1_score(y_val, predictions, average="macro", zero_division=0)
            records.append({
                "epoch": epoch,
                "train_loss": total_loss / len(y_train),
                "validation_loss": validation_loss,
                "validation_accuracy": accuracy_score(y_val, predictions),
                "validation_macro_f1": score,
            })
            if score > best_score:
                best_score = score
                weights = deepcopy(candidate.state_dict())

        return pd.DataFrame(records), weights


    tuning_rows = []
    selected_score = -1.0

    for name, settings in experiments.items():
        print(f"Checking configuration: {name}", flush=True)

        run_history, weights = train_configuration(settings)

        row = run_history.loc[run_history["validation_macro_f1"].idxmax()]
        if name == "baseline":
            run_history.to_csv(results_dir / "centralized_baseline_history.csv", index=False)
            torch.save({
                **experiment.checkpoint_metadata(),
                "model_state_dict": {key: value.cpu() for key, value in weights.items()},
                "config": settings,
                "best_epoch": int(row["epoch"]),
                "validation_macro_f1": float(row["validation_macro_f1"]),
            }, results_dir / "centralized_baseline.pt")
        tuning_rows.append({
            "configuration": name,
            "hidden_sizes": str(settings["hidden_sizes"]),
            "dropout": settings["dropout"],
            "learning_rate": settings["learning_rate"],
            "batch_size": settings["batch_size"],
            "best_epoch": int(row["epoch"]),
            "validation_accuracy": float(row["validation_accuracy"]),
            "validation_macro_f1": float(row["validation_macro_f1"]),
        })

        if row["validation_macro_f1"] > selected_score:
            selected_score = float(row["validation_macro_f1"])
            selected_name = name
            selected_epoch = int(row["epoch"])
            selected_config = dict(settings)
            selected_state = {key: value.cpu().clone() for key, value in weights.items()}
            selected_history = run_history.copy()

    tuning = pd.DataFrame(tuning_rows).sort_values(
        "validation_macro_f1", ascending=False, kind="stable"
    )
    tuning.to_csv(results_dir / "centralized_tuning_results.csv", index=False)
    selected_history.to_csv(results_dir / "centralized_selected_history.csv", index=False)

    torch.save({
        "model_state_dict": selected_state,
        "config": selected_config,
        **experiment.checkpoint_metadata(),
        "best_epoch": selected_epoch,
        "validation_macro_f1": selected_score,
    }, results_dir / "centralized_selected.pt")

    print(tuning.round(4).to_string(index=False))
    print("Selected configuration:", selected_name)
    print("Saved to:", results_dir)

    return selected_config, selected_state
