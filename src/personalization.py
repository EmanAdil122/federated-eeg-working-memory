"""Adapt the federated output layer using participant calibration trials."""

import json

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .eeg_model import build_model

CALIBRATION_PER_CONDITION = 5
PERSONALIZATION_LR = 0.001
EPOCH_OPTIONS = [0, 1, 3, 5, 10]


def calibration_split(subject, trials_per_condition):
    ordered = subject.sort_values("trial_number")
    grouped = ordered.groupby("condition", sort=False)
    if grouped.ngroups != 6 or not grouped.size().gt(trials_per_condition).all():
        raise ValueError("Expected six conditions with calibration and evaluation trials.")

    calibration = grouped.head(trials_per_condition)
    evaluation = ordered.loc[~ordered.index.isin(calibration.index)]
    assert len(calibration) == 6 * trials_per_condition
    return calibration, evaluation


def model_inputs(experiment, rows):
    return torch.from_numpy(experiment.features(rows)).to(experiment.device)


def model_targets(experiment, rows):
    return torch.from_numpy(experiment.targets(rows)).to(experiment.device)


def personalize(experiment, federated_config, best_federated_state, calibration, epochs):
    personalized = build_model(federated_config, len(experiment.feature_columns)).to(experiment.device)
    personalized.load_state_dict(best_federated_state)

    # Keep the hidden layers fixed and switch off dropout during calibration.
    personalized.eval()
    personalized.requires_grad_(False)
    output_layer = personalized[-1]
    output_layer.requires_grad_(True)

    optimizer = torch.optim.Adam(output_layer.parameters(), lr=PERSONALIZATION_LR)
    criterion = nn.CrossEntropyLoss()
    inputs = model_inputs(experiment, calibration)
    targets = model_targets(experiment, calibration)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(personalized(inputs), targets)
        loss.backward()
        optimizer.step()

    return personalized


def tune_personalization(experiment, federated_config, best_federated_state):
    results_dir = experiment.results_dir

    records = []
    for participant, subject in experiment.validation.groupby("participant_id"):
        calibration, evaluation = calibration_split(subject, CALIBRATION_PER_CONDITION)
        inputs = model_inputs(experiment, evaluation)
        targets = model_targets(experiment, evaluation).cpu().numpy()

        for epochs in EPOCH_OPTIONS:
            personalized = personalize(experiment, federated_config, best_federated_state, calibration, epochs)
            with torch.no_grad():
                predictions = personalized(inputs).argmax(dim=1).cpu().numpy()

            records.append({
                "participant_id": participant,
                "adaptation_epochs": epochs,
                "calibration_trials": len(calibration),
                "evaluation_trials": len(evaluation),
                "accuracy": accuracy_score(targets, predictions),
                "macro_f1": f1_score(
                    targets, predictions, labels=[0, 1, 2],
                    average="macro", zero_division=0,
                ),
            })

    personalization_details = pd.DataFrame(records)
    personalization_summary = (
        personalization_details.groupby("adaptation_epochs")
        .agg(
            mean_participant_accuracy=("accuracy", "mean"),
            mean_participant_macro_f1=("macro_f1", "mean"),
        )
        .reset_index()
    )

    # Select an adaptation setting, with 0 epochs kept as the FedAvg reference.
    selected_personalization = (
        personalization_summary.loc[personalization_summary["adaptation_epochs"].gt(0)]
        .sort_values(
            ["mean_participant_macro_f1", "adaptation_epochs"],
            ascending=[False, True],
        )
        .iloc[0]
    )
    personalization_epochs = int(selected_personalization["adaptation_epochs"])

    personalization_details.to_csv(
        results_dir / "personalization_validation_by_participant.csv", index=False
    )
    personalization_summary.to_csv(
        results_dir / "personalization_validation_summary.csv", index=False
    )
    (results_dir / "personalization_config.json").write_text(
        json.dumps({
            "calibration_trials_per_condition": CALIBRATION_PER_CONDITION,
            "learning_rate": PERSONALIZATION_LR,
            "adaptation_epochs": personalization_epochs,
            "trainable_layer": "output",
        }, indent=2),
        encoding="utf-8",
    )

    print(personalization_summary.round(4).to_string(index=False))
    print("Selected adaptation epochs:", personalization_epochs)

    return personalization_epochs
