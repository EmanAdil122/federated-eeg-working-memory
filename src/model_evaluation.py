"""Score the three models on the same held-out evaluation trials."""

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .eeg_model import build_model
from .personalization import CALIBRATION_PER_CONDITION, calibration_split, model_inputs, personalize


def evaluate_models(experiment, selected_config, selected_state,
                    federated_config, best_federated_state, personalization_epochs):
    primary = experiment.primary
    train = experiment.train
    validation = experiment.validation
    feature_columns = experiment.feature_columns
    device = experiment.device
    results_dir = experiment.results_dir

    test = primary.loc[primary["split"].eq("test")].copy()
    assert test["participant_id"].nunique() == 4
    assert set(test["participant_id"]).isdisjoint(
        set(train["participant_id"]) | set(validation["participant_id"])
    )

    central_test_model = build_model(selected_config, len(feature_columns)).to(device)
    central_test_model.load_state_dict(selected_state)
    central_test_model.eval()

    fed_test_model = build_model(federated_config, len(feature_columns)).to(device)
    fed_test_model.load_state_dict(best_federated_state)
    fed_test_model.eval()

    prediction_tables = []
    calibration_tables = []

    for participant, subject in test.groupby("participant_id", sort=True):
        calibration, evaluation = calibration_split(subject, CALIBRATION_PER_CONDITION)
        assert set(calibration.index).isdisjoint(evaluation.index)

        calibration_tables.append(
            calibration[["participant_id", "trial_number", "condition", "load_level"]]
        )

        personalized_model = personalize(
            experiment, federated_config, best_federated_state, calibration, personalization_epochs
        )
        inputs = model_inputs(experiment, evaluation)

        models = {
            "Centralized ANN": central_test_model,
            "FedAvg ANN": fed_test_model,
            "Personalized FedAvg ANN": personalized_model,
        }

        for name, evaluated_model in models.items():
            evaluated_model.eval()
            with torch.no_grad():
                probabilities = evaluated_model(inputs).softmax(dim=1).cpu().numpy()

            result = evaluation[
                ["participant_id", "trial_number", "task_type", "load_level"]
            ].copy()
            result["model"] = name
            result["predicted_load"] = probabilities.argmax(axis=1) + 5
            result[["prob_load_5", "prob_load_6", "prob_load_7"]] = probabilities
            prediction_tables.append(result)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    calibration_trials = pd.concat(calibration_tables, ignore_index=True)

    keys = ["participant_id", "trial_number"]
    assert len(calibration_trials) == 120
    assert not predictions.duplicated(["model", *keys]).any()

    evaluation_keys = test[keys].merge(calibration_trials[keys], on=keys, how="left",
                                       indicator=True)
    evaluation_keys = evaluation_keys.loc[evaluation_keys["_merge"].eq("left_only")]
    expected_keys = set(evaluation_keys[keys].itertuples(index=False, name=None))

    metrics = []
    for name, rows in predictions.groupby("model", sort=False):
        assert set(rows[keys].itertuples(index=False, name=None)) == expected_keys

        actual = rows["load_level"]
        predicted = rows["predicted_load"]
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=[5, 6, 7], average="macro", zero_division=0
        )

        metrics.append({
            "model": name,
            "test_trials": len(rows),
            "accuracy": accuracy_score(actual, predicted),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "macro_roc_auc_ovr": roc_auc_score(
                actual,
                rows[["prob_load_5", "prob_load_6", "prob_load_7"]].to_numpy(),
                labels=[5, 6, 7],
                multi_class="ovr",
                average="macro",
            ),
        })

    test_metrics = pd.DataFrame(metrics)
    test_metrics.to_csv(results_dir / "final_test_metrics.csv", index=False)
    predictions.to_csv(results_dir / "final_test_predictions.csv", index=False)
    calibration_trials.to_csv(results_dir / "test_calibration_trials.csv", index=False)

    print(test_metrics.round(4).to_string(index=False))
    print("Saved final test metrics and predictions to:", results_dir)

    return test_metrics
