"""Create the error analysis tables and report figures from saved results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def write_error_analysis(results_dir):
    predictions = pd.read_csv(results_dir / "final_test_predictions.csv")
    loads = [5, 6, 7]

    class_records = []
    group_records = []

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    for ax, (model_name, rows) in zip(
        axes, predictions.groupby("model", sort=False)
    ):
        actual = rows["load_level"]
        predicted = rows["predicted_load"]

        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=loads, zero_division=0
        )
        for load, p, r, score, count in zip(loads, precision, recall, f1, support):
            class_records.append({
                "model": model_name,
                "load": load,
                "precision": p,
                "recall": r,
                "f1": score,
                "trials": count,
            })

        matrix = confusion_matrix(actual, predicted, labels=loads)
        proportions = matrix / matrix.sum(axis=1, keepdims=True)
        heatmap = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)

        for i in range(len(loads)):
            for j in range(len(loads)):
                value = proportions[i, j]
                ax.text(
                    j, i, f"{matrix[i, j]}\n{value:.0%}",
                    ha="center", va="center",
                    color="white" if value > 0.55 else "black",
                )

        ax.set_title(model_name)
        ax.set_xticks(range(len(loads)), loads)
        ax.set_yticks(range(len(loads)), loads)
        ax.set_xlabel("Predicted load")
        ax.set_ylabel("Actual load")

        for group_column in ["participant_id", "task_type"]:
            for group, subset in rows.groupby(group_column):
                _, _, score, _ = precision_recall_fscore_support(
                    subset["load_level"], subset["predicted_load"],
                    labels=loads, average="macro", zero_division=0,
                )
                group_records.append({
                    "model": model_name,
                    "group_type": group_column,
                    "group": group,
                    "trials": len(subset),
                    "accuracy": accuracy_score(
                        subset["load_level"], subset["predicted_load"]
                    ),
                    "macro_f1": score,
                })

    fig.colorbar(heatmap, ax=axes, label="Share of actual load")
    fig.savefig(results_dir / "test_confusion_matrices.png", dpi=200)
    plt.close(fig)

    per_class = pd.DataFrame(class_records)
    by_group = pd.DataFrame(group_records)

    per_class.to_csv(results_dir / "test_per_class_metrics.csv", index=False)
    by_group.to_csv(results_dir / "test_group_metrics.csv", index=False)

    print("Per-class recall — diagonal of the confusion matrices:")
    print(per_class.pivot(index="load", columns="model", values="recall").round(3))

    for group_column in ["participant_id", "task_type"]:
        print(f"Macro-F1 by {group_column}:")
        print(
            by_group.loc[by_group["group_type"].eq(group_column)]
            .pivot(index="group", columns="model", values="macro_f1")
            .round(3)
        )

    print("Saved error analysis to:", results_dir)


def plot_training_curves(results_dir):
    central = pd.read_csv(results_dir / "centralized_selected_history.csv")
    federated = pd.read_csv(results_dir / "federated_history.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    runs = [
        ("Centralized ANN", central, "epoch", "train_loss", "Training loss"),
        ("FedAvg ANN", federated, "round", "mean_local_training_loss", "Mean client training loss"),
    ]

    for row, (name, history, step, train_loss, train_label) in enumerate(runs):
        selected_step = history.loc[history["validation_macro_f1"].idxmax(), step]
        loss_ax, accuracy_ax = axes[row]

        loss_ax.plot(history[step], history[train_loss], label=train_label)
        loss_ax.plot(history[step], history["validation_loss"], label="Validation loss")
        loss_ax.set(title=f"{name} — loss", xlabel=step.capitalize(), ylabel="Cross-entropy")
        loss_ax.axvline(selected_step, color="gray", linestyle=":", label="Selected checkpoint")
        loss_ax.legend()

        accuracy_ax.plot(history[step], 100 * history["validation_accuracy"])
        accuracy_ax.axhline(100 / 3, color="gray", linestyle="--", label="Random guess (33.3%)")
        accuracy_ax.axvline(selected_step, color="gray", linestyle=":", label="Selected checkpoint")
        accuracy_ax.set(
            title=f"{name} — validation accuracy",
            xlabel=step.capitalize(),
            ylabel="Accuracy (%)",
        )
        accuracy_ax.legend()

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    output = results_dir / "training_and_validation_curves.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def plot_architecture(results_dir, config):
    fig, ax = plt.subplots(figsize=(12, 4.3), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 4.3))
    ax.axis("off")

    hidden_1, hidden_2 = config["hidden_sizes"]
    blocks = [
        (1.5, "Input\n57 EEG band-power features"),
        (4.5, f"Hidden layer 1\n{hidden_1} neurons · ReLU · Dropout {config['dropout']}"),
        (7.5, f"Hidden layer 2\n{hidden_2} neurons · ReLU"),
        (10.5, "Output\n3 scores · loads 5, 6, 7"),
    ]

    for x, label in blocks:
        ax.text(
            x, 2.9, label, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.55", fc="#eaf3fb", ec="#31668a"),
        )

    for left, right in [(2.65, 3.18), (5.85, 6.43), (8.52, 9.5)]:
        ax.annotate(
            "", xy=(right, 2.9), xytext=(left, 2.9),
            arrowprops=dict(arrowstyle="->", color="#31668a", lw=1.5),
        )

    ax.text(6, 3.9, "Shared ANN architecture for EEG memory-load classification",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6, 2.05, "Training loss: cross-entropy   |   Optimizer: Adam",
            ha="center", fontsize=10)
    ax.text(6, 1.45, "Centralized: train on pooled features from 16 participants",
            ha="center", fontsize=10)
    ax.text(6, 1.0, "FedAvg: train locally on 16 participant clients and average their model weights",
            ha="center", fontsize=10)
    ax.text(6, 0.55, "Personalized FedAvg: adapt only the output layer with 30 labeled trials per new participant",
            ha="center", fontsize=10)

    output = results_dir / "eeg_ann_architecture.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output)
