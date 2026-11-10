"""Run the full EEG modeling experiment from a QC feature table and schema."""

import argparse
from pathlib import Path

import torch

from .centralized_training import train_centralized
from .federated_training import train_federated
from .model_evaluation import evaluate_models
from .model_reporting import plot_architecture, plot_training_curves, write_error_analysis
from .personalization import tune_personalization
from .preprocessing import prepare_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the QC CSV and feature schema JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("data/modeling_results"),
                        help="Destination for checkpoints, CSVs, and figures")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or --device auto.")
    if device_name == "cpu":
        torch.set_num_threads(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = prepare_experiment(args.data_dir, args.output_dir, torch.device(device_name))
    central_config, central_weights = train_centralized(experiment)
    fed_config, fed_weights = train_federated(experiment, central_config)
    epochs = tune_personalization(experiment, fed_config, fed_weights)
    evaluate_models(experiment, central_config, central_weights, fed_config, fed_weights, epochs)
    write_error_analysis(args.output_dir)
    plot_training_curves(args.output_dir)
    plot_architecture(args.output_dir, central_config)


if __name__ == "__main__":
    main()
