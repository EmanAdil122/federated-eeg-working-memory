"""Load QC-annotated features and prepare the notebook's participant split."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


@dataclass
class Experiment:
    primary: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    class_ids: dict[int, int]
    seed: int
    device: torch.device
    scaler: StandardScaler
    results_dir: Path

    def features(self, rows):
        return self.scaler.transform(rows[self.feature_columns]).astype(np.float32)

    def targets(self, rows):
        return rows[self.target_column].map(self.class_ids).to_numpy(dtype=np.int64)

    def checkpoint_metadata(self):
        return {
            "feature_columns": self.feature_columns,
            "class_ids": self.class_ids,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }


def prepare_experiment(folder, results_dir, device):
    df = pd.read_csv(folder / "eeg_features_24_participants_qc_annotated.csv")
    schema = json.loads(
        (folder / "eeg_features_24_participants_schema.json").read_text(
            encoding="utf-8"
        )
    )

    feature_columns = schema["input_feature_columns"]
    target_column = "load_level"

    primary = df.loc[df["use_in_primary_analysis"].eq(True)].copy()
    participants = sorted(primary["participant_id"].unique())

    assert len(primary) == 2878
    assert len(participants) == 24
    assert len(feature_columns) == 57
    assert set(primary[target_column]) == {5, 6, 7}
    assert not primary.duplicated(["participant_id", "trial_number"]).any()
    assert np.isfinite(primary[feature_columns].to_numpy()).all()

    seed = 42
    shuffled = np.random.default_rng(seed).permutation(participants).tolist()
    split_ids = {
        "train": sorted(shuffled[:16]),
        "validation": sorted(shuffled[16:20]),
        "test": sorted(shuffled[20:]),
    }

    split_by_id = {
        participant: split
        for split, ids in split_ids.items()
        for participant in ids
    }
    primary["split"] = primary["participant_id"].map(split_by_id)

    summary = primary.groupby("split").agg(
        participants=("participant_id", "nunique"),
        trials=("trial_number", "size"),
    ).reindex(["train", "validation", "test"])

    assert summary["participants"].tolist() == [16, 4, 4]
    print(summary.to_string())

    manifest = {
        "seed": seed,
        "target_column": target_column,
        "target_classes": [5, 6, 7],
        "input_feature_columns": feature_columns,
        "participant_splits": split_ids,
        "primary_filter": "use_in_primary_analysis == True",
    }
    output = results_dir / "eeg_modeling_split_seed42.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Split saved:", output)

    train = primary.loc[primary["split"].eq("train")]
    validation = primary.loc[primary["split"].eq("validation")]
    scaler = StandardScaler().fit(train[feature_columns])
    return Experiment(primary, train, validation, feature_columns, target_column,
                      {5: 0, 6: 1, 7: 2}, seed, device, scaler, results_dir)
