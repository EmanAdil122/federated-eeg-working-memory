
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

FLAGGED_TRIALS = {
    "sub-001": [26, 27, 44, 92, 104, 112],
    "sub-002": [70, 71],
}

for participant, trial_numbers in FLAGGED_TRIALS.items():
    participant_dir = DATA_DIR / participant
    eeg_dir = participant_dir / "eeg"

    if not eeg_dir.exists():
        eeg_dir = participant_dir

    event_files = list(eeg_dir.glob("*_events.tsv"))

    if len(event_files) != 1:
        print(f"Cannot identify one events file for {participant}")
        continue

    events = pd.read_csv(event_files[0], sep="\t")
    events["onset"] = pd.to_numeric(
        events["onset"], errors="coerce"
    )
    events = events.dropna(subset=["onset"])
    events["trial_type"] = events["trial_type"].fillna("").astype(str)
    events = events.sort_values("onset").reset_index(drop=True)

    starts = events.index[
        events["trial_type"].str.strip().str.lower()
        == "start of the trial"
    ].tolist()

    print("\n" + "=" * 75)
    print("PARTICIPANT:", participant)
    print("=" * 75)

    for trial_number in trial_numbers:
        position = trial_number - 1

        if position >= len(starts):
            continue

        start_idx = starts[position]
        start_time = events.loc[start_idx, "onset"]

        if position + 1 < len(starts):
            next_start_idx = starts[position + 1]
            end_time = events.loc[next_start_idx, "onset"]
            trial_events = events.loc[
                (events.index >= start_idx)
                & (events.index < next_start_idx)
            ].copy()
        else:
            end_time = float("inf")
            trial_events = events.loc[
                events.index >= start_idx
            ].copy()

        probe_events = trial_events[
            trial_events["trial_type"].str.strip().str.lower()
            .str.startswith("probe:")
        ]

        probe_time = (
            probe_events.iloc[0]["onset"]
            if not probe_events.empty
            else None
        )

        print(f"\n--- Trial {trial_number} ---")
        print("Trial start:", start_time)
        print("Next trial start:", end_time)
        print("Probe onset:", probe_time)

        print("\nEvents in this trial:")
        for _, event in trial_events.iterrows():
            onset = event["onset"]
            label = event["trial_type"]

            if probe_time is not None:
                relative_to_probe = round(onset - probe_time, 3)
            else:
                relative_to_probe = None

            print(
                f"{onset:10.3f} s | "
                f"relative to probe: {str(relative_to_probe):>8} s | "
                f"{label}"
            )

        # Also show nearby events around the probe, including
        # any events that may have fallen outside this trial group.
        if probe_time is not None:
            nearby = events[
                (events["onset"] >= probe_time - 1)
                & (events["onset"] <= probe_time + 5)
            ]

            print("\nAll events from 1 second before to 5 seconds after probe:")
            for _, event in nearby.iterrows():
                print(
                    f"{event['onset']:10.3f} s | "
                    f"{event['trial_type']}"
                )
