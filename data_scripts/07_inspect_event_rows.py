
import pandas as pd
from pathlib import Path

ROOT = Path("data")

flagged_trials = {
    "sub-001": [26, 27, 44, 92, 104, 112],
    "sub-002": [70, 71],
}

for participant, trial_numbers in flagged_trials.items():
    print("\n" + "=" * 90)
    print(f"PARTICIPANT: {participant}")
    print("=" * 90)

    eeg_folder = ROOT / participant / "eeg"
    event_files = list(eeg_folder.glob("*_events.tsv"))

    if not event_files:
        print("ERROR: Events TSV file not found.")
        continue

    df = pd.read_csv(event_files[0], sep="\t")
    df = df.sort_values("onset").reset_index(drop=True)

    # Locate the beginning of every trial
    start_mask = df["trial_type"].astype(str).str.contains(
        "start of the trial", case=False, na=False
    )
    start_indices = df.index[start_mask].tolist()

    columns = [
        col for col in [
            "onset",
            "duration",
            "sample",
            "trial_type",
            "response_time",
            "stim_file",
            "value",
        ]
        if col in df.columns
    ]

    for trial_number in trial_numbers:
        print(f"\n--- Trial {trial_number} ---")

        position = trial_number - 1

        if position >= len(start_indices):
            print("ERROR: Trial number exceeds available trial starts.")
            continue

        start_idx = start_indices[position]

        if position + 1 < len(start_indices):
            next_start_idx = start_indices[position + 1]
        else:
            next_start_idx = len(df)

        # Include the trial's events and the next trial-start row
        trial_rows = df.iloc[start_idx:next_start_idx].copy()

        print(
            "Trial start onset:",
            df.loc[start_idx, "onset"]
        )

        print("\nAll event rows assigned to this trial:")
        print(trial_rows[columns].to_string(index=False))

        # Show rows immediately around the trial boundary too
        context_start = max(0, start_idx - 2)
        context_end = min(len(df), next_start_idx + 2)

        context = df.iloc[context_start:context_end]

        print("\nBoundary context (including nearby rows):")
        print(context[columns].to_string(index=False))
