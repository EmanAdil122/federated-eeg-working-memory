
from pathlib import Path
import json
import mne
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

summary_rows = []
event_rows = []
missing_rows = []
error_rows = []

participants = sorted(DATA_DIR.glob("sub-*"))

for participant_dir in participants:
    participant = participant_dir.name
    eeg_dir = participant_dir / "eeg"

    if not eeg_dir.exists():
        eeg_dir = participant_dir

    set_files = list(eeg_dir.glob("*_eeg.set"))
    event_files = list(eeg_dir.glob("*_events.tsv"))

    if len(set_files) != 1 or len(event_files) != 1:
        error_rows.append({
            "participant": participant,
            "error": "Expected exactly one EEG .set file and one events.tsv file"
        })
        continue

    print(f"\nChecking {participant}...")

    try:
        # Load the EEG structure and metadata
        raw = mne.io.read_raw_eeglab(
            set_files[0],
            preload=False,
            verbose="ERROR"
        )

        sfreq = raw.info["sfreq"]
        duration = raw.n_times / sfreq
        channel_names = raw.ch_names

        # Load event timeline
        events = pd.read_csv(event_files[0], sep="\t")

        required_columns = {"onset", "trial_type"}
        if not required_columns.issubset(events.columns):
            raise ValueError("Events file lacks onset or trial_type")

        # Check missing values in important event fields
        missing_rows.append({
            "participant": participant,
            "missing_onset": int(events["onset"].isna().sum()),
            "missing_trial_type": int(events["trial_type"].isna().sum()),
            "missing_duration": (
                int(events["duration"].isna().sum())
                if "duration" in events.columns else None
            )
        })

        # Find exact duplicate event rows.
        # This is diagnostic only: we do not delete any events.
        duplicate_count = int(events.duplicated().sum())

        # Count the task events
        condition_events = events[
            events["trial_type"].astype(str).str.contains(
                r"Retention|Manipulation",
                case=False,
                na=False
            )
        ]

        counts = events["trial_type"].value_counts()

        for event_name, count in counts.items():
            event_rows.append({
                "participant": participant,
                "trial_type": event_name,
                "count": int(count)
            })

        # Check whether event onsets fall within the recording
        valid_onsets = events["onset"].dropna()
        outside_recording = int(
            ((valid_onsets < 0) | (valid_onsets > duration)).sum()
        )

        summary_rows.append({
            "participant": participant,
            "channels": len(channel_names),
            "sampling_frequency_Hz": sfreq,
            "duration_seconds": round(duration, 2),
            "event_rows": len(events),
            "unique_event_types": events["trial_type"].nunique(),
            "condition_event_rows": len(condition_events),
            "duplicate_event_rows": duplicate_count,
            "onsets_outside_recording": outside_recording,
            "channel_names": ", ".join(channel_names)
        })

        print("Channels:", len(channel_names))
        print("Sampling frequency:", sfreq, "Hz")
        print("Duration:", round(duration / 60, 2), "minutes")
        print("Event rows:", len(events))
        print("Exact duplicate event rows:", duplicate_count)
        print("Onsets outside recording:", outside_recording)

        raw.close()

    except Exception as exc:
        error_rows.append({
            "participant": participant,
            "error": str(exc)
        })
        print("ERROR:", exc)

# Save reports
reports = {
    "participant_validation.csv": summary_rows,
    "event_counts.csv": event_rows,
    "missing_values.csv": missing_rows,
    "validation_errors.csv": error_rows
}

for filename, rows in reports.items():
    pd.DataFrame(rows).to_csv(
        REPORT_DIR / filename,
        index=False
    )

print("\n========== VALIDATION COMPLETE ==========")
print("Participant summaries:", len(summary_rows))
print("Errors:", len(error_rows))
print("Reports saved in:", REPORT_DIR)
