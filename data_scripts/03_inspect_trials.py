
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

all_rows = []

for participant_dir in sorted(DATA_DIR.glob("sub-*")):
    participant = participant_dir.name
    eeg_dir = participant_dir / "eeg"

    if not eeg_dir.exists():
        eeg_dir = participant_dir

    event_files = list(eeg_dir.glob("*_events.tsv"))

    if len(event_files) != 1:
        print(f"Skipping {participant}: expected one events file")
        continue

    events = pd.read_csv(event_files[0], sep="\t")

    print(f"\n{'=' * 60}")
    print(f"PARTICIPANT: {participant}")
    print(f"{'=' * 60}")

    # Show all event types and their counts
    print("\nEvent types and counts:")
    print(
        events["trial_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Show condition-specific event counts
    condition_mask = events["trial_type"].astype(str).str.contains(
        r"(5R|6R|7R|5M|6M|7M)",
        case=False,
        na=False,
        regex=True
    )

    condition_events = events.loc[
        condition_mask,
        ["onset", "trial_type", "value"]
    ].copy()

    condition_events.insert(0, "participant", participant)
    all_rows.extend(condition_events.to_dict("records"))

    # Display a chronological sample for understanding event order
    print("\nFirst 35 events in chronological order:")
    print(
        events[["onset", "trial_type"]]
        .sort_values("onset")
        .head(35)
        .to_string(index=False)
    )

# Save condition-related events for further inspection
output_path = REPORT_DIR / "condition_event_details.csv"

pd.DataFrame(all_rows).to_csv(output_path, index=False)

print("\nSaved condition event details to:")
print(output_path)
