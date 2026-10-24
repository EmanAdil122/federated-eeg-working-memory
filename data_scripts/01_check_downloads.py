
from pathlib import Path
import pandas as pd

# Find the main project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Files needed for each participant
REQUIRED_FILES = [
    "_eeg.set",
    "_eeg.fdt",
    "_events.tsv",
    "_channels.tsv",
    "_eeg.json",
]

rows = []

# Find participant folders such as sub-001, sub-002, etc.
participants = sorted(DATA_DIR.glob("sub-*"))

if not participants:
    print("No participant folders found in:", DATA_DIR)
    print("Check that your downloaded data is inside the project's data folder.")
    raise SystemExit

for participant_dir in participants:

    # Look inside the participant's EEG folder
    eeg_dir = participant_dir / "eeg"

    # Also support files directly inside the participant folder
    if not eeg_dir.exists():
        eeg_dir = participant_dir

    files = list(eeg_dir.glob("*"))

    row = {
        "participant": participant_dir.name,
        "folder_exists": eeg_dir.exists(),
        "file_count": len(files),
    }

    for suffix in REQUIRED_FILES:
        matching_files = [
            file for file in files
            if file.name.endswith(suffix)
        ]

        row[suffix] = len(matching_files) > 0

    row["complete"] = all(
        row[suffix] for suffix in REQUIRED_FILES
    )

    # Calculate total size of files in this participant folder
    total_bytes = sum(
        file.stat().st_size
        for file in files
        if file.is_file()
    )

    row["size_MB"] = round(total_bytes / (1024 ** 2), 2)

    rows.append(row)

df = pd.DataFrame(rows)

report_path = REPORT_DIR / "download_inventory.csv"
df.to_csv(report_path, index=False)

print("\n========== DOWNLOAD INVENTORY ==========")
print(df.to_string(index=False))

print("\nParticipants found:", len(df))
print("Complete participant folders:", int(df["complete"].sum()))
print("Incomplete participant folders:", int((~df["complete"]).sum()))
print("Report saved to:", report_path)
