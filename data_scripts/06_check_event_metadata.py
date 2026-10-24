
import json
from pathlib import Path

ROOT = Path("data")

for participant in ["sub-001", "sub-002"]:
    eeg_folder = ROOT / participant / "eeg"

    print("\n" + "=" * 65)
    print("PARTICIPANT:", participant)
    print("=" * 65)

    # Inspect the EEG metadata
    json_files = list(eeg_folder.glob("*_eeg.json"))

    for file in json_files:
        print("\nEEG metadata:", file.name)

        with open(file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for key, value in metadata.items():
            print(f"{key}: {value}")

    # Inspect the events file and its columns
    events_files = list(eeg_folder.glob("*_events.tsv"))

    for file in events_files:
        print("\nEvents file:", file.name)

        with open(file, "r", encoding="utf-8") as f:
            header = f.readline().strip()

        print("Columns:", header.split("\t"))
