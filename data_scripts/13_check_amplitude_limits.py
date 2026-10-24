
from pathlib import Path
import numpy as np
import pandas as pd
import mne

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

results = []

for eeg_file in sorted(DATA_DIR.glob("sub-*/eeg/*_eeg.set")):
    participant = eeg_file.parent.parent.name

    print(f"\nChecking {participant}...")

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=False,
        verbose="ERROR"
    )

    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })

    # Read the recording in 30-second chunks.
    chunk_size = int(raw.info["sfreq"] * 30)

    for start in range(0, raw.n_times, chunk_size):
        stop = min(start + chunk_size, raw.n_times)
        data = raw.get_data(start=start, stop=stop) * 1e6

        for i, name in enumerate(raw.ch_names):
            channel_data = data[i]
            near_limit = np.abs(channel_data) >= 9000

            results.append({
                "participant_id": participant,
                "channel": name,
                "channel_type": (
                    "EOG" if name in ["EOGv", "EOGh"] else "EEG"
                ),
                "samples_checked": len(channel_data),
                "samples_abs_ge_9000_uV": int(near_limit.sum()),
                "maximum_absolute_uV": float(
                    np.max(np.abs(channel_data))
                )
            })

    raw.close()

df = pd.DataFrame(results)

summary = (
    df.groupby(
        ["participant_id", "channel", "channel_type"],
        as_index=False
    )
    .agg({
        "samples_checked": "sum",
        "samples_abs_ge_9000_uV": "sum",
        "maximum_absolute_uV": "max"
    })
)

summary["percent_near_limit"] = (
    100 * summary["samples_abs_ge_9000_uV"]
    / summary["samples_checked"]
)

output = REPORT_DIR / "amplitude_limit_check.csv"
summary.to_csv(output, index=False)

print("\nChannels with the most samples above 9000 µV:")
print(
    summary.sort_values(
        "samples_abs_ge_9000_uV",
        ascending=False
    ).head(12).to_string(index=False)
)

print(f"\nFull report saved to: {output}")
