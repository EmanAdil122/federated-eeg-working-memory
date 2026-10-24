
from pathlib import Path
import numpy as np
import mne

DATA_DIR = Path("data")

# Suspected limit from the earlier report, in microvolts.
LIMIT_UV = 10239.6875
TOLERANCE_UV = 0.5

for eeg_file in sorted(DATA_DIR.glob("sub-*/eeg/*_eeg.set")):
    print(f"\n{'=' * 55}")
    print(eeg_file.parent.parent.name)
    print("=" * 55)

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=False,
        verbose="ERROR"
    )

    raw.set_channel_types({"EOGv": "eog", "EOGh": "eog"})

    channel_names = raw.ch_names
    high_counts = np.zeros(len(channel_names), dtype=int)
    low_counts = np.zeros(len(channel_names), dtype=int)
    shared_timepoints = 0
    total_samples = 0

    chunk_size = int(raw.info["sfreq"] * 30)

    for start in range(0, raw.n_times, chunk_size):
        stop = min(start + chunk_size, raw.n_times)
        data_uv = raw.get_data(
            start=start, stop=stop
        ) * 1e6

        high = data_uv >= LIMIT_UV - TOLERANCE_UV
        low = data_uv <= -LIMIT_UV + TOLERANCE_UV

        high_counts += high.sum(axis=1)
        low_counts += low.sum(axis=1)

        # Time points where at least 5 channels approach either limit.
        near_limit = high | low
        shared_timepoints += (
            near_limit.sum(axis=0) >= 5
        ).sum()

        total_samples += data_uv.shape[1]

    print("Channel counts near positive/negative limit:")
    for i, name in enumerate(channel_names):
        if high_counts[i] or low_counts[i]:
            print(
                f"{name:6} positive={high_counts[i]:6} "
                f"negative={low_counts[i]:6}"
            )

    print(
        "Time points with 5+ channels near a limit:",
        shared_timepoints
    )
    print("Total time points:", total_samples)

    raw.close()
