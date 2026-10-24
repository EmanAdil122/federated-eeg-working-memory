
from pathlib import Path
import numpy as np
import mne

for participant in ["sub-001", "sub-002"]:
    eeg_file = next(
        Path("data", participant, "eeg").glob("*_eeg.set")
    )

    raw = mne.io.read_raw_eeglab(
        eeg_file, preload=False, verbose="ERROR"
    )

    print(f"\n{participant}")

    # Examine the same time window in both participants.
    start = int(332.5 * raw.info["sfreq"])
    stop = int(333.5 * raw.info["sfreq"])
    data_uv = raw.get_data(start=start, stop=stop) * 1e6

    for i, channel in enumerate(raw.ch_names):
        signal = data_uv[i]
        boundary_count = np.sum(signal <= -10239.0)

        if boundary_count:
            print(
                f"{channel}: {boundary_count} samples "
                "at or below -10239 uV"
            )

    raw.close()
