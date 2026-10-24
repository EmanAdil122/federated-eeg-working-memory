
from pathlib import Path

import mne
import numpy as np
import pandas as pd

DATA_DIR = Path("data")

for participant in ["sub-001", "sub-002"]:
    eeg_file = (
        DATA_DIR
        / participant
        / "eeg"
        / f"{participant}_task-VerbalWorkingMemory_eeg.set"
    )

    print("\n" + "=" * 75)
    print(f"PARTICIPANT: {participant}")
    print("=" * 75)

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose="ERROR"
    )

    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })

    print("Sampling rate:", raw.info["sfreq"], "Hz")
    print("Channel count:", len(raw.ch_names))
    print("EEG channels:", len(mne.pick_types(
        raw.info, eeg=True, eog=False, exclude=[]
    )))

    rows = []

    for channel in raw.ch_names:
        # Read one channel at a time; convert volts to microvolts.
        signal = raw.get_data(picks=[channel])[0] * 1e6

        rows.append({
            "channel": channel,
            "type": raw.get_channel_types(
                picks=[channel]
            )[0],
            "median_uV": np.median(signal),
            "std_uV": np.std(signal),
            "p99_abs_uV": np.percentile(np.abs(signal), 99),
            "p999_abs_uV": np.percentile(np.abs(signal), 99.9),
            "max_abs_uV": np.max(np.abs(signal)),
            "peak_to_peak_uV": np.ptp(signal),
        })

    result = pd.DataFrame(rows)

    print("\nChannel amplitude statistics:")
    print(
        result.sort_values("p999_abs_uV", ascending=False)
        .round(2)
        .to_string(index=False)
    )

    print("\nOverall signal statistics:")
    print(
        result[[
            "median_uV",
            "std_uV",
            "p99_abs_uV",
            "p999_abs_uV",
            "max_abs_uV",
        ]].describe().round(2).to_string()
    )
