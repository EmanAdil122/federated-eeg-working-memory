
from pathlib import Path

import mne
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
TRIAL_FILE = Path("reports/trial_table.csv")

trials = pd.read_csv(TRIAL_FILE)
results = []

for participant in ["sub-001", "sub-002"]:
    eeg_file = (
        DATA_DIR
        / participant
        / "eeg"
        / f"{participant}_task-VerbalWorkingMemory_eeg.set"
    )

    print(f"\nLoading {participant}...")
    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose="ERROR"
    )

    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })

    sfreq = raw.info["sfreq"]
    eeg_channels = [
        raw.ch_names[i]
        for i in mne.pick_types(
            raw.info, eeg=True, eog=False, exclude=[]
        )
    ]

    participant_trials = trials[
        trials["participant_id"] == participant
    ]

    for _, trial in participant_trials.iterrows():
        start = round(trial["delay_onset"] * sfreq)
        stop = round(trial["probe_onset"] * sfreq)

        if start < 0 or stop > raw.n_times or stop <= start:
            print(
                f"Skipping invalid window: "
                f"{participant}, trial {trial['trial_number']}"
            )
            continue

        segment = raw.get_data(
            picks=eeg_channels,
            start=start,
            stop=stop
        ) * 1e6

        channel_ptp = np.ptp(segment, axis=1)
        channel_abs = np.max(np.abs(segment), axis=1)

        ptp_idx = int(np.argmax(channel_ptp))
        abs_idx = int(np.argmax(channel_abs))

        results.append({
            "participant_id": participant,
            "trial_number": trial["trial_number"],
            "condition": trial["condition"],
            "max_ptp_channel": eeg_channels[ptp_idx],
            "max_ptp_uV": channel_ptp[ptp_idx],
            "max_abs_channel": eeg_channels[abs_idx],
            "max_abs_uV": channel_abs[abs_idx],
            "median_channel_ptp_uV": np.median(channel_ptp),
            "channels_over_500uV_ptp": int(
                np.sum(channel_ptp > 500)
            ),
        })

df = pd.DataFrame(results)
df.to_csv(
    "reports/trial_channel_amplitude_audit.csv",
    index=False
)

print("\n" + "=" * 65)
print("TRIAL-WINDOW AMPLITUDE SUMMARY")
print("=" * 65)

print(
    df.groupby("participant_id")[
        ["max_ptp_uV", "max_abs_uV",
         "median_channel_ptp_uV", "channels_over_500uV_ptp"]
    ].agg(["median", "mean", "max"]).round(2).to_string()
)

print("\nChannels most often responsible for maximum peak-to-peak:")
print(
    df.groupby(["participant_id", "max_ptp_channel"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nLargest-amplitude trial windows:")
print(
    df.sort_values("max_ptp_uV", ascending=False)
    .head(15)
    .round(2)
    .to_string(index=False)
)

print("\nSaved: reports/trial_channel_amplitude_audit.csv")
