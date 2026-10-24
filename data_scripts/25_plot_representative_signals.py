
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative trial windows to inspect.
# These are not being designated as bad or good in advance.
windows = {
    "sub-001": (0, 10),
    "sub-002": (0, 10),
}

channels_to_plot = ["Fp1", "Fz", "Cz", "T3", "Pz", "O1"]

for participant, (start_sec, duration_sec) in windows.items():
    eeg_file = (
        DATA_DIR
        / participant
        / "eeg"
        / f"{participant}_task-VerbalWorkingMemory_eeg.set"
    )

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=False,
        verbose="ERROR"
    )

    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })

    start_sample = round(start_sec * raw.info["sfreq"])
    stop_sample = round(
        (start_sec + duration_sec) * raw.info["sfreq"]
    )

    picks = [
        ch for ch in channels_to_plot
        if ch in raw.ch_names
    ]

    data = raw.get_data(
        picks=picks,
        start=start_sample,
        stop=stop_sample
    ) * 1e6

    times = np.arange(data.shape[1]) / raw.info["sfreq"] + start_sec

    fig, axes = plt.subplots(
        len(picks), 1,
        figsize=(12, 10),
        sharex=True
    )

    for ax, channel, signal in zip(axes, picks, data):
        ax.plot(times, signal, linewidth=0.6)
        ax.set_ylabel(f"{channel}\nµV")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(
        f"{participant}: representative raw EEG signals "
        f"({start_sec}–{start_sec + duration_sec} s)"
    )
    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{participant}_representative_signals.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")

print("\nFinished plotting representative signals.")
