
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

DATA_DIR = Path("data")
REPORT_FILE = Path("reports/amplitude_limit_check.csv")

# Find channels with the suspected amplitude limit.
summary = pd.read_csv(REPORT_FILE)
suspect = summary[
    (summary["participant_id"] == "sub-001")
    & (summary["maximum_absolute_uV"] >= 10239)
]

print("Channels with suspicious maximum amplitudes:")
print(suspect[["channel", "maximum_absolute_uV"]].to_string(index=False))

eeg_file = next(
    DATA_DIR.glob("sub-001/eeg/*_eeg.set")
)

raw = mne.io.read_raw_eeglab(
    eeg_file,
    preload=False,
    verbose="ERROR"
)

raw.set_channel_types({"EOGv": "eog", "EOGh": "eog"})

# Inspect the first 10 seconds, then look for the largest
# absolute sample in Fp1 and inspect around that time.
channel = "Fp1"
signal_uv = raw.get_data(
    picks=[channel]
)[0] * 1e6

peak_index = int(np.argmax(np.abs(signal_uv)))
peak_time = peak_index / raw.info["sfreq"]

print(f"\nChannel: {channel}")
print(f"Largest absolute sample: {signal_uv[peak_index]:.3f} µV")
print(f"Time of largest sample: {peak_time:.3f} seconds")

# Plot 2 seconds before and after the peak.
start = max(0, peak_index - int(2 * raw.info["sfreq"]))
stop = min(raw.n_times, peak_index + int(2 * raw.info["sfreq"]))

times = np.arange(start, stop) / raw.info["sfreq"]
data = raw.get_data(start=start, stop=stop) * 1e6

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Fp1 around its largest absolute sample.
fp1_index = raw.ch_names.index("Fp1")
axes[0].plot(times, data[fp1_index])
axes[0].set_title("Fp1 around its largest absolute sample")
axes[0].set_ylabel("Amplitude (µV)")
axes[0].grid(True)

# A few other channels at the same time.
for name in ["Fp2", "F3", "Cz", "O1", "EOGv"]:
    idx = raw.ch_names.index(name)
    axes[1].plot(times, data[idx], label=name)

axes[1].set_title("Other channels during the same time window")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude (µV)")
axes[1].legend(loc="upper right", ncol=3)
axes[1].grid(True)

plt.tight_layout()

output = Path("reports/sub-001_suspicious_waveform.png")
plt.savefig(output, dpi=150)
print(f"\nPlot saved to: {output}")

plt.show()
raw.close()
