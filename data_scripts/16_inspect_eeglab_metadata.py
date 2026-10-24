
from pathlib import Path
import numpy as np
import mne

eeg_file = next(Path("data/sub-001/eeg").glob("*.set"))

raw = mne.io.read_raw_eeglab(
    eeg_file,
    preload=False,
    verbose="ERROR"
)

# Examine the section where the plot showed the flat signal.
start = int(332.5 * raw.info["sfreq"])
stop = int(333.5 * raw.info["sfreq"])

data_uv = raw.get_data(start=start, stop=stop) * 1e6

print("Inspecting samples from 332.5 to 333.5 seconds")
print("Expected negative boundary: approximately -10239.6875 uV")

for i, channel in enumerate(raw.ch_names):
    signal = data_uv[i]
    boundary = signal <= -10239.0

    print(f"\n{channel}")
    print("  Samples at negative boundary:", int(boundary.sum()))
    print("  Longest consecutive boundary run:", end=" ")

    # Find the longest consecutive run of boundary samples.
    padded = np.r_[False, boundary, False].astype(int)
    changes = np.diff(padded)
    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0]

    longest = (
        int(np.max(run_ends - run_starts))
        if len(run_starts) else 0
    )

    print(longest, "samples")
    print("  Longest run duration:", round(longest / raw.info["sfreq"], 3), "seconds")

raw.close()
