
import mne
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

WINDOW_SECONDS = 2
EXPECTED_EEG_CHANNELS = 19
EXPECTED_EOG_CHANNELS = 2

all_results = []

set_files = sorted(DATA.glob("sub-*/eeg/*_eeg.set"))

if not set_files:
    raise FileNotFoundError("No EEG .set files found under data/sub-*/eeg/")

for set_file in set_files:
    participant = set_file.parents[1].name

    print(f"\nChecking {participant}: {set_file.name}")

    raw = mne.io.read_raw_eeglab(
        set_file,
        preload=False,
        verbose="ERROR"
    )

    # ADD THESE LINES
    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })

    eeg_names = [
        name for name in raw.ch_names
        if raw.get_channel_types(picks=[name])[0] == "eeg"
    ]

    eog_names = [
        name for name in raw.ch_names
        if raw.get_channel_types(picks=[name])[0] == "eog"
    ]

    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1] / 60:.2f} minutes")
    print(f"EEG channels: {len(eeg_names)}")
    print(f"EOG channels: {len(eog_names)}")

    if len(eeg_names) != EXPECTED_EEG_CHANNELS:
        print("WARNING: Unexpected EEG channel count.")

    if len(eog_names) != EXPECTED_EOG_CHANNELS:
        print("WARNING: Unexpected EOG channel count.")

    picks = eeg_names + eog_names
    window_samples = int(WINDOW_SECONDS * raw.info["sfreq"])

    channel_data = {
        name: {
            "sample_count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "max_abs": 0.0,
            "invalid_count": 0,
            "flat_windows": 0,
            "large_range_windows": 0,
            "window_count": 0,
            "max_window_range": 0.0,
        }
        for name in picks
    }

    # Read short windows to avoid loading the whole recording into memory.
    for start in range(0, raw.n_times, window_samples):
        stop = min(start + window_samples, raw.n_times)
        data = raw.get_data(picks=picks, start=start, stop=stop)

        # MNE EEG/EOG data are normally stored in volts.
        data_uv = data * 1_000_000

        for i, name in enumerate(picks):
            values = data_uv[i]
            stats = channel_data[name]

            finite = values[np.isfinite(values)]
            stats["invalid_count"] += int(
                len(values) - len(finite)
            )

            if len(finite) == 0:
                continue

            stats["sample_count"] += len(finite)
            stats["sum"] += float(np.sum(finite))
            stats["sum_squares"] += float(np.sum(finite ** 2))
            stats["min"] = min(stats["min"], float(np.min(finite)))
            stats["max"] = max(stats["max"], float(np.max(finite)))
            stats["max_abs"] = max(
                stats["max_abs"], float(np.max(np.abs(finite)))
            )

            stats["window_count"] += 1
            window_range = float(np.max(finite) - np.min(finite))
            stats["max_window_range"] = max(
                stats["max_window_range"], window_range
            )

            # Preliminary screening thresholds, not final rejection rules.
            if window_range < 0.1:
                stats["flat_windows"] += 1

            if window_range > 150:
                stats["large_range_windows"] += 1

    for name, stats in channel_data.items():
        n = stats["sample_count"]

        if n == 0:
            mean_uv = np.nan
            std_uv = np.nan
        else:
            mean_uv = stats["sum"] / n
            variance = stats["sum_squares"] / n - mean_uv ** 2
            std_uv = np.sqrt(max(variance, 0.0))

        total_windows = stats["window_count"]

        all_results.append({
            "participant_id": participant,
            "channel": name,
            "channel_type": (
                "EEG" if name in eeg_names else "EOG"
            ),
            "samples_checked": n,
            "mean_uV": mean_uv,
            "std_uV": std_uv,
            "minimum_uV": (
                stats["min"] if n else np.nan
            ),
            "maximum_uV": (
                stats["max"] if n else np.nan
            ),
            "maximum_absolute_uV": stats["max_abs"],
            "invalid_samples": stats["invalid_count"],
            "flat_windows": stats["flat_windows"],
            "large_range_windows": stats["large_range_windows"],
            "windows_checked": total_windows,
            "maximum_2s_range_uV": stats["max_window_range"],
        })

    raw.close()

results = pd.DataFrame(all_results)
output = REPORTS / "signal_quality_summary.csv"
results.to_csv(output, index=False)

print("\n" + "=" * 60)
print("SIGNAL QUALITY SCREENING SUMMARY")
print("=" * 60)
print(f"Participants: {results['participant_id'].nunique()}")
print(f"Channel rows: {len(results)}")
print(f"Channels with invalid samples: "
      f"{(results['invalid_samples'] > 0).sum()}")
print(f"Channels with flat windows: "
      f"{(results['flat_windows'] > 0).sum()}")
print(f"Channels with large-range windows: "
      f"{(results['large_range_windows'] > 0).sum()}")
print(f"Saved: {output}")
