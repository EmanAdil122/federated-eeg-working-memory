

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


# -----------------------------
# 1. Set project paths
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

TRIAL_FILE = REPORTS_DIR / "trial_table.csv"
OUTPUT_FILE = REPORTS_DIR / "eeg_features.csv"

# Previously observed suspicious interval in sub-001.
# This is a flagging rule, not proof that the data is unusable.
SUSPICIOUS_INTERVALS = {
    "sub-001": [(332.5, 333.5)]
}

# Frequency bands in Hz
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


# -----------------------------
# 2. Calculate band power
# -----------------------------

def calculate_band_power(signal_uv, sampling_rate):
    """
    signal_uv contains one EEG channel in microvolts.
    Return average Welch PSD power in each frequency band.
    """
    frequencies, psd = welch(
        signal_uv,
        fs=sampling_rate,
        nperseg=min(len(signal_uv), int(sampling_rate * 2))
    )

    features = {}

    for band_name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)

        if mask.sum() < 2:
            features[band_name] = np.nan
        else:
            # Integrate the power spectral density over the band.
            features[band_name] = float(
                np.trapezoid(psd[mask], frequencies[mask])
            )

    return features


# -----------------------------
# 3. Load trial metadata
# -----------------------------

trials = pd.read_csv(TRIAL_FILE)

required_columns = [
    "participant_id",
    "trial_number",
    "condition",
    "delay_onset",
    "probe_onset",
    "trial_quality",
    "quality_issues",
    "response_accuracy",
]

missing_columns = [
    column for column in required_columns
    if column not in trials.columns
]

if missing_columns:
    raise ValueError(
        f"Trial table is missing columns: {missing_columns}"
    )

all_features = []


# -----------------------------
# 4. Process each participant
# -----------------------------

for participant_id, participant_trials in trials.groupby(
    "participant_id", sort=True
):
    eeg_file = (
        DATA_DIR
        / participant_id
        / "eeg"
        / f"{participant_id}_task-VerbalWorkingMemory_eeg.set"
    )

    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    print(f"\nLoading {participant_id}: {eeg_file.name}")

    raw = mne.io.read_raw_eeglab(
        str(eeg_file),
        preload=True,
        verbose="ERROR"
    )
    raw.set_channel_types({
        "EOGv": "eog",
        "EOGh": "eog"
    })
    sampling_rate = float(raw.info["sfreq"])

    # Select EEG channels only; do not treat EOG as EEG.
    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=False,
        exclude=[]
    )

    if len(eeg_picks) == 0:
        raise ValueError(f"No EEG channels found for {participant_id}")

    channel_names = [raw.ch_names[i] for i in eeg_picks]

    # MNE stores EEG in volts. Convert to microvolts.
# Do not load a second copy of the entire recording.
# Extract only the current trial's samples later.
    recording_duration = raw.n_times / sampling_rate

    print(
        f"Sampling rate: {sampling_rate:g} Hz | "
        f"EEG channels: {len(channel_names)} | "
        f"Duration: {recording_duration:.1f} seconds"
    )

    # -----------------------------
    # 5. Extract features per trial
    # -----------------------------

    for _, trial in participant_trials.iterrows():
        delay_start = float(trial["delay_onset"])
        probe_start = float(trial["probe_onset"])

        row = {
            "participant_id": participant_id,
            "trial_number": int(trial["trial_number"]),
            "condition": trial["condition"],
            "trial_quality": trial["trial_quality"],
            "quality_issues": trial["quality_issues"],
            "response_accuracy": trial["response_accuracy"],
            "delay_onset": delay_start,
            "probe_onset": probe_start,
            "window_duration_seconds": probe_start - delay_start,
        }

        if (
            not np.isfinite(delay_start)
            or not np.isfinite(probe_start)
            or delay_start < 0
            or probe_start <= delay_start
            or probe_start > recording_duration
        ):
            row["feature_status"] = "invalid_window"
            row["signal_quality_flag"] = "not_evaluated"
            all_features.append(row)
            continue

        start_sample = int(round(delay_start * sampling_rate))
        stop_sample = int(round(probe_start * sampling_rate))

        # Load only this trial's EEG samples.
        segment = (
            raw.get_data(
            picks=eeg_picks,
            start=start_sample,
            stop=stop_sample
            ) * 1e6
        )
        if segment.shape[1] < int(sampling_rate):
            row["feature_status"] = "window_too_short"
            row["signal_quality_flag"] = "not_evaluated"
            all_features.append(row)
            continue

        # Basic signal summaries for review.
        peak_to_peak = np.ptp(segment, axis=1)
        max_absolute = np.max(np.abs(segment), axis=1)

        row["max_channel_peak_to_peak_uV"] = float(
            np.max(peak_to_peak)
        )
        row["max_channel_absolute_amplitude_uV"] = float(
            np.max(max_absolute)
        )

        overlaps_suspicious_interval = any(
            delay_start < interval_end
            and probe_start > interval_start
            for interval_start, interval_end
            in SUSPICIOUS_INTERVALS.get(participant_id, [])
        )

        row["overlaps_known_suspicious_interval"] = (
            overlaps_suspicious_interval
        )

        # These are screening flags, not automatic rejection criteria.
        flags = []

        if overlaps_suspicious_interval:
            flags.append("known_suspicious_interval")

        if np.max(peak_to_peak) > 500:
            flags.append("large_peak_to_peak_amplitude")

        if np.max(max_absolute) > 500:
            flags.append("large_absolute_amplitude")

        row["signal_quality_flag"] = (
            ";".join(flags) if flags else "no_screening_flag"
        )

        # Calculate band power for every EEG channel.
        for channel_index, channel_name in enumerate(channel_names):
            channel_signal = segment[channel_index]

            channel_features = calculate_band_power(
                channel_signal,
                sampling_rate
            )

            for band_name, power in channel_features.items():
                column_name = f"{channel_name}_{band_name}_power"
                row[column_name] = power

        row["feature_status"] = "extracted"
        all_features.append(row)

    # Release the current recording before loading the next one.
    del raw


# -----------------------------
# 6. Save results
# -----------------------------

features_df = pd.DataFrame(all_features)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
features_df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction finished.")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(features_df)}")
print("\nFeature status:")
print(features_df["feature_status"].value_counts(dropna=False))

print("\nSignal-quality flags:")
print(features_df["signal_quality_flag"].value_counts(dropna=False))

print("\nTrials per participant:")
print(features_df.groupby("participant_id").size())
