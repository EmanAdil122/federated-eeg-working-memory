
import pandas as pd

df = pd.read_csv("reports/eeg_features.csv")

print("=" * 60)
print("1. SIGNAL-QUALITY FLAG COUNTS")
print("=" * 60)
print(df["signal_quality_flag"].value_counts().to_string())

print("\n" + "=" * 60)
print("2. AMPLITUDE SUMMARY BY FLAG")
print("=" * 60)

amplitude_cols = [
    "max_channel_peak_to_peak_uV",
    "max_channel_absolute_amplitude_uV",
]

for col in amplitude_cols:
    if col in df.columns:
        print(f"\n{col}")
        print(
            df.groupby("signal_quality_flag")[col]
            .agg(["count", "median", "mean", "max"])
            .round(2)
            .to_string()
        )

print("\n" + "=" * 60)
print("3. FLAGGED TRIALS BY PARTICIPANT")
print("=" * 60)

flagged = df[
    df["signal_quality_flag"] != "no_screening_flag"
]

print(
    flagged.groupby(["participant_id", "signal_quality_flag"])
    .size()
    .to_string()
)

print("\n" + "=" * 60)
print("4. TRIALS OVERLAPPING THE KNOWN SUSPICIOUS INTERVAL")
print("=" * 60)

if "overlaps_known_suspicious_interval" in df.columns:
    print(
        df.groupby(
            ["participant_id", "overlaps_known_suspicious_interval"]
        ).size().to_string()
    )
else:
    print("Suspicious-interval overlap column not found.")

print("\n" + "=" * 60)
print("5. HIGHEST-AMPLITUDE TRIALS")
print("=" * 60)

sort_col = "max_channel_peak_to_peak_uV"

if sort_col in df.columns:
    cols = [
        c for c in [
            "participant_id",
            "trial_number",
            "condition",
            "trial_quality",
            "signal_quality_flag",
            "max_channel_peak_to_peak_uV",
            "max_channel_absolute_amplitude_uV",
            "overlaps_known_suspicious_interval",
        ]
        if c in df.columns
    ]

    print(
        df.sort_values(sort_col, ascending=False)[cols]
        .head(15)
        .to_string(index=False)
    )
