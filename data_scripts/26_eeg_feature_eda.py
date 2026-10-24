"""EEG feature EDA: band-power distributions, participant/condition comparisons,
amplitude screening plot and feature correlations.

Reads  reports/eeg_features.csv   (no raw EEG needed)
Writes reports/figures/*.png

Run from the repository root:  python data_scripts/26_eeg_feature_eda.py
Descriptive only: with two participants, no inferential claims are made.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "reports" / "eeg_features.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES)
bands = ["theta", "alpha", "beta"]
feat_cols = [c for c in df.columns
             if c.endswith("_power") and any(b in c.lower() for b in bands)
             and pd.api.types.is_numeric_dtype(df[c])]
if not feat_cols:
    raise SystemExit("No *_theta/alpha/beta_power columns found: " + ", ".join(df.columns))

def band_of(col):
    return next(b for b in bands if b in col.lower())

print(f"Rows: {len(df)} | band-power features: {len(feat_cols)} (expected 57)")
print("Missing values in features:", int(df[feat_cols].isna().sum().sum()))
print("Duplicate (participant_id, trial_number):", int(df.duplicated(["participant_id", "trial_number"]).sum()))

part_col, cond_col = "participant_id", "condition"

# Log10 for display only (band power is right-skewed). The CSV is not modified.
log_df = df.copy()
log_df[feat_cols] = np.log10(df[feat_cols].clip(lower=np.finfo(float).tiny))

# 1. Distribution per band
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, b in zip(axes, bands):
    vals = log_df[[c for c in feat_cols if band_of(c) == b]].to_numpy().ravel()
    ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
    ax.set_title(f"{b.capitalize()} band")
    ax.set_xlabel("log10 band power")
axes[0].set_ylabel("Count (trial x channel)")
fig.tight_layout(); fig.savefig(FIG_DIR / "band_power_distributions.png", dpi=200); plt.close(fig)

# 2. Mean band power per group
def mean_by(group_col, fname, title):
    groups = sorted(log_df[group_col].dropna().unique(), key=str)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        sub = log_df[log_df[group_col] == g]
        means = [sub[[c for c in feat_cols if band_of(c) == b]].to_numpy().mean() for b in bands]
        ax.bar(np.arange(3) + i * width, means, width, label=str(g))
    ax.set_xticks(np.arange(3) + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_ylabel("Mean log10 band power (all channels)")
    ax.set_title(title); ax.legend(title=group_col, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=200); plt.close(fig)

if part_col in df.columns:
    mean_by(part_col, "band_power_by_participant.png", "Mean band power by participant (descriptive, n=2)")
if cond_col in df.columns:
    mean_by(cond_col, "band_power_by_condition.png", "Mean band power by condition")

# 3. Window peak-to-peak amplitude by participant (relates to the amplitude screening flags)
amp = "max_channel_peak_to_peak_uV"
if amp in df.columns and part_col in df.columns:
    groups = sorted(df[part_col].unique(), key=str)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([df.loc[df[part_col] == g, amp] for g in groups], tick_labels=groups)
    ax.axhline(500, color="red", linestyle="--", label="500 uV screening heuristic")
    ax.set_yscale("log"); ax.set_ylabel("Max channel peak-to-peak per trial window (uV, log)")
    ax.set_title("Window amplitude by participant"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "amplitude_by_participant.png", dpi=200); plt.close(fig)

# 4. Correlation heatmap
corr = log_df[feat_cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Correlation of log10 band-power features (57 features)")
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=200); plt.close(fig)

print("Saved figures to", FIG_DIR)
