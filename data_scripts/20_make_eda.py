"""
Create EDA charts from the existing Phase 2 trial/response audit reports.
Run from repository root:
    python data_scripts/20_make_eda.py

Expected input:
    reports/response_probe_audit.csv
Optional:
    reports/trial_table.csv

The charts are audit-level EDA (trial/condition/response mapping), not EEG
amplitude-feature EDA. They do not resolve ambiguous response attribution.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

audit_path = REPORTS / "response_probe_audit.csv"
if not audit_path.exists():
    raise FileNotFoundError(
        f"Missing {audit_path}. Copy the project's generated "
        "response_probe_audit.csv into reports/ first."
    )

df = pd.read_csv(audit_path)
required = {"participant_id", "condition"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

# 1. Trial counts by condition and participant
counts = df.groupby(["condition", "participant_id"]).size().unstack(fill_value=0)
ax = counts.reindex(["5R", "6R", "7R", "5M", "6M", "7M"]).plot(
    kind="bar", figsize=(9, 5), rot=0
)
ax.set_title("Audited trial/probe counts by condition")
ax.set_xlabel("Condition")
ax.set_ylabel("Number of trials")
ax.legend(title="Participant")
plt.tight_layout()
plt.savefig(FIGURES / "trial_counts_by_condition.png", dpi=200)
plt.close()

# 2. Review vs pass, if the status column exists
status_col = next((c for c in ["status", "audit_status", "decision"] if c in df.columns), None)
if status_col:
    status = df.groupby(["participant_id", status_col]).size().unstack(fill_value=0)
    ax = status.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("Response/probe audit status")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Trial count")
    plt.tight_layout()
    plt.savefig(FIGURES / "response_audit_status.png", dpi=200)
    plt.close()

# 3. Response delay distribution, if delays are available.
delay_col = next((c for c in ["response_delay_s", "response_delay", "response_delays_s"] if c in df.columns), None)
if delay_col:
    values = []
    for value in df[delay_col].dropna():
        # Some audit exports contain multiple delays separated by semicolons.
        for item in str(value).split(";"):
            try:
                values.append(float(item.strip()))
            except ValueError:
                pass
    if values:
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=25)
        plt.title("Observed response-marker delays after probes")
        plt.xlabel("Delay (seconds)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIGURES / "response_delay_distribution.png", dpi=200)
        plt.close()

print(f"EDA figures saved to: {FIGURES}")
print("Interpret charts as event-audit summaries, not proof of behavioral accuracy.")
