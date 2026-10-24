
import pandas as pd
from pathlib import Path
import re

DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

LONG_RESPONSE_SECONDS = 5.0


def get_condition(text):
    match = re.search(r"([567][RM])", str(text), re.IGNORECASE)
    return match.group(1).upper() if match else None


def get_response_label(text):
    match = re.search(r"\b(correct|error)\b", str(text), re.IGNORECASE)
    return match.group(1).lower() if match else None


event_files = sorted(DATA.glob("sub-*/eeg/*_events.tsv"))

if not event_files:
    raise FileNotFoundError("No event files found.")

all_trials = []
orphan_responses = []

for event_file in event_files:
    participant = event_file.parents[1].name
    df = pd.read_csv(event_file, sep="\t")

    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["trial_type"] = df["trial_type"].fillna("").astype(str)
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    probes = df[
        df["trial_type"].str.contains(
            r"\bprobe\s*:", case=False, regex=True
        )
    ].copy().reset_index(drop=True)

    responses = df[
        df["trial_type"].str.contains(
            r"\bResponse\s*:", case=False, regex=True
        )
    ].copy().reset_index(drop=True)

    if probes.empty:
        print(f"Skipping {participant}: no probes found.")
        continue

    # A response belongs provisionally to the interval after
    # one probe and before the next probe.
    for i, probe in probes.iterrows():
        probe_onset = float(probe["onset"])
        next_probe_onset = (
            float(probes.loc[i + 1, "onset"])
            if i + 1 < len(probes)
            else float("inf")
        )

        trial_responses = responses[
            (responses["onset"] >= probe_onset)
            & (responses["onset"] < next_probe_onset)
        ]

        condition = get_condition(probe["trial_type"])
        response_times = trial_responses["onset"].astype(float).tolist()
        response_conditions = [
            get_condition(label)
            for label in trial_responses["trial_type"]
        ]
        response_labels = [
            get_response_label(label)
            for label in trial_responses["trial_type"]
        ]
        delays = [round(t - probe_onset, 3) for t in response_times]

        issues = []

        if not trial_responses.empty and condition is None:
            issues.append("probe_condition_unknown")

        if len(trial_responses) == 0:
            issues.append("missing_response")

        if len(trial_responses) > 1:
            issues.append("multiple_responses")

        if any(delay > LONG_RESPONSE_SECONDS for delay in delays):
            issues.append("response_over_5_seconds_after_probe")

        if any(delay < 0 for delay in delays):
            issues.append("response_before_probe")

        if any(
            rc is not None and condition is not None and rc != condition
            for rc in response_conditions
        ):
            issues.append("response_condition_mismatch")

        all_trials.append({
            "participant_id": participant,
            "trial_number": i + 1,
            "condition": condition,
            "probe_onset": probe_onset,
            "next_probe_onset": (
                next_probe_onset
                if next_probe_onset != float("inf")
                else None
            ),
            "response_count": len(trial_responses),
            "response_labels": "; ".join(
                label or "unknown" for label in response_labels
            ),
            "response_conditions": "; ".join(
                rc or "unknown" for rc in response_conditions
            ),
            "response_onsets_s": "; ".join(
                f"{t:.3f}" for t in response_times
            ),
            "response_delays_s": "; ".join(
                f"{d:.3f}" for d in delays
            ),
            "status": "REVIEW" if issues else "PASS",
            "issues": "; ".join(issues)
        })

    # Responses before the first probe cannot be assigned by this rule.
    first_probe = float(probes.iloc[0]["onset"])
    earlier = responses[responses["onset"] < first_probe]

    for _, response in earlier.iterrows():
        orphan_responses.append({
            "participant_id": participant,
            "response_onset": response["onset"],
            "response_label": response["trial_type"],
            "reason": "response_before_first_probe"
        })

audit = pd.DataFrame(all_trials)
audit.to_csv(REPORTS / "response_probe_audit.csv", index=False)

issues = audit[audit["status"] == "REVIEW"]
issues.to_csv(REPORTS / "response_probe_issues.csv", index=False)

orphans = pd.DataFrame(orphan_responses)
orphans.to_csv(REPORTS / "orphan_responses.csv", index=False)

print("\nRESPONSE-TO-PROBE AUDIT")
print("=" * 45)
print(f"Participants: {audit['participant_id'].nunique()}")
print(f"Probe/trial rows: {len(audit)}")
print(f"PASS: {(audit['status'] == 'PASS').sum()}")
print(f"REVIEW: {(audit['status'] == 'REVIEW').sum()}")
print(f"Missing responses: {(audit['response_count'] == 0).sum()}")
print(f"Multiple responses: {(audit['response_count'] > 1).sum()}")
print(f"Responses before first probe: {len(orphans)}")

print("\nIssues:")
counts = (
    issues.loc[issues["issues"] != "", "issues"]
    .str.split("; ")
    .explode()
    .value_counts()
)
print(counts.to_string() if not counts.empty else "None")

print("\nPer participant:")
print(
    audit.groupby(["participant_id", "status"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nSaved:")
print("reports/response_probe_audit.csv")
print("reports/response_probe_issues.csv")
print("reports/orphan_responses.csv")
