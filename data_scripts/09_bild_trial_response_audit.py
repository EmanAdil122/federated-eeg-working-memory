
import pandas as pd
from pathlib import Path
import re

ROOT = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# Delays longer than this are flagged for review, not automatically rejected.
LONG_RESPONSE_SECONDS = 5.0

# Find all participant folders automatically.
event_files = sorted(ROOT.glob("sub-*/eeg/*_events.tsv"))

if not event_files:
    raise FileNotFoundError("No *_events.tsv files found under data/sub-*/eeg/")

all_trials = []


def get_condition(text):
    """Extract a condition such as 5R, 6M, or 7R from an event label."""
    match = re.search(r"([567][RM])", str(text), re.IGNORECASE)
    return match.group(1).upper() if match else None


def get_response_label(text):
    """Extract correct/error from a response event."""
    match = re.search(r"\b(correct|error)\b", str(text), re.IGNORECASE)
    return match.group(1).lower() if match else None


for event_file in event_files:
    participant = event_file.parents[1].name

    df = pd.read_csv(event_file, sep="\t")
    required_columns = {"onset", "trial_type"}

    if not required_columns.issubset(df.columns):
        print(f"Skipping {participant}: missing required event columns.")
        continue

    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["trial_type"] = df["trial_type"].fillna("").astype(str)
    df = (
        df.dropna(subset=["onset"])
        .sort_values("onset")
        .reset_index(drop=True)
    )

    # A trial begins at each "start of the trial" event.
    start_mask = df["trial_type"].str.contains(
        r"start of the trial", case=False, regex=True
    )
    start_indices = df.index[start_mask].tolist()

    if not start_indices:
        print(f"Skipping {participant}: no trial-start events found.")
        continue

    # Each trial includes events from its start up to (but not including)
    # the next trial start.
    for position, start_idx in enumerate(start_indices):
        end_idx = (
            start_indices[position + 1]
            if position + 1 < len(start_indices)
            else len(df)
        )

        trial_events = df.iloc[start_idx:end_idx].copy()
        trial_number = position + 1
        trial_start = float(df.loc[start_idx, "onset"])

        types = trial_events["trial_type"]

        probes = trial_events[
            types.str.contains(r"\bprobe\s*:", case=False, regex=True)
        ]

        responses = trial_events[
            types.str.contains(r"\bResponse\s*:", case=False, regex=True)
        ]

        instructions = trial_events[
            types.str.contains(
                r"presentation of the task instruction",
                case=False,
                regex=True,
            )
        ]

        issues = []

        # The probe is the preferred source of the trial condition.
        condition_sources = []

        if not probes.empty:
            condition_sources.append(probes.iloc[0]["trial_type"])

        if not instructions.empty:
            condition_sources.append(instructions.iloc[0]["trial_type"])

        if not responses.empty:
            condition_sources.append(responses.iloc[0]["trial_type"])

        trial_condition = next(
            (
                condition
                for source in condition_sources
                if (condition := get_condition(source)) is not None
            ),
            None,
        )

        if trial_condition is None:
            issues.append("condition_unknown")

        if len(probes) == 0:
            issues.append("missing_probe")
        elif len(probes) > 1:
            issues.append("multiple_probes")

        probe_onset = (
            float(probes.iloc[0]["onset"])
            if not probes.empty
            else None
        )

        response_onsets = responses["onset"].astype(float).tolist()
        response_conditions = [
            get_condition(value) for value in responses["trial_type"]
        ]
        response_labels = [
            get_response_label(value) for value in responses["trial_type"]
        ]

        response_delays = (
            [round(onset - probe_onset, 3) for onset in response_onsets]
            if probe_onset is not None
            else []
        )

        if len(responses) == 0:
            issues.append("missing_response")

        if len(responses) > 1:
            issues.append("multiple_responses")

        if probe_onset is not None:
            if any(delay < 0 for delay in response_delays):
                issues.append("response_before_probe")

            if any(delay > LONG_RESPONSE_SECONDS for delay in response_delays):
                issues.append("response_over_5_seconds_after_probe")

        if trial_condition is not None:
            if any(
                condition is not None and condition != trial_condition
                for condition in response_conditions
            ):
                issues.append("response_condition_mismatch")

        # Keep the raw event labels and timings for traceability.
        response_details = [
            f"{onset:.3f}s: {label or 'unknown'}"
            for onset, label in zip(response_onsets, response_labels)
        ]

        all_trials.append(
            {
                "participant_id": participant,
                "trial_number": trial_number,
                "trial_start_onset": round(trial_start, 3),
                "condition": trial_condition,
                "probe_onset": (
                    round(probe_onset, 3)
                    if probe_onset is not None
                    else None
                ),
                "probe_count": len(probes),
                "response_count": len(responses),
                "response_onsets_s": "; ".join(
                    f"{onset:.3f}" for onset in response_onsets
                ),
                "response_labels": "; ".join(
                    label or "unknown" for label in response_labels
                ),
                "response_conditions": "; ".join(
                    condition or "unknown"
                    for condition in response_conditions
                ),
                "response_delays_after_probe_s": "; ".join(
                    f"{delay:.3f}" for delay in response_delays
                ),
                "response_details": "; ".join(response_details),
                "status": "REVIEW" if issues else "PASS",
                "issues": "; ".join(issues),
            }
        )


# Save the complete report.
trial_audit = pd.DataFrame(all_trials)

if trial_audit.empty:
    raise ValueError("No trials were processed. Check the event labels.")

full_report = REPORTS / "trial_response_audit.csv"
issues_report = REPORTS / "trial_response_issues.csv"

trial_audit.to_csv(full_report, index=False)

trial_issues = trial_audit[trial_audit["status"] == "REVIEW"].copy()
trial_issues.to_csv(issues_report, index=False)

# Print a short summary rather than every response marker.
print("\n" + "=" * 60)
print("TRIAL-LEVEL RESPONSE AUDIT")
print("=" * 60)
print(f"Participants processed: {trial_audit['participant_id'].nunique()}")
print(f"Total trials: {len(trial_audit)}")
print(f"PASS: {(trial_audit['status'] == 'PASS').sum()}")
print(f"REVIEW: {(trial_audit['status'] == 'REVIEW').sum()}")
print(f"Trials with no response: {(trial_audit['response_count'] == 0).sum()}")
print(f"Trials with multiple responses: {(trial_audit['response_count'] > 1).sum()}")

print("\nIssue counts (one trial may have multiple issues):")
issue_counts = (
    trial_audit.loc[trial_audit["issues"] != "", "issues"]
    .str.split("; ")
    .explode()
    .value_counts()
)

if issue_counts.empty:
    print("No issues flagged.")
else:
    print(issue_counts.to_string())

print("\nPer-participant summary:")
print(
    trial_audit.groupby(["participant_id", "status"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print(f"\nFull report saved: {full_report}")
print(f"Issues report saved: {issues_report}")
