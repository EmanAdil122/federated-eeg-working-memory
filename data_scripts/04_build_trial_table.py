
from pathlib import Path
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

all_trials = []
all_issues = []

def get_condition(text):
    """Extract a condition such as 5R or 6M from an event label."""
    match = re.search(r"\b([567][RM])\b", str(text))
    return match.group(1) if match else None


for participant_dir in sorted(DATA_DIR.glob("sub-*")):
    participant = participant_dir.name
    eeg_dir = participant_dir / "eeg"

    if not eeg_dir.exists():
        eeg_dir = participant_dir

    event_files = list(eeg_dir.glob("*_events.tsv"))

    if len(event_files) != 1:
        print(f"Skipping {participant}: expected one events file")
        continue

    events = pd.read_csv(event_files[0], sep="\t")
    events["trial_type"] = events["trial_type"].fillna("").astype(str)
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events.dropna(subset=["onset"])
    events = events.sort_values("onset").reset_index(drop=True)

    # Locate the beginning of every trial
    start_mask = events["trial_type"].str.strip().str.lower().eq(
        "start of the trial"
    )
    start_indices = events.index[start_mask].tolist()

    participant_trials = []

    for position, start_idx in enumerate(start_indices):
        start_onset = events.loc[start_idx, "onset"]

        # Events up to, but not including, the next trial start
        if position + 1 < len(start_indices):
            next_start_idx = start_indices[position + 1]
            trial_events = events.loc[
                (events.index >= start_idx) &
                (events.index < next_start_idx)
            ].copy()
        else:
            trial_events = events.loc[events.index >= start_idx].copy()

        labels = trial_events["trial_type"].tolist()

        def find_event(prefix):
            for _, event in trial_events.iterrows():
                if event["trial_type"].strip().lower().startswith(prefix):
                    return event
            return None

        baseline = find_event("start of the baseline period")
        encoding = find_event("encoding:")
        delay = find_event("delay period:")
        probe = find_event("probe:")

        response_events = trial_events[
            trial_events["trial_type"].str.strip().str.lower().str.startswith(
                "response:"
            )
        ]

        condition_candidates = {
            "baseline": get_condition(
                baseline["trial_type"] if baseline is not None else ""
            ),
            "encoding": get_condition(
                encoding["trial_type"] if encoding is not None else ""
            ),
            "delay": get_condition(
                delay["trial_type"] if delay is not None else ""
            ),
            "probe": get_condition(
                probe["trial_type"] if probe is not None else ""
            ),
        }

        available_conditions = [
            value for value in condition_candidates.values() if value
        ]

        condition = (
            available_conditions[0] if available_conditions else None
        )

        response_labels = response_events["trial_type"].tolist()
        response_conditions = [
            get_condition(label) for label in response_labels
        ]

        response_accuracy = []
        for label in response_labels:
            lower_label = label.lower()
            if "correct" in lower_label:
                response_accuracy.append("correct")
            elif "error" in lower_label:
                response_accuracy.append("error")
            else:
                response_accuracy.append("unknown")

        issues = []

        for stage in ["baseline", "encoding", "delay", "probe"]:
            if condition_candidates[stage] is None:
                issues.append(f"missing_{stage}")

        distinct_conditions = set(available_conditions)
        if len(distinct_conditions) > 1:
            issues.append("condition_mismatch_between_stages")

        if len(response_events) == 0:
            issues.append("missing_response")
        elif len(response_events) > 1:
            issues.append("multiple_responses")

        if condition and any(
            response_condition and response_condition != condition
            for response_condition in response_conditions
        ):
            issues.append("response_condition_mismatch")

        row = {
            "participant_id": participant,
            "trial_number": position + 1,
            "trial_start_onset": start_onset,
            "condition": condition,
            "baseline_onset": (
                baseline["onset"] if baseline is not None else None
            ),
            "encoding_onset": (
                encoding["onset"] if encoding is not None else None
            ),
            "delay_onset": (
                delay["onset"] if delay is not None else None
            ),
            "probe_onset": (
                probe["onset"] if probe is not None else None
            ),
            "response_event_count": len(response_events),
            "response_labels": " | ".join(response_labels),
            "response_accuracy": " | ".join(response_accuracy),
            "trial_quality": "check" if issues else "passed",
            "quality_issues": "; ".join(issues),
        }

        participant_trials.append(row)
        all_trials.append(row)

        for issue in issues:
            all_issues.append({
                "participant_id": participant,
                "trial_number": position + 1,
                "condition": condition,
                "issue": issue,
                "event_labels": " | ".join(labels),
            })

    participant_df = pd.DataFrame(participant_trials)
    print(f"\n{participant}: {len(participant_df)} trial rows")
    if not participant_df.empty:
        print("\nTrials per condition:")
        print(participant_df["condition"].value_counts(dropna=False))
        print("\nQuality status:")
        print(participant_df["trial_quality"].value_counts())
        print(
            "Total response markers:",
            participant_df["response_event_count"].sum()
        )

# Save trial tables and issue log
trials_df = pd.DataFrame(all_trials)
issues_df = pd.DataFrame(all_issues)

trials_df.to_csv(REPORT_DIR / "trial_table.csv", index=False)
issues_df.to_csv(REPORT_DIR / "trial_quality_issues.csv", index=False)

print("\nSaved:", REPORT_DIR / "trial_table.csv")
print("Saved:", REPORT_DIR / "trial_quality_issues.csv")
print("\nTotal trial rows:", len(trials_df))
print("Trials requiring review:", int((trials_df["trial_quality"] == "check").sum()))
