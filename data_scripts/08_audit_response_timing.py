
import pandas as pd
from pathlib import Path
import re

ROOT = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

all_results = []

for participant in ["sub-001", "sub-002"]:
    event_file = next(
        (ROOT / participant / "eeg").glob("*_events.tsv"),
        None
    )

    if event_file is None:
        print(f"Missing events file for {participant}")
        continue

    df = pd.read_csv(event_file, sep="\t")
    df = df.sort_values("onset").reset_index(drop=True)

    df["trial_type"] = df["trial_type"].fillna("").astype(str)

    probes = df[
        df["trial_type"].str.contains(r"\bprobe:", case=False, regex=True)
    ]

    responses = df[
        df["trial_type"].str.contains(r"Response:", case=False, regex=False)
    ]

    print("\n" + "=" * 90)
    print("PARTICIPANT:", participant)
    print("=" * 90)

    for response_idx, response in responses.iterrows():
        # Find the latest probe before this response
        previous_probes = probes[probes["onset"] < response["onset"]]

        if previous_probes.empty:
            previous_probe = None
            previous_probe_time = None
            previous_condition = None
            seconds_after_probe = None
        else:
            previous_probe = previous_probes.iloc[-1]
            previous_probe_time = previous_probe["onset"]
            previous_condition = re.search(
                r"probe:\s*([56][RM])",
                previous_probe["trial_type"],
                re.IGNORECASE
            )

            previous_condition = (
                previous_condition.group(1).upper()
                if previous_condition else None
            )

            seconds_after_probe = (
                response["onset"] - previous_probe_time
            )

        response_condition = re.search(
            r"Response:\s*([56][RM]):\s*(correct|error)",
            response["trial_type"],
            re.IGNORECASE
        )

        response_condition = (
            response_condition.group(1).upper()
            if response_condition else None
        )

        response_label = re.search(
            r"(correct|error)",
            response["trial_type"],
            re.IGNORECASE
        )

        response_label = (
            response_label.group(1).lower()
            if response_label else None
        )

        # Find the next probe after the response
        next_probes = probes[probes["onset"] > response["onset"]]

        next_probe_time = (
            next_probes.iloc[0]["onset"]
            if not next_probes.empty else None
        )

        seconds_until_next_probe = (
            next_probe_time - response["onset"]
            if next_probe_time is not None else None
        )

        record = {
            "participant_id": participant,
            "response_onset": response["onset"],
            "response_event": response["trial_type"],
            "response_condition": response_condition,
            "response_label": response_label,
            "previous_probe_onset": previous_probe_time,
            "previous_probe_condition": previous_condition,
            "seconds_after_previous_probe": seconds_after_probe,
            "next_probe_onset": next_probe_time,
            "seconds_until_next_probe": seconds_until_next_probe,
            "condition_matches_previous_probe": (
                response_condition == previous_condition
                if response_condition and previous_condition
                else None
            ),
        }

        all_results.append(record)

        print(
            f"Response {response['onset']:.3f}s | "
            f"{response['trial_type']} | "
            f"previous probe: {previous_probe_time} "
            f"({previous_condition}) | "
            f"delay: {seconds_after_probe} | "
            f"condition match: "
            f"{record['condition_matches_previous_probe']}"
        )

audit = pd.DataFrame(all_results)
output_file = REPORTS / "response_timing_audit.csv"
audit.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Total response markers audited:", len(audit))
print("\nCondition-match counts:")
print(audit["condition_matches_previous_probe"].value_counts(dropna=False))
