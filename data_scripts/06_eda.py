import pandas as pd
from pathlib import Path

for file in sorted(Path("data").glob("sub-*/eeg/*_events.tsv")):
    df = pd.read_csv(file, sep="\t")

    responses = df[
        df["trial_type"].astype(str).str.contains(
            r"\bResponse\s*:", case=False, regex=True
        )
    ]

    print(f"\n{file.parts[1]}")
    print(
        responses.groupby(["trial_type", "value"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .to_string(index=False)
    )