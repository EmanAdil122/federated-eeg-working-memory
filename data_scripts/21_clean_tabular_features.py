"""
Reusable cleaning and feature-scaling template for a future trial-level feature
table. This intentionally does NOT edit continuous EEG or remove EEG samples.

Input CSV must contain one row per trial and a stable participant/trial ID.
Run:
    python data_scripts/21_clean_tabular_features.py path/to/features.csv

Outputs:
    reports/features_cleaned.csv
    reports/features_cleaning_log.csv

Rules:
- Exact duplicate rows are reported and removed from the derived table.
- Missing numeric features are median-imputed; missing categorical values get
  an explicit "Unknown" category.
- Numeric outliers are flagged with the 1.5*IQR rule, not deleted or winsorized.
- StandardScaler is fitted to numeric features for a derived scaled table.
  For formal ML evaluation, fit scaling/imputation on training data only.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

if len(sys.argv) != 2:
    raise SystemExit("Usage: python data_scripts/21_clean_tabular_features.py path/to/features.csv")

source = Path(sys.argv[1])
df = pd.read_csv(source)
log_rows = []
log_rows.append({"step": "input_rows", "value": len(df)})

dups = int(df.duplicated().sum())
log_rows.append({"step": "exact_duplicate_rows", "value": dups})
df = df.drop_duplicates().copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# IDs and timing fields are not automatically treated as model features.
exclude_tokens = ("id", "trial", "onset", "time", "sample", "label", "target")
feature_numeric = [c for c in numeric_cols if not any(t in c.lower() for t in exclude_tokens)]

for col in feature_numeric:
    n_missing = int(df[col].isna().sum())
    log_rows.append({"step": f"missing_numeric:{col}", "value": n_missing})
    if n_missing:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        log_rows.append({"step": f"iqr_outliers_flagged:{col}", "value": int(mask.sum())})
        df[f"{col}_outlier_flag"] = mask

categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
for col in categorical_cols:
    n_missing = int(df[col].isna().sum())
    log_rows.append({"step": f"missing_categorical:{col}", "value": n_missing})
    if n_missing:
        df[col] = df[col].fillna("Unknown")

clean_path = REPORTS / "features_cleaned.csv"
df.to_csv(clean_path, index=False)

if feature_numeric:
    scaled = df.copy()
    scaler = StandardScaler()
    scaled[feature_numeric] = scaler.fit_transform(scaled[feature_numeric])
    scaled.to_csv(REPORTS / "features_scaled_demo.csv", index=False)
    log_rows.append({"step": "scaled_numeric_feature_count", "value": len(feature_numeric)})
    log_rows.append({"step": "scaling_warning", "value": "Demo fit on full file; for evaluation fit on train only."})

pd.DataFrame(log_rows).to_csv(REPORTS / "features_cleaning_log.csv", index=False)
print(f"Saved cleaned table: {clean_path}")
print(f"Saved cleaning log: {REPORTS / 'features_cleaning_log.csv'}")
print("Outliers were flagged, not removed. Do not use this template to alter raw EEG.")
