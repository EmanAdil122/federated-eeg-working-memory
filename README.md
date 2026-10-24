# Federated EEG Working Memory

## Project overview
This repository investigates federated learning for EEG-based working-memory decoding, with attention to differences between participants and non-IID data. The current work is **Phase 1 (problem definition) and Phase 2 (data acquisition, validation, preprocessing design and exploratory analysis)**. Model training is a later phase.

The dataset currently contains only **two participants**, so this is a pilot. No claims about generalization to the wider population are made.

## Research questions
1. How does a federated model compare with a centralized baseline for working-memory condition decoding?
2. How sensitive are results to participant-level distribution differences (non-IID data)?
3. Does a clearly defined personalization strategy improve cross-subject generalization?

## Team
| Member | Role in this deliverable |
|---|---|
| Eman Adil | <!-- TODO --> |
| Dua Sarfraz | <!-- TODO --> |
| Ayza Ahmad | <!-- TODO --> |

Course: Artificial Intelligence

## Dataset
- Dataset: OpenNeuro `ds003655`, task `VerbalWorkingMemory` (https://openneuro.org/datasets/ds003655)
- Downloaded participants: `sub-001`, `sub-002`; EEGLAB `.set` + paired `.fdt`, plus BIDS metadata
- Sampling rate 500 Hz; 21 channels = 19 EEG + 2 EOG (`EOGv`, `EOGh`)
- 120 trials per participant (20 per condition): `5R`, `6R`, `7R`, `5M`, `6M`, `7M`
- Raw EEG is **not committed** (size / data-management); see `.gitignore`.

To reproduce the raw-data steps, download `ds003655` for `sub-001` and `sub-002` and place them as:
```text
data/sub-001/eeg/sub-001_task-VerbalWorkingMemory_eeg.set (+ .fdt)
data/sub-002/eeg/sub-002_task-VerbalWorkingMemory_eeg.set (+ .fdt)
```

## Data acquisition evidence
Local file-presence and size evidence is recorded in `reports/download_inventory.csv` (8 files per participant, about 269.05 MB in total). No download API log was captured.

## Repository layout
```text
.
├── data/              # local raw data; not committed
├── data_scripts/      # Phase 2 scripts (numbered in execution order)
├── notebooks/         # exploratory notebook
├── paper/             # LaTeX report (phase1_phase2_report.tex)
├── reports/           # generated audit tables and notes
│   └── figures/       # EDA figures
├── src/               # reserved for later phases (modeling, federated training)
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline
Run from the repository root.

| Stage | Scripts | Main outputs |
|---|---|---|
| Download check, validation | `01_check_downloads.py`, `02_validate_data.py` | `download_inventory.csv`, `participant_validation.csv` |
| Trial table | `03_inspect_trials.py`, `04_build_trial_table.py` | `trial_table.csv` |
| Response audit | `05`-`10` | `response_probe_audit.csv`, `response_probe_issues.csv`, `orphan_responses.csv` |
| Signal quality | `11`-`17`, `22`-`24` | `signal_quality_decisions.md`, `trial_channel_amplitude_audit.csv` |
| Features | `18_extract_features.py` | `eeg_features.csv` (240 trials x 57 band-power features) |
| Tabular cleaning | `21_clean_tabular_features.py reports/eeg_features.csv` | `features_cleaned.csv`, `features_cleaning_log.csv` |
| EDA | `20_make_eda.py`, `25_plot_representative_signals.py`, `26_eeg_feature_eda.py` | `reports/figures/*.png` |

Steps that read raw EEG (`02`, `11`-`18`, `22`-`25`) need the local `data/` folder. Steps `20`, `21` and `26` only need the CSVs in `reports/`.

## Features
For each trial, the actual delay-to-probe window (about 6.7 s, taken from event onsets) is extracted for the 19 EEG channels. Welch power spectral density is integrated over theta (4-8 Hz), alpha (8-13 Hz) and beta (13-30 Hz), giving 19 x 3 = 57 features per trial. The table also stores trial metadata, response accuracy, trial quality and amplitude-screening columns. This is a first-pass feature set; filtering, referencing and artifact rules are not yet finalized.

## Data-centric cleaning and feature strategy
- Raw files are preserved unchanged and kept out of Git.
- Validation covers required files, sampling rate, channels, event timing, trial counts, missing stages and ambiguous response mappings.
- Duplicate-looking response markers are not deleted, and missing EEG samples are not imputed.
- Event audit: 240 probe/trial rows across two participants; the latest probe-based response audit marked 225 PASS and 15 REVIEW. This is a structural audit, not proof that every response is correctly attributed.
- Amplitude screening (500 uV heuristic) flags 122 of 240 trial windows with both amplitude flags (all 120 trials of `sub-002`) and 6 with a peak-to-peak flag only. The thresholds are not validated rejection rules and no trial has been excluded.
- A repeated boundary near -10,239.69 uV appears in all channels of `sub-001` around 332.5-333.5 s. The cause is not confirmed; do not clip, repair or exclude without investigation.
- Tabular cleaning (`21_clean_tabular_features.py`): exact duplicates removed from the derived table, median imputation for missing numeric values, "Unknown" for missing categories, IQR outliers flagged but not removed, standard scaling. For evaluation, scaling must be fitted on training data only; the local-vs-global scaling choice in federated runs must be explicit and identical across comparisons.
- Condition labels are encoded as categorical variables; response labels and quality flags are kept separate.

## Limitations / next steps
1. Explain the `sub-001` boundary anomaly and the amplitude flags on every `sub-002` trial.
2. Finalize and document filter, reference and artifact rules, then re-extract features.
3. Validate response labels and define the prediction target.
4. Expand the number of participants before any cross-subject conclusions.
5. Baseline, centralized, federated and personalized experiments.

## Report
```powershell
cd paper
pdflatex phase1_phase2_report.tex
pdflatex phase1_phase2_report.tex
```
Figures are read from `../reports/figures/`.

## Acknowledgements
Data: OpenNeuro `ds003655`; cite it according to its OpenNeuro page.
