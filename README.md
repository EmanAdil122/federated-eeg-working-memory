# Federated EEG Working Memory

## Project overview
This repository investigates federated learning for EEG-based working-memory decoding, with attention to differences between participants and non-IID data. The current work is **Phase 1 (problem definition) and Phase 2 (data acquisition, validation, preprocessing, feature extraction and exploratory analysis)**. Model training is a later phase.

The dataset currently contains only **two participants**, so this is a pilot. No claims about generalization to the wider population are made.

## Deliverable 1

- [Compiled report](paper/phase1_phase2_report.pdf) and [LaTeX source](paper/phase1_phase2_report.tex)
- [Colab preprocessing notebook](notebooks/EEG_Preprocessing_Pipeline.ipynb)
- [Pilot EDA figures](paper/figures/pilot_eda/)

The pilot retained **240 trials**, with **40 trials per condition** across
the two participants. The corrected feature table contains **57 band-power
features per trial**. All **15 response-review flags** were preserved.

## Research questions
1. How does a federated model compare with a centralized baseline for working-memory condition decoding?
2. How sensitive are results to participant-level distribution differences (non-IID data)?
3. Does a clearly defined personalization strategy improve cross-subject generalization?

## Team
- Eman Adil
- Dua Sarfraz
- Ayza Ahmad

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
├── notebooks/         # Colab preprocessing and pilot EDA
├── paper/             # LaTeX report, PDF and pilot figures
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

## Acquisition and diagnostic scripts

Run these scripts from the repository root. The feature and EDA scripts
below reproduce the initial diagnostics. For the corrected pilot results
in the Deliverable 1 report, follow the Colab notebook workflow.

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

## Colab workflow for the corrected pilot

1. Run the notebook setup and load the two recordings and event tables.
2. Crop around the task, apply the 50 Hz notch and 1–40 Hz band-pass filters,
   and resample to 100 Hz. Retain the original linked-earlobe reference.
3. Fit and review ICA separately for each participant. Review candidate
   components using EOG association, scalp maps and time courses.
4. Audit the corrected delay windows, extract band powers, encode the
   task conditions, validate the feature table, and run the scaling demo.
5. Generate trial-balance, band-power distribution and correlation plots.

The notebook saves reviewed ICA solutions, corrected recordings and
derived feature tables in Drive checkpoints. The report figures are
included under `paper/figures/pilot_eda/`.

## Features
Each trial uses its actual delay-to-probe interval (about 6.7 s) and
19 EEG channels. Welch power spectral density is integrated over theta
(4–8 Hz), alpha (8–13 Hz) and beta (13–30 Hz), giving 57 band powers in
µV². IDs, labels, response-review fields and quality flags are stored
separately from the input feature list.

## Data validation and preprocessing

- The probe audit marked 225 trials PASS and 15 REVIEW. These flags
  describe the response markers. Condition labels come from the task events.
- The suspicious boundary interval in `sub-001` preceded the task crop.
  Its cause remains unconfirmed. After correction, none of the 240 delay
  windows exceeded the 500 µV peak-to-peak review threshold.
- The corrected feature audit found zero missing, infinite or negative
  band powers and zero duplicate trial keys or complete rows.
- Participant-wise IQR screening flagged at least one feature in 56 trials
  for `sub-001` and 57 for `sub-002`. All 240 trials were retained.
- Conditions `5R`, `6R`, `7R`, `5M`, `6M`, and `7M` map to nominal IDs
  0–5, respectively. Set size and task type are also available separately.
- The scaling demonstration fits means and standard deviations on
  `sub-001` and applies the same parameters to both participants.

## Limitations and next steps

1. Validate ambiguous response labels and choose the prediction target.
2. Expand the participant sample before drawing cross-subject conclusions.
3. Define how ICA will be fitted or calibrated for evaluation. The pilot
   fits used each participant's full task crop. Fit scaling parameters
   within each training split.
4. Evaluate centralized, federated and personalized models with a
   documented split and comparison protocol.

## Report
```powershell
cd paper
pdflatex phase1_phase2_report.tex
pdflatex phase1_phase2_report.tex
```
From `paper/`, the report reads the pilot EDA figures from
`figures/pilot_eda/` and the response audit figure from
`../reports/figures/`.

## Acknowledgements
Data: OpenNeuro `ds003655`; cite it according to its OpenNeuro page.
