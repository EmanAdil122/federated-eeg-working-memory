# Federated EEG Working Memory

## Project overview
This project compares centralized, federated and personalized neural
networks for EEG-based working-memory load prediction. Deliverable 1
established the preprocessing workflow on two pilot participants.
Deliverable 2 expands the cohort to 24 participants and evaluates the
models on held-out participants.

## Deliverable 1

- [Compiled report](paper/phase1_phase2_report.pdf) and [LaTeX source](paper/phase1_phase2_report.tex)
- [Colab preprocessing notebook](notebooks/EEG_Preprocessing_Pipeline.ipynb)
- [Pilot EDA figures](paper/figures/pilot_eda/)

The pilot retained **240 trials**, with **40 trials per condition** across
the two participants. The corrected feature table contains **57 band-power
features per trial**. All **15 response-review flags** were preserved.

## Deliverable 2: Modeling and evaluation

- [Report PDF](paper/deliverable2_report.pdf) · [LaTeX source](paper/deliverable2_report.tex)
- [Modeling and evaluation notebook](notebooks/EEG_Modeling.ipynb)
- [Architecture, training curves, and confusion matrices](paper/figures/modeling/)

The classifier predicts memory load (5, 6, or 7 letters) from 57 EEG
band-power features. The primary dataset contains 2,878 of 2,880 trials
from 24 participants. Two trials with abrupt residual signal jumps are
flagged out of the primary analysis. The participant split uses seed 42
and assigns 16 participants to training, 4 to validation, and 4 to test.
Feature scaling is fitted on training participants only.

The three experiments share a PyTorch MLP with 57 inputs, 64 and 32 hidden
units, ReLU activations, dropout of 0.4 after the first hidden layer, and
three output scores. The centralized model trains on pooled features.
FedAvg simulates one client per training participant and aggregates client
weights by trial count. Personalization adapts the FedAvg output layer
using the first five labeled primary trials per condition (30 per
participant). All three models are scored on the same 359 test trials,
with calibration trials held out from scoring.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Macro ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| Centralized ANN | 0.3036 | 0.3098 | 0.3035 | 0.3036 | 0.4752 |
| FedAvg ANN | 0.3538 | 0.3573 | 0.3543 | 0.3276 | 0.4750 |
| Personalized FedAvg ANN | 0.3538 | 0.3643 | 0.3541 | 0.3397 | 0.4797 |

Personalization increased macro-F1 relative to FedAvg on this split.
Overall accuracy remained close to the 33.3% uniform-guess reference.
The report includes per-class errors and performance by participant and
task type. Results are descriptive for four test participants.

### Reproduce the modeling run

1. Obtain the feature table and schema from the team's Google Drive
   folder. Ask a repository maintainer for access if needed.
2. In your Drive, place the files in
   `My Drive/EEG_Project/checkpoints/combined_features/`:
   - `eeg_features_24_participants_qc_annotated.csv`
   - `eeg_features_24_participants_schema.json`
   The notebook creates `eeg_modeling_split_seed42.json` using seed 42.
   For a different Drive layout, update the folder variable in the first cell.
3. Open the modeling notebook in Colab. Run the setup and split, then
   centralized baseline, centralized tuning, FedAvg, personalization,
   final test evaluation, and error analysis in order. The final plotting
   cells save the report figures. These cells use PyTorch, NumPy, pandas,
   scikit-learn and Matplotlib.
4. Results are saved under `modeling_results/` in the shared Drive layout.
   The standalone model scripts and environment requirements should be
   updated together when the team's script version is ready.

### Model weights

The Colab run saves `centralized_selected.pt` and `federated_selected.pt` in `modeling_results/`.
These checkpoints include the model parameters, feature order and scaling
statistics. The personalized model is produced from the FedAvg checkpoint
and each participant's calibration trials.

**Checkpoint downloads:**

- [Centralized ANN weights](https://drive.google.com/file/d/1Iw1S1ulvC6ThpbhFYT8lDBbSQpAOqsxV/view?usp=sharing)
- [FedAvg ANN weights](https://drive.google.com/file/d/1ceewAjjzVzAMFGVSp3Q2NN6JZ17HJFVN/view?usp=sharing)

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
- Pilot: `sub-001` and `sub-002`. Expanded cohort: `sub-001` through `sub-024`; EEGLAB `.set` + paired `.fdt`, plus BIDS metadata
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
├── notebooks/         # Colab preprocessing, modeling and evaluation
├── paper/             # Deliverable 1 and 2 reports and figures
├── reports/           # generated audit tables and notes
│   └── figures/       # EDA figures
├── src/               # location for standalone project modules
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

- Model comparison uses one participant split and four held-out people.
  Performance differences should be checked on additional participants.
- ICA was fitted separately on each participant's full task recording.
  The study therefore evaluates already-preprocessed recordings.
- FedAvg is a simulation using a shared training-derived scaler.
  Personalization requires labeled calibration trials from the same session.
- Response accuracy is provisional where markers are ambiguous. The model
  target comes from the task-condition markers.
- The subgroup analysis describes participant and task differences.
  Demographic fairness was not assessed.

## Build the reports

From the repository root, use a LaTeX installation with the required packages:

```powershell
cd paper
pdflatex phase1_phase2_report.tex
pdflatex phase1_phase2_report.tex
pdflatex deliverable2_report.tex
pdflatex deliverable2_report.tex
```

Figure paths are relative to `paper/`. Deliverable 2 includes the pilot figures
and the modeling figures under `figures/modeling/`.

## Acknowledgements
Data: OpenNeuro `ds003655`; cite it according to its OpenNeuro page.
