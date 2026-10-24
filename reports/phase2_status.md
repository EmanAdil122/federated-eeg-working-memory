# Phase 2 status checklist

## Completed according to the current project audit
- [x] Downloaded two participants from OpenNeuro ds003655
- [x] File inventory: 8 files per participant; approximately 269.05 MB total
- [x] Basic recording validation: 21 channels, 500 Hz, plausible durations
- [x] Event/trial structure audit: 120 trials per participant, 20 per condition
- [x] Response-to-probe audit with uncertain cases preserved for review
- [x] Channel type correction: EOGv/EOGh marked as EOG
- [x] Preliminary amplitude-boundary investigation and decision note
- [x] Raw data excluded from Git

## Partially completed / needs real output files
- [ ] Ensure the actual `reports/download_inventory.csv`, validation CSVs, and response audit CSVs are present in the repository
- [ ] Run `20_make_eda.py` and commit the generated figures
- [ ] Merge these files with the existing Phase 2 scripts; avoid duplicate scripts with overlapping purposes
- [ ] Add screenshots/download notes if required by the instructor as acquisition evidence

## Still required before claiming Phase 2 complete
- [ ] Resolve or clearly document the `sub-001` boundary anomaly before applying final signal cleaning
- [ ] Document the justified filter and artifact-handling choices
- [ ] Produce a reproducible trial-level feature table from a justified time window
- [ ] Demonstrate missing-value/duplicate/outlier handling on the derived table, without silently altering raw EEG
- [ ] Document categorical encoding and numeric scaling; fit preprocessing on training data only for later ML evaluation
- [ ] Add EEG-derived EDA plots such as feature distributions and feature correlations once features exist
- [ ] Insert actual Phase 1 project content into the LaTeX report
- [ ] Compile and inspect the final PDF
- [ ] Check team commit history for individual contributions

## Important distinction
Current condition-count charts are event-audit EDA. They are not feature-correlation charts and do not substitute for EEG-feature EDA.
