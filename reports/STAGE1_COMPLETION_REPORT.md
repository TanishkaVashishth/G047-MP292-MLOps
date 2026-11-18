# MP292 Stage 1 Completion Report

## Commands run

- `python src/mp292/data_checks.py`
- `python src/mp292/verify_remaining.py`
- `dvc status`
- `dvc add` was not run
- `git status --short --branch`
- Exact evidence checks for row count, full-row and record-ID duplicates, leakage correlation, sentinel count, and text levels

## Stage 1 findings

| Finding | Result | Evidence |
|---|---|---|
| leaking_column | Present | `litres_paid_in_passbook` is absent from `holdout_features.csv`; correlation with `litres_per_day` is 0.998526 |
| unit_mixup | Not present | Numeric summaries for `fodder_price_rs` and `feed_kg` are consistent across all village clusters |
| sentinel_value | Present | `temp_c == 999` occurs 42 times in `train.csv` |
| duplicate_rows | Not present | 0 extra full duplicate rows; 0 extra duplicate `record_id` values |
| stuck_sensor | Present | `buffaloes == 114` for every one of 102 rows from 2026-03-12 through 2026-03-28 |
| target_typos | Not present | Top targets remain close to `litres_paid_in_passbook`; no target group is 10 times larger |
| spelling_variants | Not present | Clean levels are 3 in `breed_mix` and 3 in `village_cluster`; train and later spellings agree |

Counts are from the original `train.csv` as given: 1000 rows.

## DVC and Git

- DVC is initialized: `.dvc/config` exists.
- `dvc status`: data and pipelines are up to date.
- `data/raw/train.csv.dvc` and `data/raw/later.csv.dvc` are present and tracked.
- Raw CSV files are ignored by Git and are not tracked.
- Existing `.dvc` hashes are present for both raw CSV files.

## README and group details

- `GROUP.md` is present and contains Group G047, Problem MP292, faculty/batch, and all four enrolment numbers.
- `README.md` now contains Group G047, Problem MP292, faculty/batch, and all four enrolment numbers.
- The repository URL remains a placeholder and must be replaced with the public repository link before submission.

## Git status

- Only `.gitignore` and `findings.yaml` had tracked modifications observed before this report.
- Untracked areas include `reports/`, `scripts/`, and `src/`.
- No raw CSV was modified.

## Missing requirements and blockers

- README group details are complete; only the public repository URL is unknown and is marked as a placeholder.
- Both verification scripts ran without errors during the final rerun.
- Stage 1 data-part values in `findings.yaml` are populated from original-data evidence; Stage 2-4 tracking and monitoring values remain intentionally null.
- No Stage 2 work was started, no model was trained, and no commit or push was made.
