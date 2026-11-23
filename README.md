# MP292 - Village Milk Collection (Dehradun)

## Project Details

- Group ID: G047
- Problem ID: MP292
- Faculty: Avantika Saklani
- Batch: 24
- Repository: `https://github.com/TanishkaVashishth/G047-MP292-MLOps`

## Group Members

| Enrolment Number | Name |
|---|---|
| S24CSEU1374 | Parv Gurung |
| S24CSEU0016 | Atharv Kundu |
| S24CSEU1384 | Parth Singh |
| S24CSEU1294 | Tanishka Vashishth |

## Stage 1

This repository contains the original data metadata managed by DVC and the evidence-based investigation of the seven MP292 data issues.

- Data investigations: `src/mp292/data_checks.py`
- Remaining checks: `src/mp292/verify_remaining.py`
- Findings: `findings.yaml`

## Stage 2: ML model and MLflow

Stage 2 code provides reproducible cleaning, a train-fitted scikit-learn
preprocessing/model pipeline, three validation model configurations, MAE-based
selection, MLflow tracking and model registry helpers. Configuration lives in
`params.yaml`; see `reports/STAGE2_COMPLETION_REPORT.md` for implementation
status and the exact reproduction commands.

The original CSV files are kept locally and ignored by Git; the tracked DVC
pointer files preserve their data metadata. On a fresh checkout, restore the
files with DVC using your configured remote. With the pinned requirements
installed and the data available, run:

```powershell
py -3.12 -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
py -3.12 -m mp292.train
py -3.12 -m pytest -v
```
