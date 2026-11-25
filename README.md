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

## Stage 3 service

Install the pinned dependencies with Python 3.12, then run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --port 8000
```

Open `http://localhost:8000/docs` to try the eight-field prediction request.
The service uses the exported v1 pipeline. Date-derived inputs missing from an
API request use the pipeline's training-fitted median imputation.

```powershell
docker build -t mp292-api:1.0 .
docker run --rm -p 8000:8000 mp292-api:1.0
```

Kubernetes manifests are under `k8s/`. For a local cluster, load the image into
the cluster before applying them.
