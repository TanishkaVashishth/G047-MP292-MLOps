# MP292 Stage 2 Completion Report

## Scope

Stage 2 uses only `data/raw/train.csv` for model fitting and validation. It does
not run the Stage 3 service/container/deployment work or the Stage 4 monitoring,
retraining, or holdout prediction work. The original CSV files, Stage 1
`findings.yaml`, and DVC pointer files were left unchanged.

## Reproducible pipeline

- `params.yaml` supplies the train path, target, leakage column, stuck sensor
  interval, seed (`42`), validation fraction (`0.20`), candidate settings,
  SQLite URI (`sqlite:///mlflow.db`), model registry name, and output path.
- `clean_data` copies its input, changes `temp_c == 999` to missing, and
  changes `buffaloes == 114` to missing only from 2026-03-12 through
  2026-03-28. The supplied training CSV has 42 sentinel temperatures and 102
  readings in that stuck sensor interval.
- `prepare_features` removes `litres_paid_in_passbook`, the target, and
  identifiers. The train-fitted sklearn pipeline applies median numeric
  imputation, most-frequent categorical imputation, and one-hot encoding.
- All candidates use the same `train_test_split` with `random_state=42` and
  `test_size=0.20`. Random forest estimators use one worker for reliable
  execution in the Windows environment.
- Every successful run logs the original training file's SHA-256 as
  `data_sha256` and validation MAE (along with RMSE and R-squared).

## Dataset identity

`data/raw/train.csv` is 70,229 bytes. Its SHA-256, calculated from the original
file bytes, is:

`fcda4bd5e65c8c0040ef227e932e51a1670c02223c2aeeaffb5b7107b97981b8`

The train and later CSV MD5 hashes also match their existing DVC pointers.

## Validation results

| Candidate | Validation MAE (litres) | MLflow run ID |
|---|---:|---|
| Random forest baseline | 38.2366128188 | `dbd19b8ff2434165abc86389100fd565` |
| Random forest tuned | 38.0961080524 | `5a078752e69f48a086eb9f674abf9270` |
| Extra Trees | 36.8284836389 | `c39c96b36a7a46f6bece2e1afb37717f` |

Extra Trees had the lowest validation MAE and was selected. Its fitted pipeline
is saved at `models/model.joblib`.

## MLflow and registry verification

- Tracking URI: `sqlite:///mlflow.db`
- Three finished candidate runs are present in `mlflow.db`; each has the
  dataset SHA-256 parameter and an MAE metric.
- The registry contains `MP292_Milk_Collection` version `1`, in READY state.
- The `champion` alias resolves to version `1`, the Extra Trees run above.
- `mlflow.db` and `models/model.joblib` were created and loaded for verification.

The database also contains three failed setup attempts from debugging the
SQLAlchemy compatibility and restricted temporary-directory behavior. They
are not included in the validation comparison above. The three finished runs
and registered model were verified from the tracking database after training.

## Tests

Command: `py -3.12 -m pytest -q -p no:cacheprovider tests`

Result: **4 passed**. Coverage includes non-mutating cleaning of both faulty
fields, leakage exclusion and date features, a fitted preprocessing/model
pipeline with missing values and unseen categories, and FastAPI `TestClient`
behavior for valid input, health response, invalid field values, and extra
fields. The default pytest invocation also passed all four tests but emitted
cache warnings; the final explicit-directory invocation passed without
warnings.

## Dependencies and execution notes

Direct dependencies are pinned in `requirements.txt`. `SQLAlchemy==2.0.36` is
also pinned because MLflow 2.19 is incompatible with the SQLAlchemy 2.1 build
that pip initially selected. Dependencies were installed in Python 3.12.

MLflow model logging required broader filesystem access because the restricted
execution sandbox denied writes to MLflow's `tempfile` staging directories.
After that access was granted, all three model runs, registry registration,
and output artifacts completed. A JSON serialization issue in the training
summary was corrected afterward so future successful calls return only
JSON-serializable result fields; retraining into this existing version-1
registry is intentionally prevented by the registry helper.
