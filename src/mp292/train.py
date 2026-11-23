"""Train three reproducible regressors, track runs, and register the MAE winner."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ROOT, load_config
from .data_loader import load_csv, training_data_sha256
from .evaluate import regression_metrics
from .model_registry import register_champion
from .preprocessing import build_preprocessor, clean_data, prepare_features


def _estimator(name: str, params: dict, seed: int):
    opts = {k: v for k, v in params.items() if k != "estimator"}
    opts["random_state"] = seed
    if params["estimator"] == "RandomForestRegressor":
        return RandomForestRegressor(**opts, n_jobs=1)
    if params["estimator"] == "ExtraTreesRegressor":
        return ExtraTreesRegressor(**opts, n_jobs=1)
    raise ValueError(f"Unsupported estimator in params.yaml: {params['estimator']}")


def train(config_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    raw = load_csv(cfg["data"]["train_path"])
    data_hash = training_data_sha256(cfg["data"]["train_path"])
    target = cfg["data"]["target"]
    cleaned = clean_data(raw, cfg).dropna(subset=[target])
    X = prepare_features(cleaned, cfg)
    y = cleaned[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=cfg["seed"]
    )
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    best = None
    results = []
    for name, params in cfg["models"].items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)),
                             ("regressor", _estimator(name, params, cfg["seed"]))])
        with mlflow.start_run(run_name=name) as active:
            pipeline.fit(X_train, y_train)
            metrics = regression_metrics(y_val, pipeline.predict(X_val))
            mlflow.log_param("model_name", name)
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params({f"model.{k}": str(v) for k, v in params.items()})
            mlflow.log_params({"seed": cfg["seed"], "test_size": cfg["split"]["test_size"]})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("feature_columns", json.dumps(list(X.columns)))
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            row = {"name": name, "run_id": active.info.run_id, **metrics, "params": params}
            results.append(row)
            if best is None or metrics["mae"] < best["mae"]:
                best = row
                best["pipeline"] = pipeline
    model_path = ROOT / cfg["model_output"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], model_path)
    version = register_champion(best["run_id"], cfg["registered_model_name"])
    return {"results": [{k: v for k, v in row.items() if k not in {"params", "pipeline"}}
                         for row in results],
            "best_run_id": best["run_id"], "best_model": best["name"],
            "registered_model_name": cfg["registered_model_name"], "champion_version": version,
            "data_sha256": data_hash, "model_path": str(model_path)}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
