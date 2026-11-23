"""Strict MLflow registry operations for the champion model."""
import mlflow
from mlflow.tracking import MlflowClient


def register_champion(run_id: str, model_name: str) -> int:
    client = MlflowClient()
    existing = client.search_model_versions(f"name='{model_name}'")
    if existing:
        raise RuntimeError(
            f"{model_name} already has registered versions; refusing to label a later "
            "version as the Stage 2 v1. Use a fresh tracking database for the initial registration."
        )
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, model_name)
    if str(registered.version) != "1":
        raise RuntimeError(f"Expected initial model version 1, got {registered.version}")
    client.set_registered_model_alias(model_name, "champion", registered.version)
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [v for v in versions if v.version == registered.version]
    if not matching:
        raise RuntimeError(f"Registered version {registered.version} was not found")
    alias_version = client.get_model_version_by_alias(model_name, "champion")
    if str(alias_version.version) != str(registered.version):
        raise RuntimeError("champion alias verification failed")
    mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    return int(registered.version)
