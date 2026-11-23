"""Raw data access; source CSV files are never modified."""
from pathlib import Path
import hashlib

import pandas as pd

from .config import ROOT, load_config


def load_csv(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}. Restore it with DVC before training.")
    return pd.read_csv(resolved)


def training_data_sha256(path: str | Path | None = None) -> str:
    """Hash the original CSV bytes, before pandas or preprocessing touches them."""
    if path is None:
        path = load_config()["data"]["train_path"]
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return hashlib.sha256(resolved.read_bytes()).hexdigest()
