"""Configuration loading for the MP292 pipeline."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load params.yaml with safe YAML parsing and resolve relative paths."""
    config_path = Path(path) if path else ROOT / "params.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
