"""Small, explicit helpers for reproducible local MLflow baseline logging."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from typing import Any

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash without logging the complete configuration as an artifact."""
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_baseline_run(
    *,
    config: Mapping[str, Any],
    model_name: str,
    model_params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    seed: int,
) -> None:
    """Log approved baseline metadata and scalar metrics to local MLflow."""
    import mlflow

    mlflow_settings = config.get("mlflow")
    if not isinstance(mlflow_settings, Mapping):
        raise TypeError("model configuration mlflow must be a mapping.")
    tracking_uri = mlflow_settings.get("tracking_uri")
    experiment_name = mlflow_settings.get("experiment_name")
    if not isinstance(tracking_uri, str) or not isinstance(experiment_name, str):
        raise TypeError("model configuration mlflow needs tracking_uri and experiment_name strings.")

    # MLflow 3.17 blocks its legacy local file store unless this explicit opt-in
    # is set.  The project roadmap deliberately keeps local runs in ./mlruns.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    scalar_metrics = {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and name != "top_k_count"
    }
    safe_params = {name: str(value) for name, value in model_params.items()}
    safe_params.update({"config_hash": config_hash(config), "seed": str(seed)})
    with mlflow.start_run(run_name=f"baseline_{model_name}"):
        mlflow.set_tags({"split": "train/valid", "variant": "baseline", "model": model_name})
        mlflow.set_tag("git_commit", current_git_commit())
        mlflow.log_params(safe_params)
        mlflow.log_metrics(scalar_metrics)
    LOGGER.info("Logged baseline %s to MLflow experiment %s", model_name, experiment_name)
