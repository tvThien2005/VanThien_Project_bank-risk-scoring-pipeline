"""Shared small-data fixture for pipeline integration tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipelines.training_pipeline import run_training
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_config(path: Path, report_dir: Path, bundle_dir: Path, tracking_dir: Path) -> dict[str, Any]:
    """Create a tiny deterministic model configuration suitable for sample.csv."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["optuna"].update({"n_trials": 1, "timeout_seconds": 30, "tune_sample_size": None})
    config["lightgbm"].update(
        {"n_estimators": 20, "early_stopping_rounds": 5, "min_child_samples": 1, "n_jobs": 1}
    )
    config["mlflow"].update({"tracking_uri": str(tracking_dir), "experiment_name": "integration"})
    for name in config["reports"]:
        config["reports"][name] = str(report_dir / f"{path.stem}_{name}.html")
    config["bundle"].update({"version": "v1.0.0", "directory": str(bundle_dir)})
    return config


@pytest.fixture(scope="session")
def trained_sample_project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Train both tiny task models from sample.csv once for integration checks."""
    workspace = tmp_path_factory.mktemp("sample_pipeline")
    config = copy.deepcopy(load_config(PROJECT_ROOT / "configs/config.yaml"))
    report_dir = workspace / "reports"
    model_dir = workspace / "models"
    tracking_dir = workspace / "mlruns"
    config["data"].update(
        {
            "raw_path": str(PROJECT_ROOT / "tests/data/sample.csv"),
            "interim_path": str(workspace / "interim/customers.parquet"),
        }
    )
    config["splits"]["id_paths"] = {
        name: str(workspace / f"splits/{name}_ids.npy") for name in ("train", "valid", "test")
    }
    config["features"].update(
        {
            "output_dir": str(workspace / "features"),
            "preprocessor_path": str(workspace / "features/preprocessor.joblib"),
            "train_target_correlation_path": str(workspace / "features/correlations.parquet"),
        }
    )
    config["data_quality"].update(
        {
            "config_path": str(PROJECT_ROOT / "configs/data_quality.yaml"),
            "report_path": str(report_dir / "quality.html"),
        }
    )
    config["training"]["max_mlflow_artifact_bytes"] = 1_000_000
    config["models"]["credit_config_path"] = str(workspace / "model_credit.yaml")
    config["models"]["fraud_config_path"] = str(workspace / "model_fraud.yaml")
    config["governance"]["reports"] = {
        name: str(report_dir / f"{name}.html") for name in config["governance"]["reports"]
    }

    credit = _model_config(
        PROJECT_ROOT / "configs/model_credit.yaml",
        report_dir,
        model_dir / "credit_risk/v1.0.0",
        tracking_dir,
    )
    fraud = _model_config(
        PROJECT_ROOT / "configs/model_fraud.yaml",
        report_dir,
        model_dir / "fraud_risk/v1.0.0",
        tracking_dir,
    )
    with Path(config["models"]["credit_config_path"]).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(credit, handle, sort_keys=False)
    with Path(config["models"]["fraud_config_path"]).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(fraud, handle, sort_keys=False)
    results = run_training(config, final=False)
    return {"workspace": workspace, "config": config, "results": results}
