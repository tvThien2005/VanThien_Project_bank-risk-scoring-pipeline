"""Train and persist the XGBoost challenger for credit risk without test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.data.validate import load_data_quality_config
from src.features.preprocess import load_preprocessor
from src.models.bundle import save_model_bundle
from src.models.evaluate import evaluate_by_segment
from src.models.mlflow_utils import config_hash, current_git_commit, log_model_run
from src.models.train_credit import train_credit_xgboost
from src.pipelines.training_pipeline import (
    _sample_train_indices,
    _segment_frame,
    _target,
)
from src.utils.config import load_config
from src.utils.io import read_parquet_safe
from src.utils.logger import get_logger
from src.utils.seeds import set_seed

LOGGER = get_logger(__name__)


def _arguments() -> argparse.Namespace:
    """Parse the explicitly non-production XGBoost smoke option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use 2 Optuna trials on a 20k-row train/tune sample.")
    return parser.parse_args()


def _load_credit_config(path: str) -> dict[str, Any]:
    """Load the credit model YAML configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Credit model configuration must be a mapping.")
    return config


def main() -> None:
    """Train XGBoost on train/validation and save a versioned challenger bundle."""
    arguments = _arguments()
    config = load_config()
    seed = int(config["project"]["seed"])
    set_seed(seed)
    credit_config = _load_credit_config(str(config["models"]["credit_config_path"]))
    if arguments.smoke:
        credit_config["optuna"] = {
            **credit_config["optuna"],
            "n_trials": 2,
            "timeout_seconds": 120,
            "tune_sample_size": 20_000,
        }
        credit_config["xgboost"] = {
            **credit_config["xgboost"],
            "n_estimators": 300,
            "early_stopping_rounds": 30,
        }
    output_dir = Path(str(config["features"]["output_dir"]))
    X_train = read_parquet_safe(output_dir / "X_train.parquet")
    X_valid = read_parquet_safe(output_dir / "X_valid.parquet")
    y_train = _target(output_dir / "y_credit_train.parquet", "credit_risk")
    y_valid = _target(output_dir / "y_credit_valid.parquet", "credit_risk")
    selected = _sample_train_indices(y_train, 20_000 if arguments.smoke else None, seed)
    X_model_train = X_train.iloc[selected]
    y_model_train = y_train.iloc[selected]
    result = train_credit_xgboost(
        X_model_train, y_model_train, X_valid, y_valid, credit_config, seed=seed
    )
    segments = evaluate_by_segment(
        y_valid,
        result.model.predict(X_valid.to_numpy(dtype=float)),
        _segment_frame(config),
        task="credit_risk",
        report_path=str(credit_config["reports"]["xgboost_segments"]),
    )
    quality_config = load_data_quality_config(str(config["data_quality"]["config_path"]))
    metadata = {
        "git_commit": current_git_commit(),
        "config_hash": config_hash({"main": config, "model": credit_config}),
        "classes": sorted(y_model_train.unique()),
        "inference": {
            "features": {
                key: config["features"].get(key)
                for key in ("reference_date", "age_bins", "income_bins", "credit_risk_experiment")
            },
            "data_quality": dict(quality_config),
        },
    }
    bundle_settings = credit_config["xgboost_bundle"]
    bundle_dir = Path(str(bundle_settings["directory"]))
    if arguments.smoke:
        bundle_dir = bundle_dir.parent / "smoke"
    saved_dir = save_model_bundle(
        task="credit_risk",
        version=str(bundle_settings["version"]),
        directory=bundle_dir,
        model=result.model,
        preprocessor=load_preprocessor(str(config["features"]["preprocessor_path"])),
        feature_list=X_model_train.columns.tolist(),
        metrics={
            "validation": result.metrics,
            "best_params": result.best_params,
            "best_iteration": result.best_iteration,
        },
        metadata=metadata,
        algorithm="xgboost",
    )
    log_model_run(
        config=credit_config,
        model_name="xgboost",
        model_params={**credit_config["xgboost"], **result.best_params, "best_iteration": result.best_iteration},
        metrics=result.metrics,
        seed=seed,
        artifact_paths={"model": saved_dir / "model.json"},
        max_artifact_bytes=int(config["training"]["max_mlflow_artifact_bytes"]),
    )
    LOGGER.info("Credit XGBoost result: %s; segment rows: %d", result.metrics, len(segments))


if __name__ == "__main__":
    main()
