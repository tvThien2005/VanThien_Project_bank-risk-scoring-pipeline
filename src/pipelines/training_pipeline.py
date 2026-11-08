"""Giai đoạn 6 orchestration for feature-artifact LightGBM training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.data.load import load_raw
from src.data.split import make_splits
from src.data.validate import load_data_quality_config
from src.features.build_features import add_bins
from src.features.preprocess import load_preprocessor
from src.models.bundle import save_model_bundle
from src.models.evaluate import (
    evaluate_by_segment,
    evaluate_classification,
    evaluate_regression,
)
from src.models.mlflow_utils import config_hash, current_git_commit, log_model_run
from src.models.train_credit import train_credit_lightgbm
from src.models.train_fraud import train_fraud_lightgbm
from src.utils.io import read_parquet_safe
from src.utils.logger import get_logger
from src.utils.seeds import set_seed

LOGGER = get_logger(__name__)
TEST_EVALUATION_FILES = ("test_metrics.json", "threshold_test_metrics.json")


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a model YAML file and require a mapping root."""
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise TypeError(f"Model configuration must be a mapping: {path}")
    return settings


def _target(path: Path, column: str) -> pd.Series:
    """Load one isolated target vector while preserving feature-row order."""
    frame = read_parquet_safe(path)
    if frame.columns.tolist() != [column]:
        raise ValueError(f"Expected only target {column!r} in {path}.")
    return frame[column]


def _sample_train_indices(y_credit: pd.Series, size: int | None, seed: int) -> list[int]:
    """Return optional stratified train positions for an explicitly non-production smoke run."""
    if size is None or size >= len(y_credit):
        return list(range(len(y_credit)))
    if size < y_credit.nunique():
        raise ValueError("smoke train sample must include every credit-risk class.")
    selected, _ = train_test_split(
        range(len(y_credit)), train_size=size, random_state=seed, stratify=y_credit
    )
    return sorted(selected)


def _segment_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    """Rebuild the saved validation split only for configured segment labels."""
    training_settings = config.get("training")
    quality_settings = config.get("data_quality")
    features = config.get("features")
    if not all(isinstance(value, Mapping) for value in (training_settings, quality_settings, features)):
        raise TypeError("training, data_quality, and features configurations must be mappings.")
    segment_columns = training_settings.get("segment_columns")
    if not isinstance(segment_columns, list):
        raise TypeError("training.segment_columns must be a list.")
    raw = load_raw(config)
    quality_config = load_data_quality_config(str(quality_settings["config_path"]))
    cleaned = add_missing_flags(handle_outliers(apply_entry_error_rules(raw, quality_config), quality_config))
    _, valid, _ = make_splits(cleaned, config)
    featured = add_bins(valid, features["age_bins"], features["income_bins"])
    missing = sorted(set(segment_columns) - set(featured.columns))
    if missing:
        raise ValueError(f"Configured segment columns are missing: {missing}")
    return featured.loc[:, segment_columns].reset_index(drop=True)


def _metadata(
    config: Mapping[str, Any], quality_config: Mapping[str, Any], model_config: Mapping[str, Any], classes: list[int] | None
) -> dict[str, Any]:
    """Create reproducible bundle metadata including only inference-relevant settings."""
    features = config["features"]
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": current_git_commit(),
        "config_hash": config_hash({"main": config, "model": model_config}),
        "classes": classes,
        "inference": {
            "features": {
                "reference_date": features["reference_date"],
                "age_bins": features["age_bins"],
                "income_bins": features["income_bins"],
                "credit_risk_experiment": features.get("credit_risk_experiment", {}),
            },
            "data_quality": dict(quality_config),
        },
    }


def _write_test_metrics(bundle_dir: Path, metrics: Mapping[str, Any], force: bool) -> Path:
    """Persist one final-test metric file, refusing accidental repeat evaluation."""
    path = bundle_dir / "test_metrics.json"
    if any((bundle_dir / name).is_file() for name in TEST_EVALUATION_FILES) and not force:
        raise FileExistsError("test đã được đánh giá, dùng --force nếu thực sự cần đánh giá lại")
    path.write_text(json.dumps(dict(metrics), indent=2, default=str), encoding="utf-8")
    return path


def run_training(
    config: Mapping[str, Any],
    *,
    final: bool = False,
    force: bool = False,
    smoke_train_size: int | None = None,
    smoke_tune_size: int | None = None,
    smoke_trials: int | None = None,
    smoke_timeout_seconds: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune main models on train/valid and optionally evaluate the held-out test once."""
    project = config.get("project")
    features = config.get("features")
    model_paths = config.get("models")
    targets = config.get("targets")
    training_settings = config.get("training")
    quality_settings = config.get("data_quality")
    if not all(
        isinstance(value, Mapping)
        for value in (project, features, model_paths, targets, training_settings, quality_settings)
    ):
        raise TypeError("Main training configuration is incomplete.")
    if final and smoke_train_size is not None:
        raise ValueError("--final cannot be combined with a smoke training sample.")
    if force and not final:
        raise ValueError("--force is only valid together with --final.")
    seed = int(project["seed"])
    set_seed(seed)
    credit_config = _load_yaml_mapping(str(model_paths["credit_config_path"]))
    fraud_config = _load_yaml_mapping(str(model_paths["fraud_config_path"]))
    if final and not force:
        for model_config in (credit_config, fraud_config):
            bundle_dir = Path(str(model_config["bundle"]["directory"]))
            if any((bundle_dir / name).is_file() for name in TEST_EVALUATION_FILES):
                raise FileExistsError("test đã được đánh giá, dùng --force nếu thực sự cần đánh giá lại")
    for model_config in (credit_config, fraud_config):
        optuna_settings = model_config["optuna"]
        if smoke_tune_size is not None:
            optuna_settings["tune_sample_size"] = smoke_tune_size
        if smoke_trials is not None:
            optuna_settings["n_trials"] = smoke_trials
        if smoke_timeout_seconds is not None:
            optuna_settings["timeout_seconds"] = smoke_timeout_seconds

    output_dir = Path(str(features["output_dir"]))
    X_train = read_parquet_safe(output_dir / "X_train.parquet")
    X_valid = read_parquet_safe(output_dir / "X_valid.parquet")
    credit_target = str(targets["classification"])
    fraud_target = str(targets["regression"])
    y_credit_train = _target(output_dir / "y_credit_train.parquet", credit_target)
    y_credit_valid = _target(output_dir / "y_credit_valid.parquet", credit_target)
    y_fraud_train = _target(output_dir / "y_fraud_train.parquet", fraud_target)
    y_fraud_valid = _target(output_dir / "y_fraud_valid.parquet", fraud_target)
    selected = _sample_train_indices(y_credit_train, smoke_train_size, seed)
    LOGGER.info("Training main models on %d/%d train rows", len(selected), len(X_train))
    credit_result = train_credit_lightgbm(
        X_train.iloc[selected], y_credit_train.iloc[selected], X_valid, y_credit_valid, credit_config, seed=seed
    )
    fraud_result = train_fraud_lightgbm(
        X_train.iloc[selected], y_fraud_train.iloc[selected], X_valid, y_fraud_valid, fraud_config, seed=seed
    )

    segment_frame = _segment_frame(config)
    if len(segment_frame) != len(X_valid):
        raise ValueError("Validation segment frame does not align with validation features.")
    credit_segments = evaluate_by_segment(
        y_credit_valid,
        credit_result.model.predict(X_valid.to_numpy(dtype=float)),
        segment_frame,
        task=credit_target,
        report_path=str(credit_config["reports"]["segments"]),
    )
    fraud_segments = evaluate_by_segment(
        y_fraud_valid,
        fraud_result.model.predict(X_valid.to_numpy(dtype=float)),
        segment_frame,
        task=fraud_target,
        top_k=float(fraud_config["evaluation"]["top_k"]),
        report_path=str(fraud_config["reports"]["segments"]),
    )
    preprocessor = load_preprocessor(str(features["preprocessor_path"]))
    quality_config = load_data_quality_config(str(quality_settings["config_path"]))
    bundle_specs = (
        (credit_target, credit_config, credit_result, credit_segments, sorted(y_credit_train.unique())),
        (fraud_target, fraud_config, fraud_result, fraud_segments, None),
    )
    results: dict[str, dict[str, Any]] = {}
    for task, model_config, result, segment_metrics, classes in bundle_specs:
        bundle_settings = model_config["bundle"]
        bundle_dir = Path(str(bundle_settings["directory"]))
        if smoke_train_size is not None:
            bundle_dir = bundle_dir.parent / "smoke"
        metadata = _metadata(config, quality_config, model_config, classes)
        bundle_metrics = {
            "validation": result.metrics,
            "best_params": result.best_params,
            "best_iteration": result.best_iteration,
        }
        saved_dir = save_model_bundle(
            task=task,
            version=str(bundle_settings["version"]),
            directory=bundle_dir,
            model=result.model,
            preprocessor=preprocessor,
            feature_list=X_train.columns.tolist(),
            metrics=bundle_metrics,
            metadata=metadata,
        )
        artifact_paths: dict[str, Path] = {"model": saved_dir / "model.txt"}
        if final:
            X_test = read_parquet_safe(output_dir / "X_test.parquet")
            target_file = "y_credit_test.parquet" if task == credit_target else "y_fraud_test.parquet"
            y_test = _target(output_dir / target_file, task)
            test_metrics = (
                evaluate_classification(y_test, result.model.predict(X_test.to_numpy(dtype=float)))
                if task == credit_target
                else evaluate_regression(
                    y_test,
                    result.model.predict(X_test.to_numpy(dtype=float)),
                    top_k=float(model_config["evaluation"]["top_k"]),
                )
            )
            artifact_paths["test_metrics"] = _write_test_metrics(saved_dir, test_metrics, force)
            bundle_metrics["test"] = test_metrics
            (saved_dir / "metrics.json").write_text(
                json.dumps(bundle_metrics, indent=2, default=str), encoding="utf-8"
            )
        log_model_run(
            config=model_config,
            model_name="lightgbm",
            model_params={**model_config["lightgbm"], **result.best_params, "best_iteration": result.best_iteration},
            metrics=result.metrics,
            seed=seed,
            artifact_paths=artifact_paths,
            max_artifact_bytes=int(training_settings["max_mlflow_artifact_bytes"]),
        )
        results[task] = {"validation": result.metrics, "segment_rows": len(segment_metrics)}
    return results
