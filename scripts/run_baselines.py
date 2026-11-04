"""Run Giai đoạn 5 baseline models, defaulting to a small stratified train sample."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.models.train_credit import train_credit_baselines
from src.models.train_fraud import train_fraud_baselines
from src.utils.config import load_config
from src.utils.io import read_parquet_safe
from src.utils.logger import get_logger
from src.utils.report import build_html_report, write_html_report

LOGGER = get_logger(__name__)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a model YAML configuration and require a mapping at its root."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"Model configuration must be a mapping: {config_path}")
    return config


def _load_target(path: Path, target_name: str) -> pd.Series:
    """Load exactly one target column and preserve row order with the feature matrix."""
    frame = read_parquet_safe(path)
    if frame.columns.tolist() != [target_name]:
        raise ValueError(f"Expected only target column {target_name!r} in {path}.")
    return frame[target_name]


def stratified_sample_indices(
    y_credit: pd.Series, sample_size: int | None, seed: int
) -> np.ndarray:
    """Return deterministic train indices stratified by credit-risk class when sampling."""
    total_rows = len(y_credit)
    if sample_size is None or sample_size >= total_rows:
        return np.arange(total_rows)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive or omitted for the complete train split.")
    if sample_size < y_credit.nunique():
        raise ValueError("sample_size must be at least the number of credit-risk classes.")
    selected, _ = train_test_split(
        np.arange(total_rows),
        train_size=sample_size,
        random_state=seed,
        stratify=y_credit,
    )
    return np.sort(selected)


def _write_summary(
    credit_results: Mapping[str, Any], fraud_results: Mapping[str, Any], report_path: str | Path
) -> None:
    """Write a compact, synthetic-data baseline comparison table."""
    rows: list[dict[str, Any]] = []
    for task, results in (("credit_risk", credit_results), ("fraud_risk_score", fraud_results)):
        for model_name, result in results.items():
            rows.append({"task": task, "model": model_name, **result.metrics})
    table = pd.DataFrame(rows).to_html(index=False, float_format=lambda value: f"{value:.6f}")
    write_html_report(build_html_report("Kết quả baseline trên validation", [("Metrics", table)]), report_path)


def run_baselines(sample_size: int | None, *, log_mlflow: bool = True) -> dict[str, dict[str, Any]]:
    """Load G4 artifacts, train only a train sample, and evaluate on untouched validation."""
    config = load_config()
    feature_settings = config["features"]
    model_paths = config.get("models")
    baseline_settings = config.get("baselines")
    if not isinstance(feature_settings, Mapping):
        raise TypeError("features configuration must be a mapping.")
    if not isinstance(model_paths, Mapping) or not isinstance(baseline_settings, Mapping):
        raise TypeError("models and baselines configuration must be mappings.")

    output_dir = Path(str(feature_settings["output_dir"]))
    credit_target = str(config["targets"]["classification"])
    fraud_target = str(config["targets"]["regression"])
    X_train = read_parquet_safe(output_dir / "X_train.parquet")
    X_valid = read_parquet_safe(output_dir / "X_valid.parquet")
    y_credit_train = _load_target(output_dir / "y_credit_train.parquet", credit_target)
    y_credit_valid = _load_target(output_dir / "y_credit_valid.parquet", credit_target)
    y_fraud_train = _load_target(output_dir / "y_fraud_train.parquet", fraud_target)
    y_fraud_valid = _load_target(output_dir / "y_fraud_valid.parquet", fraud_target)
    if not (len(X_train) == len(y_credit_train) == len(y_fraud_train)):
        raise ValueError("Training feature and target artifacts have inconsistent row counts.")

    seed = int(config["project"]["seed"])
    selected = stratified_sample_indices(y_credit_train, sample_size, seed)
    LOGGER.info(
        "Training G5 baselines on %d/%d rows; validation remains %d untouched rows",
        len(selected),
        len(X_train),
        len(X_valid),
    )
    credit_config = _load_yaml_mapping(str(model_paths["credit_config_path"]))
    fraud_config = _load_yaml_mapping(str(model_paths["fraud_config_path"]))
    credit_results = train_credit_baselines(
        X_train.iloc[selected],
        y_credit_train.iloc[selected],
        X_valid,
        y_credit_valid,
        credit_config,
        seed=seed,
        log_mlflow=log_mlflow,
    )
    fraud_results = train_fraud_baselines(
        X_train.iloc[selected],
        y_fraud_train.iloc[selected],
        X_valid,
        y_fraud_valid,
        fraud_config,
        seed=seed,
        log_mlflow=log_mlflow,
    )
    _write_summary(credit_results, fraud_results, str(baseline_settings["summary_report_path"]))
    return {
        "credit_risk": {name: result.metrics for name, result in credit_results.items()},
        "fraud_risk_score": {name: result.metrics for name, result in fraud_results.items()},
    }


def parse_args() -> argparse.Namespace:
    """Parse baseline command options without exposing the test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20_000,
        help="Number of stratified train rows; omit with --full-train to use all rows.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use every train row instead of the default smoke-test sample.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable local MLflow logging.")
    return parser.parse_args()


def main() -> None:
    """Run baseline smoke training and log results through the project logger."""
    arguments = parse_args()
    results = run_baselines(
        None if arguments.full_train else arguments.sample_size,
        log_mlflow=not arguments.no_mlflow,
    )
    LOGGER.info("Baseline validation results: %s", results)


if __name__ == "__main__":
    main()
