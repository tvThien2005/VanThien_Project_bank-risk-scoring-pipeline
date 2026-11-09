"""Create leak-free train, validation, and test feature artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.data.load import load_raw
from src.data.split import make_splits
from src.data.validate import load_data_quality_config
from src.features.build_features import (
    build_stateless_features,
)
from src.features.feature_selection import target_correlations_on_train
from src.features.preprocess import (
    build_preprocessor,
    fit_preprocessor,
    save_preprocessor,
    transform_features,
)
from src.utils.config import load_config
from src.utils.io import write_parquet_atomic
from src.utils.logger import get_logger
from src.utils.seeds import set_seed

LOGGER = get_logger(__name__)
SPLIT_NAMES = ("train", "valid", "test")


def _feature_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return output and stateless-feature settings from the main configuration."""
    settings = config.get("features")
    if not isinstance(settings, Mapping):
        raise TypeError("features configuration must be a mapping.")
    required = {
        "reference_date",
        "age_bins",
        "income_bins",
        "output_dir",
        "preprocessor_path",
        "train_target_correlation_path",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing feature configuration keys: {missing}")
    return settings


def _build_features(df: pd.DataFrame, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Apply only stateless date, ratio, and bin feature transformations."""
    experiment = settings.get("credit_risk_experiment", {})
    if not isinstance(experiment, Mapping):
        raise TypeError("features.credit_risk_experiment must be a mapping.")
    enabled = experiment.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("features.credit_risk_experiment.enabled must be a boolean.")
    return build_stateless_features(
        df,
        reference_date=settings["reference_date"],
        age_bins=settings["age_bins"],
        income_bins=settings["income_bins"],
        credit_risk_experiment_enabled=enabled,
    )


def _write_split_artifacts(
    X: pd.DataFrame,
    source: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    targets: Mapping[str, Any],
) -> None:
    """Write transformed X plus isolated target vectors for one data split."""
    credit_target = str(targets["classification"])
    fraud_target = str(targets["regression"])
    write_parquet_atomic(X.reset_index(drop=True), output_dir / f"X_{split_name}.parquet")
    write_parquet_atomic(
        source[[credit_target]].reset_index(drop=True),
        output_dir / f"y_credit_{split_name}.parquet",
    )
    write_parquet_atomic(
        source[[fraud_target]].reset_index(drop=True),
        output_dir / f"y_fraud_{split_name}.parquet",
    )


def run_feature_engineering(config: Mapping[str, Any]) -> dict[str, int]:
    """Build split feature artifacts from raw data, fitting state only on train."""
    project = config.get("project")
    targets = config.get("targets")
    if not isinstance(project, Mapping) or "seed" not in project:
        raise ValueError("Missing project.seed in configuration.")
    if not isinstance(targets, Mapping):
        raise TypeError("targets configuration must be a mapping.")
    set_seed(int(project["seed"]))
    settings = _feature_settings(config)
    quality_settings = config.get("data_quality")
    if not isinstance(quality_settings, Mapping) or "config_path" not in quality_settings:
        raise ValueError("Missing data_quality.config_path in configuration.")

    raw = load_raw(config)
    quality_config = load_data_quality_config(str(quality_settings["config_path"]))
    cleaned = add_missing_flags(handle_outliers(apply_entry_error_rules(raw, quality_config), quality_config))
    splits = dict(zip(SPLIT_NAMES, make_splits(cleaned, config), strict=True))
    featured_splits = {
        name: _build_features(frame, settings) for name, frame in splits.items()
    }

    preprocessor = fit_preprocessor(build_preprocessor(config), featured_splits["train"])
    save_preprocessor(preprocessor, str(settings["preprocessor_path"]))
    output_dir = Path(str(settings["output_dir"]))
    row_counts: dict[str, int] = {}
    for name in SPLIT_NAMES:
        transformed = transform_features(preprocessor, featured_splits[name])
        _write_split_artifacts(transformed, splits[name], name, output_dir, targets)
        if name == "train":
            diagnostics = target_correlations_on_train(
                transformed,
                splits[name][str(targets["classification"])],
                splits[name][str(targets["regression"])],
            )
            write_parquet_atomic(diagnostics, str(settings["train_target_correlation_path"]))
        LOGGER.info("Wrote %d transformed %s rows", len(transformed), name)
        row_counts[name] = len(transformed)
    return row_counts


def main() -> None:
    """Fit preprocessing on train only and write all configured feature artifacts."""
    run_feature_engineering(load_config())


if __name__ == "__main__":
    main()
