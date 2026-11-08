"""Compare selected credit model fairness with a no-sensitive-feature refit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.data.load import load_raw
from src.data.split import make_splits
from src.data.validate import load_data_quality_config
from src.evaluation.fairness import compute_fairness_metrics
from src.features.build_features import build_stateless_features
from src.features.preprocess import (
    build_preprocessor,
    fit_preprocessor,
    transform_features,
)
from src.models.bundle import save_model_bundle
from src.models.mlflow_utils import config_hash, current_git_commit, log_model_run
from src.models.predict import load_model_bundle
from src.models.train_credit import train_credit_lightgbm_fixed
from src.utils.config import load_config
from src.utils.io import read_parquet_safe
from src.utils.report import build_html_report, write_html_report
from src.utils.seeds import set_seed


def _load_model_config(path: str) -> dict[str, Any]:
    """Load the credit-model configuration used to construct a fixed refit."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Credit model configuration must be a mapping.")
    return config


def _stateless(frame: pd.DataFrame, features: dict[str, Any]) -> pd.DataFrame:
    """Build row-local features with the configuration captured for this variant."""
    experiment = features.get("credit_risk_experiment", {})
    if not isinstance(experiment, dict):
        raise TypeError("features.credit_risk_experiment must be a mapping.")
    return build_stateless_features(
        frame,
        reference_date=features["reference_date"],
        age_bins=features["age_bins"],
        income_bins=features["income_bins"],
        credit_risk_experiment_enabled=bool(experiment.get("enabled", False)),
    )


def _inference_metadata(
    config: dict[str, Any], quality_config: dict[str, Any], credit_config: dict[str, Any]
) -> dict[str, Any]:
    """Record the alternate feature policy necessary for raw-batch inference."""
    features = config["features"]
    return {
        "git_commit": current_git_commit(),
        "config_hash": config_hash({"main": config, "model": credit_config}),
        "variant": "no_sensitive",
        "classes": [0, 1, 2, 3],
        "inference": {
            "features": {
                key: features.get(key)
                for key in ("reference_date", "age_bins", "income_bins", "credit_risk_experiment")
            },
            "data_quality": quality_config,
        },
    }


def main() -> None:
    """Run validation-only fairness diagnostics; no Optuna and no test-set reads."""
    config = load_config()
    set_seed(int(config["project"]["seed"]))
    governance = config["governance"]
    feature_dir = Path(str(config["features"]["output_dir"]))
    y_valid = read_parquet_safe(feature_dir / "y_credit_valid.parquet").iloc[:, 0]
    X_valid = read_parquet_safe(feature_dir / "X_valid.parquet")
    selected = load_model_bundle("v1.0.0", "credit_risk")
    selected_output = np.asarray(selected.model.predict(X_valid.to_numpy(dtype=float)))
    selected_predictions = (
        selected_output.argmax(axis=1) if selected_output.ndim == 2 else selected_output.astype(int)
    )

    quality_config = load_data_quality_config(str(config["data_quality"]["config_path"]))
    raw = load_raw(config)
    cleaned = add_missing_flags(handle_outliers(apply_entry_error_rules(raw, quality_config), quality_config))
    train_raw, valid_raw, _ = make_splits(cleaned, config)
    sensitive_names = list(governance["sensitive_columns"])
    sensitive = _stateless(valid_raw, config["features"])[sensitive_names].reset_index(drop=True)

    variant_config = copy.deepcopy(config)
    exclusions = set(variant_config["features"].get("exclude", []))
    exclusions.update(governance["no_sensitive_exclude"])
    variant_config["features"]["exclude"] = sorted(exclusions)
    train_features = _stateless(train_raw, variant_config["features"])
    valid_features = _stateless(valid_raw, variant_config["features"])
    variant_preprocessor = fit_preprocessor(build_preprocessor(variant_config), train_features)
    X_train_variant = transform_features(variant_preprocessor, train_features)
    X_valid_variant = transform_features(variant_preprocessor, valid_features)
    y_train = train_raw["credit_risk"].reset_index(drop=True)
    credit_config = _load_model_config(str(config["models"]["credit_config_path"]))
    selected_metrics = json.loads(
        (Path("models/credit_risk/v1.0.0") / "metrics.json").read_text(encoding="utf-8")
    )
    best_params = selected_metrics.get("best_params")
    if not isinstance(best_params, dict):
        raise TypeError("Selected credit bundle metrics must include best_params.")
    variant_result = train_credit_lightgbm_fixed(
        X_train_variant,
        y_train,
        X_valid_variant,
        y_valid,
        credit_config,
        params=best_params,
        seed=int(config["project"]["seed"]),
    )
    variant_predictions = variant_result.model.predict(X_valid_variant.to_numpy(dtype=float))
    high_risk_class = int(governance["credit_high_risk_class"])
    selected_by_group, selected_summary = compute_fairness_metrics(
        y_valid, selected_predictions, sensitive, high_risk_class=high_risk_class
    )
    variant_by_group, variant_summary = compute_fairness_metrics(
        y_valid, variant_predictions, sensitive, high_risk_class=high_risk_class
    )
    selected_by_group.insert(0, "variant", "selected")
    variant_by_group.insert(0, "variant", "no_sensitive")
    selected_summary.insert(0, "variant", "selected")
    variant_summary.insert(0, "variant", "no_sensitive")
    bundle_dir = Path("models/credit_risk/experiments/no_sensitive")
    saved_dir = save_model_bundle(
        task="credit_risk",
        version="no_sensitive",
        directory=bundle_dir,
        model=variant_result.model,
        preprocessor=variant_preprocessor,
        feature_list=X_train_variant.columns.tolist(),
        metrics={
            "validation": variant_result.metrics,
            "best_params": best_params,
            "best_iteration": variant_result.best_iteration,
            "excluded_features": variant_config["features"]["exclude"],
        },
        metadata=_inference_metadata(variant_config, dict(quality_config), credit_config),
    )
    log_model_run(
        config=credit_config,
        model_name="lightgbm",
        model_params={**credit_config["lightgbm"], **best_params, "best_iteration": variant_result.best_iteration},
        metrics=variant_result.metrics,
        seed=int(config["project"]["seed"]),
        artifact_paths={"model": saved_dir / "model.txt"},
        max_artifact_bytes=int(config["training"]["max_mlflow_artifact_bytes"]),
        variant="no_sensitive",
    )
    comparison = pd.DataFrame(
        [
            {"variant": "selected", **selected_metrics["validation"]},
            {"variant": "no_sensitive", **variant_result.metrics},
        ]
    ).drop(columns=["confusion_matrix"])
    report = build_html_report(
        "Credit fairness report",
        [
            ("Validation performance comparison", comparison.to_html(index=False, float_format=lambda value: f"{value:.6f}")),
            ("Fairness metrics by group", pd.concat([selected_by_group, variant_by_group]).to_html(index=False, float_format=lambda value: f"{value:.6f}")),
            ("Group disparity summary", pd.concat([selected_summary, variant_summary]).to_html(index=False, float_format=lambda value: f"{value:.6f}")),
            ("Policy note", "Age, age_bin, and marital_status are excluded only in the no_sensitive comparison. This is a validation-only diagnostic on synthetic data; it does not establish legal or production fairness."),
        ],
    )
    write_html_report(report, governance["reports"]["fairness"])


if __name__ == "__main__":
    main()
