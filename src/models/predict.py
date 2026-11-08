"""Load model bundles and predict from raw customer rows without feature leakage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.features.build_features import (
    build_stateless_features,
)
from src.features.preprocess import transform_features
from src.models.bundle import ModelBundle, load_saved_bundle


def load_model_bundle(
    version: str, task: str, model_root: str | Path = "models", model_name: str = "lightgbm"
) -> ModelBundle:
    """Load a versioned task bundle from the configured model-root convention."""
    task_directory = {"credit_risk": "credit_risk", "fraud_risk_score": "fraud_risk"}.get(task)
    if task_directory is None:
        raise ValueError("task must be 'credit_risk' or 'fraud_risk_score'.")
    if model_name == "lightgbm":
        directory = Path(model_root) / task_directory / version
    elif model_name == "xgboost" and task == "credit_risk":
        directory = Path(model_root) / task_directory / "xgboost" / version
    else:
        raise ValueError("Only credit_risk has an xgboost challenger bundle.")
    return load_saved_bundle(directory)


def _available_quality_rules(
    quality_config: Mapping[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    """Retain only entry/outlier rules whose referenced columns are present at inference."""
    rules = quality_config.get("rules")
    if not isinstance(rules, list):
        raise TypeError("Bundle inference data-quality configuration needs rules.")
    available: list[Mapping[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        column = rule.get("column")
        other_column = rule.get("other_column")
        if column in frame and (other_column is None or other_column in frame):
            available.append(rule)
    return {"rules": available}


def _inference_features(X_raw: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Apply the bundle's stateless cleaning and features before its fitted preprocessor."""
    if "customer_id" not in X_raw:
        raise ValueError("X_raw must preserve customer_id for prediction joins.")
    inference = bundle.metadata.get("inference")
    if not isinstance(inference, Mapping):
        raise TypeError("Bundle metadata is missing inference settings.")
    feature_settings = inference.get("features")
    quality_config = inference.get("data_quality")
    if not isinstance(feature_settings, Mapping) or not isinstance(quality_config, Mapping):
        raise TypeError("Bundle inference settings are malformed.")
    raw = X_raw.copy()
    for column in ("account_creation_date", "last_transaction_date"):
        if column in raw:
            raw[column] = pd.to_datetime(raw[column], errors="raise")
    rules = _available_quality_rules(quality_config, raw)
    cleaned = add_missing_flags(handle_outliers(apply_entry_error_rules(raw, rules), rules))
    experiment = feature_settings.get("credit_risk_experiment", {})
    if not isinstance(experiment, Mapping):
        raise TypeError("Bundle credit-risk experiment settings are malformed.")
    enabled = experiment.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("Bundle credit-risk experiment switch must be a boolean.")
    featured = build_stateless_features(
        cleaned,
        reference_date=feature_settings["reference_date"],
        age_bins=feature_settings["age_bins"],
        income_bins=feature_settings["income_bins"],
        credit_risk_experiment_enabled=enabled,
    )
    expected_columns = getattr(bundle.preprocessor, "feature_names_in_", ())
    for column in expected_columns:
        if column not in featured:
            featured[column] = np.nan
    return featured


def predict_batch(X_raw: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Transform raw rows with a fitted bundle and return predictions joined to IDs."""
    featured = _inference_features(X_raw, bundle)
    transformed = transform_features(bundle.preprocessor, featured)
    missing = sorted(set(bundle.feature_list) - set(transformed.columns))
    if missing:
        raise ValueError(f"Inference preprocessing is missing trained features: {missing}")
    matrix = transformed.loc[:, bundle.feature_list]
    values = matrix.to_numpy(dtype=float)
    algorithm = str(bundle.metadata.get("algorithm", "lightgbm"))
    prediction = (
        bundle.model.predict_proba(values)
        if algorithm == "xgboost" and bundle.task == "credit_risk"
        else bundle.model.predict(values)
    )
    result = pd.DataFrame({"customer_id": X_raw["customer_id"].to_numpy()})
    if bundle.task == "credit_risk":
        probabilities = np.asarray(prediction)
        if probabilities.ndim != 2:
            raise ValueError("Credit-risk model must return class probabilities.")
        classes = bundle.metadata.get("classes")
        if not isinstance(classes, list) or len(classes) != probabilities.shape[1]:
            raise ValueError("Credit-risk bundle is missing valid class labels.")
        result["credit_risk_prediction"] = np.asarray(classes)[probabilities.argmax(axis=1)]
        for index, class_label in enumerate(classes):
            result[f"credit_risk_probability_{class_label}"] = probabilities[:, index]
    elif bundle.task == "fraud_risk_score":
        result["fraud_risk_score_prediction"] = np.asarray(prediction, dtype=float)
    else:
        raise ValueError(f"Unsupported bundle task: {bundle.task}")
    return result
