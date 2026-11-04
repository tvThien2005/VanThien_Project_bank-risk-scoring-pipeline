"""Train deterministic fraud-score baseline regressors on train and validation data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.models.evaluate import evaluate_regression
from src.models.mlflow_utils import log_baseline_run
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FraudBaselineResult:
    """A fitted fraud baseline and its validation metrics."""

    model: DummyRegressor | Ridge
    metrics: dict[str, Any]


def _settings(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate and return model, report, and evaluation settings from a fraud config."""
    models = config.get("models")
    reports = config.get("reports")
    evaluation = config.get("evaluation")
    if not all(isinstance(value, Mapping) for value in (models, reports, evaluation)):
        raise TypeError("fraud model configuration needs models, reports, and evaluation mappings.")
    if {"dummy", "ridge"} - set(models) or {"dummy", "ridge"} - set(reports):
        raise ValueError("fraud configuration is missing a baseline model or report path.")
    return models, reports, evaluation


def train_fraud_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    seed: int,
    log_mlflow: bool = True,
) -> dict[str, FraudBaselineResult]:
    """Fit mean Dummy and Ridge baselines, then score validation-only performance."""
    models, reports, evaluation = _settings(config)
    dummy_settings = models["dummy"]
    ridge_settings = models["ridge"]
    if not isinstance(dummy_settings, Mapping) or not isinstance(ridge_settings, Mapping):
        raise TypeError("fraud baseline model settings must be mappings.")
    top_k = float(evaluation["top_k"])
    estimators: dict[str, DummyRegressor | Ridge] = {
        "dummy": DummyRegressor(strategy=str(dummy_settings["strategy"])),
        "ridge": Ridge(alpha=float(ridge_settings["alpha"])),
    }
    results: dict[str, FraudBaselineResult] = {}
    for model_name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        metrics = evaluate_regression(
            y_valid,
            estimator.predict(X_valid),
            top_k=top_k,
            report_path=str(reports[model_name]),
            model_name=model_name,
        )
        if log_mlflow:
            params = dummy_settings if model_name == "dummy" else ridge_settings
            log_baseline_run(
                config=config,
                model_name=model_name,
                model_params=params,
                metrics=metrics,
                seed=seed,
            )
        results[model_name] = FraudBaselineResult(model=estimator, metrics=metrics)
        LOGGER.info("Fraud baseline %s validation metrics: %s", model_name, metrics)
    return results
