"""Train deterministic credit-risk baseline classifiers on train and validation data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import evaluate_classification
from src.models.mlflow_utils import log_baseline_run
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CreditBaselineResult:
    """A fitted credit baseline and its validation metrics."""

    model: DummyClassifier | LogisticRegression
    metrics: dict[str, Any]


def _settings(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate and return model and report settings from a credit config."""
    models = config.get("models")
    reports = config.get("reports")
    if not isinstance(models, Mapping) or not isinstance(reports, Mapping):
        raise TypeError("credit model configuration needs models and reports mappings.")
    if {"dummy", "logistic_regression"} - set(models) or {
        "dummy",
        "logistic_regression",
    } - set(reports):
        raise ValueError("credit configuration is missing a baseline model or report path.")
    return models, reports


def train_credit_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    seed: int,
    log_mlflow: bool = True,
) -> dict[str, CreditBaselineResult]:
    """Fit Dummy and balanced multinomial logistic baselines, then score validation."""
    models, reports = _settings(config)
    dummy_settings = models["dummy"]
    logistic_settings = models["logistic_regression"]
    if not isinstance(dummy_settings, Mapping) or not isinstance(logistic_settings, Mapping):
        raise TypeError("credit baseline model settings must be mappings.")

    estimators: dict[str, DummyClassifier | LogisticRegression] = {
        "dummy": DummyClassifier(strategy=str(dummy_settings["strategy"])),
        "logistic_regression": LogisticRegression(
            C=float(logistic_settings["C"]),
            class_weight=str(logistic_settings["class_weight"]),
            max_iter=int(logistic_settings["max_iter"]),
            # sklearn >= 1.9 infers multinomial multiclass handling for lbfgs.
            random_state=seed,
            solver=str(logistic_settings["solver"]),
        ),
    }
    results: dict[str, CreditBaselineResult] = {}
    for model_name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        metrics = evaluate_classification(
            y_valid,
            estimator.predict(X_valid),
            report_path=str(reports[model_name]),
            model_name=model_name,
        )
        if log_mlflow:
            params = dummy_settings if model_name == "dummy" else logistic_settings
            log_baseline_run(
                config=config,
                model_name=model_name,
                model_params=params,
                metrics=metrics,
                seed=seed,
            )
        results[model_name] = CreditBaselineResult(model=estimator, metrics=metrics)
        LOGGER.info("Credit baseline %s validation metrics: %s", model_name, metrics)
    return results
