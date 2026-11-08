"""Train deterministic fraud-score baseline regressors on train and validation data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.models.evaluate import evaluate_regression
from src.models.mlflow_utils import log_baseline_run
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FraudBaselineResult:
    """A fitted fraud baseline and its validation metrics."""

    model: DummyRegressor | Ridge
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FraudLightGBMResult:
    """The tuned, train-only LightGBM regressor and its validation diagnostics."""

    model: lgb.LGBMRegressor
    metrics: dict[str, Any]
    best_params: dict[str, Any]
    best_iteration: int


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


def _lightgbm_settings(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return validated LightGBM, Optuna, search-space, and report settings."""
    lightgbm_settings = config.get("lightgbm")
    optuna_settings = config.get("optuna")
    search_space = config.get("search_space")
    reports = config.get("reports")
    if not all(
        isinstance(value, Mapping)
        for value in (lightgbm_settings, optuna_settings, search_space, reports)
    ):
        raise TypeError("fraud LightGBM configuration is incomplete.")
    if "lightgbm" not in reports:
        raise ValueError("fraud reports.lightgbm is required.")
    return lightgbm_settings, optuna_settings, search_space, reports


def _fraud_tuning_indices(total_rows: int, sample_size: int | None, seed: int) -> list[int]:
    """Return reproducible random row positions for optional regression tuning samples."""
    if sample_size is None or sample_size >= total_rows:
        return list(range(total_rows))
    if sample_size <= 0:
        raise ValueError("fraud tune_sample_size must be positive when provided.")
    selected, _ = train_test_split(range(total_rows), train_size=sample_size, random_state=seed)
    return sorted(selected)


def _suggest_lightgbm_params(trial: optuna.Trial, search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest all tuned regressor parameters from configuration-provided bounds."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", *map(int, search_space["num_leaves"])),
        "max_depth": trial.suggest_int("max_depth", *map(int, search_space["max_depth"])),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *map(int, search_space["min_child_samples"])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *map(float, search_space["learning_rate"])
        ),
        "subsample": trial.suggest_float("subsample", *map(float, search_space["subsample"])),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *map(float, search_space["colsample_bytree"])
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *map(float, search_space["reg_alpha"]), log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *map(float, search_space["reg_lambda"]), log=True
        ),
    }


def _fraud_estimator(
    settings: Mapping[str, Any], params: Mapping[str, Any], seed: int
) -> lgb.LGBMRegressor:
    """Create a deterministic Huber LightGBM regressor from config and trial params."""
    fixed = dict(settings)
    fixed.pop("early_stopping_rounds")
    fixed.update(params)
    fixed.update(
        {
            "random_state": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )
    return lgb.LGBMRegressor(**fixed)


def _matrix(frame: pd.DataFrame) -> Any:
    """Convert named engineered features to a LightGBM-safe dense numeric matrix."""
    return frame.to_numpy(dtype=float)


def train_fraud_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> FraudLightGBMResult:
    """Tune on train/valid only, then refit the chosen fraud model on all train rows."""
    settings, optuna_settings, search_space, reports = _lightgbm_settings(config)
    primary_metric = str(config.get("primary_metric"))
    if primary_metric not in {"rmse", "mae"}:
        raise ValueError("fraud primary_metric must be rmse or mae.")
    tune_indices = _fraud_tuning_indices(len(y_train), optuna_settings.get("tune_sample_size"), seed)
    X_tune = X_train.iloc[tune_indices]
    y_tune = y_train.iloc[tune_indices]
    callbacks = [
        lgb.early_stopping(int(settings["early_stopping_rounds"]), verbose=False),
        lgb.log_evaluation(period=0),
    ]
    top_k_settings = config.get("evaluation")
    if not isinstance(top_k_settings, Mapping):
        raise TypeError("fraud evaluation settings must be a mapping.")
    top_k = float(top_k_settings["top_k"])

    def objective(trial: optuna.Trial) -> float:
        """Fit one train-only trial and return the configured validation error."""
        candidate = _fraud_estimator(settings, _suggest_lightgbm_params(trial, search_space), seed)
        candidate.fit(
            _matrix(X_tune), y_tune, eval_set=[(_matrix(X_valid), y_valid)], callbacks=callbacks
        )
        trial_metrics = evaluate_regression(y_valid, candidate.predict(_matrix(X_valid)), top_k=top_k)
        return float(trial_metrics[primary_metric])

    study = optuna.create_study(
        direction=str(optuna_settings["direction"]), sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        objective,
        n_trials=int(optuna_settings["n_trials"]),
        timeout=int(optuna_settings["timeout_seconds"]),
        show_progress_bar=False,
    )
    model = _fraud_estimator(settings, study.best_params, seed)
    model.fit(
        _matrix(X_train), y_train, eval_set=[(_matrix(X_valid), y_valid)], callbacks=callbacks
    )
    metrics = evaluate_regression(
        y_valid,
        model.predict(_matrix(X_valid)),
        top_k=top_k,
        report_path=str(reports["lightgbm"]),
        model_name="lightgbm",
    )
    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    LOGGER.info(
        "Fraud LightGBM validation %s=%.6f after %d trials", primary_metric, metrics[primary_metric], len(study.trials)
    )
    return FraudLightGBMResult(
        model=model,
        metrics=metrics,
        best_params=dict(study.best_params),
        best_iteration=best_iteration,
    )
