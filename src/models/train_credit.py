"""Train deterministic credit-risk baseline classifiers on train and validation data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.models.evaluate import evaluate_classification
from src.models.mlflow_utils import log_baseline_run
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

def _xgboost_module() -> Any:
    """Import XGBoost only when its challenger is requested."""
    try:
        import xgboost
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "XGBoost is required for this challenger. Install requirements.txt first."
        ) from error
    return xgboost


@dataclass(frozen=True)
class CreditBaselineResult:
    """A fitted credit baseline and its validation metrics."""

    model: DummyClassifier | LogisticRegression
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CreditLightGBMResult:
    """The tuned, train-only LightGBM classifier and its validation diagnostics."""

    model: lgb.LGBMClassifier
    metrics: dict[str, Any]
    best_params: dict[str, Any]
    best_iteration: int


@dataclass(frozen=True)
class CreditXGBoostResult:
    """The tuned, train-only XGBoost challenger and its validation diagnostics."""

    model: Any
    metrics: dict[str, Any]
    best_params: dict[str, Any]
    best_iteration: int


@dataclass(frozen=True)
class CreditFixedLightGBMResult:
    """One LightGBM refit with already-selected parameters and no Optuna study."""

    model: lgb.LGBMClassifier
    metrics: dict[str, Any]
    best_iteration: int


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
        raise TypeError("credit LightGBM configuration is incomplete.")
    if "lightgbm" not in reports:
        raise ValueError("credit reports.lightgbm is required.")
    return lightgbm_settings, optuna_settings, search_space, reports


def _credit_tuning_indices(y_train: pd.Series, sample_size: int | None, seed: int) -> list[int]:
    """Return reproducible stratified row positions for optional Optuna acceleration."""
    if sample_size is None or sample_size >= len(y_train):
        return list(range(len(y_train)))
    if sample_size < y_train.nunique():
        raise ValueError("credit tune_sample_size must cover every class.")
    selected, _ = train_test_split(
        range(len(y_train)), train_size=sample_size, random_state=seed, stratify=y_train
    )
    return sorted(selected)


def _suggest_lightgbm_params(trial: optuna.Trial, search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest all tuned classifier parameters from configuration-provided bounds."""
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


def _credit_estimator(
    settings: Mapping[str, Any], params: Mapping[str, Any], seed: int
) -> lgb.LGBMClassifier:
    """Create a deterministic multiclass LightGBM classifier from config and trial params."""
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
    return lgb.LGBMClassifier(**fixed)


def _matrix(frame: pd.DataFrame) -> Any:
    """Convert named engineered features to a LightGBM-safe dense numeric matrix."""
    return frame.to_numpy(dtype=float)


def train_credit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> CreditLightGBMResult:
    """Tune on train/valid only, then refit the chosen credit model on all train rows."""
    settings, optuna_settings, search_space, reports = _lightgbm_settings(config)
    primary_metric = str(config.get("primary_metric"))
    if primary_metric not in {"qwk", "macro_f1"}:
        raise ValueError("credit primary_metric must be qwk or macro_f1.")
    tune_indices = _credit_tuning_indices(
        y_train, optuna_settings.get("tune_sample_size"), seed
    )
    X_tune = X_train.iloc[tune_indices]
    y_tune = y_train.iloc[tune_indices]
    callbacks = [
        lgb.early_stopping(int(settings["early_stopping_rounds"]), verbose=False),
        lgb.log_evaluation(period=0),
    ]

    def objective(trial: optuna.Trial) -> float:
        """Fit one train-only trial and return the configured validation score."""
        candidate = _credit_estimator(settings, _suggest_lightgbm_params(trial, search_space), seed)
        candidate.fit(
            _matrix(X_tune), y_tune, eval_set=[(_matrix(X_valid), y_valid)], callbacks=callbacks
        )
        trial_metrics = evaluate_classification(y_valid, candidate.predict(_matrix(X_valid)))
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
    model = _credit_estimator(settings, study.best_params, seed)
    model.fit(
        _matrix(X_train), y_train, eval_set=[(_matrix(X_valid), y_valid)], callbacks=callbacks
    )
    metrics = evaluate_classification(
        y_valid,
        model.predict(_matrix(X_valid)),
        report_path=str(reports["lightgbm"]),
        model_name="lightgbm",
    )
    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    LOGGER.info(
        "Credit LightGBM validation %s=%.6f after %d trials", primary_metric, metrics[primary_metric], len(study.trials)
    )
    return CreditLightGBMResult(
        model=model,
        metrics=metrics,
        best_params=dict(study.best_params),
        best_iteration=best_iteration,
    )


def train_credit_lightgbm_fixed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    seed: int,
) -> CreditFixedLightGBMResult:
    """Refit selected LightGBM parameters once; intended for fairness variants only."""
    settings, _, _, _reports = _lightgbm_settings(config)
    model = _credit_estimator(settings, params, seed)
    callbacks = [
        lgb.early_stopping(int(settings["early_stopping_rounds"]), verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        _matrix(X_train), y_train, eval_set=[(_matrix(X_valid), y_valid)], callbacks=callbacks
    )
    metrics = evaluate_classification(y_valid, model.predict(_matrix(X_valid)))
    return CreditFixedLightGBMResult(
        model=model,
        metrics=metrics,
        best_iteration=int(model.best_iteration_ or model.n_estimators_),
    )


def _xgboost_settings(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return validated XGBoost, Optuna, search-space, and report settings."""
    xgboost_settings = config.get("xgboost")
    optuna_settings = config.get("optuna")
    search_space = config.get("xgboost_search_space")
    reports = config.get("reports")
    if not all(
        isinstance(value, Mapping)
        for value in (xgboost_settings, optuna_settings, search_space, reports)
    ):
        raise TypeError("credit XGBoost configuration is incomplete.")
    if "xgboost" not in reports:
        raise ValueError("credit reports.xgboost is required.")
    return xgboost_settings, optuna_settings, search_space, reports


def _suggest_xgboost_params(trial: optuna.Trial, search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest all tuned XGBoost classifier parameters from configured bounds."""
    return {
        "max_depth": trial.suggest_int("max_depth", *map(int, search_space["max_depth"])),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", *map(float, search_space["min_child_weight"])
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
        "gamma": trial.suggest_float("gamma", *map(float, search_space["gamma"])),
    }


def _credit_xgboost_estimator(
    settings: Mapping[str, Any], params: Mapping[str, Any], seed: int
) -> Any:
    """Create a reproducible multiclass XGBoost classifier from config and trial params."""
    fixed = dict(settings)
    fixed.update(params)
    fixed.update({"random_state": seed, "seed": seed})
    return _xgboost_module().XGBClassifier(**fixed)


def _balanced_sample_weights(y: pd.Series) -> np.ndarray:
    """Give every observed credit-risk class identical total training weight."""
    labels = np.asarray(y)
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {label: len(labels) / (len(classes) * count) for label, count in zip(classes, counts, strict=True)}
    return np.asarray([class_weights[label] for label in labels], dtype=float)


def train_credit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> CreditXGBoostResult:
    """Tune an XGBoost challenger using train/validation only and QWK or macro-F1."""
    settings, optuna_settings, search_space, reports = _xgboost_settings(config)
    primary_metric = str(config.get("primary_metric"))
    if primary_metric not in {"qwk", "macro_f1"}:
        raise ValueError("credit primary_metric must be qwk or macro_f1.")
    tune_indices = _credit_tuning_indices(y_train, optuna_settings.get("tune_sample_size"), seed)
    X_tune = X_train.iloc[tune_indices]
    y_tune = y_train.iloc[tune_indices]
    tune_weights = _balanced_sample_weights(y_tune)

    def objective(trial: optuna.Trial) -> float:
        """Fit one train-only XGBoost trial and return its validation QWK or macro-F1."""
        candidate = _credit_xgboost_estimator(
            settings, _suggest_xgboost_params(trial, search_space), seed
        )
        candidate.fit(
            _matrix(X_tune),
            y_tune,
            sample_weight=tune_weights,
            eval_set=[(_matrix(X_valid), y_valid)],
            verbose=False,
        )
        trial_metrics = evaluate_classification(y_valid, candidate.predict(_matrix(X_valid)))
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
    model = _credit_xgboost_estimator(settings, study.best_params, seed)
    model.fit(
        _matrix(X_train),
        y_train,
        sample_weight=_balanced_sample_weights(y_train),
        eval_set=[(_matrix(X_valid), y_valid)],
        verbose=False,
    )
    metrics = evaluate_classification(
        y_valid,
        model.predict(_matrix(X_valid)),
        report_path=str(reports["xgboost"]),
        model_name="xgboost",
    )
    best_iteration = int(model.best_iteration if model.best_iteration is not None else settings["n_estimators"])
    LOGGER.info(
        "Credit XGBoost validation %s=%.6f after %d trials",
        primary_metric,
        metrics[primary_metric],
        len(study.trials),
    )
    return CreditXGBoostResult(
        model=model,
        metrics=metrics,
        best_params=dict(study.best_params),
        best_iteration=best_iteration,
    )
