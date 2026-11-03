"""Train-only feature-selection diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_on_train(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    scoring: str,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Calculate permutation importance on training data only for a fitted model."""
    result = permutation_importance(
        model,
        X_train,
        y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def correlation_filter(X_train: pd.DataFrame, threshold: float) -> list[str]:
    """Return highly correlated numeric columns to drop using training data only."""
    if not 0 < threshold < 1:
        raise ValueError("Correlation threshold must be between 0 and 1.")
    numeric = X_train.select_dtypes(include=np.number)
    correlation = numeric.corr().abs()
    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    )
    return [column for column in upper_triangle.columns if (upper_triangle[column] > threshold).any()]


def target_correlations_on_train(
    X_train: pd.DataFrame, y_credit_train: pd.Series, y_fraud_train: pd.Series
) -> pd.DataFrame:
    """Record train-only linear target correlations to detect obvious feature leakage."""
    return pd.DataFrame(
        {
            "feature": X_train.columns,
            "credit_risk_correlation": X_train.corrwith(y_credit_train).to_numpy(),
            "fraud_risk_score_correlation": X_train.corrwith(y_fraud_train).to_numpy(),
        }
    ).assign(
        max_abs_target_correlation=lambda frame: frame[
            ["credit_risk_correlation", "fraud_risk_score_correlation"]
        ].abs().max(axis=1)
    ).sort_values("max_abs_target_correlation", ascending=False, ignore_index=True)
