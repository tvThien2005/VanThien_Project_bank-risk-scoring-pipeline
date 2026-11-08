"""Tests for stateless feature building and train-fitted preprocessing."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features.build_features import (
    add_bins,
    add_credit_risk_experiment_features,
    add_date_features,
    add_ratio_features,
)
from src.features.feature_selection import (
    correlation_filter,
    permutation_importance_on_train,
    target_correlations_on_train,
)
from src.features.preprocess import (
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_features,
)


def _preprocessor_config() -> dict[str, Any]:
    """Create a minimal valid configuration for preprocessing tests."""
    return {
        "features": {
            "skew_threshold": 1.0,
            "exclude": [],
            "education_order": ["High School", "Some College", "Bachelor", "Master", "PhD"],
            "preprocessor_path": "unused.joblib",
        },
        "targets": {
            "classification": "credit_risk",
            "regression": "fraud_risk_score",
        },
    }


def _feature_frame() -> pd.DataFrame:
    """Create a compact raw-feature frame with target and excluded columns included."""
    return pd.DataFrame(
        {
            "customer_id": ["FN0000001", "FN0000002", "FN0000003", "FN0000004"],
            "account_creation_date": pd.to_datetime(["2020-01-01"] * 4),
            "last_transaction_date": pd.to_datetime(["2024-01-01"] * 4),
            "age": [20, 30, 40, 50],
            "income": [10.0, np.nan, 30.0, 10_000.0],
            "employment_status": pd.Categorical(
                ["Full-time", "Part-time", None, "Retired"]
            ),
            "education_level": pd.Categorical(
                ["High School", "Bachelor", None, "PhD"]
            ),
            "credit_risk": [0, 1, 2, 3],
            "fraud_risk_score": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_stateless_date_ratio_and_bin_features() -> None:
    """Create configured date, annualized-ratio, and categorical-bin features."""
    frame = pd.DataFrame(
        {
            "account_creation_date": pd.to_datetime(["2020-01-01"]),
            "last_transaction_date": pd.to_datetime(["2024-12-01"]),
            "monthly_spending": [1_000.0],
            "income": [120_000.0],
            "investment_value": [60_000.0],
            "num_loans": [2],
            "num_credit_cards": [0],
            "late_payment_months": [6],
            "account_age_months": [0],
            "credit_utilization": [0.5],
            "debt_to_income": [0.4],
            "age": [30],
        }
    )

    featured = add_bins(
        add_ratio_features(add_date_features(frame, "2025-01-01")),
        age_bins=[0, 25, 35, float("inf")],
        income_bins=[0, 50_000, 150_000, float("inf")],
    )

    assert featured.loc[0, "account_age_days"] == 1827
    assert featured.loc[0, "days_since_last_txn"] == 31
    assert featured.loc[0, "spending_to_income"] == 0.1
    assert featured.loc[0, "investment_to_income"] == 0.5
    assert pd.isna(featured.loc[0, "loans_per_card"])
    assert isinstance(featured["age_bin"].dtype, pd.CategoricalDtype)
    assert isinstance(featured["income_bin"].dtype, pd.CategoricalDtype)


def test_credit_risk_experiment_features_are_optional_and_safe() -> None:
    """Build optional credit-behaviour interactions without dividing by zero."""
    frame = pd.DataFrame(
        {
            "late_payment_months": [6.0, 2.0],
            "account_age_months": [24.0, 0.0],
            "credit_utilization": [0.5, 0.8],
            "num_credit_cards": [2.0, 0.0],
            "debt_to_income": [0.4, 0.3],
        }
    )

    featured = add_credit_risk_experiment_features(frame, enabled=True)
    disabled = add_credit_risk_experiment_features(frame, enabled=False)

    assert featured.loc[0, "late_payment_rate"] == 0.25
    assert featured.loc[0, "utilization_per_card"] == 0.25
    assert featured.loc[0, "debt_utilization_interaction"] == 0.2
    assert featured.loc[0, "payment_credit_interaction"] == 3.0
    assert pd.isna(featured.loc[1, "late_payment_rate"])
    assert pd.isna(featured.loc[1, "utilization_per_card"])
    assert set(disabled.columns) == set(frame.columns)


def test_preprocessor_is_train_fitted_excludes_leakage_and_round_trips(tmp_path: Path) -> None:
    """Fit only training features and reproduce outputs after joblib serialization."""
    train = _feature_frame()
    valid = _feature_frame().iloc[:2].copy()
    preprocessor = fit_preprocessor(build_preprocessor(_preprocessor_config()), train)

    transformed_train = transform_features(preprocessor, train)
    transformed_valid = transform_features(preprocessor, valid)
    saved_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, saved_path)
    reloaded_valid = transform_features(load_preprocessor(saved_path), valid)

    names = transformed_train.columns.tolist()
    assert transformed_train.shape[0] == len(train)
    assert transformed_valid.shape[0] == len(valid)
    assert all("customer_id" not in name for name in names)
    assert all("account_creation_date" not in name for name in names)
    assert all("last_transaction_date" not in name for name in names)
    assert all("credit_risk" not in name for name in names)
    assert all("fraud_risk_score" not in name for name in names)
    np.testing.assert_allclose(transformed_valid.to_numpy(), reloaded_valid.to_numpy())
    skew_transformer = preprocessor.named_transformers_["numeric"].named_steps["skew_log"]
    assert "income" in skew_transformer.skewed_columns_


def test_train_only_feature_selection_helpers() -> None:
    """Calculate correlation and permutation diagnostics from training rows only."""
    X_train = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "duplicate_signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "noise": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    model = LinearRegression().fit(X_train, y_train)

    importance = permutation_importance_on_train(
        model,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        n_repeats=3,
        random_state=42,
    )

    assert "duplicate_signal" in correlation_filter(X_train, threshold=0.99)
    assert set(importance["feature"]) == set(X_train.columns)
    correlations = target_correlations_on_train(X_train, y_train, y_train)
    assert set(correlations["feature"]) == set(X_train.columns)
