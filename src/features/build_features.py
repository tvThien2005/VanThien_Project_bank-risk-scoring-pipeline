"""Stateless feature transformations applied consistently to every data split."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_date_features(df: pd.DataFrame, reference_date: str | pd.Timestamp) -> pd.DataFrame:
    """Add account and transaction recency in days from a fixed reference date."""
    featured = df.copy()
    reference = pd.Timestamp(reference_date)
    account_creation = pd.to_datetime(featured["account_creation_date"], errors="raise")
    last_transaction = pd.to_datetime(featured["last_transaction_date"], errors="raise")
    featured["account_age_days"] = (reference - account_creation).dt.days.astype("Int64")
    featured["days_since_last_txn"] = (reference - last_transaction).dt.days.astype("Int64")
    return featured


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series while returning missing values for zero denominators."""
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").mask(
        pd.to_numeric(denominator, errors="coerce") == 0
    )
    return pd.to_numeric(numerator, errors="coerce") / denominator_numeric


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add annualized spending, investment, and loan-per-card ratios."""
    featured = df.copy()
    featured["spending_to_income"] = _safe_divide(
        pd.to_numeric(featured["monthly_spending"], errors="coerce") * 12,
        featured["income"],
    )
    featured["investment_to_income"] = _safe_divide(
        featured["investment_value"], featured["income"]
    )
    featured["loans_per_card"] = _safe_divide(featured["num_loans"], featured["num_credit_cards"])
    return featured


def add_credit_risk_experiment_features(df: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    """Add optional, row-local credit-behaviour features with safe denominators.

    These features use only fields known at prediction time.  Keeping them behind
    one configuration switch makes the baseline and experimental feature sets
    directly comparable.
    """
    if not enabled:
        return df.copy()
    featured = df.copy()
    late_payment_months = pd.to_numeric(featured["late_payment_months"], errors="coerce")
    account_age_months = pd.to_numeric(featured["account_age_months"], errors="coerce")
    credit_utilization = pd.to_numeric(featured["credit_utilization"], errors="coerce")
    debt_to_income = pd.to_numeric(featured["debt_to_income"], errors="coerce")

    featured["late_payment_rate"] = _safe_divide(late_payment_months, account_age_months)
    featured["utilization_per_card"] = _safe_divide(
        credit_utilization, featured["num_credit_cards"]
    )
    featured["debt_utilization_interaction"] = debt_to_income * credit_utilization
    featured["payment_credit_interaction"] = late_payment_months * credit_utilization
    return featured


def build_stateless_features(
    df: pd.DataFrame,
    *,
    reference_date: str | pd.Timestamp,
    age_bins: Sequence[float],
    income_bins: Sequence[float],
    credit_risk_experiment_enabled: bool = False,
) -> pd.DataFrame:
    """Apply every row-local feature transformation used by train and inference."""
    featured = add_date_features(df, reference_date)
    featured = add_ratio_features(featured)
    featured = add_credit_risk_experiment_features(
        featured, enabled=credit_risk_experiment_enabled
    )
    return add_bins(featured, age_bins, income_bins)


def _bin_column(values: pd.Series, bins: Sequence[float]) -> pd.Series:
    """Create a string categorical interval feature using configured bin boundaries."""
    if len(bins) < 2:
        raise ValueError("Feature bins require at least two boundaries.")
    intervals = pd.cut(values, bins=bins, right=False, include_lowest=True)
    return intervals.astype("string").astype("category")


def add_bins(
    df: pd.DataFrame, age_bins: Sequence[float], income_bins: Sequence[float]
) -> pd.DataFrame:
    """Add categorical age and income bands using configuration-supplied boundaries."""
    featured = df.copy()
    featured["age_bin"] = _bin_column(featured["age"], age_bins)
    featured["income_bin"] = _bin_column(featured["income"], income_bins)
    return featured
