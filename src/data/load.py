"""Load raw customer data into a documented dataframe schema."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_csv_safe, write_parquet_atomic
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
DATE_COLUMNS = ("account_creation_date", "last_transaction_date")
CATEGORICAL_COLUMNS = (
    "employment_status",
    "education_level",
    "marital_status",
    "home_ownership",
    "region",
)
INTEGER_COLUMNS = (
    "age",
    "credit_score",
    "account_age_months",
    "num_credit_cards",
    "num_loans",
    "late_payment_months",
    "credit_risk",
)
FLOAT_COLUMNS = (
    "income",
    "employment_years",
    "monthly_spending",
    "credit_utilization",
    "debt_to_income",
    "savings_rate",
    "investment_value",
    "fraud_risk_score",
)
EXPECTED_COLUMNS = (
    "customer_id",
    *INTEGER_COLUMNS[:2],
    *FLOAT_COLUMNS[:2],
    *CATEGORICAL_COLUMNS,
    *INTEGER_COLUMNS[2:6],
    *FLOAT_COLUMNS[2:7],
    *DATE_COLUMNS,
    "credit_risk",
    "fraud_risk_score",
)


def _get_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Return a configured data path or raise a clear configuration error."""
    data_config = cfg.get("data")
    if not isinstance(data_config, Mapping) or key not in data_config:
        raise ValueError(f"Missing data.{key} in configuration.")
    return Path(str(data_config[key]))


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure the input has precisely the 23 documented source columns."""
    columns = set(df.columns)
    missing = sorted(set(EXPECTED_COLUMNS) - columns)
    unexpected = sorted(columns - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"Raw data schema mismatch: {', '.join(details)}")


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw columns to string, nullable numeric, datetime, and category."""
    typed = df.loc[:, EXPECTED_COLUMNS].copy()
    typed["customer_id"] = typed["customer_id"].astype("string")
    for column in DATE_COLUMNS:
        typed[column] = pd.to_datetime(typed[column], errors="raise")
    for column in CATEGORICAL_COLUMNS:
        typed[column] = typed[column].astype("string").astype("category")
    for column in INTEGER_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="raise").astype("Int64")
    for column in FLOAT_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="raise").astype("Float64")
    return typed


def load_raw(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Read, type, and persist raw customer data as an interim Parquet file."""
    raw_path = _get_path(cfg, "raw_path")
    interim_path = _get_path(cfg, "interim_path")
    raw = read_csv_safe(raw_path)
    _validate_columns(raw)
    typed = _coerce_dtypes(raw)
    write_parquet_atomic(typed, interim_path)
    LOGGER.info("Loaded %d customer records from %s", len(typed), raw_path)
    return typed
