"""Stateless data-cleaning transformations based on configured fixed rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data.validate import rule_violation_mask
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _get_rules(cfg: Mapping[str, Any], action: str) -> list[Mapping[str, Any]]:
    """Select rules with the requested configured action."""
    rules = cfg.get("rules")
    if not isinstance(rules, list):
        raise TypeError("Data-quality config rules must be a list.")
    return [rule for rule in rules if isinstance(rule, Mapping) and rule["action"] == action]


def apply_entry_error_rules(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Replace fixed, implausible entry errors with missing values."""
    cleaned = df.copy()
    for rule in _get_rules(cfg, "to_nan"):
        mask = rule_violation_mask(cleaned, rule)
        column = str(rule["column"])
        cleaned.loc[mask, column] = pd.NA
        LOGGER.info("Set %d invalid %s values to missing", int(mask.sum()), column)
    return cleaned


def handle_outliers(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Add boolean outlier flags without changing observed feature values."""
    flagged = df.copy()
    for rule in _get_rules(cfg, "flag"):
        column = str(rule["column"])
        flag_column = f"{column}_outlier_flag"
        mask = rule_violation_mask(flagged, rule)
        if flag_column in flagged:
            flagged[flag_column] = flagged[flag_column] | mask
        else:
            flagged[flag_column] = mask.astype(bool)
        LOGGER.info("Flagged %d outliers in %s", int(mask.sum()), column)
    return flagged


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness indicator columns for original columns containing nulls."""
    flagged = df.copy()
    source_columns = [
        column
        for column in df.columns
        if not column.endswith("_missing_flag") and not column.endswith("_outlier_flag")
    ]
    for column in source_columns:
        missing = flagged[column].isna()
        if missing.any():
            flagged[f"{column}_missing_flag"] = missing.astype(bool)
            LOGGER.info("Added missingness flag for %s", column)
    return flagged
