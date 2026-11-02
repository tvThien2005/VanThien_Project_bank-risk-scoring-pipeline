"""Tests for stateless entry-error cleaning and quality flags."""

from pathlib import Path

import pandas as pd

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.data.validate import load_data_quality_config, run_rules


def _sample_and_config() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the typed sample fixture and project data-quality configuration."""
    project_root = Path(__file__).resolve().parents[2]
    sample = pd.read_csv(
        project_root / "tests" / "data" / "sample.csv",
        parse_dates=["account_creation_date", "last_transaction_date"],
    )
    config = load_data_quality_config(project_root / "configs" / "data_quality.yaml")
    return sample, config


def test_entry_error_rules_replace_only_invalid_values_with_missing() -> None:
    """Replace documented invalid values, including invalid date ordering, with nulls."""
    sample, config = _sample_and_config()

    cleaned = apply_entry_error_rules(sample, config)

    assert pd.isna(cleaned.loc[2, "credit_score"])
    assert pd.isna(cleaned.loc[3, "age"])
    assert pd.isna(cleaned.loc[4, "credit_utilization"])
    assert pd.isna(cleaned.loc[5, "last_transaction_date"])
    assert pd.isna(cleaned.loc[6, "income"])
    remaining = run_rules(cleaned, config)
    entry_rule_names = {
        str(rule["name"]) for rule in config["rules"] if rule["action"] == "to_nan"
    }
    assert all(remaining[name]["count"] == 0 for name in entry_rule_names)


def test_outlier_and_missing_flags_preserve_source_values() -> None:
    """Flag fixed-threshold outliers and nulls without modifying observed values."""
    sample, config = _sample_and_config()
    sample.loc[7, "income"] = 300_000.0

    with_outliers = handle_outliers(sample, config)
    final = add_missing_flags(with_outliers)

    assert with_outliers.loc[7, "income"] == 300_000.0
    assert with_outliers.loc[7, "income_outlier_flag"]
    assert final.loc[0, "employment_years_missing_flag"]
    assert final.loc[1, "investment_value_missing_flag"]
