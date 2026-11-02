"""Tests for data-quality rule execution on the committed sample fixture."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.validate import load_data_quality_config, run_rules


def test_run_rules_counts_intentional_sample_violations() -> None:
    """Count each deliberately injected entry error in sample data once."""
    project_root = Path(__file__).resolve().parents[2]
    sample = pd.read_csv(
        project_root / "tests" / "data" / "sample.csv",
        parse_dates=["account_creation_date", "last_transaction_date"],
    )
    config = load_data_quality_config(project_root / "configs" / "data_quality.yaml")

    results = run_rules(sample, config)

    assert results["age_range"]["count"] == 1
    assert results["credit_score_range"]["count"] == 1
    assert results["credit_utilization_range"]["count"] == 1
    assert results["income_nonnegative"]["count"] == 1
    assert results["transaction_after_account_creation"]["count"] == 1
    assert results["age_range"]["rate"] == pytest.approx(1 / 50)
