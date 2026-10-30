"""Tests for the deterministic sample-data fixture generator."""

from scripts.make_sample_data import make_sample_data


def test_make_sample_data_contains_schema_and_intentional_errors() -> None:
    """Generate the expected schema and all documented quality violations."""
    sample = make_sample_data(seed=42, rows=50)

    assert sample.shape == (50, 23)
    assert sample["employment_years"].isna().iloc[0]
    assert sample["investment_value"].isna().iloc[1]
    assert sample.loc[2, "credit_score"] == 999
    assert sample.loc[3, "age"] == 200
    assert sample.loc[4, "credit_utilization"] > 1
    assert sample.loc[5, "last_transaction_date"] < sample.loc[5, "account_creation_date"]
    assert sample.loc[6, "income"] < 0
