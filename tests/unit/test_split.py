"""Tests for the single deterministic stratified train/valid/test split."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import make_splits


def _split_config(tmp_path: Path) -> dict[str, Any]:
    """Create an isolated split configuration with temporary ID artifact paths."""
    return {
        "project": {"seed": 42},
        "splits": {
            "train_size": 0.70,
            "valid_size": 0.15,
            "test_size": 0.15,
            "stratify_column": "credit_risk",
            "id_paths": {
                "train": str(tmp_path / "train_ids.npy"),
                "valid": str(tmp_path / "valid_ids.npy"),
                "test": str(tmp_path / "test_ids.npy"),
            },
        },
    }


def _balanced_customers() -> pd.DataFrame:
    """Create enough examples for exact 70/15/15 stratification across four classes."""
    credit_risk = np.repeat(np.arange(4), 500)
    return pd.DataFrame(
        {
            "customer_id": [f"FN{index:07d}" for index in range(len(credit_risk))],
            "credit_risk": credit_risk,
            "fraud_risk_score": np.linspace(0.1, 1.0, len(credit_risk)),
        }
    )


def test_make_splits_is_complete_stratified_and_reproducible(tmp_path: Path) -> None:
    """Persist one non-overlapping 70/15/15 split with matching class proportions."""
    customers = _balanced_customers()
    config = _split_config(tmp_path)

    train, valid, test = make_splits(customers, config)
    second_train, second_valid, second_test = make_splits(customers, config)

    all_ids = set(train["customer_id"]) | set(valid["customer_id"]) | set(test["customer_id"])
    assert len(train) == 1400
    assert len(valid) == 300
    assert len(test) == 300
    assert set(train["customer_id"]).isdisjoint(valid["customer_id"])
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
    assert set(valid["customer_id"]).isdisjoint(test["customer_id"])
    assert all_ids == set(customers["customer_id"])
    assert train["customer_id"].tolist() == second_train["customer_id"].tolist()
    assert valid["customer_id"].tolist() == second_valid["customer_id"].tolist()
    assert test["customer_id"].tolist() == second_test["customer_id"].tolist()
    expected_rates = customers["credit_risk"].value_counts(normalize=True).sort_index()
    for split in (train, valid, test):
        actual_rates = split["credit_risk"].value_counts(normalize=True).sort_index()
        assert (actual_rates - expected_rates).abs().max() < 0.01
    for split_name in ("train", "valid", "test"):
        assert Path(config["splits"]["id_paths"][split_name]).is_file()
