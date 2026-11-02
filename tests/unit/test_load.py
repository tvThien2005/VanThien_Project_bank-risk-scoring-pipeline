"""Tests for documented raw-data schema coercion."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import load_raw


def test_load_raw_types_and_persists_sample_data(tmp_path: Path) -> None:
    """Type the committed sample fixture and save its interim Parquet output."""
    project_root = Path(__file__).resolve().parents[2]
    interim_path = tmp_path / "customers.parquet"
    config: dict[str, Any] = {
        "data": {
            "raw_path": str(project_root / "tests" / "data" / "sample.csv"),
            "interim_path": str(interim_path),
        }
    }

    customers = load_raw(config)

    assert interim_path.is_file()
    assert str(customers["customer_id"].dtype) == "string"
    assert isinstance(customers["employment_status"].dtype, pd.CategoricalDtype)
    assert str(customers["age"].dtype) == "Int64"
    assert str(customers["income"].dtype) == "Float64"
    assert pd.api.types.is_datetime64_any_dtype(customers["account_creation_date"])
