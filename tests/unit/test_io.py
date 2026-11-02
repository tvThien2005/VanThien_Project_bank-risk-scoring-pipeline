"""Tests for safe dataframe file input/output."""

from pathlib import Path

import pandas as pd

from src.utils.io import read_csv_safe, read_parquet_safe, write_parquet_atomic


def test_csv_and_atomic_parquet_round_trip(tmp_path: Path) -> None:
    """Read CSV and round-trip its rows through an atomically written Parquet file."""
    csv_path = tmp_path / "customers.csv"
    parquet_path = tmp_path / "customers.parquet"
    expected = pd.DataFrame({"customer_id": ["FN0000001"], "income": [100.0]})
    expected.to_csv(csv_path, index=False)

    loaded_csv = read_csv_safe(csv_path)
    write_parquet_atomic(loaded_csv, parquet_path)

    pd.testing.assert_frame_equal(read_parquet_safe(parquet_path), expected)
