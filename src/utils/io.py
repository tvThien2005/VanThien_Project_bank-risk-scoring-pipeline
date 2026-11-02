"""Safe dataframe input/output utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def read_csv_safe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file after checking that its path is a regular file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    LOGGER.info("Reading CSV: %s", csv_path)
    return pd.read_csv(csv_path, **kwargs)


def read_parquet_safe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file after checking that its path is a regular file."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    LOGGER.info("Reading Parquet: %s", parquet_path)
    return pd.read_parquet(parquet_path, **kwargs)


def write_parquet_atomic(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write Parquet to a same-directory temporary file before replacing output."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".parquet", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        df.to_parquet(temporary_path, index=False, **kwargs)
        os.replace(temporary_path, destination)
        LOGGER.info("Wrote Parquet atomically: %s", destination)
    finally:
        temporary_path.unlink(missing_ok=True)
