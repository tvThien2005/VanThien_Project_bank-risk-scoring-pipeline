"""Tests for raw dataset availability checks."""

from typing import Any

import pytest

from scripts.download_data import check_raw_data


def test_check_raw_data_explains_missing_file() -> None:
    """Refuse to download and explain a missing external raw dataset."""
    config: dict[str, Any] = {"data": {"raw_path": "does-not-exist.csv"}}

    with pytest.raises(FileNotFoundError, match="never downloads data"):
        check_raw_data(config)
