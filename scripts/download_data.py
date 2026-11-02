"""Check that the externally supplied raw dataset is available locally."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def check_raw_data(cfg: Mapping[str, Any]) -> Path:
    """Return the raw data path or raise if a user must place it locally."""
    data_config = cfg.get("data")
    if not isinstance(data_config, Mapping) or "raw_path" not in data_config:
        raise ValueError("Missing data.raw_path in configuration.")
    raw_path = Path(str(data_config["raw_path"]))
    if not raw_path.is_file():
        message = (
            f"Raw dataset not found at {raw_path}. Place the supplied CSV there or "
            "update data.raw_path in configs/config.yaml. This script never downloads data."
        )
        LOGGER.error(message)
        raise FileNotFoundError(message)
    LOGGER.info("Raw dataset is available: %s", raw_path)
    return raw_path


def main() -> None:
    """Validate the configured raw dataset and exit nonzero when it is missing."""
    try:
        check_raw_data(load_config())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
