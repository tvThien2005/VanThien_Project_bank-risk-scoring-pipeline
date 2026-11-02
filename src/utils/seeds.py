"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set process-level random seeds for Python and NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    LOGGER.info("Set random seed to %d", seed)
