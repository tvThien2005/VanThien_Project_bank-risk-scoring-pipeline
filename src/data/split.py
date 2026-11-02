"""Deterministic, single train/validation/test splitting for both tasks."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
SPLIT_NAMES = ("train", "valid", "test")


def _split_settings(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return the split-specific section of the main config."""
    settings = cfg.get("splits")
    if not isinstance(settings, Mapping):
        raise TypeError("splits configuration must be a mapping.")
    required = {"train_size", "valid_size", "test_size", "stratify_column", "id_paths"}
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing split configuration keys: {missing}")
    id_paths = settings["id_paths"]
    if not isinstance(id_paths, Mapping):
        raise TypeError("splits.id_paths must be a mapping.")
    path_missing = sorted(set(SPLIT_NAMES) - set(id_paths))
    if path_missing:
        raise ValueError(f"Missing split ID paths: {path_missing}")
    total = sum(float(settings[name]) for name in ("train_size", "valid_size", "test_size"))
    if not isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("train_size, valid_size, and test_size must sum to 1.0.")
    if any(float(settings[name]) <= 0 for name in ("train_size", "valid_size", "test_size")):
        raise ValueError("Each split size must be positive.")
    return settings


def _id_paths(settings: Mapping[str, Any]) -> dict[str, Path]:
    """Return configured split-ID paths as Path objects."""
    id_paths = settings["id_paths"]
    if not isinstance(id_paths, Mapping):
        raise TypeError("splits.id_paths must be a mapping.")
    return {name: Path(str(id_paths[name])) for name in SPLIT_NAMES}


def _validate_input(df: pd.DataFrame, stratify_column: str) -> None:
    """Validate columns, identifiers, and class support before splitting."""
    for column in ("customer_id", stratify_column):
        if column not in df:
            raise ValueError(f"Dataframe is missing required split column: {column}")
    if df["customer_id"].isna().any() or not df["customer_id"].is_unique:
        raise ValueError("customer_id must be non-null and unique before splitting.")
    if df[stratify_column].isna().any():
        raise ValueError(f"{stratify_column} cannot contain missing values for stratification.")
    if df[stratify_column].nunique() < 2:
        raise ValueError(f"{stratify_column} needs at least two classes for stratification.")


def _frames_from_ids(df: pd.DataFrame, split_ids: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, ...]:
    """Restore split dataframes in the saved identifier order."""
    indexed = df.assign(customer_id=df["customer_id"].astype(str)).set_index(
        "customer_id", drop=False
    )
    expected_ids = set(indexed.index)
    found_ids = {identifier for ids in split_ids.values() for identifier in ids.astype(str)}
    if expected_ids != found_ids:
        raise ValueError("Saved split IDs do not match the supplied dataframe customer IDs.")
    if sum(len(ids) for ids in split_ids.values()) != len(found_ids):
        raise ValueError("Saved split IDs overlap across train, valid, and test.")
    return tuple(indexed.loc[split_ids[name].astype(str)].reset_index(drop=True) for name in SPLIT_NAMES)


def _load_existing_splits(
    df: pd.DataFrame, paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, ...] | None:
    """Load a complete existing split so data are never repartitioned silently."""
    exists = {name: path.is_file() for name, path in paths.items()}
    if any(exists.values()) and not all(exists.values()):
        raise FileExistsError("Split ID files are incomplete; restore or remove all three together.")
    if not all(exists.values()):
        return None
    split_ids = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    LOGGER.info("Loaded existing train/valid/test split IDs from configured paths")
    return _frames_from_ids(df, split_ids)


def _save_split_ids(splits: Mapping[str, pd.DataFrame], paths: Mapping[str, Path]) -> None:
    """Persist customer identifiers for the one shared split across both tasks."""
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        customer_ids = splits[name]["customer_id"].astype(str).to_numpy(dtype=str)
        np.save(path, customer_ids)
        LOGGER.info("Saved %d %s split IDs to %s", len(customer_ids), name, path)


def make_splits(
    df: pd.DataFrame, cfg: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create or reload one 70/15/15 stratified split shared by both tasks."""
    settings = _split_settings(cfg)
    stratify_column = str(settings["stratify_column"])
    _validate_input(df, stratify_column)
    paths = _id_paths(settings)
    existing = _load_existing_splits(df, paths)
    if existing is not None:
        return existing

    seed_config = cfg.get("project")
    if not isinstance(seed_config, Mapping) or "seed" not in seed_config:
        raise ValueError("Missing project.seed in configuration.")
    seed = int(seed_config["seed"])
    train_size = float(settings["train_size"])
    valid_size = float(settings["valid_size"])
    test_size = float(settings["test_size"])
    train, holdout = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df[stratify_column],
    )
    valid_share_of_holdout = valid_size / (valid_size + test_size)
    valid, test = train_test_split(
        holdout,
        train_size=valid_share_of_holdout,
        random_state=seed,
        stratify=holdout[stratify_column],
    )
    splits = {"train": train, "valid": valid, "test": test}
    _save_split_ids(splits, paths)
    LOGGER.info(
        "Created split with sizes train=%d, valid=%d, test=%d",
        len(train),
        len(valid),
        len(test),
    )
    return train, valid, test
