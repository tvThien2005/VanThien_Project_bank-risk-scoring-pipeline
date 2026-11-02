"""Rule-based validation for raw customer data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
VALID_ACTIONS = {"to_nan", "flag"}
VALID_RULES = {"in_range", "regex", "relation"}
VALID_RELATION_OPERATORS = {"gte", "lte"}


def load_data_quality_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the standalone data-quality rules YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Data-quality config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
        raise TypeError("Data-quality config rules must be a list.")
    names: set[str] = set()
    for rule in config["rules"]:
        _validate_rule_definition(rule, names)
    return config


def _validate_rule_definition(rule: Any, names: set[str] | None = None) -> None:
    """Validate a rule definition before it is evaluated against a dataframe."""
    if not isinstance(rule, Mapping):
        raise TypeError("Each data-quality rule must be a mapping.")
    required = {"name", "column", "rule", "action"}
    missing = sorted(required - set(rule))
    if missing:
        raise ValueError(f"Rule is missing fields: {missing}")
    name = str(rule["name"])
    if names is not None:
        if name in names:
            raise ValueError(f"Duplicate data-quality rule name: {name}")
        names.add(name)
    rule_type = rule["rule"]
    if rule_type not in VALID_RULES:
        raise ValueError(f"Unsupported rule type: {rule_type}")
    if rule["action"] not in VALID_ACTIONS:
        raise ValueError(f"Unsupported rule action: {rule['action']}")
    if rule_type == "in_range" and "min" not in rule and "max" not in rule:
        raise ValueError(f"Range rule {name} needs min and/or max.")
    if rule_type == "regex" and "pattern" not in rule:
        raise ValueError(f"Regex rule {name} needs pattern.")
    if rule_type == "relation":
        if "other_column" not in rule:
            raise ValueError(f"Relation rule {name} needs other_column.")
        if rule.get("operator") not in VALID_RELATION_OPERATORS:
            raise ValueError(f"Relation rule {name} needs a supported operator.")


def rule_violation_mask(df: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Series:
    """Return a boolean mask for non-null values violating one configured rule."""
    _validate_rule_definition(rule)
    column = str(rule["column"])
    if column not in df:
        raise ValueError(f"Rule {rule['name']} references missing column: {column}")
    values = df[column]
    rule_type = rule["rule"]
    if rule_type == "in_range":
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = pd.Series(False, index=df.index)
        if "min" in rule:
            invalid |= numeric < rule["min"]
        if "max" in rule:
            invalid |= numeric > rule["max"]
    elif rule_type == "regex":
        invalid = ~values.astype("string").str.fullmatch(str(rule["pattern"]), na=False)
    else:
        other_column = str(rule["other_column"])
        if other_column not in df:
            raise ValueError(
                f"Rule {rule['name']} references missing column: {other_column}"
            )
        other_values = df[other_column]
        if rule["operator"] == "gte":
            invalid = values < other_values
        else:
            invalid = values > other_values
    return values.notna() & invalid.fillna(False)


def run_rules(df: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, dict[str, int | float]]:
    """Return violation count and rate for every configured data-quality rule."""
    rules = cfg.get("rules")
    if not isinstance(rules, list):
        raise TypeError("Data-quality config rules must be a list.")
    total_rows = len(df)
    results: dict[str, dict[str, int | float]] = {}
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise TypeError("Each data-quality rule must be a mapping.")
        mask = rule_violation_mask(df, rule)
        count = int(mask.sum())
        name = str(rule["name"])
        results[name] = {
            "count": count,
            "rate": count / total_rows if total_rows else 0.0,
        }
        LOGGER.info("Rule %s found %d violations", name, count)
    return results
