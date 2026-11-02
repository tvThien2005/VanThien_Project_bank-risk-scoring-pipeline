"""Configuration loading, environment overrides, and validation."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ENV_PREFIX = "CREDIT_FRAUD__"
REQUIRED_CONFIG_KEYS = (
    "project.seed",
    "data.raw_path",
    "data.interim_path",
    "targets.classification",
    "targets.regression",
)


def _parse_environment_value(value: str) -> Any:
    """Parse an environment value with YAML scalar semantics."""
    parsed = yaml.safe_load(value)
    return value if parsed is None else parsed


def apply_env_overrides(
    config: MutableMapping[str, Any],
    *,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    environment: Mapping[str, str] | None = None,
) -> MutableMapping[str, Any]:
    """Merge nested overrides such as ``CREDIT_FRAUD__DATA__RAW_PATH``."""
    source = os.environ if environment is None else environment
    for name, value in source.items():
        if not name.startswith(env_prefix):
            continue
        keys = [key.lower() for key in name.removeprefix(env_prefix).split("__") if key]
        if not keys:
            continue
        current: MutableMapping[str, Any] = config
        for key in keys[:-1]:
            nested = current.get(key)
            if not isinstance(nested, MutableMapping):
                nested = {}
                current[key] = nested
            current = nested
        current[keys[-1]] = _parse_environment_value(value)
    return config


def validate_config(
    config: Mapping[str, Any], required_keys: tuple[str, ...] = REQUIRED_CONFIG_KEYS
) -> None:
    """Raise ``ValueError`` when a required dot-delimited setting is absent."""
    missing: list[str] = []
    for dotted_key in required_keys:
        current: Any = config
        for key in dotted_key.split("."):
            if not isinstance(current, Mapping) or key not in current:
                missing.append(dotted_key)
                break
            current = current[key]
        if current is None:
            missing.append(dotted_key)
    if missing:
        message = f"Missing required configuration keys: {', '.join(sorted(set(missing)))}"
        raise ValueError(message)


def load_config(
    path: str | Path = "configs/config.yaml", *, env_prefix: str = DEFAULT_ENV_PREFIX
) -> dict[str, Any]:
    """Load YAML configuration, apply environment overrides, and validate it."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping.")
    apply_env_overrides(config, env_prefix=env_prefix)
    validate_config(config)
    return config
