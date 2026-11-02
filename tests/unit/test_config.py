"""Tests for YAML loading, validation, and environment overrides."""

from pathlib import Path

import pytest

from src.utils.config import load_config, validate_config


def test_load_config_applies_nested_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment values override nested YAML settings with parsed types."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  seed: 42
data:
  raw_path: input.csv
  interim_path: output.parquet
targets:
  classification: credit_risk
  regression: fraud_risk_score
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CREDIT_FRAUD__PROJECT__SEED", "99")

    config = load_config(config_path)

    assert config["project"]["seed"] == 99


def test_validate_config_rejects_missing_required_key() -> None:
    """Configuration validation identifies absent nested settings."""
    with pytest.raises(ValueError, match="data.interim_path"):
        validate_config(
            {
                "project": {"seed": 42},
                "data": {"raw_path": "input.csv"},
                "targets": {
                    "classification": "credit_risk",
                    "regression": "fraud_risk_score",
                },
            }
        )
