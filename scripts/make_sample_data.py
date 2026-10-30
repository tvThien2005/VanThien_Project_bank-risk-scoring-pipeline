"""Create a deterministic 50-row synthetic data-quality fixture.

The fixture uses the schema of the raw customer file.  ``employment_years`` and
``investment_value`` contain the two intended nullable values.  It also contains
one invalid value each for credit score, age, credit utilization, date ordering,
and income so later data-quality tests have known violations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_settings(config_path: Path) -> tuple[int, Path, int]:
    """Load seed, output path, and row count from the shared configuration."""
    with config_path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    return (
        int(config["project"]["seed"]),
        PROJECT_ROOT / config["sample_data"]["output_path"],
        int(config["sample_data"]["rows"]),
    )


def make_sample_data(seed: int, rows: int) -> pd.DataFrame:
    """Build a 50-row dataframe matching the documented raw-data schema."""
    if rows < 6:
        msg = "rows must be at least 6 to include all intended quality cases"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    creation_dates = pd.Timestamp("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 1_800, size=rows), unit="D"
    )
    transaction_dates = creation_dates + pd.to_timedelta(
        rng.integers(1, 366, size=rows), unit="D"
    )
    frame = pd.DataFrame(
        {
            "customer_id": [f"FN{index:07d}" for index in range(1, rows + 1)],
            "age": rng.integers(18, 86, size=rows),
            "income": rng.uniform(20_000, 180_000, size=rows).round(2),
            "credit_score": rng.integers(300, 851, size=rows),
            "employment_years": rng.uniform(0, 40, size=rows).round(1),
            "employment_status": rng.choice(
                ["Unemployed", "Part-time", "Self-employed", "Full-time", "Retired"],
                size=rows,
            ),
            "education_level": rng.choice(
                ["High School", "Some College", "Bachelor", "Master", "PhD"], size=rows
            ),
            "marital_status": rng.choice(
                ["Single", "Married", "Divorced", "Widowed"], size=rows
            ),
            "home_ownership": rng.choice(["Rent", "Mortgage", "Own", "Other"], size=rows),
            "region": rng.choice(
                ["Northeast", "Southeast", "Midwest", "Southwest", "West"], size=rows
            ),
            "account_age_months": rng.integers(1, 361, size=rows),
            "monthly_spending": rng.uniform(0, 8_000, size=rows).round(2),
            "credit_utilization": rng.uniform(0, 1, size=rows).round(3),
            "num_credit_cards": rng.integers(0, 11, size=rows),
            "num_loans": rng.integers(0, 6, size=rows),
            "late_payment_months": rng.integers(0, 25, size=rows),
            "debt_to_income": rng.uniform(0, 1.5, size=rows).round(3),
            "savings_rate": rng.uniform(0, 1, size=rows).round(3),
            "investment_value": rng.uniform(0, 500_000, size=rows).round(2),
            "account_creation_date": creation_dates.strftime("%Y-%m-%d"),
            "last_transaction_date": transaction_dates.strftime("%Y-%m-%d"),
            "credit_risk": rng.integers(0, 4, size=rows),
            "fraud_risk_score": rng.uniform(0, 1, size=rows).round(3),
        }
    )
    frame.loc[0, "employment_years"] = np.nan
    frame.loc[1, "investment_value"] = np.nan
    frame.loc[2, "credit_score"] = 999
    frame.loc[3, "age"] = 200
    frame.loc[4, "credit_utilization"] = 1.2
    frame.loc[5, "last_transaction_date"] = "2018-12-31"
    frame.loc[6, "income"] = -1_000.0
    return frame


def main() -> None:
    """Write the committed sample fixture to ``tests/data/sample.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed, output_path, rows = load_settings(CONFIG_PATH)
    sample = make_sample_data(seed, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(sample), output_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
