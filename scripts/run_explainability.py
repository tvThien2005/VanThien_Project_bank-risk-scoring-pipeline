"""Create bounded SHAP reports for the selected LightGBM credit and fraud bundles."""

from __future__ import annotations

from pathlib import Path

from src.explainability.shap_explain import create_shap_report
from src.models.predict import load_model_bundle
from src.utils.config import load_config
from src.utils.io import read_parquet_safe


def main() -> None:
    """Explain the validation sample without reading labels or test data."""
    config = load_config()
    governance = config["governance"]
    X_valid = read_parquet_safe(Path(str(config["features"]["output_dir"])) / "X_valid.parquet")
    common = {
        "X": X_valid,
        "sample_size": int(governance["shap_sample_size"]),
        "waterfall_cases": int(governance["shap_waterfall_cases"]),
        "seed": int(config["project"]["seed"]),
    }
    credit = load_model_bundle("v1.0.0", "credit_risk")
    create_shap_report(
        credit.model, task="credit_risk", report_path=governance["reports"]["credit_shap"], **common
    )
    fraud = load_model_bundle("v1.0.0", "fraud_risk_score")
    create_shap_report(
        fraud.model, task="fraud_risk_score", report_path=governance["reports"]["fraud_shap"], **common
    )


if __name__ == "__main__":
    main()
