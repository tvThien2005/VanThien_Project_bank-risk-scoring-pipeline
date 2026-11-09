"""Render a bounded exploratory data-analysis report from a reproducible raw-data sample."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.load import load_raw
from src.utils.config import load_config
from src.utils.report import build_html_report, figure_to_html, write_html_report


def _target_figures(sample: pd.DataFrame) -> tuple[str, str]:
    """Create compact class-distribution and fraud-score diagnostic charts."""
    credit_figure, credit_axis = plt.subplots(figsize=(7, 4))
    sample["credit_risk"].value_counts().sort_index().plot.bar(ax=credit_axis, color="#2563eb")
    credit_axis.set(xlabel="Credit-risk class", ylabel="Customers", title="Credit-risk distribution")
    credit_figure.tight_layout()
    credit_html = figure_to_html(credit_figure)
    plt.close(credit_figure)

    fraud_figure, fraud_axis = plt.subplots(figsize=(7, 4))
    fraud_axis.hist(sample["fraud_risk_score"].dropna(), bins=40, color="#d97706")
    fraud_axis.set(xlabel="Fraud-risk score", ylabel="Customers", title="Fraud-risk score distribution")
    fraud_figure.tight_layout()
    fraud_html = figure_to_html(fraud_figure)
    plt.close(fraud_figure)
    return credit_html, fraud_html


def main() -> None:
    """Sample at most 50k rows, summarize data, and write a synthetic EDA report."""
    config = load_config()
    settings = config["eda"]
    raw = load_raw(config)
    sample_size = min(int(settings["sample_size"]), len(raw))
    sample = raw.sample(n=sample_size, random_state=int(config["project"]["seed"]))
    missingness = (
        sample.isna()
        .mean()
        .rename_axis("column")
        .reset_index(name="missing_rate")
        .sort_values("missing_rate", ascending=False)
    )
    numeric = sample.select_dtypes(include="number").describe().T.reset_index(names="feature")
    categorical = pd.DataFrame(
        {
            "feature": sample.select_dtypes(include=["category", "string", "object"]).columns,
            "unique_values": [sample[column].nunique(dropna=True) for column in sample.select_dtypes(include=["category", "string", "object"]).columns],
        }
    )
    credit_chart, fraud_chart = _target_figures(sample)
    report = build_html_report(
        "Exploratory data analysis",
        [
            ("Sample", f"<p>Phân tích {sample_size:,}/{len(raw):,} dòng raw với seed cố định.</p>"),
            ("Credit-risk distribution", credit_chart),
            ("Fraud-risk distribution", fraud_chart),
            ("Missingness", missingness.to_html(index=False, float_format=lambda value: f"{value:.2%}")),
            ("Numeric summary", numeric.to_html(index=False, float_format=lambda value: f"{value:.4f}")),
            ("Categorical cardinality", categorical.to_html(index=False)),
        ],
    )
    write_html_report(report, Path(str(settings["report_path"])))


if __name__ == "__main__":
    main()
