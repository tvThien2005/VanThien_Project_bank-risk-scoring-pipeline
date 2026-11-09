"""Render a bounded exploratory data-analysis report from a reproducible raw-data sample."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.clean import apply_entry_error_rules
from src.data.load import load_raw
from src.data.validate import load_data_quality_config
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


def _business_segments(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Summarize risk outcomes by interpretable, non-causal customer segments."""
    segmented = df.assign(
        high_credit_risk=(df["credit_risk"] >= 2).astype(float),
        income_band=pd.cut(df["income"], [0, 50_000, 100_000, 150_000, 250_000, float("inf")], right=False),
        credit_score_band=pd.cut(
            df["credit_score"], [300, 580, 670, 740, 851], right=False,
            labels=["300–579", "580–669", "670–739", "740–850"],
        ),
        utilization_band=pd.cut(
            df["credit_utilization"], [0, 0.3, 0.6, 1.000001], right=False,
            labels=["<30%", "30–59%", "≥60%"],
        ),
        late_payment_band=pd.cut(
            df["late_payment_months"], [-1, 0, 2, 24], labels=["0", "1–2", "3–24"],
        ),
    )

    def summarize(column: str) -> pd.DataFrame:
        return (
            segmented.groupby(column, observed=True)
            .agg(
                customers=("customer_id", "size"),
                high_credit_risk_rate=("high_credit_risk", "mean"),
                mean_credit_risk=("credit_risk", "mean"),
                mean_fraud_score=("fraud_risk_score", "mean"),
            )
            .reset_index()
        )

    return [
        ("Risk by income band", summarize("income_band")),
        ("Risk by credit-score band", summarize("credit_score_band")),
        ("Risk by credit-utilization band", summarize("utilization_band")),
        ("Risk by late-payment band", summarize("late_payment_band")),
    ]


def _business_table_html(table: pd.DataFrame) -> str:
    """Render segment rates as percentages while retaining score scales as decimals."""
    rendered = table.copy()
    rendered["high_credit_risk_rate"] = rendered["high_credit_risk_rate"].map(
        lambda value: f"{value:.2%}"
    )
    for column in ("mean_credit_risk", "mean_fraud_score"):
        rendered[column] = rendered[column].map(lambda value: f"{value:.4f}")
    return rendered.to_html(index=False)


def main() -> None:
    """Summarize all raw data or a configured reproducible sample in an EDA report."""
    config = load_config()
    settings = config["eda"]
    raw = load_raw(config)
    quality_config = load_data_quality_config(config["data_quality"]["config_path"])
    configured_sample_size = settings.get("sample_size")
    sample_size = len(raw) if configured_sample_size is None else min(int(configured_sample_size), len(raw))
    sample = raw if sample_size == len(raw) else raw.sample(
        n=sample_size, random_state=int(config["project"]["seed"])
    )
    scope = "toàn bộ" if sample_size == len(raw) else "mẫu ngẫu nhiên cố định"
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
    business_sections = [
        (title, _business_table_html(table))
        for title, table in _business_segments(apply_entry_error_rules(sample, quality_config))
    ]
    report = build_html_report(
        "Exploratory data analysis",
        [
            ("Scope", f"<p>Phân tích {scope}: {sample_size:,}/{len(raw):,} dòng raw.</p>"),
            ("Credit-risk distribution", credit_chart),
            ("Fraud-risk distribution", fraud_chart),
            ("Missingness", missingness.to_html(index=False, float_format=lambda value: f"{value:.2%}")),
            ("Numeric summary", numeric.to_html(index=False, float_format=lambda value: f"{value:.4f}")),
            ("Categorical cardinality", categorical.to_html(index=False)),
            (
                "Business-oriented segment analysis",
                "<p>Descriptive associations after entry-error cleaning; they are not causal findings and do not define a lending or fraud policy.</p>",
            ),
            *business_sections,
        ],
    )
    write_html_report(report, Path(str(settings["report_path"])))


if __name__ == "__main__":
    main()
