"""Run configured data-quality checks and render an HTML report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.clean import add_missing_flags, apply_entry_error_rules, handle_outliers
from src.data.load import load_raw
from src.data.validate import load_data_quality_config, run_rules
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.report import build_html_report, figure_to_html, write_html_report

LOGGER = get_logger(__name__)


def _data_quality_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the report settings needed by this script from the main config."""
    settings = config.get("data_quality")
    if not isinstance(settings, Mapping):
        raise TypeError("data_quality settings must be a mapping.")
    required = {"config_path", "report_path", "income_bins"}
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing data_quality settings: {missing}")
    return settings


def _rule_results_frame(results: Mapping[str, Mapping[str, int | float]]) -> pd.DataFrame:
    """Convert rule results into a report-ready sorted dataframe."""
    frame = pd.DataFrame.from_dict(results, orient="index").reset_index(names="rule")
    return frame.sort_values("count", ascending=False, ignore_index=True)


def _missingness_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing counts and rates without imputing any values."""
    missing_count = df.isna().sum()
    frame = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_rate": (missing_count / len(df) if len(df) else 0.0).values,
        }
    )
    return frame.sort_values("missing_count", ascending=False, ignore_index=True)


def _missingness_by_group(
    df: pd.DataFrame, group_column: str, missing_columns: list[str]
) -> pd.DataFrame:
    """Summarize missingness rates by one configured diagnostic grouping."""
    rates = df[missing_columns].isna().astype(float)
    grouped = rates.groupby(df[group_column], observed=True).mean().reset_index()
    return grouped


def _missingness_observations(
    df: pd.DataFrame, missing_columns: list[str]
) -> str:
    """Summarize the largest group-level missingness gaps as an HTML list."""
    observations: list[str] = []
    for group_column in ("employment_status", "income_bin", "credit_risk"):
        grouped = _missingness_by_group(df, group_column, missing_columns)
        rates = grouped[missing_columns]
        gaps = rates.max() - rates.min()
        largest_gap = gaps.max()
        if largest_gap > 0:
            column = str(gaps.idxmax())
            observations.append(
                f"<li><strong>{group_column}</strong>: missing <code>{column}</code> "
                f"chênh {largest_gap:.2%} giữa các nhóm.</li>"
            )
    if not observations:
        return "<p>Không có missingness để so sánh theo nhóm.</p>"
    return (
        "<p>Các chênh lệch dưới đây cho thấy missingness không hoàn toàn ngẫu nhiên; "
        "không được suy diễn quan hệ nhân quả từ dữ liệu synthetic.</p><ul>"
        f"{''.join(observations)}</ul>"
    )


def _violation_chart(results: pd.DataFrame) -> str:
    """Render a bar chart of configured validation-rule violations."""
    figure, axis = plt.subplots(figsize=(10, 7))
    ordered = results.sort_values("count")
    axis.barh(ordered["rule"], ordered["count"], color="#2563eb")
    axis.set_title("Số vi phạm theo rule")
    axis.set_xlabel("Số dòng")
    figure.tight_layout()
    html = figure_to_html(figure)
    plt.close(figure)
    return html


def _missingness_chart(missingness: pd.DataFrame) -> str:
    """Render a bar chart of up to ten columns with the most missing values."""
    figure, axis = plt.subplots(figsize=(10, 5))
    top = missingness.head(10).sort_values("missing_count")
    axis.barh(top["column"], top["missing_count"], color="#9333ea")
    axis.set_title("Mười cột có missing nhiều nhất")
    axis.set_xlabel("Số dòng thiếu")
    figure.tight_layout()
    html = figure_to_html(figure)
    plt.close(figure)
    return html


def main() -> None:
    """Run rules, diagnose missingness, and write the synthetic-data report."""
    config = load_config()
    settings = _data_quality_settings(config)
    rules_config = load_data_quality_config(Path(str(settings["config_path"])))
    raw = load_raw(config)
    violations = run_rules(raw, rules_config)
    cleaned = apply_entry_error_rules(raw, rules_config)
    cleaned_with_outliers = handle_outliers(cleaned, rules_config)
    final = add_missing_flags(cleaned_with_outliers)
    post_clean = run_rules(cleaned, rules_config)

    violation_frame = _rule_results_frame(violations)
    missingness = _missingness_frame(raw)
    missing_columns = missingness.loc[missingness["missing_count"] > 0, "column"].tolist()
    income_bins = [*settings["income_bins"], float("inf")]
    diagnostic = raw.assign(income_bin=pd.cut(raw["income"], bins=income_bins, right=False))
    group_sections: list[tuple[str, str]] = []
    if missing_columns:
        for group_column in ("employment_status", "income_bin", "credit_risk"):
            grouped = _missingness_by_group(diagnostic, group_column, missing_columns)
            group_sections.append(
                (
                    f"Missingness theo {group_column}",
                    grouped.to_html(index=False, float_format="{:.2%}".format),
                )
            )

    remaining = _rule_results_frame(post_clean)
    entry_rule_names = {
        str(rule["name"])
        for rule in rules_config["rules"]
        if rule["action"] == "to_nan"
    }
    remaining_entry_errors = remaining.loc[
        (remaining["rule"].isin(entry_rule_names)) & (remaining["count"] > 0), "rule"
    ].tolist()
    if remaining_entry_errors:
        raise ValueError(f"Entry-error rules still have violations: {remaining_entry_errors}")
    sections = [
        ("Vi phạm trước khi làm sạch", violation_frame.to_html(index=False, float_format="{:.4%}".format)),
        ("Biểu đồ vi phạm", _violation_chart(violation_frame)),
        ("Missingness tổng quan", missingness.to_html(index=False, float_format="{:.4%}".format)),
        ("Biểu đồ missingness", _missingness_chart(missingness)),
        (
            "Kiểm tra sau làm sạch",
            "<p>Không còn vi phạm entry-error.</p>",
        ),
        (
            "Kết quả cờ dữ liệu",
            f"<p>Dataset sau làm sạch có {len(final.columns)} cột, gồm các cờ outlier và missingness.</p>",
        ),
        ("Nhận định về missingness", _missingness_observations(diagnostic, missing_columns)),
        *group_sections,
    ]
    report = build_html_report("Báo cáo chất lượng dữ liệu", sections)
    write_html_report(report, Path(str(settings["report_path"])))
    LOGGER.info("Data-quality workflow completed for %d rows", len(raw))


if __name__ == "__main__":
    main()
