"""Evaluate baseline models and optionally render synthetic-data HTML reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    classification_confusion_matrix,
    macro_f1,
    mae,
    per_class_prf,
    precision_recall_at_k,
    qwk,
    r2,
    rmse,
    spearman,
)
from src.utils.report import build_html_report, figure_to_html, write_html_report


def _metric_table(metrics: dict[str, Any]) -> str:
    """Render scalar report metrics without serializing matrix-valued entries."""
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str, np.number)) and key != "top_k_count"
    }
    return pd.DataFrame([scalar_metrics]).T.rename(columns={0: "value"}).to_html(
        float_format=lambda value: f"{value:.6f}"
    )


def _confusion_figure(matrix: np.ndarray, labels: np.ndarray) -> str:
    """Render the classification confusion matrix as an embedded figure."""
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Dự đoán",
        ylabel="Thực tế",
        title="Confusion matrix",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    html = figure_to_html(figure)
    plt.close(figure)
    return html


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    report_path: str | Path | None = None,
    model_name: str = "classification model",
) -> dict[str, Any]:
    """Return credit metrics and write a synthetic-data report when requested."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    labels = np.unique(np.concatenate((actual, predicted)))
    matrix = classification_confusion_matrix(actual, predicted)
    metrics: dict[str, Any] = {
        "macro_f1": macro_f1(actual, predicted),
        "qwk": qwk(actual, predicted),
        **per_class_prf(actual, predicted),
        "confusion_matrix": matrix.tolist(),
    }
    if report_path is not None:
        html = build_html_report(
            f"Đánh giá credit risk — {model_name}",
            [
                ("Metrics", _metric_table(metrics)),
                ("Confusion matrix", _confusion_figure(matrix, labels)),
            ],
        )
        write_html_report(html, report_path)
    return metrics


def _regression_figure(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Render an observed-versus-predicted diagnostic figure."""
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(y_true, y_pred, alpha=0.35, s=12)
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    axis.plot([lower, upper], [lower, upper], color="tab:red", linestyle="--")
    axis.set(
        xlabel="Fraud risk score thực tế",
        ylabel="Fraud risk score dự đoán",
        title="Observed vs predicted",
    )
    figure.tight_layout()
    html = figure_to_html(figure)
    plt.close(figure)
    return html


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    top_k: float,
    report_path: str | Path | None = None,
    model_name: str = "regression model",
) -> dict[str, Any]:
    """Return fraud metrics and write a synthetic-data report when requested."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    metrics: dict[str, Any] = {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "r2": r2(actual, predicted),
        "spearman": spearman(actual, predicted),
        **precision_recall_at_k(actual, predicted, top_k),
    }
    if report_path is not None:
        html = build_html_report(
            f"Đánh giá fraud risk — {model_name}",
            [
                ("Metrics", _metric_table(metrics)),
                ("Observed vs predicted", _regression_figure(actual, predicted)),
            ],
        )
        write_html_report(html, report_path)
    return metrics


def evaluate_by_segment(
    y_true: Sequence[int] | Sequence[float],
    y_pred: Sequence[int] | Sequence[float],
    segments: pd.DataFrame,
    *,
    task: str,
    top_k: float | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute validation metrics independently for each configured customer segment."""
    if len(y_true) != len(y_pred) or len(y_true) != len(segments):
        raise ValueError("Segment frame and prediction arrays must have equal row counts.")
    if task not in {"credit_risk", "fraud_risk_score"}:
        raise ValueError("task must be credit_risk or fraud_risk_score.")
    if task == "fraud_risk_score" and top_k is None:
        raise ValueError("top_k is required for fraud segment evaluation.")
    true_series = pd.Series(y_true, index=segments.index)
    prediction_series = pd.Series(y_pred, index=segments.index)
    rows: list[dict[str, Any]] = []
    for segment_name in segments.columns:
        grouped = segments[segment_name].astype("string").fillna("<missing>")
        for value, indexes in grouped.groupby(grouped, observed=True).groups.items():
            actual = true_series.loc[indexes]
            predicted = prediction_series.loc[indexes]
            metrics = (
                evaluate_classification(actual, predicted)
                if task == "credit_risk"
                else evaluate_regression(actual, predicted, top_k=float(top_k))
            )
            rows.append(
                {
                    "segment": segment_name,
                    "value": value,
                    "rows": len(indexes),
                    **{key: metric for key, metric in metrics.items() if key != "confusion_matrix"},
                }
            )
    result = pd.DataFrame(rows)
    if report_path is not None:
        write_html_report(
            build_html_report(
                f"Đánh giá theo phân khúc — {task}",
                [("Validation segment metrics", result.to_html(index=False, float_format=lambda value: f"{value:.6f}"))],
            ),
            report_path,
        )
    return result
