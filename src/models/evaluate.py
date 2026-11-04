"""Evaluate baseline models and optionally render synthetic-data HTML reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
