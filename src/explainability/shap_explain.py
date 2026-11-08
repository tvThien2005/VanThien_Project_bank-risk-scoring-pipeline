"""Bounded SHAP explanations for fitted tree models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.report import build_html_report, figure_to_html, write_html_report


@dataclass(frozen=True)
class ShapExplanation:
    """Compact explanation output suitable for a report and automated checks."""

    importance: pd.DataFrame
    sampled_rows: int


def _shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Compute TreeSHAP values, normalizing old and new SHAP result shapes."""
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    array = np.asarray(values)
    if array.ndim == 2:
        return array[:, :, np.newaxis]
    if array.ndim == 3:
        return array
    raise ValueError("Unsupported SHAP value shape.")


def create_shap_report(
    model: Any,
    X: pd.DataFrame,
    *,
    task: str,
    report_path: str,
    sample_size: int = 10_000,
    waterfall_cases: int = 3,
    seed: int = 42,
) -> ShapExplanation:
    """Write global and per-case TreeSHAP charts using at most ``sample_size`` rows."""
    if sample_size <= 0 or waterfall_cases <= 0:
        raise ValueError("sample_size and waterfall_cases must be positive.")
    sample = X.sample(n=min(sample_size, len(X)), random_state=seed).reset_index(drop=True)
    values = _shap_values(model, sample)
    mean_absolute = np.abs(values).mean(axis=(0, 2))
    importance = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_absolute})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    global_figure, global_axis = plt.subplots(figsize=(9, 6))
    top = importance.head(20).iloc[::-1]
    global_axis.barh(top["feature"], top["mean_abs_shap"])
    global_axis.set(xlabel="mean(|SHAP|)", title=f"Global SHAP summary — {task}")
    global_figure.tight_layout()
    sections: list[tuple[str, str]] = [
        ("Global SHAP summary", figure_to_html(global_figure)),
        ("Global importance", importance.head(30).to_html(index=False, float_format=lambda value: f"{value:.6f}")),
    ]
    plt.close(global_figure)
    case_order = np.argsort(np.abs(values).sum(axis=(1, 2)))[::-1][:waterfall_cases]
    for rank, row_index in enumerate(case_order, start=1):
        contribution = values[row_index].mean(axis=1)
        order = np.argsort(np.abs(contribution))[-10:]
        case_figure, case_axis = plt.subplots(figsize=(9, 5))
        case_axis.barh(sample.columns[order], contribution[order], color=np.where(contribution[order] >= 0, "#d97706", "#2563eb"))
        case_axis.axvline(0, color="black", linewidth=0.8)
        case_axis.set(xlabel="SHAP contribution", title=f"Waterfall case {rank} — row {row_index}")
        case_figure.tight_layout()
        sections.append((f"Waterfall case {rank}", figure_to_html(case_figure)))
        plt.close(case_figure)
    write_html_report(build_html_report(f"SHAP explanation — {task}", sections), report_path)
    return ShapExplanation(importance=importance, sampled_rows=len(sample))
