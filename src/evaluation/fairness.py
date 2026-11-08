"""Group-level fairness diagnostics for credit-risk decision support."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import f1_score


def compute_fairness_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    sensitive_cols: pd.DataFrame,
    *,
    high_risk_class: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute group selection, error-rate, and F1 diagnostics with Fairlearn MetricFrame."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if len(actual) != len(predicted) or len(actual) != len(sensitive_cols):
        raise ValueError("y_true, y_pred, and sensitive_cols must have identical row counts.")
    if sensitive_cols.empty:
        raise ValueError("At least one sensitive column is required.")
    actual_high = (actual >= high_risk_class).astype(int)
    predicted_high = (predicted >= high_risk_class).astype(int)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }
    for column in sensitive_cols.columns:
        groups = sensitive_cols[column].astype("string").fillna("<missing>")
        frame = MetricFrame(
            metrics=metrics,
            y_true=actual_high,
            y_pred=predicted_high,
            sensitive_features=groups,
        ).by_group.reset_index().rename(columns={column: "group"})
        f1_by_group = {
            group: f1_score(actual[indexes], predicted[indexes], average="macro", zero_division=0)
            for group, indexes in groups.groupby(groups, observed=True).groups.items()
        }
        frame["macro_f1"] = frame["group"].map(f1_by_group)
        frame.insert(0, "sensitive_feature", column)
        frame["rows"] = frame["group"].map(groups.value_counts(dropna=False)).astype(int)
        rows.append(frame)
        for metric in ("selection_rate", "true_positive_rate", "false_positive_rate", "macro_f1"):
            values = frame[metric].to_numpy(dtype=float)
            minimum, maximum = float(values.min()), float(values.max())
            summaries.append(
                {
                    "sensitive_feature": column,
                    "metric": metric,
                    "min_group_value": minimum,
                    "max_group_value": maximum,
                    "difference": maximum - minimum,
                    "ratio": minimum / maximum if maximum > 0 else np.nan,
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)
