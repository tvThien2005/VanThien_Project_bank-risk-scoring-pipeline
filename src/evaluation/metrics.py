"""Metrics for the credit-risk classification and fraud-score regression tasks."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from math import ceil

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return macro F1, treating unsupported predicted classes as zero."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def qwk(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return quadratic weighted kappa for the ordered credit-risk classes."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def per_class_prf(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Return precision, recall, F1, and support for every observed class."""
    labels = np.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, float] = {}
    for label, class_precision, class_recall, class_f1, class_support in zip(
        labels, precision, recall, f1, support, strict=True
    ):
        numeric_label = float(label)
        label_name = str(int(numeric_label)) if numeric_label.is_integer() else str(label)
        prefix = f"class_{label_name}"
        metrics[f"{prefix}_precision"] = float(class_precision)
        metrics[f"{prefix}_recall"] = float(class_recall)
        metrics[f"{prefix}_f1"] = float(class_f1)
        metrics[f"{prefix}_support"] = float(class_support)
    return metrics


def classification_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> np.ndarray:
    """Return a confusion matrix with the same ordered labels in rows and columns."""
    labels = np.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred))))
    return sklearn_confusion_matrix(y_true, y_pred, labels=labels)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return root mean squared error for fraud-risk predictions."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean absolute error for fraud-risk predictions."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return coefficient of determination for fraud-risk predictions."""
    return float(r2_score(y_true, y_pred))


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return rank correlation, or NaN when one input is constant."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        correlation = spearmanr(y_true, y_pred).statistic
    return float(correlation)


def precision_recall_at_k(
    y_true: Sequence[float], scores: Sequence[float], k: float = 0.05
) -> dict[str, float]:
    """Measure alert quality at top ``k`` predicted scores.

    A true positive is a customer whose observed ``fraud_risk_score`` lies in the
    top ``k`` proportion of the evaluation population.  The predicted-positive set
    is the equally sized group with the highest predicted score; ties are resolved
    deterministically by original row order.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be in the interval (0, 1].")
    actual = np.asarray(y_true, dtype=float)
    predicted_scores = np.asarray(scores, dtype=float)
    if actual.ndim != 1 or predicted_scores.ndim != 1 or len(actual) != len(predicted_scores):
        raise ValueError("y_true and scores must be one-dimensional arrays of equal length.")
    if len(actual) == 0:
        raise ValueError("precision_recall_at_k requires at least one row.")

    count = ceil(len(actual) * k)
    actual_order = np.argsort(-actual, kind="mergesort")
    predicted_order = np.argsort(-predicted_scores, kind="mergesort")
    actual_positive = set(actual_order[:count].tolist())
    predicted_positive = set(predicted_order[:count].tolist())
    true_positives = len(actual_positive & predicted_positive)
    rate = true_positives / count
    return {
        "precision_at_k": float(rate),
        "recall_at_k": float(rate),
        "k": float(k),
        "top_k_count": float(count),
    }
