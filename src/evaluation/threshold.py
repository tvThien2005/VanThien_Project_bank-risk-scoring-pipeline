"""Validation-only decision policies for ordered credit and fraud-risk outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import precision_recall_at_k, qwk


@dataclass(frozen=True)
class CreditThresholdResult:
    """Thresholds fitted on validation labels and their QWK comparison with argmax."""

    thresholds: tuple[float, ...]
    qwk_thresholded: float
    qwk_argmax: float


def expected_class_score(y_proba: np.ndarray) -> np.ndarray:
    """Return the ordinal expected class score from multiclass probabilities."""
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("y_proba must be a two-dimensional probability matrix with >=2 classes.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each predicted probability row must sum to one.")
    return probabilities @ np.arange(probabilities.shape[1], dtype=float)


def apply_qwk_thresholds(scores: Sequence[float], thresholds: Sequence[float]) -> np.ndarray:
    """Map continuous ordinal expected scores to ordered class labels."""
    values = np.asarray(scores, dtype=float)
    cuts = np.asarray(thresholds, dtype=float)
    if cuts.ndim != 1 or len(cuts) == 0 or np.any(np.diff(cuts) <= 0):
        raise ValueError("thresholds must be a strictly increasing, non-empty sequence.")
    return np.digitize(values, cuts, right=False).astype(int)


def _candidate_thresholds(scores: np.ndarray, count: int) -> np.ndarray:
    """Use a bounded quantile grid, keeping threshold search deterministic and fast."""
    quantiles = np.linspace(0.0, 1.0, count)
    return np.unique(np.quantile(scores, quantiles))


def find_qwk_thresholds(
    y_true: Sequence[int], y_proba: np.ndarray, n_classes: int, *, candidate_quantiles: int = 201
) -> CreditThresholdResult:
    """Coordinate-search ordered cuts on validation only to maximize quadratic kappa."""
    actual = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.shape != (len(actual), n_classes):
        raise ValueError("y_proba shape must be (len(y_true), n_classes).")
    if n_classes < 2 or candidate_quantiles < 3:
        raise ValueError("n_classes must be >=2 and candidate_quantiles must be >=3.")
    scores = expected_class_score(probabilities)
    candidates = _candidate_thresholds(scores, candidate_quantiles)
    if len(candidates) < n_classes:
        raise ValueError("Predicted scores do not contain enough variation to fit thresholds.")
    thresholds = np.linspace(0.5, n_classes - 1.5, n_classes - 1)
    best = float(qwk(actual, apply_qwk_thresholds(scores, thresholds)))
    epsilon = np.finfo(float).eps
    for _ in range(10):
        improved = False
        for index in range(len(thresholds)):
            lower = thresholds[index - 1] + epsilon if index else -np.inf
            upper = thresholds[index + 1] - epsilon if index < len(thresholds) - 1 else np.inf
            eligible = candidates[(candidates > lower) & (candidates < upper)]
            for candidate in eligible:
                trial = thresholds.copy()
                trial[index] = candidate
                score = float(qwk(actual, apply_qwk_thresholds(scores, trial)))
                if score > best:
                    thresholds, best, improved = trial, score, True
        if not improved:
            break
    return CreditThresholdResult(
        thresholds=tuple(float(value) for value in thresholds),
        qwk_thresholded=best,
        qwk_argmax=float(qwk(actual, probabilities.argmax(axis=1))),
    )


def top_k_threshold(scores: Sequence[float], k: float = 0.05) -> float:
    """Return the predicted-score cutoff corresponding to the top ``k`` fraction."""
    if not 0 < k < 1:
        raise ValueError("k must be strictly between zero and one.")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty, finite one-dimensional sequence.")
    return float(np.quantile(values, 1 - k))


def fraud_top_k_metrics(
    y_true: Sequence[float], scores: Sequence[float], k: float = 0.05
) -> dict[str, float]:
    """Report the valid-selected fraud cutoff alongside the existing exact top-k metric."""
    threshold = top_k_threshold(scores, k)
    return {"top_k_threshold": threshold, **precision_recall_at_k(y_true, scores, k)}
