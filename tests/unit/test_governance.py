"""Tests for validation-only threshold and fairness diagnostics."""

import numpy as np
import pandas as pd

from src.evaluation.fairness import compute_fairness_metrics
from src.evaluation.threshold import (
    apply_qwk_thresholds,
    expected_class_score,
    find_qwk_thresholds,
    top_k_threshold,
)


def test_credit_thresholds_preserve_order_and_do_not_lose_to_argmax() -> None:
    """Optimize monotonic ordinal cuts against the validation-only argmax policy."""
    actual = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    probabilities = np.array(
        [
            [0.70, 0.25, 0.04, 0.01],
            [0.55, 0.35, 0.08, 0.02],
            [0.42, 0.46, 0.10, 0.02],
            [0.25, 0.55, 0.17, 0.03],
            [0.08, 0.35, 0.49, 0.08],
            [0.04, 0.20, 0.56, 0.20],
            [0.02, 0.08, 0.35, 0.55],
            [0.01, 0.03, 0.20, 0.76],
        ]
    )

    result = find_qwk_thresholds(actual, probabilities, n_classes=4, candidate_quantiles=21)
    predicted = apply_qwk_thresholds(expected_class_score(probabilities), result.thresholds)

    assert len(result.thresholds) == 3
    assert np.all(np.diff(result.thresholds) > 0)
    assert set(predicted).issubset({0, 1, 2, 3})
    assert result.qwk_thresholded >= result.qwk_argmax


def test_top_k_threshold_and_fairness_tables_are_grouped() -> None:
    """Produce stable review cutoffs and Fairlearn group diagnostics."""
    assert top_k_threshold([0.1, 0.2, 0.3, 0.4], k=0.25) == 0.325
    groups, summary = compute_fairness_metrics(
        [0, 1, 2, 3, 0, 3],
        [0, 2, 2, 3, 1, 1],
        pd.DataFrame({"age_bin": ["young", "young", "mid", "mid", "old", "old"]}),
        high_risk_class=2,
    )

    assert set(groups["group"]) == {"young", "mid", "old"}
    assert {"selection_rate", "true_positive_rate", "false_positive_rate", "macro_f1"}.issubset(
        summary["metric"]
    )
