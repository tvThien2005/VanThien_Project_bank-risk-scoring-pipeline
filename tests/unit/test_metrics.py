"""Unit tests for task metrics with small known-answer examples."""

from __future__ import annotations

import pandas as pd
import pytest

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
from src.models.train_credit import train_credit_baselines
from src.models.train_fraud import train_fraud_baselines


def test_classification_metrics_for_perfect_ordered_predictions() -> None:
    """Perfect predictions produce perfect class and ordered metrics."""
    actual = [0, 1, 2, 3]
    predicted = [0, 1, 2, 3]

    assert macro_f1(actual, predicted) == pytest.approx(1.0)
    assert qwk(actual, predicted) == pytest.approx(1.0)
    assert per_class_prf(actual, predicted)["class_2_recall"] == pytest.approx(1.0)
    assert classification_confusion_matrix(actual, predicted).tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_regression_metrics_and_top_k_definition() -> None:
    """Regression metrics and top-k alerts match a hand-calculated example."""
    actual = [0.0, 1.0, 2.0, 3.0]
    predicted = [0.0, 2.0, 2.0, 3.0]

    assert rmse(actual, predicted) == pytest.approx(0.5)
    assert mae(actual, predicted) == pytest.approx(0.25)
    assert r2(actual, predicted) == pytest.approx(0.8)
    assert spearman(actual, predicted) == pytest.approx(0.9486832981)

    top_k = precision_recall_at_k(actual, [0.2, 0.9, 0.1, 0.8], k=0.5)
    assert top_k["precision_at_k"] == pytest.approx(0.5)
    assert top_k["recall_at_k"] == pytest.approx(0.5)
    assert top_k["top_k_count"] == 2.0


def test_top_k_rejects_invalid_fraction() -> None:
    """An invalid operating percentage is rejected before scoring."""
    with pytest.raises(ValueError, match="interval"):
        precision_recall_at_k([0.1], [0.2], k=0.0)


def test_baseline_trainers_fit_small_feature_matrix_without_mlflow(tmp_path) -> None:
    """Both baseline trainers fit and evaluate a small validation-only example."""
    X_train = pd.DataFrame({"first": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "second": [1.0] * 6})
    X_valid = pd.DataFrame({"first": [0.5, 1.5, 3.5, 4.5], "second": [1.0] * 4})
    reports = {
        "dummy": str(tmp_path / "dummy.html"),
        "logistic_regression": str(tmp_path / "logistic.html"),
        "ridge": str(tmp_path / "ridge.html"),
    }
    credit_config = {
        "models": {
            "dummy": {"strategy": "most_frequent"},
            "logistic_regression": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 200,
                "solver": "lbfgs",
            },
        },
        "reports": reports,
    }
    fraud_config = {
        "models": {"dummy": {"strategy": "mean"}, "ridge": {"alpha": 1.0}},
        "evaluation": {"top_k": 0.5},
        "reports": reports,
    }

    credit = train_credit_baselines(
        X_train,
        pd.Series([0, 0, 1, 1, 2, 2]),
        X_valid,
        pd.Series([0, 1, 1, 2]),
        credit_config,
        seed=42,
        log_mlflow=False,
    )
    fraud = train_fraud_baselines(
        X_train,
        pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        X_valid,
        pd.Series([0.05, 0.15, 0.35, 0.45]),
        fraud_config,
        seed=42,
        log_mlflow=False,
    )

    assert set(credit) == {"dummy", "logistic_regression"}
    assert set(fraud) == {"dummy", "ridge"}
    assert (tmp_path / "logistic.html").is_file()
    assert (tmp_path / "ridge.html").is_file()
