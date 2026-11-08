"""Fit validation-only credit and fraud decision thresholds; test is opt-in via --final."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.threshold import (
    apply_qwk_thresholds,
    expected_class_score,
    find_qwk_thresholds,
    fraud_top_k_metrics,
)
from src.models.evaluate import evaluate_classification, evaluate_regression
from src.models.predict import load_model_bundle
from src.pipelines.training_pipeline import TEST_EVALUATION_FILES
from src.utils.config import load_config
from src.utils.io import read_parquet_safe
from src.utils.report import build_html_report, write_html_report


def _arguments() -> argparse.Namespace:
    """Parse explicit final-test evaluation safeguards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Evaluate fixed validation policies once on test.")
    parser.add_argument("--force", action="store_true", help="Deliberately replace prior threshold test metrics.")
    return parser.parse_args()


def _target(directory: Path, name: str, split: str) -> pd.Series:
    """Read an isolated target vector from the feature artifact directory."""
    frame = read_parquet_safe(directory / f"y_{name}_{split}.parquet")
    return frame.iloc[:, 0]


def _write_once(path: Path, payload: Mapping[str, Any], force: bool) -> None:
    """Prevent repeatedly using test data unless a deliberate override is supplied."""
    if path.is_file() and not force:
        raise FileExistsError("test đã được đánh giá, dùng --force nếu thực sự cần đánh giá lại")
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")


def main() -> None:
    """Select thresholds on validation and optionally evaluate those frozen policies on test."""
    arguments = _arguments()
    if arguments.force and not arguments.final:
        raise ValueError("--force is only valid together with --final.")
    config = load_config()
    features = Path(str(config["features"]["output_dir"]))
    governance = config["governance"]
    credit = load_model_bundle("v1.0.0", "credit_risk")
    fraud = load_model_bundle("v1.0.0", "fraud_risk_score")
    if arguments.final and not arguments.force:
        for directory in (Path("models/credit_risk/v1.0.0"), Path("models/fraud_risk/v1.0.0")):
            if any((directory / name).is_file() for name in TEST_EVALUATION_FILES):
                raise FileExistsError("test đã được đánh giá, dùng --force nếu thực sự cần đánh giá lại")
    X_valid = read_parquet_safe(features / "X_valid.parquet")
    y_credit_valid = _target(features, "credit", "valid")
    y_fraud_valid = _target(features, "fraud", "valid")
    credit_proba = credit.model.predict(X_valid.to_numpy(dtype=float))
    threshold_result = find_qwk_thresholds(
        y_credit_valid,
        credit_proba,
        n_classes=len(credit.metadata["classes"]),
        candidate_quantiles=int(governance["threshold_candidate_quantiles"]),
    )
    fraud_scores = fraud.model.predict(X_valid.to_numpy(dtype=float))
    fraud_result = fraud_top_k_metrics(y_fraud_valid, fraud_scores, k=0.05)
    credit_payload = {
        "selection_split": "validation",
        "thresholds": list(threshold_result.thresholds),
        "qwk_argmax": threshold_result.qwk_argmax,
        "qwk_thresholded": threshold_result.qwk_thresholded,
    }
    fraud_payload = {"selection_split": "validation", **fraud_result}
    Path("models/credit_risk/v1.0.0/thresholds.json").write_text(
        json.dumps(credit_payload, indent=2), encoding="utf-8"
    )
    Path("models/fraud_risk/v1.0.0/thresholds.json").write_text(
        json.dumps(fraud_payload, indent=2), encoding="utf-8"
    )
    sections = [
        ("Credit threshold policy", pd.DataFrame([credit_payload]).to_html(index=False)),
        ("Fraud top-k policy", pd.DataFrame([fraud_payload]).to_html(index=False)),
    ]
    if arguments.final:
        X_test = read_parquet_safe(features / "X_test.parquet")
        y_credit_test = _target(features, "credit", "test")
        y_fraud_test = _target(features, "fraud", "test")
        credit_test_pred = apply_qwk_thresholds(
            expected_class_score(credit.model.predict(X_test.to_numpy(dtype=float))),
            threshold_result.thresholds,
        )
        credit_test = evaluate_classification(y_credit_test, credit_test_pred)
        fraud_test = evaluate_regression(
            y_fraud_test,
            fraud.model.predict(X_test.to_numpy(dtype=float)),
            top_k=0.05,
        )
        _write_once(Path("models/credit_risk/v1.0.0/threshold_test_metrics.json"), credit_test, arguments.force)
        _write_once(Path("models/fraud_risk/v1.0.0/threshold_test_metrics.json"), fraud_test, arguments.force)
        sections.extend(
            [
                ("Credit frozen-policy test metrics", pd.DataFrame([credit_test]).to_html(index=False)),
                ("Fraud frozen-policy test metrics", pd.DataFrame([fraud_test]).to_html(index=False)),
            ]
        )
    write_html_report(build_html_report("Threshold governance report", sections), governance["reports"]["thresholds"])


if __name__ == "__main__":
    main()
