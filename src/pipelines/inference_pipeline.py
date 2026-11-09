"""Batch inference orchestration for raw customer CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.models.predict import load_model_bundle, predict_batch
from src.utils.io import read_csv_safe
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
InferenceTask = Literal["credit", "fraud", "both"]


def run_batch_inference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    task: InferenceTask = "both",
    version: str = "v1.0.0",
    model_root: str | Path = "models",
) -> pd.DataFrame:
    """Load versioned bundles, predict raw rows, and atomically write a CSV result."""
    raw = read_csv_safe(input_path)
    if "customer_id" not in raw:
        raise ValueError("Batch input must contain customer_id.")
    prediction = pd.DataFrame({"customer_id": raw["customer_id"].to_numpy()})
    if task in {"credit", "both"}:
        credit = load_model_bundle(version, "credit_risk", model_root)
        prediction = prediction.merge(predict_batch(raw, credit), on="customer_id", validate="one_to_one")
    if task in {"fraud", "both"}:
        fraud = load_model_bundle(version, "fraud_risk_score", model_root)
        prediction = prediction.merge(predict_batch(raw, fraud), on="customer_id", validate="one_to_one")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    prediction.to_csv(temporary, index=False)
    temporary.replace(destination)
    LOGGER.info("Wrote %d %s batch predictions to %s", len(prediction), task, destination)
    return prediction
