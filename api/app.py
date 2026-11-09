"""FastAPI application serving the saved credit and fraud model bundles."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    BothPrediction,
    CreditPrediction,
    CustomerFeatures,
    FraudPrediction,
    HealthResponse,
)
from src.models.bundle import ModelBundle
from src.models.predict import load_model_bundle, predict_batch
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ModelRegistry:
    """Bundles and policy loaded once for the lifetime of one API process."""

    credit: ModelBundle | None
    fraud: ModelBundle | None
    fraud_threshold: float | None
    detail: str | None

    @property
    def ready(self) -> bool:
        """Return whether every prediction dependency is available."""
        return self.credit is not None and self.fraud is not None and self.fraud_threshold is not None


def _load_registry() -> ModelRegistry:
    """Load model bundles and the validation-selected fraud policy from environment paths."""
    root = Path(os.environ.get("MODEL_DIR", "models"))
    version = os.environ.get("MODEL_VERSION", "v1.0.0")
    try:
        credit = load_model_bundle(version, "credit_risk", root)
        fraud = load_model_bundle(version, "fraud_risk_score", root)
        threshold_path = root / "fraud_risk" / version / "thresholds.json"
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(payload["top_k_threshold"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        detail = f"Model bundles or fraud threshold policy are unavailable: {error}"
        LOGGER.warning(detail)
        return ModelRegistry(None, None, None, detail)
    LOGGER.info("Loaded credit and fraud bundles version %s from %s", version, root)
    return ModelRegistry(credit, fraud, threshold, None)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load prediction dependencies once while keeping the health endpoint available."""
    app.state.registry = _load_registry()
    yield


app = FastAPI(title="Credit and Fraud Risk API", version="1.0.0", lifespan=lifespan)


def _registry(request: Request) -> ModelRegistry:
    """Return ready dependencies or an actionable service-unavailable response."""
    registry: ModelRegistry = request.app.state.registry
    if not registry.ready:
        raise HTTPException(status_code=503, detail=registry.detail or "Model bundles are not loaded.")
    return registry


def _raw_frame(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a validated request into one raw row for the shared inference pipeline."""
    return pd.DataFrame([customer.model_dump(mode="json")])


def _credit_prediction(frame: pd.DataFrame, bundle: ModelBundle) -> CreditPrediction:
    """Map the common inference output to the public credit response schema."""
    prediction = predict_batch(frame, bundle).iloc[0]
    probabilities = {
        column.removeprefix("credit_risk_probability_"): float(prediction[column])
        for column in prediction.index
        if column.startswith("credit_risk_probability_")
    }
    return CreditPrediction(
        customer_id=str(prediction["customer_id"]),
        credit_risk_prediction=int(prediction["credit_risk_prediction"]),
        probabilities=probabilities,
    )


def _fraud_prediction(frame: pd.DataFrame, bundle: ModelBundle, threshold: float) -> FraudPrediction:
    """Map the common inference output to the public fraud response schema."""
    prediction = predict_batch(frame, bundle).iloc[0]
    score = float(prediction["fraud_risk_score_prediction"])
    return FraudPrediction(
        customer_id=str(prediction["customer_id"]),
        fraud_risk_score_prediction=score,
        review_flag=score >= threshold,
        review_threshold=threshold,
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report readiness without failing startup when bundles have not been mounted."""
    registry: ModelRegistry = request.app.state.registry
    return HealthResponse(status="ok" if registry.ready else "model_not_loaded", detail=registry.detail)


@app.post("/predict/credit", response_model=CreditPrediction)
def predict_credit(customer: CustomerFeatures, request: Request) -> CreditPrediction:
    """Predict one customer's ordered credit-risk class and probabilities."""
    registry = _registry(request)
    if registry.credit is None:
        raise AssertionError("Ready registry must contain the credit bundle.")
    return _credit_prediction(_raw_frame(customer), registry.credit)


@app.post("/predict/fraud", response_model=FraudPrediction)
def predict_fraud(customer: CustomerFeatures, request: Request) -> FraudPrediction:
    """Predict one customer's fraud score and frozen review flag."""
    registry = _registry(request)
    if registry.fraud is None or registry.fraud_threshold is None:
        raise AssertionError("Ready registry must contain fraud dependencies.")
    return _fraud_prediction(_raw_frame(customer), registry.fraud, registry.fraud_threshold)


@app.post("/predict/both", response_model=BothPrediction)
def predict_both(customer: CustomerFeatures, request: Request) -> BothPrediction:
    """Return credit and fraud predictions from one validated raw customer row."""
    registry = _registry(request)
    if registry.credit is None or registry.fraud is None or registry.fraud_threshold is None:
        raise AssertionError("Ready registry must contain both model dependencies.")
    frame = _raw_frame(customer)
    return BothPrediction(
        credit=_credit_prediction(frame, registry.credit),
        fraud=_fraud_prediction(frame, registry.fraud, registry.fraud_threshold),
    )
