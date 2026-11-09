"""End-to-end API coverage using the temporary sample-trained model bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.app import app
from api.schemas import CustomerFeatures

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _valid_payload() -> dict[str, object]:
    """Read one complete valid raw sample row without the training targets."""
    sample = pd.read_csv(PROJECT_ROOT / "tests/data/sample.csv")
    candidates = sample.drop(columns=["credit_risk", "fraud_risk_score"]).dropna()
    for _, row in candidates.iterrows():
        payload = json.loads(row.to_json())
        try:
            CustomerFeatures.model_validate(payload)
        except ValidationError:
            continue
        return payload
    raise AssertionError("The sample fixture must contain a valid API row.")


def test_api_predicts_all_tasks_from_saved_bundles(trained_sample_project, monkeypatch) -> None:
    """Serve one raw row through each endpoint and preserve its customer identifier."""
    workspace = Path(trained_sample_project["workspace"])
    threshold_path = workspace / "models/fraud_risk/v1.0.0/thresholds.json"
    threshold_path.write_text('{"top_k_threshold": 0.5}', encoding="utf-8")
    monkeypatch.setenv("MODEL_DIR", str(workspace / "models"))
    monkeypatch.setenv("MODEL_VERSION", "v1.0.0")
    payload = _valid_payload()

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        credit = client.post("/predict/credit", json=payload)
        fraud = client.post("/predict/fraud", json=payload)
        both = client.post("/predict/both", json=payload)

    assert credit.status_code == 200
    assert fraud.status_code == 200
    assert both.status_code == 200
    assert credit.json()["customer_id"] == payload["customer_id"]
    assert set(credit.json()["probabilities"]) == {"0", "1", "2", "3"}
    assert isinstance(fraud.json()["review_flag"], bool)
    assert both.json()["fraud"] == fraud.json()


def test_api_rejects_invalid_request_and_reports_missing_models(monkeypatch, tmp_path) -> None:
    """Return Pydantic 422 errors and a non-crashing readiness response without models."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "model_not_loaded"
        invalid = client.post("/predict/credit", json={**_valid_payload(), "age": 17})
        unavailable = client.post("/predict/credit", json=_valid_payload())

    assert invalid.status_code == 422
    assert unavailable.status_code == 503
