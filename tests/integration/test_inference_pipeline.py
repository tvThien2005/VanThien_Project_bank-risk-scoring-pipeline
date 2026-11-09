"""Integration coverage for raw sample batch inference."""

from pathlib import Path

from src.pipelines.inference_pipeline import run_batch_inference

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_batch_inference_predicts_both_tasks_from_raw_sample(trained_sample_project) -> None:
    """Load sample-trained bundles and preserve one prediction row per raw customer."""
    workspace = Path(trained_sample_project["workspace"])
    output = workspace / "predictions.csv"

    prediction = run_batch_inference(
        PROJECT_ROOT / "tests/data/sample.csv",
        output,
        task="both",
        model_root=workspace / "models",
    )

    assert output.is_file()
    assert len(prediction) == 50
    assert {"credit_risk_prediction", "fraud_risk_score_prediction"}.issubset(prediction.columns)
