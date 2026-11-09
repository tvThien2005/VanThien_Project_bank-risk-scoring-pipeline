"""Integration coverage for raw sample to versioned model bundles."""

from pathlib import Path


def test_training_pipeline_builds_both_sample_bundles(trained_sample_project) -> None:
    """Run the complete training path from sample.csv without using the test split."""
    results = trained_sample_project["results"]
    workspace = Path(trained_sample_project["workspace"])

    assert set(results) == {"credit_risk", "fraud_risk_score"}
    assert (workspace / "features/X_train.parquet").is_file()
    assert (workspace / "models/credit_risk/v1.0.0/model.txt").is_file()
    assert (workspace / "models/fraud_risk/v1.0.0/model.txt").is_file()
    assert not (workspace / "models/credit_risk/v1.0.0/test_metrics.json").exists()
