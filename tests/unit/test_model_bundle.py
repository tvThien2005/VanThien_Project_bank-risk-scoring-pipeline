"""Tests for serializing and reloading a fitted LightGBM model bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.models.bundle import load_saved_bundle, save_model_bundle


def test_saved_lightgbm_bundle_reloads_with_identical_predictions(tmp_path) -> None:
    """A persisted bundle yields exactly the original class-probability predictions."""
    raw = pd.DataFrame(
        {
            "first": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
            "second": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        }
    )
    target = pd.Series([0, 0, 1, 1, 2, 2, 3, 3])
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), ["first", "second"])])
    transformed = pd.DataFrame(
        preprocessor.fit_transform(raw), columns=preprocessor.get_feature_names_out()
    )
    model = LGBMClassifier(n_estimators=10, n_jobs=1, random_state=42, verbosity=-1).fit(
        transformed, target
    )
    directory = save_model_bundle(
        task="credit_risk",
        version="test",
        directory=tmp_path / "bundle",
        model=model,
        preprocessor=preprocessor,
        feature_list=transformed.columns.tolist(),
        metrics={"validation": {"qwk": 1.0}},
        metadata={
            "inference": {"features": {}, "data_quality": {"rules": []}},
            "classes": [0, 1, 2, 3],
        },
    )

    loaded = load_saved_bundle(directory)
    np.testing.assert_allclose(model.predict_proba(transformed), loaded.model.predict(transformed))
