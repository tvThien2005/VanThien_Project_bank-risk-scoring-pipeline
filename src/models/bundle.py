"""Persist and reload self-contained LightGBM and XGBoost model bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import yaml
from sklearn.compose import ColumnTransformer

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _xgboost_module() -> Any:
    """Import XGBoost only while saving or loading an XGBoost bundle."""
    try:
        import xgboost
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "XGBoost is required for this bundle. Install requirements.txt first."
        ) from error
    return xgboost


@dataclass(frozen=True)
class ModelBundle:
    """All objects required to transform raw customer rows and make predictions."""

    task: str
    version: str
    model: Any
    preprocessor: ColumnTransformer
    feature_list: tuple[str, ...]
    metadata: Mapping[str, Any]


def _json_default(value: Any) -> Any:
    """Serialize NumPy-compatible scalar values in bundle JSON files."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_model_bundle(
    *,
    task: str,
    version: str,
    directory: str | Path,
    model: Any,
    preprocessor: ColumnTransformer,
    feature_list: Sequence[str],
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    algorithm: str = "lightgbm",
) -> Path:
    """Write the required model, preprocessing, metadata, and metric bundle files."""
    if algorithm not in {"lightgbm", "xgboost"}:
        raise ValueError("algorithm must be lightgbm or xgboost.")
    bundle_dir = Path(directory)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "lightgbm":
        if not isinstance(model, lgb.LGBMModel) or not hasattr(model, "booster_"):
            raise ValueError("A fitted LightGBM model is required for a LightGBM bundle.")
        model.booster_.save_model(str(bundle_dir / "model.txt"))
    else:
        if not isinstance(model, _xgboost_module().XGBClassifier):
            raise ValueError("A fitted XGBClassifier is required for an XGBoost bundle.")
        model.save_model(str(bundle_dir / "model.json"))
    joblib.dump(preprocessor, bundle_dir / "preprocessor.joblib")
    (bundle_dir / "feature_list.json").write_text(
        json.dumps(list(feature_list), indent=2), encoding="utf-8"
    )
    (bundle_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, default=_json_default), encoding="utf-8"
    )
    payload = {"task": task, "version": version, **dict(metadata), "algorithm": algorithm}
    yaml_payload = json.loads(json.dumps(payload, default=_json_default))
    (bundle_dir / "metadata.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    LOGGER.info("Saved %s model bundle version %s to %s", task, version, bundle_dir)
    return bundle_dir


def load_saved_bundle(directory: str | Path) -> ModelBundle:
    """Load and validate an on-disk model bundle from a single directory."""
    bundle_dir = Path(directory)
    metadata_path = bundle_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete model bundle at {bundle_dir}: missing ['metadata.yaml']")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("Model metadata must be a YAML mapping.")
    algorithm = str(metadata.get("algorithm", "lightgbm"))
    model_filename = {"lightgbm": "model.txt", "xgboost": "model.json"}.get(algorithm)
    if model_filename is None:
        raise ValueError(f"Unsupported bundle algorithm: {algorithm}")
    required = (
        model_filename,
        "preprocessor.joblib",
        "feature_list.json",
        "metrics.json",
        "metadata.yaml",
    )
    missing = [name for name in required if not (bundle_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model bundle at {bundle_dir}: missing {missing}")
    feature_list = json.loads((bundle_dir / "feature_list.json").read_text(encoding="utf-8"))
    if not isinstance(feature_list, list) or not all(isinstance(item, str) for item in feature_list):
        raise TypeError("feature_list.json must contain a list of strings.")
    preprocessor = joblib.load(bundle_dir / "preprocessor.joblib")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("Bundle preprocessor is not a ColumnTransformer.")
    task = metadata.get("task")
    version = metadata.get("version")
    if not isinstance(task, str) or not isinstance(version, str):
        raise TypeError("Bundle metadata requires task and version strings.")
    if algorithm == "lightgbm":
        model: Any = lgb.Booster(model_file=str(bundle_dir / model_filename))
    else:
        model = _xgboost_module().XGBClassifier()
        model.load_model(str(bundle_dir / model_filename))
    return ModelBundle(
        task=task,
        version=version,
        model=model,
        preprocessor=preprocessor,
        feature_list=tuple(feature_list),
        metadata=metadata,
    )
