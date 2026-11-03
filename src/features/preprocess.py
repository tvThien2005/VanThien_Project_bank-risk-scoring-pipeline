"""Train-fitted preprocessing that excludes identifiers, raw dates, and targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
IDENTIFIER_COLUMN = "customer_id"
RAW_DATE_COLUMNS = ("account_creation_date", "last_transaction_date")
EDUCATION_COLUMN = "education_level"


class TrainSkewedLogTransformer(BaseEstimator, TransformerMixin):
    """Log-transform nonnegative numeric columns whose skew exceeds a train threshold."""

    def __init__(self, skew_threshold: float) -> None:
        """Store the configured absolute skew threshold without fitting data."""
        self.skew_threshold = skew_threshold

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> TrainSkewedLogTransformer:
        """Identify eligible skewed columns using only the supplied training matrix."""
        frame = self._to_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        skewness = frame.apply(pd.Series.skew, numeric_only=True)
        nonnegative = frame.apply(
            lambda column: not pd.api.types.is_bool_dtype(column)
            and (column.dropna() >= 0).all(),
            axis=0,
        )
        self.skewed_columns_ = [
            column
            for column in frame.columns
            if bool(nonnegative[column])
            and pd.notna(skewness.get(column))
            and abs(float(skewness[column])) > self.skew_threshold
        ]
        LOGGER.info("Selected %d skewed numeric columns for log1p", len(self.skewed_columns_))
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Apply log1p only to the train-selected columns while preserving missing values."""
        check_is_fitted(self, "skewed_columns_")
        if isinstance(X, pd.DataFrame):
            transformed = X.copy()
            for column in self.skewed_columns_:
                transformed[column] = np.log1p(pd.to_numeric(transformed[column], errors="coerce"))
            return transformed
        transformed_array = np.asarray(X, dtype=float).copy()
        for column in self.skewed_columns_:
            index = list(self.feature_names_in_).index(column)
            transformed_array[:, index] = np.log1p(transformed_array[:, index])
        return transformed_array

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return unchanged feature names because this transformer adds no columns."""
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _to_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Normalize transformer input to a dataframe with stable feature names."""
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X)
        return pd.DataFrame(array, columns=[f"feature_{index}" for index in range(array.shape[1])])


class NumericMissingNormalizer(BaseEstimator, TransformerMixin):
    """Convert pandas nullable numeric values to float arrays with NumPy missing values."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> NumericMissingNormalizer:
        """Record feature names without learning any distributional statistic."""
        self.feature_names_in_ = self._feature_names(X)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Normalize pandas ``pd.NA`` values to ``np.nan`` for sklearn imputers."""
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce").astype(float)
        return np.asarray(X, dtype=float)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return unchanged names because normalization does not add columns."""
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)

    @staticmethod
    def _feature_names(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract stable column names for dataframe or array input."""
        if isinstance(X, pd.DataFrame):
            return np.asarray(X.columns, dtype=object)
        return np.asarray([f"feature_{index}" for index in range(np.asarray(X).shape[1])])


class CategoricalMissingNormalizer(BaseEstimator, TransformerMixin):
    """Convert nullable pandas categorical values to object values with ``np.nan`` missingness."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> CategoricalMissingNormalizer:
        """Record feature names without learning any distributional statistic."""
        self.feature_names_in_ = NumericMissingNormalizer._feature_names(X)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Normalize pandas category/string missingness for sklearn imputers."""
        if isinstance(X, pd.DataFrame):
            normalized = X.astype(object)
            return normalized.where(pd.notna(normalized), np.nan)
        array = np.asarray(X, dtype=object)
        return np.where(pd.isna(array), np.nan, array)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return unchanged names because normalization does not add columns."""
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


def _feature_settings(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return validated feature-engineering settings from the main configuration."""
    settings = cfg.get("features")
    if not isinstance(settings, Mapping):
        raise TypeError("features configuration must be a mapping.")
    required = {
        "skew_threshold",
        "exclude",
        "education_order",
        "preprocessor_path",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing feature configuration keys: {missing}")
    if not isinstance(settings["exclude"], list):
        raise TypeError("features.exclude must be a list.")
    if not isinstance(settings["education_order"], list):
        raise TypeError("features.education_order must be a list.")
    return settings


def _excluded_columns(cfg: Mapping[str, Any]) -> frozenset[str]:
    """Build the set of columns permanently excluded from the model matrix."""
    settings = _feature_settings(cfg)
    targets = cfg.get("targets")
    if not isinstance(targets, Mapping):
        raise TypeError("targets configuration must be a mapping.")
    target_columns = {str(target) for target in targets.values()}
    configured_exclusions = {str(column) for column in settings["exclude"]}
    return frozenset(
        {IDENTIFIER_COLUMN, *RAW_DATE_COLUMNS, *target_columns, *configured_exclusions}
    )


def _numeric_columns(frame: pd.DataFrame, excluded: frozenset[str]) -> list[str]:
    """Select model-eligible numeric and boolean dataframe columns."""
    return [
        column
        for column in frame.columns
        if column not in excluded
        and (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]


def _education_columns(frame: pd.DataFrame, excluded: frozenset[str]) -> list[str]:
    """Select education when present and not explicitly excluded."""
    return [EDUCATION_COLUMN] if EDUCATION_COLUMN in frame and EDUCATION_COLUMN not in excluded else []


def _other_categorical_columns(frame: pd.DataFrame, excluded: frozenset[str]) -> list[str]:
    """Select model-eligible non-education string and categorical columns."""
    return [
        column
        for column in frame.columns
        if column not in excluded
        and column != EDUCATION_COLUMN
        and (
            isinstance(frame[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_object_dtype(frame[column])
        )
    ]


def build_preprocessor(cfg: Mapping[str, Any]) -> ColumnTransformer:
    """Build an unfitted feature preprocessor with train-fitted log/impute/encode steps."""
    settings = _feature_settings(cfg)
    excluded = _excluded_columns(cfg)
    numeric_pipeline = Pipeline(
        steps=[
            ("skew_log", TrainSkewedLogTransformer(float(settings["skew_threshold"]))),
            ("normalize_missing", NumericMissingNormalizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    education_pipeline = Pipeline(
        steps=[
            ("normalize_missing", CategoricalMissingNormalizer()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[settings["education_order"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("normalize_missing", CategoricalMissingNormalizer()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, partial(_numeric_columns, excluded=excluded)),
            ("education", education_pipeline, partial(_education_columns, excluded=excluded)),
            (
                "categorical",
                categorical_pipeline,
                partial(_other_categorical_columns, excluded=excluded),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit all stateful preprocessing decisions strictly on the training features."""
    preprocessor.fit(X_train)
    LOGGER.info("Fitted preprocessor on %d training rows", len(X_train))
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw feature frames into named model matrices using a fitted preprocessor."""
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    """Serialize a fitted preprocessor to a configured joblib artifact path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    LOGGER.info("Saved preprocessor to %s", output_path)


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    """Load a serialized fitted preprocessor from a joblib artifact path."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found: {input_path}")
    preprocessor = joblib.load(input_path)
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("Loaded preprocessor is not a ColumnTransformer.")
    return preprocessor
