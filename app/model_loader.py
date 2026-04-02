"""Load MLflow pyfunc model or a sklearn placeholder with the same feature schema."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Column order must match training and PredictRequest → dict mapping
FEATURE_COLUMNS: list[str] = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "walk_score",
    "transit_score",
    "median_zip_rent",
]


@dataclass
class LoadedModel:
    """Wraps either MLflow pyfunc or a sklearn estimator."""

    source: str  # "mlflow" | "placeholder"
    sklearn_model: object | None
    pyfunc_model: object | None
    version: str | None
    load_message: str | None = None

    def predict_row(self, features: dict[str, float]) -> float:
        row = pd.DataFrame([{c: features[c] for c in FEATURE_COLUMNS}])
        if self.pyfunc_model is not None:
            raw = self.pyfunc_model.predict(row)
        else:
            assert self.sklearn_model is not None
            X = row[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            raw = self.sklearn_model.predict(X)
        out = np.asarray(raw).ravel()
        return float(out[0])


def build_placeholder_model() -> LinearRegression:
    """Train the placeholder sklearn regressor (same as offline fallback)."""
    return _build_placeholder()


def _build_placeholder() -> LinearRegression:
    rng = np.random.default_rng(42)
    n = 500
    X = rng.uniform(
        low=[0.5, 0.5, 400, 20, 20, 1500],
        high=[4.0, 3.0, 2500, 98, 98, 6000],
        size=(n, len(FEATURE_COLUMNS)),
    )
    # Synthetic rent: base + weighted features + noise
    rent = (
        800
        + 350 * X[:, 0]
        + 120 * X[:, 1]
        + 0.45 * X[:, 2]
        + 4.0 * X[:, 3]
        + 3.0 * X[:, 4]
        + 0.12 * X[:, 5]
        + rng.normal(0, 80, size=n)
    )
    model = LinearRegression()
    model.fit(X, rent)
    return model


def load_model() -> LoadedModel:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    model_uri = os.environ.get("MLFLOW_MODEL_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri:
        try:
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            version = os.environ.get("MLFLOW_MODEL_VERSION")
            msg = f"Loaded MLflow model from {model_uri}"
            logger.info(msg)
            return LoadedModel(
                source="mlflow",
                sklearn_model=None,
                pyfunc_model=pyfunc_model,
                version=version,
                load_message=msg,
            )
        except Exception as e:
            logger.warning(
                "MLFLOW_MODEL_URI set but load failed (%s); using placeholder.",
                e,
            )

    sk = _build_placeholder()
    msg = "Using placeholder LinearRegression (set MLFLOW_TRACKING_URI + MLFLOW_MODEL_URI for registry)"
    logger.info(msg)
    return LoadedModel(
        source="placeholder",
        sklearn_model=sk,
        pyfunc_model=None,
        version=None,
        load_message=msg,
    )
