"""Load MLflow rent model or fallback placeholder."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.rent_predictor import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Wraps either MLflow pyfunc or a sklearn estimator."""

    source: str  # "mlflow" | "placeholder"
    sklearn_model: object | None
    pyfunc_model: object | None
    version: str | None
    load_message: str | None = None

    @staticmethod
    def _coerce_input(row: pd.DataFrame) -> pd.DataFrame:
        out = row.copy()
        out["zip_code"] = out["zip_code"].fillna("UNK").astype(str)
        for col in FEATURE_COLUMNS:
            if col != "zip_code":
                if col not in out.columns:
                    out[col] = np.nan
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        return out

    def predict_row(self, features: dict[str, Any]) -> dict[str, Any]:
        row = pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])
        row = self._coerce_input(row)
        if self.pyfunc_model is not None:
            raw = self.pyfunc_model.predict(row)
            if isinstance(raw, pd.DataFrame):
                rec = raw.iloc[0].to_dict()
                return {
                    "predicted_rent_usd": float(rec.get("predicted_rent_usd", float("nan"))),
                    "fair_rent_p10": (
                        float(rec["fair_rent_p10"]) if "fair_rent_p10" in rec and pd.notna(rec["fair_rent_p10"]) else None
                    ),
                    "fair_rent_p90": (
                        float(rec["fair_rent_p90"]) if "fair_rent_p90" in rec and pd.notna(rec["fair_rent_p90"]) else None
                    ),
                }
            arr = np.asarray(raw).ravel()
            return {"predicted_rent_usd": float(arr[0]), "fair_rent_p10": None, "fair_rent_p90": None}

        # Placeholder predictor — used only when MLFLOW_MODEL_URI isn't set.
        #
        # We *don't* use the LinearRegression instance fitted in `_build_placeholder`
        # because it was trained on synthetic data and produces garbage on real SF
        # listings (extreme negative or saturated values). Instead we use a small
        # transparent formula anchored on `zori_baseline` (the ZIP-level Zillow rent
        # index, which is a real SF benchmark) and scaled by the unit size and a
        # few amenity adjustments. This still varies per listing in a plausible way
        # so the SPA can be demoed end-to-end without the real model.
        feats = row.iloc[0].to_dict()

        def _f(key: str, default: float) -> float:
            v = feats.get(key)
            try:
                if v is None or pd.isna(v):
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        bedrooms = max(0.0, min(_f("bedrooms", 1.0), 5.0))
        bathrooms = max(1.0, min(_f("bathrooms", 1.0), 4.0))
        walk_score = max(0.0, min(_f("walk_score", 80.0), 100.0))
        transit_score = max(0.0, min(_f("transit_score", 80.0), 100.0))
        zori = _f("zori_baseline", 3500.0)
        if zori <= 0:
            zori = 3500.0

        size_mult = {0: 0.70, 1: 0.90, 2: 1.10, 3: 1.35, 4: 1.55, 5: 1.70}.get(int(bedrooms), 1.0)
        amenity_mult = 1.0 + 0.0015 * (walk_score - 70) + 0.0010 * (transit_score - 70)
        bath_bonus = 200.0 * max(0.0, bathrooms - 1.0)

        predicted = float(np.clip(zori * size_mult * amenity_mult + bath_bonus, 1500.0, 12000.0))
        return {
            "predicted_rent_usd": predicted,
            "fair_rent_p10": None,
            "fair_rent_p90": None,
        }

    def flag_overpriced(self, listing: dict[str, Any]) -> dict[str, Any]:
        if self.pyfunc_model is not None:
            row = pd.DataFrame([{c: listing.get(c) for c in FEATURE_COLUMNS}])
            row = self._coerce_input(row)
            if "actual_rent_usd" in listing:
                row["actual_rent_usd"] = float(listing["actual_rent_usd"])
            elif "rent_usd" in listing:
                row["actual_rent_usd"] = float(listing["rent_usd"])
            out = self.pyfunc_model.predict(row)
            if isinstance(out, pd.DataFrame) and not out.empty:
                rec = out.iloc[0].to_dict()
                return {
                    "predicted_rent_usd": float(rec.get("predicted_rent_usd", float("nan"))),
                    "fair_rent_p10": (
                        float(rec["fair_rent_p10"]) if "fair_rent_p10" in rec and pd.notna(rec["fair_rent_p10"]) else None
                    ),
                    "fair_rent_p90": (
                        float(rec["fair_rent_p90"]) if "fair_rent_p90" in rec and pd.notna(rec["fair_rent_p90"]) else None
                    ),
                    "delta_usd": (
                        float(rec["delta_usd"]) if "delta_usd" in rec and pd.notna(rec["delta_usd"]) else None
                    ),
                    "delta_pct": (
                        float(rec["delta_pct"]) if "delta_pct" in rec and pd.notna(rec["delta_pct"]) else None
                    ),
                    "flag_overpriced": bool(rec.get("flag_overpriced", False)),
                    "flag_reason": rec.get("flag_reason"),
                    "top_shap_contributors": rec.get("top_shap_contributors") or [],
                }

        prediction = self.predict_row(listing)
        predicted = float(prediction["predicted_rent_usd"])
        actual = listing.get("actual_rent_usd", listing.get("rent_usd"))
        delta_usd = (float(actual) - predicted) if actual is not None else None
        delta_pct = (delta_usd / predicted) if (delta_usd is not None and predicted > 0) else None

        # IMPORTANT: the placeholder regressor is trained on synthetic data and is not
        # accurate for real SF listings (it's only a smoke-test stand-in for the
        # MLflow rent model). We therefore *never* assert "overpriced" from it — that
        # verdict is reserved for the real registered model. The relative price gap
        # is still surfaced in `delta_*` so /rank can use it for batch-relative
        # price-fairness scoring.
        return {
            "predicted_rent_usd": round(predicted, 2),
            "fair_rent_p10": prediction.get("fair_rent_p10"),
            "fair_rent_p90": prediction.get("fair_rent_p90"),
            "delta_usd": round(delta_usd, 2) if delta_usd is not None else None,
            "delta_pct": round(delta_pct, 4) if delta_pct is not None else None,
            "flag_overpriced": False,
            "flag_reason": "placeholder_model_no_overprice_verdict",
            "top_shap_contributors": [],
        }


def build_placeholder_model() -> LinearRegression:
    """Train the placeholder sklearn regressor (same as offline fallback)."""
    return _build_placeholder()


def _build_placeholder() -> LinearRegression:
    rng = np.random.default_rng(42)
    n = 500
    data = pd.DataFrame(
        {
            "zip_code": [f"941{d}" for d in rng.integers(2, 9, size=n)],
            "bedrooms": rng.uniform(0, 4, size=n),
            "bathrooms": rng.uniform(1, 3, size=n),
            "walk_score": rng.uniform(40, 100, size=n),
            "transit_score": rng.uniform(40, 100, size=n),
            "census_median_income": rng.uniform(60_000, 180_000, size=n),
            "census_renter_ratio": rng.uniform(0.25, 0.9, size=n),
            "census_vacancy_rate": rng.uniform(0.02, 0.15, size=n),
            "crime_total_month_zip_log1p_latest": rng.uniform(0.0, 8.0, size=n),
            "zori_baseline": rng.uniform(2_000, 5_500, size=n),
            "zhvi_level": rng.uniform(600_000, 2_000_000, size=n),
            "zhvi_12mo_delta": rng.uniform(-80_000, 120_000, size=n),
            "redfin_mom_pct": rng.uniform(-0.05, 0.08, size=n),
            "redfin_yoy_pct": rng.uniform(-0.1, 0.2, size=n),
        }
    )
    data["bedrooms_x_census_income"] = data["bedrooms"] * data["census_median_income"]
    data["walk_score_x_transit_score"] = data["walk_score"] * data["transit_score"]
    data = data[FEATURE_COLUMNS]

    rent = (
        500
        + 320 * data["bedrooms"]
        + 180 * data["bathrooms"]
        + 6 * data["walk_score"]
        + 0.02 * data["census_median_income"]
        + 0.45 * data["zori_baseline"]
        + 0.0005 * data["zhvi_level"]
        + 2200 * data["redfin_yoy_pct"]
        - 150 * data["crime_total_month_zip_log1p_latest"]
        + rng.normal(0, 80, size=n)
    )
    encoded = data.copy()
    encoded["zip_code"] = encoded["zip_code"].astype("category").cat.codes
    model = LinearRegression()
    model.fit(encoded, rent)
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
