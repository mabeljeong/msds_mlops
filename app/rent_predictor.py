"""Trainable rent predictor bundle with overpriced-flag explanations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBRegressor

FEATURE_COLUMNS: list[str] = [
    "zip_code",
    "bedrooms",
    "bathrooms",
    "walk_score",
    "transit_score",
    "census_median_income",
    "census_renter_ratio",
    "census_vacancy_rate",
    "crime_total_month_zip_log1p_latest",
    "zori_baseline",
    "zhvi_level",
    "zhvi_12mo_delta",
    "redfin_mom_pct",
    "redfin_yoy_pct",
    "bedrooms_x_census_income",
    "walk_score_x_transit_score",
]

CATEGORICAL_FEATURES = ["zip_code"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ]
    )


def build_training_pipeline(
    random_state: int = 42,
    *,
    n_estimators: int = 220,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_child_weight: float = 3,
    subsample: float = 0.85,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 2.0,
    objective: str = "reg:squarederror",
    quantile_alpha: float | None = None,
) -> Pipeline:
    kwargs: dict[str, Any] = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective=objective,
        random_state=random_state,
    )
    if quantile_alpha is not None:
        kwargs["quantile_alpha"] = quantile_alpha
    regressor = XGBRegressor(**kwargs)
    return Pipeline([("preprocessor", _build_preprocessor()), ("regressor", regressor)])


@dataclass
class RentPredictor:
    model: Pipeline
    feature_columns: list[str]
    q25_model: Pipeline | None = None
    q75_model: Pipeline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _frame(self, data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.DataFrame([dict(data)])

        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = np.nan

        frame = frame[self.feature_columns].copy()
        frame["zip_code"] = frame["zip_code"].astype("string").fillna("UNK")
        for col in self.feature_columns:
            if col != "zip_code":
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    @staticmethod
    def _expm1_clip(values: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(np.asarray(values, dtype=float)), 0.0)

    def predict_rent(self, data: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        X = self._frame(data)
        return self._expm1_clip(self.model.predict(X))

    def predict_interval(
        self, data: pd.DataFrame | Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        X = self._frame(data)
        out: dict[str, np.ndarray] = {"point": self._expm1_clip(self.model.predict(X))}
        if self.q25_model is not None:
            out["p25"] = self._expm1_clip(self.q25_model.predict(X))
        if self.q75_model is not None:
            out["p75"] = self._expm1_clip(self.q75_model.predict(X))
        return out

    def top_shap_contributors(self, listing: Mapping[str, Any], top_k: int = 5) -> list[dict[str, float | str]]:
        X = self._frame(listing)
        preprocessor = self.model.named_steps["preprocessor"]
        regressor: XGBRegressor = self.model.named_steps["regressor"]
        transformed = preprocessor.transform(X)
        contrib = regressor.get_booster().predict(DMatrix(transformed), pred_contribs=True)[0]
        names = list(preprocessor.get_feature_names_out()) + ["bias"]

        pairs = [
            {"feature": names[i], "shap_contribution_log_rent": float(contrib[i])}
            for i in range(min(len(names), len(contrib) - 1))
        ]
        pairs.sort(key=lambda row: abs(float(row["shap_contribution_log_rent"])), reverse=True)
        return pairs[:top_k]

    def flag_overpriced(
        self,
        listing: Mapping[str, Any],
        threshold_pct: float = 0.10,
        threshold_usd: float = 250.0,
    ) -> dict[str, Any]:
        intervals = self.predict_interval(listing)
        predicted = float(intervals["point"][0])
        p25 = float(intervals["p25"][0]) if "p25" in intervals else None
        p75 = float(intervals["p75"][0]) if "p75" in intervals else None

        actual = listing.get("actual_rent_usd", listing.get("rent_usd"))
        actual_val = float(actual) if actual is not None else np.nan
        delta_usd = actual_val - predicted if np.isfinite(actual_val) else np.nan
        delta_pct = (delta_usd / predicted) if predicted > 0 and np.isfinite(delta_usd) else np.nan

        if p75 is not None and np.isfinite(actual_val):
            is_flagged = bool(actual_val > p75)
            flag_reason = "actual_rent_above_p75"
        else:
            is_flagged = bool(
                np.isfinite(delta_usd)
                and np.isfinite(delta_pct)
                and delta_usd >= threshold_usd
                and delta_pct >= threshold_pct
            )
            flag_reason = "delta_pct_exceeds_threshold"

        return {
            "predicted_rent_usd": round(predicted, 2),
            "fair_rent_p25": round(p25, 2) if p25 is not None else None,
            "fair_rent_p75": round(p75, 2) if p75 is not None else None,
            "delta_usd": round(float(delta_usd), 2) if np.isfinite(delta_usd) else None,
            "delta_pct": round(float(delta_pct), 4) if np.isfinite(delta_pct) else None,
            "flag_overpriced": is_flagged,
            "flag_reason": flag_reason,
            "top_shap_contributors": self.top_shap_contributors(listing),
        }


class RentPredictorPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper that supports prediction and overpriced flagging."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.predictor: RentPredictor = joblib.load(context.artifacts["predictor_bundle"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        frame = model_input.copy()
        if "actual_rent_usd" in frame.columns:
            records = [self.predictor.flag_overpriced(row.to_dict()) for _, row in frame.iterrows()]
            return pd.DataFrame(records)
        intervals = self.predictor.predict_interval(frame)
        out = pd.DataFrame({"predicted_rent_usd": intervals["point"]})
        if "p25" in intervals:
            out["fair_rent_p25"] = intervals["p25"]
        if "p75" in intervals:
            out["fair_rent_p75"] = intervals["p75"]
        return out
