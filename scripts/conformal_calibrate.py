#!/usr/bin/env python3
"""Conformalized Quantile Regression calibration for the q25/q75 heads.

Runs once to estimate per-side offsets that expand the q25/q75 intervals so
empirical coverage on a held-out slice matches the nominal level (0.50 by
default). Does not modify the registered MLflow model; instead writes
`demo/conformal_calibration.json` with offsets, observed coverage, and a small
note that V2 deployment can apply these at serve-time.

Reference: Romano et al. 2019, "Conformalized Quantile Regression".

CLI:
  python scripts/conformal_calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rent_predictor import FEATURE_COLUMNS, build_training_pipeline  # noqa: E402

FEATURE_PATH = ROOT / "data" / "features" / "listings_features.csv"
OUTPUT_PATH = ROOT / "demo" / "conformal_calibration.json"
HOLDOUT_FRACTION = 0.15
RANDOM_STATE = 13
NOMINAL_COVERAGE = 0.50
WINNING_HP = dict(max_depth=3, min_child_weight=3, n_estimators=220, learning_rate=0.05)


def _scope_2000_8000(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["rent_usd"] >= 2_000) & (df["rent_usd"] <= 8_000)].copy()


def _coerce(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["zip_code"] = out["zip_code"].astype("string").fillna("UNK").astype(str)
    for col in FEATURE_COLUMNS:
        if col != "zip_code":
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out[FEATURE_COLUMNS]


def _coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((actual >= lo) & (actual <= hi)))


def main() -> None:
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    df = _scope_2000_8000(df)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    X = _coerce(df)
    y = df["rent_usd"].astype(float).to_numpy()
    y_log = np.log1p(y)

    X_fit, X_cal, ylog_fit, _, y_fit_raw, y_cal_raw = train_test_split(
        X, y_log, y, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE
    )
    print(f"Fit rows: {len(X_fit)}  Calibration rows: {len(X_cal)}")

    point = build_training_pipeline(**WINNING_HP, objective="reg:squarederror")
    q25 = build_training_pipeline(**WINNING_HP, objective="reg:quantileerror", quantile_alpha=0.25)
    q75 = build_training_pipeline(**WINNING_HP, objective="reg:quantileerror", quantile_alpha=0.75)

    point.fit(X_fit, ylog_fit)
    q25.fit(X_fit, ylog_fit)
    q75.fit(X_fit, ylog_fit)

    p25_pred = np.maximum(np.expm1(q25.predict(X_cal)), 0.0)
    p75_pred = np.maximum(np.expm1(q75.predict(X_cal)), 0.0)
    point_pred = np.maximum(np.expm1(point.predict(X_cal)), 0.0)

    raw_coverage = _coverage(y_cal_raw, p25_pred, p75_pred)
    raw_width = float(np.mean(p75_pred - p25_pred))

    # Symmetric CQR nonconformity score on the dollar scale (rent is on a roughly
    # log-symmetric distribution after scoping; dollar-side residuals are easy to
    # interpret in the model card / API).
    scores = np.maximum(p25_pred - y_cal_raw, y_cal_raw - p75_pred)
    n_cal = len(scores)
    target_index = int(np.ceil((n_cal + 1) * NOMINAL_COVERAGE)) - 1
    target_index = min(max(target_index, 0), n_cal - 1)
    sorted_scores = np.sort(scores)
    q_hat = float(sorted_scores[target_index])
    q_hat = max(q_hat, 0.0)

    cal_lo = np.maximum(p25_pred - q_hat, 0.0)
    cal_hi = p75_pred + q_hat
    calibrated_coverage = _coverage(y_cal_raw, cal_lo, cal_hi)
    calibrated_width = float(np.mean(cal_hi - cal_lo))

    point_mae = float(np.mean(np.abs(y_cal_raw - point_pred)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "Conformalized Quantile Regression (Romano et al. 2019), symmetric dollar offset",
        "nominal_coverage": NOMINAL_COVERAGE,
        "holdout_fraction": HOLDOUT_FRACTION,
        "random_state": RANDOM_STATE,
        "n_fit": int(len(X_fit)),
        "n_calibration": n_cal,
        "winning_hp": WINNING_HP,
        "raw": {
            "coverage": raw_coverage,
            "mean_width_usd": raw_width,
        },
        "calibrated": {
            "offset_usd": q_hat,
            "coverage": calibrated_coverage,
            "mean_width_usd": calibrated_width,
        },
        "point_mae_usd": point_mae,
        "usage_note": (
            "Apply at serve-time as p25' = max(p25 - offset_usd, 0), p75' = p75 + offset_usd. "
            "Offsets should be re-estimated whenever the underlying model is retrained."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote calibration offsets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
