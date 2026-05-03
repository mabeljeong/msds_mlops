#!/usr/bin/env python3
"""Build a reproducible demo pack for `/flag_overpriced`.

Loads the registered MLflow rent model, scores every in-scope listing from the
feature table (deduped on zip/beds/baths/rent), then hand-picks 5 representative
cases:
  - one clearly overpriced (actual >> p75)
  - one clearly underpriced (actual << p25)
  - two in-band examples
  - one edge case where the interval barely contains the actual

Outputs:
  - demo/example_listings.json (the 5 demo records with full responses)
  - demo/scored_listings.csv   (every scored real listing in scope)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_loader import load_model  # noqa: E402
from app.rent_predictor import FEATURE_COLUMNS  # noqa: E402

FEATURE_PATH = ROOT / "data" / "features" / "listings_features.csv"
DEMO_DIR = ROOT / "demo"
DEMO_JSON = DEMO_DIR / "example_listings.json"
DEMO_SAMPLE_CSV = DEMO_DIR / "scored_listings.csv"

DEDUP_KEYS = ["zip_code", "bedrooms", "bathrooms", "rent_usd"]


def _scope_2000_8000(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["rent_usd"] >= 2_000) & (df["rent_usd"] <= 8_000)].copy()


def _build_listing_payload(row: pd.Series) -> dict:
    payload: dict = {}
    for col in FEATURE_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            continue
        payload[col] = float(value) if col != "zip_code" else str(value)
    payload["actual_rent_usd"] = float(row["rent_usd"])
    return payload


def _classify(actual: float, p25: float | None, p75: float | None, point: float) -> str:
    if p25 is None or p75 is None:
        if actual > point * 1.10:
            return "overpriced"
        if actual < point * 0.90:
            return "underpriced"
        return "in_band"
    if actual > p75 * 1.10:
        return "overpriced_clear"
    if actual < p25 * 0.90:
        return "underpriced_clear"
    if abs(actual - p75) <= 0.05 * (p75 - p25) or abs(actual - p25) <= 0.05 * (p75 - p25):
        return "edge_case"
    if p25 <= actual <= p75:
        return "in_band"
    return "out_of_band_minor"


def _select_demo_cases(scored: pd.DataFrame) -> list[dict]:
    chosen: list[dict] = []
    used_zips: set[str] = set()

    overpriced = scored.loc[scored["category"] == "overpriced_clear"].sort_values(
        "delta_above_p75", ascending=False
    )
    underpriced = scored.loc[scored["category"] == "underpriced_clear"].sort_values(
        "delta_below_p25", ascending=False
    )
    in_band = scored.loc[scored["category"] == "in_band"].sort_values("interval_width", ascending=False)
    edge = scored.loc[scored["category"] == "edge_case"]

    def _pick(group: pd.DataFrame, label: str, n: int = 1) -> None:
        for _, row in group.iterrows():
            if row["zip_code"] in used_zips:
                continue
            chosen.append({**row.to_dict(), "demo_label": label})
            used_zips.add(row["zip_code"])
            if len([c for c in chosen if c["demo_label"] == label]) >= n:
                return

    _pick(overpriced, "overpriced", n=1)
    _pick(underpriced, "underpriced", n=1)
    _pick(in_band, "in_band", n=2)
    if not edge.empty:
        _pick(edge, "edge_case", n=1)
    else:
        backup = in_band.assign(distance_to_p75=lambda d: (d["fair_rent_p75"] - d["actual_rent_usd"]).abs())
        backup = backup.sort_values("distance_to_p75").reset_index(drop=True)
        _pick(backup, "edge_case", n=1)

    return chosen


def main() -> None:
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = (ROOT / "mlruns").as_uri()
    model_uri = os.environ.get("MLFLOW_MODEL_URI")
    if not model_uri:
        raise SystemExit(
            "MLFLOW_MODEL_URI must point to the registered RentIQ model "
            "(e.g. runs:/<id>/model)."
        )

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    df = _scope_2000_8000(df)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    # Score every in-scope listing, deduped on the keys that identify a unit.
    # `dropna=False` keeps rows where bathrooms is NaN instead of collapsing them.
    sample = (
        df.drop_duplicates(subset=DEDUP_KEYS)
        .sort_values(["zip_code", "bedrooms", "bathrooms", "rent_usd"], na_position="last")
        .reset_index(drop=True)
    )

    loaded = load_model()
    if loaded.source != "mlflow":
        raise SystemExit("Model registry not loaded; cannot generate demo pack.")

    records: list[dict] = []
    for _, row in sample.iterrows():
        payload = _build_listing_payload(row)
        response = loaded.flag_overpriced(payload)
        actual = float(payload["actual_rent_usd"])
        p25 = response.get("fair_rent_p25")
        p75 = response.get("fair_rent_p75")
        point = float(response["predicted_rent_usd"])
        category = _classify(actual, p25, p75, point)
        records.append(
            {
                "zip_code": payload["zip_code"],
                "bedrooms": payload.get("bedrooms"),
                "bathrooms": payload.get("bathrooms"),
                "actual_rent_usd": actual,
                "predicted_rent_usd": point,
                "fair_rent_p25": p25,
                "fair_rent_p75": p75,
                "delta_usd": response.get("delta_usd"),
                "delta_pct": response.get("delta_pct"),
                "flag_overpriced": response.get("flag_overpriced"),
                "flag_reason": response.get("flag_reason"),
                "category": category,
                "delta_above_p75": (actual - p75) if p75 is not None else None,
                "delta_below_p25": (p25 - actual) if p25 is not None else None,
                "interval_width": (p75 - p25) if p25 is not None and p75 is not None else None,
                "request_payload": payload,
                "response": response,
            }
        )

    scored = pd.DataFrame(records)
    scored.to_csv(DEMO_SAMPLE_CSV, index=False)
    print(f"Scored {len(scored)} listings -> {DEMO_SAMPLE_CSV}")

    demo_cases = _select_demo_cases(scored)
    if len(demo_cases) < 5:
        # Fall back: top up with strongest in-band examples we haven't used.
        used_zips = {c["zip_code"] for c in demo_cases}
        extra = scored.loc[
            (scored["category"] == "in_band") & (~scored["zip_code"].isin(used_zips))
        ].head(5 - len(demo_cases))
        for _, row in extra.iterrows():
            demo_cases.append({**row.to_dict(), "demo_label": "in_band"})

    payload_out = {
        "generated_with": {
            "feature_table": str(FEATURE_PATH.relative_to(ROOT)),
            "n_sampled_listings": int(len(sample)),
            "dedup_keys": DEDUP_KEYS,
            "model_uri": model_uri,
            "model_version": os.environ.get("MLFLOW_MODEL_VERSION"),
            "scope": "rent_usd in [2000, 8000]",
        },
        "examples": [
            {
                "demo_label": case["demo_label"],
                "zip_code": case["zip_code"],
                "actual_rent_usd": case["actual_rent_usd"],
                "predicted_rent_usd": case["predicted_rent_usd"],
                "fair_rent_p25": case.get("fair_rent_p25"),
                "fair_rent_p75": case.get("fair_rent_p75"),
                "flag_overpriced": case.get("flag_overpriced"),
                "flag_reason": case.get("flag_reason"),
                "delta_usd": case.get("delta_usd"),
                "delta_pct": case.get("delta_pct"),
                "request_payload": case["request_payload"],
                "response": case["response"],
            }
            for case in demo_cases
        ],
    }
    DEMO_JSON.write_text(
        json.dumps(payload_out, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)),
        encoding="utf-8",
    )
    print(f"Wrote {len(demo_cases)} demo examples -> {DEMO_JSON}")
    for case in demo_cases:
        print(
            f"  - {case['demo_label']:<13} zip={case['zip_code']} "
            f"actual=${case['actual_rent_usd']:.0f} "
            f"point=${case['predicted_rent_usd']:.0f} "
            f"p25=${case.get('fair_rent_p25') or 0:.0f} "
            f"p75=${case.get('fair_rent_p75') or 0:.0f}"
        )


if __name__ == "__main__":
    main()
