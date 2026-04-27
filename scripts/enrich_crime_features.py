#!/usr/bin/env python3
"""
Build richer ZIP-month crime features and merge them into cleaned listings.

Inputs by default:
    data/processed/crime_zip_month.csv
    data/raw/sf_zip_polygons.json
    data/raw/sf_crime_incidents.csv
    data/processed/listings_clean.csv

Outputs by default:
    data/processed/crime_features_v2.csv
    data/processed/crime_category_breakdown.csv
    data/processed/listings_clean.csv

Run from repo root:
    python scripts/enrich_crime_features.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_cleaning import _assign_zip_by_point, standardize_zip

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DEFAULT_CRIME_ZIP_MONTH = PROCESSED_DIR / "crime_zip_month.csv"
DEFAULT_POLYGONS = RAW_DIR / "sf_zip_polygons.json"
DEFAULT_INCIDENTS = RAW_DIR / "sf_crime_incidents.csv"
DEFAULT_FEATURES_OUT = PROCESSED_DIR / "crime_features_v2.csv"
DEFAULT_BREAKDOWN_OUT = PROCESSED_DIR / "crime_category_breakdown.csv"
DEFAULT_LISTINGS = PROCESSED_DIR / "listings_clean.csv"

COUNT_COLUMNS = (
    "crime_total_month_zip",
    "crime_violent_month_zip",
    "crime_property_month_zip",
)

ROLLING_FEATURE_COLUMNS = (
    "crime_total_3mo_avg",
    "crime_total_12mo_avg",
    "crime_violent_3mo_avg",
    "crime_violent_12mo_avg",
    "crime_property_3mo_avg",
    "crime_property_12mo_avg",
)

NORMALIZED_FEATURE_COLUMNS = (
    "violent_share",
    "crime_trend",
    "crime_rate_per_1k",
)

MERGE_FEATURE_COLUMNS = (
    *COUNT_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
    *NORMALIZED_FEATURE_COLUMNS,
)


def _load_zip_population(path: Path | str) -> pd.DataFrame:
    """Return ``zip_code`` and population from the deck.gl ZIP polygon file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for entry in raw:
        zip_code = standardize_zip(pd.Series([entry.get("zipcode")])).iloc[0]
        if pd.isna(zip_code):
            continue
        rows.append(
            {
                "zip_code": str(zip_code),
                "zip_population": pd.to_numeric(entry.get("population"), errors="coerce"),
            }
        )
    pop = pd.DataFrame(rows, columns=["zip_code", "zip_population"])
    if pop.empty:
        return pop
    pop = pop.dropna(subset=["zip_population"]).drop_duplicates("zip_code", keep="first")
    pop["zip_population"] = pop["zip_population"].astype("float64")
    return pop


def _complete_monthly_grid(crime: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ZIP-month count rows with zeroes before rolling windows."""
    if crime.empty:
        return crime.copy()

    work = crime.copy()
    work["zip_code"] = standardize_zip(work["zip_code"])
    work["month"] = pd.to_datetime(work["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work = work.dropna(subset=["zip_code", "month"])

    for col in COUNT_COLUMNS:
        if col not in work.columns:
            raise ValueError(f"crime_zip_month.csv missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype("int64")

    min_month = work["month"].min()
    max_month = work["month"].max()
    months = pd.date_range(min_month, max_month, freq="MS")
    zips = sorted(set(work["zip_code"].dropna()) | set(population.get("zip_code", pd.Series(dtype="string")).dropna()))

    grid = pd.MultiIndex.from_product([zips, months], names=["zip_code", "month"]).to_frame(index=False)
    out = grid.merge(work[["zip_code", "month", *COUNT_COLUMNS]], on=["zip_code", "month"], how="left")
    for col in COUNT_COLUMNS:
        out[col] = out[col].fillna(0).astype("int64")
    return out


def build_crime_features(
    crime_zip_month_path: Path | str = DEFAULT_CRIME_ZIP_MONTH,
    zip_polygons_path: Path | str = DEFAULT_POLYGONS,
) -> pd.DataFrame:
    """Build rolling and population-normalized ZIP-month crime features."""
    crime_zip_month_path = Path(crime_zip_month_path)
    zip_polygons_path = Path(zip_polygons_path)
    if not crime_zip_month_path.is_file():
        raise FileNotFoundError(f"Missing crime ZIP-month input: {crime_zip_month_path}")
    if not zip_polygons_path.is_file():
        raise FileNotFoundError(f"Missing ZIP polygons input: {zip_polygons_path}")

    crime = pd.read_csv(crime_zip_month_path, low_memory=False)
    required = {"zip_code", "month", *COUNT_COLUMNS}
    missing = required - set(crime.columns)
    if missing:
        raise ValueError(f"crime_zip_month.csv missing required columns: {sorted(missing)}")

    population = _load_zip_population(zip_polygons_path)
    features = _complete_monthly_grid(crime, population)
    if features.empty:
        return pd.DataFrame(columns=["zip_code", "month", "zip_population", *MERGE_FEATURE_COLUMNS])

    features = features.sort_values(["zip_code", "month"]).reset_index(drop=True)
    grouped = features.groupby("zip_code", group_keys=False)

    rolling_specs = {
        "crime_total_month_zip": "crime_total",
        "crime_violent_month_zip": "crime_violent",
        "crime_property_month_zip": "crime_property",
    }
    for source_col, prefix in rolling_specs.items():
        features[f"{prefix}_3mo_avg"] = grouped[source_col].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        features[f"{prefix}_12mo_avg"] = grouped[source_col].transform(
            lambda s: s.rolling(window=12, min_periods=1).mean()
        )

    total = features["crime_total_month_zip"].astype("float64")
    violent = features["crime_violent_month_zip"].astype("float64")
    features["violent_share"] = np.divide(violent, total, out=np.zeros(len(features), dtype="float64"), where=total > 0)

    trend_denominator = features["crime_total_12mo_avg"].astype("float64")
    trend_numerator = features["crime_total_3mo_avg"].astype("float64") - trend_denominator
    features["crime_trend"] = np.divide(
        trend_numerator,
        trend_denominator,
        out=np.zeros(len(features), dtype="float64"),
        where=trend_denominator > 0,
    )

    features = features.merge(population, on="zip_code", how="left")
    population_values = features["zip_population"].astype("float64")
    features["crime_rate_per_1k"] = np.divide(
        total * 1000.0,
        population_values,
        out=np.full(len(features), np.nan, dtype="float64"),
        where=population_values > 0,
    )

    ordered_cols = ["zip_code", "month", "zip_population", *MERGE_FEATURE_COLUMNS]
    return features[ordered_cols]


def _load_zip_polygons_for_assignment(path: Path | str) -> dict[str, object]:
    """Load polygons for point-in-polygon assignment without requiring population."""
    from shapely.geometry import Polygon

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons: dict[str, object] = {}
    for entry in raw:
        zip_code = standardize_zip(pd.Series([entry.get("zipcode")])).iloc[0]
        contour = entry.get("contour") or []
        if pd.isna(zip_code) or len(contour) < 4:
            continue
        try:
            poly = Polygon([(float(lon), float(lat)) for lon, lat in contour])
        except (TypeError, ValueError):
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons[str(zip_code)] = poly
    return polygons


def _incidents_with_zip(incidents_path: Path | str, zip_polygons_path: Path | str) -> pd.DataFrame:
    """Load incidents, ensuring a standardized ``zip_code`` column exists."""
    incidents = pd.read_csv(incidents_path, low_memory=False)
    required = {"incident_date", "incident_category"}
    missing = required - set(incidents.columns)
    if missing:
        raise ValueError(f"sf_crime_incidents.csv missing required columns: {sorted(missing)}")

    incidents["incident_date"] = pd.to_datetime(incidents["incident_date"], errors="coerce")
    incidents = incidents.loc[incidents["incident_date"].notna()].copy()

    if "zip_code" in incidents.columns:
        incidents["zip_code"] = standardize_zip(incidents["zip_code"])
    else:
        coord_required = {"latitude", "longitude"}
        missing_coords = coord_required - set(incidents.columns)
        if missing_coords:
            raise ValueError(
                "sf_crime_incidents.csv needs either zip_code or latitude/longitude columns; "
                f"missing {sorted(missing_coords)}"
            )
        polygons = _load_zip_polygons_for_assignment(zip_polygons_path)
        incidents["zip_code"] = _assign_zip_by_point(incidents["latitude"], incidents["longitude"], polygons)

    incidents["incident_category"] = incidents["incident_category"].astype("string").fillna("Unknown")
    return incidents.loc[incidents["zip_code"].notna()].reset_index(drop=True)


def build_category_breakdown(
    incidents_path: Path | str = DEFAULT_INCIDENTS,
    zip_polygons_path: Path | str = DEFAULT_POLYGONS,
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    """Aggregate top incident categories per ZIP over the latest 12 months."""
    incidents_path = Path(incidents_path)
    zip_polygons_path = Path(zip_polygons_path)
    if not incidents_path.is_file():
        raise FileNotFoundError(f"Missing incidents input: {incidents_path}")
    if not zip_polygons_path.is_file():
        raise FileNotFoundError(f"Missing ZIP polygons input: {zip_polygons_path}")

    incidents = _incidents_with_zip(incidents_path, zip_polygons_path)
    if incidents.empty:
        return pd.DataFrame(columns=["zip_code", "category", "count_12mo", "share_of_total"])

    max_date = incidents["incident_date"].max()
    start_date = max_date - pd.DateOffset(months=12)
    recent = incidents.loc[(incidents["incident_date"] > start_date) & (incidents["incident_date"] <= max_date)].copy()
    if recent.empty:
        return pd.DataFrame(columns=["zip_code", "category", "count_12mo", "share_of_total"])

    counts = (
        recent.groupby(["zip_code", "incident_category"], as_index=False)
        .size()
        .rename(columns={"incident_category": "category", "size": "count_12mo"})
    )
    totals = counts.groupby("zip_code", as_index=False)["count_12mo"].sum().rename(columns={"count_12mo": "_zip_total"})
    counts = counts.merge(totals, on="zip_code", how="left")
    counts["share_of_total"] = counts["count_12mo"] / counts["_zip_total"]
    counts = counts.sort_values(["zip_code", "count_12mo", "category"], ascending=[True, False, True])
    counts = counts.groupby("zip_code", group_keys=False).head(top_n)
    return counts.drop(columns=["_zip_total"]).reset_index(drop=True)


def merge_features_into_listings(
    listings_path: Path | str = DEFAULT_LISTINGS,
    features: Optional[pd.DataFrame] = None,
    features_path: Path | str = DEFAULT_FEATURES_OUT,
    output_path: Path | str = DEFAULT_LISTINGS,
) -> pd.DataFrame:
    """Left-join crime v2 features into listings by ZIP and listing month."""
    listings_path = Path(listings_path)
    features_path = Path(features_path)
    output_path = Path(output_path)
    if not listings_path.is_file():
        raise FileNotFoundError(f"Missing listings input: {listings_path}")

    listings = pd.read_csv(listings_path, low_memory=False)
    if features is None:
        if not features_path.is_file():
            raise FileNotFoundError(f"Missing crime features input: {features_path}")
        features = pd.read_csv(features_path, low_memory=False)

    if "zip_code" not in listings.columns or "date" not in listings.columns:
        raise ValueError("listings_clean.csv must contain zip_code and date columns")

    out = listings.copy()
    out["zip_code"] = standardize_zip(out["zip_code"])
    out["_listing_month"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    feat = features.copy()
    feat["zip_code"] = standardize_zip(feat["zip_code"])
    feat["_listing_month"] = pd.to_datetime(feat["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    feat_cols = ["zip_code", "_listing_month", *MERGE_FEATURE_COLUMNS]
    feat = feat[feat_cols].drop_duplicates(["zip_code", "_listing_month"], keep="last")

    existing_feature_cols = [col for col in MERGE_FEATURE_COLUMNS if col in out.columns]
    if existing_feature_cols:
        out = out.drop(columns=existing_feature_cols)

    n_rows = len(out)
    merged = out.merge(feat, on=["zip_code", "_listing_month"], how="left")
    assert len(merged) == n_rows, "left-join changed listings row count"

    for col in COUNT_COLUMNS:
        merged[col] = merged[col].fillna(0).astype("int64")
    for col in ROLLING_FEATURE_COLUMNS + NORMALIZED_FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype("float64")

    merged = merged.drop(columns=["_listing_month"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crime v2 features and merge them into listings_clean.csv.")
    parser.add_argument("--crime-zip-month", type=Path, default=DEFAULT_CRIME_ZIP_MONTH)
    parser.add_argument("--zip-polygons", type=Path, default=DEFAULT_POLYGONS)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES_OUT)
    parser.add_argument("--breakdown-output", type=Path, default=DEFAULT_BREAKDOWN_OUT)
    parser.add_argument("--listings", type=Path, default=DEFAULT_LISTINGS)
    parser.add_argument("--listings-output", type=Path, default=DEFAULT_LISTINGS)
    parser.add_argument("--top-categories", type=int, default=6)
    parser.add_argument("--skip-listings-merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    args.breakdown_output.parent.mkdir(parents=True, exist_ok=True)

    features = build_crime_features(args.crime_zip_month, args.zip_polygons)
    features.to_csv(args.features_output, index=False)
    print(f"[Write] {args.features_output} ({len(features)} rows)")

    breakdown = build_category_breakdown(args.incidents, args.zip_polygons, top_n=args.top_categories)
    breakdown.to_csv(args.breakdown_output, index=False)
    print(f"[Write] {args.breakdown_output} ({len(breakdown)} rows)")

    if not args.skip_listings_merge:
        listings = merge_features_into_listings(
            args.listings,
            features=features,
            output_path=args.listings_output,
        )
        print(f"[Write] {args.listings_output} ({len(listings)} rows)")


if __name__ == "__main__":
    main()
