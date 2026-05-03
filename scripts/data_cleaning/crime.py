"""SFPD incidents → ZIP-month crime feature aggregation + listings merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    CRIME_FEATURE_COLUMNS,
    CRIME_PROPERTY_CATEGORIES,
    CRIME_VIOLENT_CATEGORIES,
    PATH_CRIME,
    PATH_ZIP_POLYGONS,
)
from .geo import _assign_zip_by_point, _load_zip_polygons
from .utils import _filter_from_start_date, standardize_zip


def clean_crime(
    crime_path: Path | str = PATH_CRIME,
    zip_polygons_path: Path | str = PATH_ZIP_POLYGONS,
) -> pd.DataFrame | None:
    """
    SFPD incidents → (zip_code, month) feature table.

    Steps:
      1. Load ``sf_crime_incidents.csv`` (output of ``scripts/fetch_sf_crime.py``).
      2. Drop rows missing ``latitude`` / ``longitude`` or ``incident_date``.
      3. Filter to ``incident_date >= START_DATE``.
      4. Assign ``zip_code`` via point-in-polygon against SF ZIP polygons.
      5. Bucket ``incident_category`` into violent / property flags.
      6. Aggregate to (zip_code, month) with total + violent + property counts;
         add a ``log1p`` of total for skew control.

    Returns ``None`` when either input file is missing.
    """
    crime_path = Path(crime_path)
    zip_polygons_path = Path(zip_polygons_path)

    if not crime_path.is_file():
        print(f"[Crime] Skip — file not found: {crime_path}")
        return None
    if not zip_polygons_path.is_file():
        print(f"[Crime] Skip — ZIP polygons file not found: {zip_polygons_path}")
        return None

    df = pd.read_csv(crime_path, low_memory=False)
    print(f"[Crime] Loaded: {len(df)} rows")

    required = {"incident_date", "incident_category", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Crime CSV missing required columns: {sorted(missing)}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    dropped = int((~has_coords).sum())
    df = df.loc[has_coords].reset_index(drop=True)
    print(f"  Dropped {dropped} rows missing lat/lon → {len(df)}")

    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df = df.loc[df["incident_date"].notna()].reset_index(drop=True)
    df = _filter_from_start_date(df, "incident_date")
    if df.empty:
        print("[Crime] No incidents after date filter; returning empty feature table.")
        return pd.DataFrame(columns=["zip_code", "month", *CRIME_FEATURE_COLUMNS])

    polygons = _load_zip_polygons(zip_polygons_path)
    print(f"  Loaded {len(polygons)} ZIP polygons: {sorted(polygons)[:6]}...")

    df["zip_code"] = _assign_zip_by_point(df["latitude"], df["longitude"], polygons)
    assigned = df["zip_code"].notna().sum()
    print(f"  Assigned ZIP for {assigned}/{len(df)} incidents ({assigned / max(len(df), 1):.1%})")
    df = df.loc[df["zip_code"].notna()].reset_index(drop=True)

    df["month"] = df["incident_date"].dt.to_period("M").dt.to_timestamp()
    cat = df["incident_category"].astype("string").fillna("")
    df["_is_violent"] = cat.isin(CRIME_VIOLENT_CATEGORIES)
    df["_is_property"] = cat.isin(CRIME_PROPERTY_CATEGORIES)

    agg = (
        df.groupby(["zip_code", "month"], as_index=False)
        .agg(
            crime_total_month_zip=("incident_date", "size"),
            crime_violent_month_zip=("_is_violent", "sum"),
            crime_property_month_zip=("_is_property", "sum"),
        )
    )
    agg["crime_violent_month_zip"] = agg["crime_violent_month_zip"].astype("int64")
    agg["crime_property_month_zip"] = agg["crime_property_month_zip"].astype("int64")
    agg["crime_total_month_zip_log1p"] = np.log1p(agg["crime_total_month_zip"]).astype("float64")

    print(
        f"[Crime] Done: {len(agg)} (zip, month) rows; "
        f"zips={agg['zip_code'].nunique()}, "
        f"months {agg['month'].min().date()}..{agg['month'].max().date()}"
    )
    return agg


def merge_crime_into_listings(
    listings: pd.DataFrame,
    crime: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Left-join ZIP-month crime features onto listings.

    Listings keep their row count; missing (zip, month) combinations get ``0``
    for count features and ``log1p(0) = 0.0`` for the log feature so models
    receive a stable, non-null schema.
    """
    out = listings.copy()
    if "date" not in out.columns or "zip_code" not in out.columns:
        for col in CRIME_FEATURE_COLUMNS:
            out[col] = pd.NA
        return out

    out["_listing_month"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["zip_code"] = standardize_zip(out["zip_code"])

    if crime is None or crime.empty:
        for col in CRIME_FEATURE_COLUMNS:
            out[col] = 0 if col != "crime_total_month_zip_log1p" else 0.0
        out = out.drop(columns=["_listing_month"])
        print("[Crime/Merge] No crime features available; filled with zeros.")
        return out

    feat = crime.rename(columns={"month": "_listing_month"}).copy()
    feat["zip_code"] = standardize_zip(feat["zip_code"])

    n0 = len(out)
    merged = out.merge(feat, on=["zip_code", "_listing_month"], how="left")
    assert len(merged) == n0, "left-join changed listings row count"

    fill_zero = {
        "crime_total_month_zip": 0,
        "crime_violent_month_zip": 0,
        "crime_property_month_zip": 0,
        "crime_total_month_zip_log1p": 0.0,
    }
    for col, val in fill_zero.items():
        if col not in merged.columns:
            merged[col] = val
        else:
            merged[col] = merged[col].fillna(val)
    merged["crime_total_month_zip"] = merged["crime_total_month_zip"].astype("int64")
    merged["crime_violent_month_zip"] = merged["crime_violent_month_zip"].astype("int64")
    merged["crime_property_month_zip"] = merged["crime_property_month_zip"].astype("int64")
    merged["crime_total_month_zip_log1p"] = merged["crime_total_month_zip_log1p"].astype("float64")

    matched = (merged["crime_total_month_zip"] > 0).sum()
    print(f"[Crime/Merge] Joined crime features into {matched}/{n0} listing rows")
    return merged.drop(columns=["_listing_month"])
