#!/usr/bin/env python3
"""
Build a unified training table for listing-level rent modeling.

Outputs:
  - data/features/listings_features.csv
  - data/features/listings_features.parquet
  - data/features/listings_features_manifest.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import data_cleaning as dc

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SCRAPED_DIR = REPO_ROOT / "data" / "scraped"
FEATURES_DIR = REPO_ROOT / "data" / "features"

LISTINGS_PROCESSED_PATH = PROCESSED_DIR / "listings_clean.csv"
APARTMENTS_CLEAN_PATH = SCRAPED_DIR / "sf_apartments_listings_clean.csv"
CENSUS_PATH = PROCESSED_DIR / "census_clean.csv"
CRIME_PATH = PROCESSED_DIR / "crime_zip_month.csv"
REDFIN_PATH = PROCESSED_DIR / "redfin_metro_clean.csv"
ZORI_RAW_PATH = REPO_ROOT / "data" / "raw" / "zillow_observed_rent_index.csv"
ZHVI_RAW_PATH = REPO_ROOT / "data" / "raw" / "zillow_home_value_index.csv"

OUT_CSV = FEATURES_DIR / "listings_features.csv"
OUT_PARQUET = FEATURES_DIR / "listings_features.parquet"
OUT_MANIFEST = FEATURES_DIR / "listings_features_manifest.json"

ZIP_RE = re.compile(r"\b(\d{5})\b")
RENT_RE = re.compile(r"\$\s*([\d,]+)")
BED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Beds?\+?|\bStudio\b", re.IGNORECASE)
BATH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Baths?", re.IGNORECASE)


def _extract_zip_from_text(text: object) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    m = ZIP_RE.search(str(text))
    return m.group(1) if m else None


def _extract_rent_from_text(text: object) -> float | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    m = RENT_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_beds_baths(text: object) -> tuple[float | None, float | None]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None, None
    s = str(text)
    bed_match = BED_RE.search(s)
    bath_match = BATH_RE.search(s)

    bedrooms = None
    bathrooms = None

    if bed_match:
        token = bed_match.group(0)
        if "studio" in token.lower():
            bedrooms = 0.0
        else:
            try:
                bedrooms = float(bed_match.group(1))
            except (TypeError, ValueError):
                bedrooms = None

    if bath_match:
        try:
            bathrooms = float(bath_match.group(1))
        except (TypeError, ValueError):
            bathrooms = None

    return bedrooms, bathrooms


def _load_base_listings() -> pd.DataFrame:
    if not LISTINGS_PROCESSED_PATH.is_file():
        raise FileNotFoundError(f"Missing listings file: {LISTINGS_PROCESSED_PATH}")
    base = pd.read_csv(LISTINGS_PROCESSED_PATH, low_memory=False)
    base["zip_code"] = dc.standardize_zip(base.get("zip_code"))
    base["date"] = pd.to_datetime(base.get("date"), errors="coerce")
    base["rent_usd"] = pd.to_numeric(base.get("rent_usd"), errors="coerce")
    for col in ("walk_score", "transit_score"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        else:
            base[col] = np.nan
    return base


def _load_apartments_augmented() -> pd.DataFrame:
    cols = ["zip_code", "date", "rent_usd", "beds_baths", "url", "walk_score", "transit_score"]
    if not APARTMENTS_CLEAN_PATH.is_file():
        return pd.DataFrame(columns=cols)

    apt = pd.read_csv(APARTMENTS_CLEAN_PATH, low_memory=False)
    apt["rent_usd"] = apt.get("pricing", pd.Series(index=apt.index)).map(_extract_rent_from_text)
    apt["zip_code"] = dc.standardize_zip(
        apt.get("zip_code", pd.Series(index=apt.index)).fillna(
            apt.get("address", pd.Series(index=apt.index)).map(_extract_zip_from_text)
        )
    )
    apt["date"] = pd.to_datetime(apt.get("scraped_at"), errors="coerce")
    apt["beds_baths"] = apt.get("beds_baths", pd.Series(index=apt.index)).astype("string")
    apt["url"] = apt.get("url", pd.Series(index=apt.index))
    for col in ("walk_score", "transit_score"):
        if col in apt.columns:
            apt[col] = pd.to_numeric(apt[col], errors="coerce")
        else:
            apt[col] = np.nan

    apt = apt[cols].copy()
    good = apt["zip_code"].notna() & apt["rent_usd"].notna()
    apt = apt.loc[good].reset_index(drop=True)
    return apt


def _prepare_listings() -> pd.DataFrame:
    base = _load_base_listings()
    apartments = _load_apartments_augmented()

    base_cols = ["zip_code", "date", "rent_usd", "beds_baths", "url", "walk_score", "transit_score"]
    base_min = base.reindex(columns=base_cols)
    apartments = apartments.reindex(columns=base_cols)
    merged = pd.concat([base_min, apartments], ignore_index=True)
    merged["zip_code"] = dc.standardize_zip(merged["zip_code"])
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["rent_usd"] = pd.to_numeric(merged["rent_usd"], errors="coerce")
    merged["beds_baths"] = merged["beds_baths"].astype("string")
    for col in ("walk_score", "transit_score"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    valid = merged["zip_code"].notna() & merged["rent_usd"].notna()
    merged = merged.loc[valid].reset_index(drop=True)

    merged = merged.drop_duplicates(subset=["zip_code", "date", "rent_usd", "beds_baths", "url"], keep="first")

    parsed = merged["beds_baths"].map(_parse_beds_baths)
    merged["bedrooms"] = parsed.map(lambda x: x[0])
    merged["bathrooms"] = parsed.map(lambda x: x[1])
    merged["bedrooms"] = pd.to_numeric(merged["bedrooms"], errors="coerce")
    merged["bathrooms"] = pd.to_numeric(merged["bathrooms"], errors="coerce")
    return merged


def _crime_panel() -> pd.DataFrame:
    """Return zip+month crime values plus a per-zip latest fallback."""
    if not CRIME_PATH.is_file():
        return pd.DataFrame(columns=["zip_code", "month", "crime_total_month_zip_log1p_latest"])
    crime = pd.read_csv(CRIME_PATH, low_memory=False)
    crime["zip_code"] = dc.standardize_zip(crime["zip_code"])
    crime["month"] = pd.to_datetime(crime["month"], errors="coerce")
    crime["crime_total_month_zip_log1p"] = pd.to_numeric(crime["crime_total_month_zip_log1p"], errors="coerce")
    crime = crime.dropna(subset=["zip_code", "month"])
    crime = crime.rename(columns={"crime_total_month_zip_log1p": "crime_total_month_zip_log1p_latest"})
    return crime[["zip_code", "month", "crime_total_month_zip_log1p_latest"]]


def _zillow_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-format ZORI and ZHVI panels keyed by (zip_code, month)."""
    zori = dc.clean_zori(ZORI_RAW_PATH)
    zhvi = dc.clean_zhvi(ZHVI_RAW_PATH)

    zori_panel = pd.DataFrame(columns=["zip_code", "month", "zori_baseline"])
    zhvi_panel = pd.DataFrame(columns=["zip_code", "month", "zhvi_level", "zhvi_12mo_delta"])

    if zori is not None:
        zori = zori[["zip_code", "date", "zori_rent"]].copy()
        zori["zip_code"] = dc.standardize_zip(zori["zip_code"])
        zori["month"] = pd.to_datetime(zori["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        zori = zori.dropna(subset=["zip_code", "month"])
        zori_panel = (
            zori.groupby(["zip_code", "month"], as_index=False)["zori_rent"].mean()
            .rename(columns={"zori_rent": "zori_baseline"})
        )

    if zhvi is not None:
        zhvi = zhvi[["zip_code", "date", "zhvi_value"]].copy()
        zhvi["zip_code"] = dc.standardize_zip(zhvi["zip_code"])
        zhvi["month"] = pd.to_datetime(zhvi["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        zhvi = zhvi.dropna(subset=["zip_code", "month"])
        zhvi = zhvi.sort_values(["zip_code", "month"])
        zhvi["zhvi_12mo_delta"] = zhvi.groupby("zip_code")["zhvi_value"].diff(12)
        zhvi_panel = zhvi.rename(columns={"zhvi_value": "zhvi_level"})[
            ["zip_code", "month", "zhvi_level", "zhvi_12mo_delta"]
        ]

    return zori_panel, zhvi_panel


def _redfin_panel() -> pd.DataFrame:
    """Return per-month metro momentum keyed by listing_month."""
    cols = ["month", "redfin_mom_pct", "redfin_yoy_pct"]
    if not REDFIN_PATH.is_file():
        return pd.DataFrame(columns=cols)
    redfin = pd.read_csv(REDFIN_PATH, low_memory=False)
    redfin["period_end"] = pd.to_datetime(redfin.get("period_end"), errors="coerce")
    redfin["median_asking_rent_mom_pct"] = pd.to_numeric(redfin.get("median_asking_rent_mom_pct"), errors="coerce")
    redfin["median_asking_rent_yoy_pct"] = pd.to_numeric(redfin.get("median_asking_rent_yoy_pct"), errors="coerce")

    if "region" in redfin.columns:
        mask = redfin["region"].astype("string").str.contains("San Francisco", case=False, na=False)
        if mask.any():
            redfin = redfin.loc[mask].copy()

    redfin = redfin.dropna(subset=["period_end"])
    if redfin.empty:
        return pd.DataFrame(columns=cols)

    redfin["month"] = redfin["period_end"].dt.to_period("M").dt.to_timestamp()
    panel = (
        redfin.groupby("month", as_index=False)
        .agg(
            redfin_mom_pct=("median_asking_rent_mom_pct", "mean"),
            redfin_yoy_pct=("median_asking_rent_yoy_pct", "mean"),
        )
    )
    return panel[cols]


def _point_in_time_zip_month_join(
    out: pd.DataFrame,
    panel: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Left-merge a (zip, month) panel onto listings, then fill gaps with the per-zip
    most-recent observation that occurred on or before each listing's month."""
    if panel.empty:
        for col in value_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.dropna(subset=["zip_code", "month"]).sort_values(["zip_code", "month"])
    out = out.merge(
        panel,
        left_on=["zip_code", "listing_month"],
        right_on=["zip_code", "month"],
        how="left",
    )
    out = out.drop(columns=[c for c in ["month"] if c in out.columns])

    needs_fill = pd.Series(False, index=out.index)
    for col in value_cols:
        needs_fill = needs_fill | out[col].isna()
    needs_fill = needs_fill & out["listing_month"].notna() & out["zip_code"].notna()
    if needs_fill.any():
        left = out.loc[needs_fill, ["zip_code", "listing_month"]].copy()
        left = left.reset_index().sort_values(["zip_code", "listing_month"])
        right = panel.sort_values(["month", "zip_code"]).copy()
        filled = pd.merge_asof(
            left,
            right,
            left_on="listing_month",
            right_on="month",
            by="zip_code",
            direction="backward",
        )
        filled = filled.set_index("index")
        for col in value_cols:
            out.loc[filled.index, col] = out.loc[filled.index, col].fillna(filled[col])

    # Per-zip latest fallback for rows with missing listing_month or zip without earlier panel rows.
    latest = (
        panel.sort_values(["zip_code", "month"]).groupby("zip_code", as_index=False).tail(1)
    )[["zip_code", *value_cols]]
    for col in value_cols:
        latest_map = dict(zip(latest["zip_code"], latest[col]))
        mask = out[col].isna() & out["zip_code"].notna()
        if mask.any():
            out.loc[mask, col] = out.loc[mask, "zip_code"].map(latest_map)
    return out


def _point_in_time_month_join(
    out: pd.DataFrame,
    panel: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    if panel.empty:
        for col in value_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.dropna(subset=["month"]).sort_values("month")
    out = out.merge(panel, left_on="listing_month", right_on="month", how="left")
    out = out.drop(columns=[c for c in ["month"] if c in out.columns])

    needs_fill = pd.Series(False, index=out.index)
    for col in value_cols:
        needs_fill = needs_fill | out[col].isna()
    needs_fill = needs_fill & out["listing_month"].notna()
    if needs_fill.any():
        left = out.loc[needs_fill, ["listing_month"]].copy()
        left = left.reset_index().sort_values("listing_month")
        filled = pd.merge_asof(
            left,
            panel,
            left_on="listing_month",
            right_on="month",
            direction="backward",
        )
        filled = filled.set_index("index")
        for col in value_cols:
            out.loc[filled.index, col] = out.loc[filled.index, col].fillna(filled[col])

    # Single-row fallback to most recent observation for rows with missing listing_month.
    latest_row = panel.sort_values("month").tail(1)
    if not latest_row.empty:
        for col in value_cols:
            value = latest_row[col].iloc[0]
            mask = out[col].isna()
            if mask.any() and pd.notna(value):
                out.loc[mask, col] = value
    return out


def _join_features(listings: pd.DataFrame) -> pd.DataFrame:
    out = listings.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["listing_month"] = out["date"].dt.to_period("M").dt.to_timestamp()

    if CENSUS_PATH.is_file():
        census = pd.read_csv(CENSUS_PATH, low_memory=False)
        census["zip_code"] = dc.standardize_zip(census["zip_code"])
        census_cols = ["zip_code", "census_median_income", "census_renter_ratio", "census_vacancy_rate"]
        out = out.merge(census[census_cols], on="zip_code", how="left")
    else:
        for col in ("census_median_income", "census_renter_ratio", "census_vacancy_rate"):
            out[col] = np.nan

    out = _point_in_time_zip_month_join(out, _crime_panel(), ["crime_total_month_zip_log1p_latest"])

    zori_panel, zhvi_panel = _zillow_panel()
    out = _point_in_time_zip_month_join(out, zori_panel, ["zori_baseline"])
    out = _point_in_time_zip_month_join(out, zhvi_panel, ["zhvi_level", "zhvi_12mo_delta"])

    out = _point_in_time_month_join(out, _redfin_panel(), ["redfin_mom_pct", "redfin_yoy_pct"])

    out["bedrooms_x_census_income"] = (
        pd.to_numeric(out.get("bedrooms"), errors="coerce")
        * pd.to_numeric(out.get("census_median_income"), errors="coerce")
    )
    out["walk_score_x_transit_score"] = (
        pd.to_numeric(out.get("walk_score"), errors="coerce")
        * pd.to_numeric(out.get("transit_score"), errors="coerce")
    )

    ordered = [
        "zip_code",
        "date",
        "listing_month",
        "rent_usd",
        "beds_baths",
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
    for col in ordered:
        if col not in out.columns:
            out[col] = np.nan
    out = out[ordered].copy()
    return out


def _build_manifest(df: pd.DataFrame) -> dict:
    null_rates = {c: float(df[c].isna().mean()) for c in df.columns}

    target = pd.to_numeric(df["rent_usd"], errors="coerce")
    target_distribution = {
        "count": int(target.notna().sum()),
        "mean": float(target.mean()) if target.notna().any() else None,
        "median": float(target.median()) if target.notna().any() else None,
        "std": float(target.std()) if target.notna().any() else None,
        "min": float(target.min()) if target.notna().any() else None,
        "p10": float(target.quantile(0.10)) if target.notna().any() else None,
        "p25": float(target.quantile(0.25)) if target.notna().any() else None,
        "p75": float(target.quantile(0.75)) if target.notna().any() else None,
        "p90": float(target.quantile(0.90)) if target.notna().any() else None,
        "max": float(target.max()) if target.notna().any() else None,
    }

    numeric = df.select_dtypes(include=[np.number]).copy()
    corr_entries: list[dict[str, float | str]] = []
    if "rent_usd" in numeric.columns:
        corr = numeric.corr(numeric_only=True)["rent_usd"].drop(labels=["rent_usd"], errors="ignore").dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
        for feature_name, value in corr.items():
            corr_entries.append({"feature": feature_name, "corr_with_rent_usd": float(value)})

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "null_rates": null_rates,
        "target_distribution": target_distribution,
        "basic_correlation_check": corr_entries,
    }


def build() -> pd.DataFrame:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    listings = _prepare_listings()
    features = _join_features(listings)

    features = features.sort_values(["date", "zip_code", "rent_usd"], na_position="last").reset_index(drop=True)
    features.to_csv(OUT_CSV, index=False)
    features.to_parquet(OUT_PARQUET, index=False)

    manifest = _build_manifest(features)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return features


if __name__ == "__main__":
    out_df = build()
    print(f"Wrote {len(out_df):,} rows -> {OUT_CSV}")
    print(f"Wrote parquet -> {OUT_PARQUET}")
    print(f"Wrote manifest -> {OUT_MANIFEST}")
