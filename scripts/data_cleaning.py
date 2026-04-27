#!/usr/bin/env python3
"""
Single-file data cleaning for SF real-estate ML: ZORI, ZHVI, Census, Redfin, listings.

Scraped exports (e.g. apartments.com) live under ``data/scraped/*.csv``. Each CSV with
columns ``address`` and ``beds_baths`` is cleaned and concatenated into ``listings_clean.csv``.

Run from repo root:
  python scripts/data_cleaning.py

Expects raw paths under REPO_ROOT (see Config). Missing files are skipped with a message.
Final panel: inner join ZORI + ZHVI on (zip_code, date).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# 1. Imports (pandas + stdlib above)
# =============================================================================


# =============================================================================
# 2. Config
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

START_DATE = "2018-01-01"

SF_ZIPCODES = [
    "94102",
    "94103",
    "94104",
    "94105",
    "94107",
    "94108",
    "94109",
    "94110",
    "94111",
    "94112",
    "94114",
    "94115",
    "94116",
    "94117",
    "94118",
    "94121",
    "94122",
    "94123",
    "94124",
    "94127",
    "94128",
    "94129",
    "94130",
    "94131",
    "94132",
    "94133",
    "94134",
    "94158",
]

SF_ZIP_SET = frozenset(SF_ZIPCODES)

# ZORI: drop rows for these ZIPs after melt (sparse / missing in source)
ZORI_DROP_ZIPS = frozenset({"94104", "94111", "94127", "94128", "94130"})

# Zillow wide tables: preferred id columns (only those present in the file are used)
ZILLOW_ID_COLS = [
    "RegionName",
    "SizeRank",
    "City",
    "Metro",
    "CountyName",
    "State",
    "StateName",
]

DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Default input paths (override by editing this block or passing paths into cleaners)
PATH_ZORI = REPO_ROOT / "data" / "raw" / "zillow_observed_rent_index.csv"
PATH_ZHVI = REPO_ROOT / "data" / "raw" / "zillow_home_value_index.csv"
PATH_CENSUS = REPO_ROOT / "data" / "raw" / "census_acs_sf.csv"
PATH_REDFIN = REPO_ROOT / "data" / "raw" / "redfin_median_asking_rent.csv"
PATH_CRIME = REPO_ROOT / "data" / "raw" / "sf_crime_incidents.csv"
PATH_ZIP_POLYGONS = REPO_ROOT / "data" / "raw" / "sf_zip_polygons.json"

# All scrape outputs (CSV) go here; cleaning picks up every listings-shaped file
DIR_SCRAPED = REPO_ROOT / "data" / "scraped"
LISTINGS_SCRAPE_REQUIRED_COLS = frozenset({"address", "beds_baths"})

OUT_PROCESSED = REPO_ROOT / "data" / "processed"

# Listings: optional rent bounds (set to None to disable)
LISTINGS_RENT_MIN = 500.0
LISTINGS_RENT_MAX = 10000.0

# WalkScore enrichment columns appended to listings_clean.csv
WALKSCORE_COLUMNS = (
    "walk_score",
    "walk_description",
    "transit_score",
    "transit_description",
    "bike_score",
    "bike_description",
)

CENSUS_SENTINEL = -666666666

# SFPD ``incident_category`` values grouped into broad signal types. Anything
# not listed here counts toward ``crime_total_month_zip`` only.
CRIME_VIOLENT_CATEGORIES = frozenset({
    "Assault",
    "Robbery",
    "Sex Offense",
    "Homicide",
    "Weapons Offense",
    "Weapons Carrying Etc",
    "Weapons Offence",
    "Human Trafficking, Commercial Sex Acts",
    "Human Trafficking (A), Commercial Sex Acts",
    "Human Trafficking (B), Involuntary Servitude",
    "Human Trafficking",
    "Rape",
    "Offences Against The Family And Children",
})
CRIME_PROPERTY_CATEGORIES = frozenset({
    "Larceny Theft",
    "Burglary",
    "Motor Vehicle Theft",
    "Motor Vehicle Theft?",
    "Vandalism",
    "Malicious Mischief",
    "Arson",
    "Stolen Property",
    "Embezzlement",
    "Forgery And Counterfeiting",
    "Fraud",
    "Recovered Vehicle",
})

# Crime feature columns appended to listings after ZIP-month merge.
CRIME_FEATURE_COLUMNS = (
    "crime_total_month_zip",
    "crime_violent_month_zip",
    "crime_property_month_zip",
    "crime_total_month_zip_log1p",
)


# =============================================================================
# 3. Utility functions
# =============================================================================


def standardize_zip(zip_code: pd.Series | object) -> pd.Series:
    """Convert ZIP codes to string dtype, zero-padded to 5 digits (e.g. 9414 -> 09414)."""
    s = zip_code if isinstance(zip_code, pd.Series) else pd.Series([zip_code])
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    mask = num.notna() & (num >= 0) & (num <= 99_999)
    out.loc[mask] = num.loc[mask].round(0).astype("Int64").astype(str).str.zfill(5)
    return out


def parse_date(df: pd.DataFrame, column: str, *, errors: str = "coerce") -> pd.DataFrame:
    """Return a copy of ``df`` with ``column`` converted to timezone-naive datetime."""
    out = df.copy()
    dt = pd.to_datetime(out[column], errors=errors)
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    out[column] = dt
    return out


def _date_columns(columns: list[str]) -> list[str]:
    return sorted(c for c in columns if DATE_COL_RE.match(str(c)))


def _melt_zillow_wide(
    df: pd.DataFrame,
    *,
    zip_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Wide Zillow row → long rows with zip_code, date, value_name."""
    cols = list(df.columns)
    if zip_col not in cols:
        raise ValueError(f"Missing zip column {zip_col!r}")
    id_vars = [c for c in ZILLOW_ID_COLS if c in cols and c != zip_col]
    date_cols = _date_columns(cols)
    if not date_cols:
        raise ValueError("No YYYY-MM-DD columns found for melt")
    work = df.assign(zip_code=standardize_zip(df[zip_col]))
    long_df = work.melt(
        id_vars=id_vars + ["zip_code"],
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )
    long_df["date"] = pd.to_datetime(long_df["date"], format="%Y-%m-%d", errors="coerce")
    return long_df


def _filter_from_start_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    start = pd.Timestamp(START_DATE)
    out = df.loc[df[column] >= start].reset_index(drop=True)
    print(f"  After START_DATE (>={START_DATE}): {len(out)} rows")
    return out


# =============================================================================
# 4. Cleaning functions (one per dataset)
# =============================================================================


def clean_zori(path: Path | str) -> Optional[pd.DataFrame]:
    """
    ZORI wide CSV → long with zip_code, date, zori_rent.
    SF ZIPs only; START_DATE; drops known sparse ZIPs.
    """
    path = Path(path)
    if not path.is_file():
        print(f"[ZORI] Skip — file not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    print(f"[ZORI] Loaded wide: {len(df)} rows")

    z = standardize_zip(df["RegionName"])
    df = df.assign(_z=z)
    df = df.loc[df["_z"].isin(SF_ZIP_SET)].drop(columns=["_z"]).reset_index(drop=True)
    print(f"  After SF ZIP filter: {len(df)} wide rows")

    long_df = _melt_zillow_wide(df, zip_col="RegionName", value_name="zori_rent")
    print(f"  After melt: {len(long_df)} rows")

    long_df = _filter_from_start_date(long_df, "date")

    before = len(long_df)
    long_df = long_df.loc[~long_df["zip_code"].isin(ZORI_DROP_ZIPS)].reset_index(drop=True)
    print(f"  After dropping sparse ZIPs {sorted(ZORI_DROP_ZIPS)}: {len(long_df)} rows (dropped {before - len(long_df)})")

    keep_cols = [c for c in ["zip_code", "date", "zori_rent"] if c in long_df.columns]
    extra = [c for c in long_df.columns if c not in keep_cols]
    out = long_df[keep_cols + extra].copy()
    print(f"[ZORI] Done: {len(out)} rows, columns {list(out.columns)[:6]}...")
    return out


def clean_zhvi(path: Path | str) -> Optional[pd.DataFrame]:
    """
    ZHVI wide CSV (ISO-8859-1) → long with zip_code, date, zhvi_value.
    SF ZIP filter BEFORE melt; START_DATE filter after melt.
    """
    path = Path(path)
    if not path.is_file():
        print(f"[ZHVI] Skip — file not found: {path}")
        return None

    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    print(f"[ZHVI] Loaded wide: {len(df)} rows")

    z = standardize_zip(df["RegionName"])
    df = df.assign(_z=z)
    df = df.loc[df["_z"].isin(SF_ZIP_SET)].drop(columns=["_z"]).reset_index(drop=True)
    print(f"  After SF ZIP filter (pre-melt): {len(df)} wide rows")

    long_df = _melt_zillow_wide(df, zip_col="RegionName", value_name="zhvi_value")
    print(f"  After melt: {len(long_df)} rows")

    long_df = _filter_from_start_date(long_df, "date")

    keep_cols = [c for c in ["zip_code", "date", "zhvi_value"] if c in long_df.columns]
    extra = [c for c in long_df.columns if c not in keep_cols]
    out = long_df[keep_cols + extra].copy()
    print(f"[ZHVI] Done: {len(out)} rows")
    return out


def clean_census(path: Path | str) -> Optional[pd.DataFrame]:
    """Census ZIP-level static features: standardize zip_code; drop ZIP 94104."""
    path = Path(path)
    if not path.is_file():
        print(f"[Census] Skip — file not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    print(f"[Census] Loaded: {len(df)} rows")

    if "zip_code" not in df.columns:
        raise ValueError("Census CSV must contain column 'zip_code'")

    df = df.copy()
    df["zip_code"] = standardize_zip(df["zip_code"])

    before = len(df)
    df = df.loc[df["zip_code"] != "94104"].reset_index(drop=True)
    print(f"  Dropped ZIP 94104: {before} → {len(df)} rows")

    for col in df.columns:
        if col == "zip_code":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] == CENSUS_SENTINEL, col] = pd.NA

    print(f"[Census] Done: {len(df)} rows")
    return df


def _strip_money(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("$", "").replace(",", "").replace("%", "")
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _strip_pct(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("%", "").replace("$", "")
    s = re.sub(r"^[+]", "", s)
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def clean_redfin(path: Path | str) -> Optional[pd.DataFrame]:
    """
    Redfin metro-level TSV: utf-16, tab. No zip_code.
    Cleans $ / % numerics; parses period dates; drops Property Type, Bedrooms, size.
    """
    path = Path(path)
    if not path.is_file():
        print(f"[Redfin] Skip — file not found: {path}")
        return None

    df = pd.read_csv(path, encoding="utf-16", sep="\t", low_memory=False)
    print(f"[Redfin] Loaded: {len(df)} rows")

    drop_cols = [c for c in ("Property Type", "Bedrooms", "size") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped columns: {drop_cols}")

    # Date columns: StartMonth/EndMonth or Start/End
    start_col = end_col = None
    for a, b in (("StartMonth", "EndMonth"), ("Start", "End")):
        if a in df.columns and b in df.columns:
            start_col, end_col = a, b
            break
    if start_col is None:
        raise ValueError("Redfin file needs StartMonth/EndMonth or Start/End columns")

    df = parse_date(df, start_col)
    df = df.rename(columns={start_col: "period_start"})
    df = parse_date(df, end_col)
    df = df.rename(columns={end_col: "period_end"})

    rent_col = "Median Asking Rent"
    if rent_col in df.columns:
        df["median_asking_rent_usd"] = df[rent_col].map(_strip_money)
        df = df.drop(columns=[rent_col])
    mom = "Median Asking Rent MoM"
    if mom in df.columns:
        df["median_asking_rent_mom_pct"] = df[mom].map(_strip_pct)
        df = df.drop(columns=[mom])
    yoy = "Median Asking Rent YoY"
    if yoy in df.columns:
        df["median_asking_rent_yoy_pct"] = df[yoy].map(_strip_pct)
        df = df.drop(columns=[yoy])

    if "Region" in df.columns:
        df = df.rename(columns={"Region": "region"})

    print(f"[Redfin] Done: {len(df)} rows (metro-level; not joinable on zip_code)")
    return df


# =============================================================================
# 4b. Crime cleaning (SFPD incidents → ZIP-month features)
# =============================================================================


def _load_zip_polygons(path: Path | str) -> dict[str, "object"]:
    """
    Load a deck.gl-format SF ZIP polygons JSON into ``{zip_code: shapely.Polygon}``.

    Input format (``data/raw/sf_zip_polygons.json``)::

        [
            {"zipcode": 94110, "population": ..., "area": ...,
             "contour": [[lon, lat], [lon, lat], ...]},
            ...
        ]
    """
    from shapely.geometry import Polygon  # local import: keep module import light

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for entry in raw:
        zip_series = standardize_zip(pd.Series([entry.get("zipcode")]))
        zip_str = zip_series.iloc[0]
        if pd.isna(zip_str):
            continue
        contour = entry.get("contour") or []
        if len(contour) < 4:
            continue
        try:
            poly = Polygon([(float(lon), float(lat)) for lon, lat in contour])
        except (TypeError, ValueError):
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        out[str(zip_str)] = poly
    return out


def _assign_zip_by_point(
    latitude: pd.Series,
    longitude: pd.Series,
    polygons: dict[str, "object"],
) -> pd.Series:
    """
    Vectorized point-in-polygon ZIP assignment using shapely's STRtree.

    ``latitude`` / ``longitude`` are float Series (NaN allowed). Returns a
    string Series ``zip_code`` aligned with the input index, ``NA`` when no
    polygon contains the point or coordinates are missing.
    """
    import shapely
    from shapely.strtree import STRtree

    lat = pd.to_numeric(latitude, errors="coerce").to_numpy()
    lon = pd.to_numeric(longitude, errors="coerce").to_numpy()
    out = pd.array([pd.NA] * len(lat), dtype="string")

    if not polygons:
        return pd.Series(out, index=latitude.index, name="zip_code")

    zip_codes = list(polygons.keys())
    polys = list(polygons.values())
    tree = STRtree(polys)

    valid_idx = np.where(~(np.isnan(lat) | np.isnan(lon)))[0]
    if len(valid_idx) == 0:
        return pd.Series(out, index=latitude.index, name="zip_code")

    points = shapely.points(lon[valid_idx], lat[valid_idx])
    point_idxs, poly_idxs = tree.query(points, predicate="within")

    # One SF point should match at most one ZIP polygon. If polygon edges ever
    # produce duplicates, keeping the first match is deterministic.
    matched_points: set[int] = set()
    for point_idx, poly_idx in zip(point_idxs, poly_idxs):
        p = int(point_idx)
        if p in matched_points:
            continue
        out[int(valid_idx[p])] = zip_codes[int(poly_idx)]
        matched_points.add(p)

    return pd.Series(out, index=latitude.index, name="zip_code")


def clean_crime(
    crime_path: Path | str = PATH_CRIME,
    zip_polygons_path: Path | str = PATH_ZIP_POLYGONS,
) -> Optional[pd.DataFrame]:
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
    crime: Optional[pd.DataFrame],
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


ZIP_IN_ADDRESS = re.compile(r"\bCA\s+(\d{5})\b", re.IGNORECASE)
FIRST_RENT = re.compile(r"\$\s*([\d,]+)\+?")
# All floorplan + rent pairs in a pricing line (same listing, multiple units)
FLOORPLAN_RENT_RE = re.compile(
    r"(Studio|\d+\s+Beds\+|\d+\s+Beds|\d+\s+Bed)\s+\$\s*([\d,]+)\+?",
    re.IGNORECASE,
)
SQFT_RE = re.compile(r"([\d,]+)\s*sq\.?\s*ft\.?", re.IGNORECASE)
RAW_SQFT_COLUMNS = ("sqft", "square_feet", "sq_ft", "square_feet_listed")
LISTINGS_OUTPUT_COLUMNS = ["zip_code", "date", "rent_usd", "beds_baths", "sqft", "address", "url"]


def _extract_sqft_from_text(text: object) -> Optional[float]:
    """Parse first ``942 sq ft`` / ``1,100 sq ft`` style value from a string; ``None`` if absent."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    m = SQFT_RE.search(str(text))
    if not m:
        return None
    raw_num = m.group(1).replace(",", "")
    try:
        v = float(raw_num)
    except ValueError:
        return None
    return v if v > 0 else None


def _strip_sqft_from_label(label: str) -> str:
    """
    Remove ``942 sq ft`` / ``1,100 sq ft`` phrasing from a unit description.

    Fallback parsing used ``text before $`` as ``beds_baths``, which often still
    contained sqft; that number belongs in ``sqft`` only.
    """
    s = re.sub(r",?\s*[\d,]+\s*sq\.?\s*ft\.?", "", str(label), flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


def _parse_floorplans_from_text(text: object) -> list[tuple[str, float]]:
    """Return [(floorplan_label, rent_usd), ...] from a pricing or beds_baths string."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s = str(text).strip()
    if not s:
        return []
    pairs: list[tuple[str, float]] = []
    for m in FLOORPLAN_RENT_RE.finditer(s):
        label = m.group(1).strip()
        rent = float(m.group(2).replace(",", ""))
        pairs.append((label, rent))
    return pairs


def clean_listings(path: Path | str) -> Optional[pd.DataFrame]:
    """
    Scraped listings: one output row per (address, floorplan, rent).

    Parses all ``Studio`` / ``N Bed`` / ``N Beds`` / ``N Beds+`` segments from
    ``pricing`` when present (full multi-unit string), else falls back to ``beds_baths``.
    Adds ``sqft`` when a raw column (``sqft``, ``square_feet``, …) exists or when
    text contains patterns like ``942 sq ft``. ``beds_baths`` is layout-only (sqft
    phrasing stripped so it does not duplicate the ``sqft`` column).
    """
    path = Path(path)
    if not path.is_file():
        print(f"[Listings] Skip — file not found: {path}")
        return None

    raw = pd.read_csv(path, low_memory=False)
    n0 = len(raw)
    print(f"[Listings] Loaded: {n0} scrape rows")

    z = raw["address"].astype("string").str.extract(ZIP_IN_ADDRESS, expand=False).astype("string")
    if "title" in raw.columns:
        z = z.fillna(raw["title"].astype("string").str.extract(ZIP_IN_ADDRESS, expand=False).astype("string"))

    z_std = standardize_zip(z)

    dt = pd.to_datetime(raw["scraped_at"], errors="coerce") if "scraped_at" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)

    sqft_col = next((c for c in RAW_SQFT_COLUMNS if c in raw.columns), None)

    records: list[dict] = []
    for i in range(len(raw)):
        row = raw.iloc[i]
        zip_c = z_std.iloc[i]
        date_v = dt.iloc[i] if hasattr(dt, "iloc") else dt[i]

        pricing = row["pricing"] if "pricing" in raw.columns else pd.NA
        beds_baths = row["beds_baths"]
        text_for_units = pricing if pd.notna(pricing) and str(pricing).strip() else beds_baths
        plans = _parse_floorplans_from_text(text_for_units)

        if not plans:
            m = FIRST_RENT.search(str(beds_baths)) if pd.notna(beds_baths) else None
            if m:
                label = str(beds_baths).split("$", 1)[0].strip() or "unknown"
                plans = [(label, float(m.group(1).replace(",", "")))]

        explicit_sqft: Optional[float] = None
        if sqft_col is not None:
            v = pd.to_numeric(row[sqft_col], errors="coerce")
            if pd.notna(v):
                explicit_sqft = float(v)

        for floorplan, rent in plans:
            sqft_val = explicit_sqft
            if sqft_val is None:
                sqft_val = _extract_sqft_from_text(text_for_units)
            if sqft_val is None:
                sqft_val = _extract_sqft_from_text(beds_baths)

            beds_label = _strip_sqft_from_label(str(floorplan))

            records.append(
                {
                    "zip_code": zip_c,
                    "date": date_v,
                    "rent_usd": rent,
                    "beds_baths": beds_label,
                    "sqft": sqft_val,
                    "address": row["address"],
                    "url": row["url"] if "url" in raw.columns else pd.NA,
                }
            )

    out = pd.DataFrame.from_records(records, columns=LISTINGS_OUTPUT_COLUMNS)
    print(f"  Expanded scrape rows → floorplan rows: {n0} → {len(out)} (one row per unit type / price)")

    if out.empty:
        print(f"[Listings] Done ({Path(path).name}): 0 rows (no parsable rents)")
        return out

    out["rent_usd"] = pd.to_numeric(out["rent_usd"], errors="coerce")
    out["sqft"] = pd.to_numeric(out["sqft"], errors="coerce")

    n1 = len(out)
    key_ok = out["zip_code"].notna() & out["rent_usd"].notna() & out["date"].notna()
    dropped_missing = int((~key_ok).sum())
    out = out.loc[key_ok].reset_index(drop=True)
    print(f"  Dropped {dropped_missing} rows missing zip_code, rent_usd, or date ({n1} → {len(out)})")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if LISTINGS_RENT_MIN is not None:
        b = out["rent_usd"] >= LISTINGS_RENT_MIN
        d = int((~b).sum())
        out = out.loc[b].reset_index(drop=True)
        if d:
            print(f"  Dropped {d} rows with rent_usd < {LISTINGS_RENT_MIN}")
    if LISTINGS_RENT_MAX is not None:
        b = out["rent_usd"] <= LISTINGS_RENT_MAX
        d = int((~b).sum())
        out = out.loc[b].reset_index(drop=True)
        if d:
            print(f"  Dropped {d} rows with rent_usd > {LISTINGS_RENT_MAX}")

    out = _filter_from_start_date(out, "date")

    print(f"[Listings] Done ({Path(path).name}): {len(out)} rows (sqft non-null: {out['sqft'].notna().sum()})")
    return out


def discover_scraped_listing_csvs() -> list[Path]:
    """Return sorted paths under ``data/scraped/`` that look like apartments.com listing scrapes."""
    DIR_SCRAPED.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []
    for p in sorted(DIR_SCRAPED.glob("*.csv")):
        if p.stem.endswith("_clean"):
            print(f"[Listings] Skip {p.name} (derived clean listing file)")
            continue
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"[Listings] Skip unreadable {p.name}: {exc}")
            continue
        if LISTINGS_SCRAPE_REQUIRED_COLS <= cols:
            found.append(p)
        else:
            missing = LISTINGS_SCRAPE_REQUIRED_COLS - cols
            print(f"[Listings] Skip {p.name} (not a listings scrape; missing columns {sorted(missing)})")
    return found


def enrich_listings_with_walkscore(listings: pd.DataFrame) -> pd.DataFrame:
    """
    Append WalkScore / Transit / Bike score columns to cleaned listings.

    Geocodes ``address`` to lat/lon (Nominatim), then calls the WalkScore API
    via :mod:`fetch_walkscore`. With ``ensure_complete=True``, missing scores
    are imputed from ZIP/global means so every row has at least a
    ``walk_score`` populated.

    Failure modes:
      * No API key configured -> log and return null-filled columns (allows
        the rest of the pipeline to keep running locally).
      * Permanent API errors (invalid key, quota exceeded, IP blocked) ->
        re-raised as :class:`WalkScoreAPIError` so we never silently write a
        listings file with empty score columns.
      * Other transient errors -> logged with a non-null walk_score check at
        the end (which raises if any row is still missing a walk_score).
    """
    out = listings.copy()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("WALKSCORE_API_KEY"):
        print("[Listings] WALKSCORE_API_KEY not set; skipping WalkScore enrichment.")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    try:
        from fetch_walkscore import WalkScoreAPIError, enrich_walkscore  # local import
    except ImportError as exc:
        print(f"[Listings] WalkScore module unavailable ({exc}); skipping enrichment.")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    print(f"[Listings] Enriching {len(out)} rows with WalkScore (geocode + API)...")
    try:
        enriched = enrich_walkscore(out, ensure_complete=True)
    except WalkScoreAPIError:
        # Invalid key / quota / IP block — never silently write blank scores.
        raise
    except Exception as exc:
        print(f"[Listings] WalkScore enrichment failed: {exc}")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    n_scored = int(enriched["walk_score"].notna().sum()) if "walk_score" in enriched.columns else 0
    print(f"[Listings] WalkScore: scored {n_scored}/{len(enriched)} rows")

    missing = {col: int(enriched[col].isna().sum()) for col in WALKSCORE_COLUMNS if col in enriched.columns}
    incomplete = {col: n for col, n in missing.items() if n > 0}
    if incomplete:
        raise ValueError(
            "WalkScore enrichment incomplete; nulls remain in: "
            + ", ".join(f"{c}={n}" for c, n in incomplete.items())
        )
    print("[Listings] WalkScore: all six score columns are non-null.")
    return enriched


def clean_all_scraped_listings() -> Optional[pd.DataFrame]:
    """Clean every qualifying CSV in ``data/scraped/`` and concatenate."""
    paths = discover_scraped_listing_csvs()
    if not paths:
        print(f"[Listings] No listing scrapes in {DIR_SCRAPED} (add *.csv with address + beds_baths).")
        return None
    print(f"[Listings] Found {len(paths)} scrape file(s): {[p.name for p in paths]}")
    parts: list[pd.DataFrame] = []
    for p in paths:
        part = clean_listings(p)
        if part is not None and len(part) > 0:
            parts.append(part)
    if not parts:
        return None
    merged = pd.concat(parts, ignore_index=True)
    print(f"[Listings] Combined scrapes: {len(merged)} total rows")
    return merged


# =============================================================================
# 5. Pipeline
# =============================================================================


def build_clean_dataset() -> Optional[pd.DataFrame]:
    """
    Load and clean all sources; inner-join ZORI + ZHVI on (zip_code, date).

    Returns the merged panel, or None if ZORI or ZHVI is missing.
    """
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

    zori = clean_zori(PATH_ZORI)
    zhvi = clean_zhvi(PATH_ZHVI)
    census = clean_census(PATH_CENSUS)
    redfin = clean_redfin(PATH_REDFIN)
    crime = clean_crime(PATH_CRIME, PATH_ZIP_POLYGONS)
    listings = clean_all_scraped_listings()

    if census is not None:
        census.to_csv(OUT_PROCESSED / "census_clean.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'census_clean.csv'}")
    if redfin is not None:
        redfin.to_csv(OUT_PROCESSED / "redfin_metro_clean.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'redfin_metro_clean.csv'}")
    if crime is not None:
        crime.to_csv(OUT_PROCESSED / "crime_zip_month.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'crime_zip_month.csv'}")
    if listings is not None:
        listings = enrich_listings_with_walkscore(listings)
        listings = merge_crime_into_listings(listings, crime)
        listings.to_csv(OUT_PROCESSED / "listings_clean.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'listings_clean.csv'}")
    if zori is not None:
        zori.to_csv(OUT_PROCESSED / "zori_long.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'zori_long.csv'}")
    if zhvi is not None:
        zhvi.to_csv(OUT_PROCESSED / "zhvi_long.csv", index=False)
        print(f"[Write] {OUT_PROCESSED / 'zhvi_long.csv'}")

    if zori is None or zhvi is None:
        print("[Panel] Cannot inner-join ZORI + ZHVI — one or both inputs missing.")
        return None

    z_sub = zori[["zip_code", "date", "zori_rent"]].copy()
    h_sub = zhvi[["zip_code", "date", "zhvi_value"]].copy()
    panel = z_sub.merge(h_sub, on=["zip_code", "date"], how="inner")

    print(f"[Panel] Inner join ZORI + ZHVI on (zip_code, date): {len(panel)} rows")
    panel.to_csv(OUT_PROCESSED / "panel_zori_zhvi.csv", index=False)
    print(f"[Write] {OUT_PROCESSED / 'panel_zori_zhvi.csv'}")
    return panel


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    final_df = build_clean_dataset()
    if final_df is not None:
        print(f"\nFinal panel shape: {final_df.shape}")
        print(f"Columns ({len(final_df.columns)}): {list(final_df.columns)}")
    else:
        print("\nFinal panel not built (see messages above). Listings/Census/Redfin may still have been written.")
