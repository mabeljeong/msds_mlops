"""Census + Redfin metro panel cleaners."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .constants import CENSUS_SENTINEL
from .utils import _strip_money, _strip_pct, parse_date, standardize_zip


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


def clean_redfin(path: Path | str) -> Optional[pd.DataFrame]:
    """Redfin metro-level TSV: utf-16, tab. No zip_code. Cleans $ / % numerics; parses dates."""
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
