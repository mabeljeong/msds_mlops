"""ZORI / ZHVI wide-CSV cleaners → long ``(zip_code, date, value)`` tables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .constants import SF_ZIP_SET, ZORI_DROP_ZIPS
from .utils import _filter_from_start_date, _melt_zillow_wide, standardize_zip


def clean_zori(path: Path | str) -> Optional[pd.DataFrame]:
    """ZORI wide CSV → long with zip_code, date, zori_rent. SF ZIPs only; START_DATE; drops sparse ZIPs."""
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
    """ZHVI wide CSV (ISO-8859-1) → long with zip_code, date, zhvi_value. SF ZIP filter pre-melt."""
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
