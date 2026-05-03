"""Pure helpers used across the data_cleaning package: ZIP / date / money / sqft / floorplan parsing."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .constants import (
    DATE_COL_RE,
    FLOORPLAN_RENT_RE,
    SQFT_RE,
    START_DATE,
    ZILLOW_ID_COLS,
)


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


__all__ = [
    "standardize_zip",
    "parse_date",
    "_date_columns",
    "_melt_zillow_wide",
    "_filter_from_start_date",
    "_strip_money",
    "_strip_pct",
    "_extract_sqft_from_text",
    "_strip_sqft_from_label",
    "_parse_floorplans_from_text",
]
