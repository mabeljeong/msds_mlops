"""Apartments.com / similar scraped listings → cleaned per-floorplan rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import (
    FIRST_RENT,
    LISTINGS_OUTPUT_COLUMNS,
    LISTINGS_RENT_MAX,
    LISTINGS_RENT_MIN,
    LISTINGS_SCRAPE_REQUIRED_COLS,
    RAW_SQFT_COLUMNS,
    ZIP_IN_ADDRESS,
)
from .utils import (
    _extract_sqft_from_text,
    _filter_from_start_date,
    _parse_floorplans_from_text,
    _strip_sqft_from_label,
    standardize_zip,
)


def clean_listings(path: Path | str) -> pd.DataFrame | None:
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

        explicit_sqft: float | None = None
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
    """Return sorted paths under ``DIR_SCRAPED`` that look like apartments.com listing scrapes.

    ``DIR_SCRAPED`` is read from the package namespace at call time so tests can
    monkeypatch ``data_cleaning.DIR_SCRAPED``.
    """
    import data_cleaning as _pkg

    dir_scraped: Path = _pkg.DIR_SCRAPED
    dir_scraped.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []
    for p in sorted(dir_scraped.glob("*.csv")):
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


def clean_all_scraped_listings() -> pd.DataFrame | None:
    """Clean every qualifying CSV in ``DIR_SCRAPED`` and concatenate."""
    import data_cleaning as _pkg

    paths = discover_scraped_listing_csvs()
    if not paths:
        print(f"[Listings] No listing scrapes in {_pkg.DIR_SCRAPED} (add *.csv with address + beds_baths).")
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
