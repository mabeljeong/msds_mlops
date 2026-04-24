#!/usr/bin/env python3
"""
Clean sf_apartments_listings_combined.csv:
  1. Explode each property row into one row per unit type (Studio, 1 Bed, 2 Beds, …)
  2. Extract zip_code from address
  3. Fill missing address from title
  4. pricing column → clean price only (e.g. "$3,862+")
  5. beds_baths column → unit type label only (e.g. "1 Bed", "2 Beds")

Output: data/scraped/sf_apartments_listings_clean.csv
Run from repo root: python scripts/clean_combined_listings.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT = Path(__file__).resolve().parents[1] / "data" / "scraped" / "sf_apartments_listings_combined.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "scraped" / "sf_apartments_listings_clean.csv"

# Matches: "Studio $3,200+", "1 Bed $3,862+", "2 Beds $4,487", "3 Beds+ $7,642"
UNIT_PRICE_RE = re.compile(
    r"(Studio|\d+\s+Beds?\+?)\s+\$\s*([\d,]+)(\+?)",
    re.IGNORECASE,
)

# ZIP code: 5-digit number
ZIP_RE = re.compile(r"\b(\d{5})\b")

# Address starts at the first digit in a title (e.g. "Isle House 39 Bruton St…" → "39 Bruton St…")
ADDR_FROM_TITLE_RE = re.compile(r"(\d+.+)")


def fill_address_from_title(title: object) -> str | None:
    if pd.isna(title):
        return None
    m = ADDR_FROM_TITLE_RE.search(str(title))
    return m.group(1).strip() if m else None


def extract_zip(text: object) -> str | None:
    if pd.isna(text):
        return None
    m = ZIP_RE.search(str(text))
    return m.group(1) if m else None


def normalize_unit_type(raw: str) -> str:
    """'2 Beds+' → '2 Beds+', '1 Bed ' → '1 Bed', etc."""
    return re.sub(r"\s+", " ", raw).strip()


def explode_pricing(row: pd.Series) -> list[dict]:
    """Return one dict per unit type found in the pricing string."""
    pricing = row["pricing"]
    records = []

    if pd.isna(pricing):
        return records

    for m in UNIT_PRICE_RE.finditer(str(pricing)):
        unit_type = normalize_unit_type(m.group(1))
        price_num = m.group(2)           # e.g. "3,862"
        plus = m.group(3)                # "+" or ""
        price_str = f"${price_num}{plus}"  # e.g. "$3,862+"

        records.append({
            "title": row["title"],
            "beds_baths": unit_type,
            "pricing": price_str,
            "address": row["address"],
            "zip_code": row["zip_code"],
            "url": row.get("url"),
            "scraped_at": row.get("scraped_at"),
        })

    return records


def main() -> None:
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT.name}")

    # Step 3: fill missing address from title
    missing_mask = df["address"].isna()
    df.loc[missing_mask, "address"] = df.loc[missing_mask, "title"].apply(fill_address_from_title)
    filled = missing_mask.sum()
    still_missing = df["address"].isna().sum()
    print(f"Filled {filled} missing addresses from title ({still_missing} still missing)")

    # Step 2: extract zip_code
    df["zip_code"] = df["address"].apply(extract_zip)
    print(f"ZIP code extracted: {df['zip_code'].notna().sum()}/{len(df)} rows have a ZIP")

    # Steps 1, 4, 5: explode by unit type, clean pricing, clean beds_baths
    records = []
    skipped = 0
    for _, row in df.iterrows():
        expanded = explode_pricing(row)
        if expanded:
            records.extend(expanded)
        else:
            skipped += 1

    out = pd.DataFrame(records, columns=["title", "beds_baths", "pricing", "address", "zip_code", "url", "scraped_at"])
    print(f"Skipped {skipped} rows with no parseable price (e.g. 'Call for Rent')")
    print(f"Exploded {len(df) - skipped} properties → {len(out)} unit-type rows")

    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
