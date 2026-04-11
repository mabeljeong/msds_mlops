"""
Enrich a listings DataFrame with Walk Score, Transit Score, and Bike Score.

Expected input CSV columns:
    address     – street address string (e.g. "39 Bruton St")
    latitude    – float
    longitude   – float
    city        – optional; defaults to "San Francisco" if missing

Usage (once the apartments.com scrape is available):
    from scripts.fetch_walkscore import enrich_walkscore
    enriched_df = enrich_walkscore(listings_df)

Or standalone:
    python scripts/fetch_walkscore.py --input data/raw/listings.csv \
                                      --output data/processed/listings_walkscore.csv
"""

import os
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["WALKSCORE_API_KEY"]
WALKSCORE_URL = "https://api.walkscore.com/score"
DEFAULT_CITY = "San Francisco"

_NULL_SCORES = {
    "walk_score": None,
    "walk_description": None,
    "transit_score": None,
    "transit_description": None,
    "bike_score": None,
    "bike_description": None,
}


def get_scores(address: str, city: str, lat: float, lon: float) -> dict:
    """
    Fetch walk/transit/bike scores for a single address.

    Returns a dict with keys:
        walk_score, walk_description,
        transit_score, transit_description,
        bike_score, bike_description

    Returns all-None on any exception so the pipeline never crashes.
    """
    try:
        full_address = f"{address}, {city}"
        params = {
            "format": "json",
            "address": full_address,
            "lat": lat,
            "lon": lon,
            "transit": 1,
            "bike": 1,
            "wsapikey": API_KEY,
        }
        resp = requests.get(WALKSCORE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "walk_score": data.get("walkscore"),
            "walk_description": data.get("description"),
            "transit_score": data.get("transit", {}).get("score"),
            "transit_description": data.get("transit", {}).get("description"),
            "bike_score": data.get("bike", {}).get("score"),
            "bike_description": data.get("bike", {}).get("description"),
        }
    except Exception:
        return dict(_NULL_SCORES)


def enrich_walkscore(df: pd.DataFrame, delay: float = 0.25) -> pd.DataFrame:
    """
    Add walk/transit/bike score columns to a listings DataFrame.

    Args:
        df:     DataFrame with columns: address, latitude, longitude.
                Optional column: city (defaults to "San Francisco").
        delay:  Seconds to sleep between API calls (free tier: 5 000/day).

    Returns:
        df with 6 new columns appended in-place (copy returned).
    """
    df = df.copy()
    has_city = "city" in df.columns

    score_rows = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lat = getattr(row, "latitude")
        lon = getattr(row, "longitude")

        if pd.isna(lat) or pd.isna(lon):
            score_rows.append(dict(_NULL_SCORES))
            continue

        address = getattr(row, "address")
        city = getattr(row, "city", DEFAULT_CITY) if has_city else DEFAULT_CITY

        scores = get_scores(address, city, lat, lon)
        score_rows.append(scores)

        if i % 25 == 0:
            print(f"  WalkScore: processed {i}/{len(df)} rows...")

        time.sleep(delay)

    scores_df = pd.DataFrame(score_rows, index=df.index)
    return pd.concat([df, scores_df], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich listings with WalkScore data")
    parser.add_argument("--input", required=True, help="Path to input listings CSV")
    parser.add_argument("--output", required=True, help="Path to save enriched CSV")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between API calls (default: 0.25)",
    )
    args = parser.parse_args()

    print(f"Loading listings from {args.input}...")
    listings = pd.read_csv(args.input)
    print(f"  {len(listings):,} rows loaded")

    print("Fetching WalkScore data...")
    enriched = enrich_walkscore(listings, delay=args.delay)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"\nSaved {len(enriched):,} rows → {out_path}")

    scored = enriched["walk_score"].notna().sum()
    print(f"Successfully scored: {scored:,}/{len(enriched):,} rows")
