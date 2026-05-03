"""
Fetch ZIP-level ACS 5-Year (2022) data for San Francisco from the Census API
and save to data/raw/census_acs_sf.csv.

Usage:
    python scripts/fetch_census.py
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["CENSUS_API_KEY"]
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"
OUT_PATH = Path(__file__).parents[1] / "data" / "raw" / "census_acs_sf.csv"

CENSUS_SENTINEL = -666666666

# ZCTAs that fall within San Francisco (county 075).
# The Census API does not support filtering ZCTAs by county, so we fetch all
# California ZCTAs and keep only these.
SF_ZIPCODES = {
    "94102", "94103", "94104", "94105", "94107", "94108", "94109", "94110",
    "94111", "94112", "94114", "94115", "94116", "94117", "94118", "94121",
    "94122", "94123", "94124", "94127", "94128", "94129", "94130", "94131",
    "94132", "94133", "94134", "94158",
}

VARIABLES = {
    "B25064_001E": "census_median_rent",
    "B19013_001E": "census_median_income",
    "B25002_003E": "census_vacant_units",
    "B01003_001E": "census_total_population",
    "B25003_002E": "census_owner_units",
    "B25003_003E": "census_renter_units",
    "B25034_002E": "census_units_built_2020_later",
    "B08301_001E": "census_total_commuters",
}


def fetch_all_zctas() -> pd.DataFrame:
    """
    Fetch all ZCTAs nationally.

    ZCTAs are a national geography in the Census API — they cannot be filtered
    by state or county at query time. We filter to SF ZIP codes after fetching.
    """
    params = {
        "get": "NAME," + ",".join(VARIABLES.keys()),
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data[1:], columns=data[0])


def fetch_sf_census() -> pd.DataFrame:
    print("  Fetching all national ZCTAs (Census ZCTAs are a national geography)...")
    combined = fetch_all_zctas()
    print(f"  Total ZCTAs fetched: {len(combined)}")

    # Filter to SF ZIP codes only
    combined = combined.rename(columns={"zip code tabulation area": "zip_code"})
    combined = combined[combined["zip_code"].isin(SF_ZIPCODES)].copy()
    print(f"  SF ZCTAs matched: {len(combined)}")

    combined = combined.drop_duplicates(subset="zip_code")

    # Drop raw NAME column
    combined = combined.drop(columns=["NAME"], errors="ignore")

    # Rename Census variable codes to human-readable names
    combined = combined.rename(columns=VARIABLES)

    # Cast numeric columns to float, replace sentinel with NaN
    numeric_cols = list(VARIABLES.values())
    combined[numeric_cols] = (
        combined[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace(CENSUS_SENTINEL, float("nan"))
    )

    # Derived columns
    occupied = combined["census_owner_units"] + combined["census_renter_units"]
    combined["census_renter_ratio"] = combined["census_renter_units"] / occupied
    combined["census_vacancy_rate"] = combined["census_vacant_units"] / (
        occupied + combined["census_vacant_units"]
    )

    return combined


if __name__ == "__main__":
    print("Fetching ACS 5-Year (2022) data for San Francisco ZCTAs...")
    df = fetch_sf_census()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows → {OUT_PATH}")
    print(f"ZIP codes: {df['zip_code'].nunique()}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nSample:")
    sample_cols = ["zip_code", "census_median_rent", "census_median_income", "census_renter_ratio"]
    print(df[sample_cols].head())
