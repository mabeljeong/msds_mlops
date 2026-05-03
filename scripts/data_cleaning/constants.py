"""Configuration constants shared across the data_cleaning package.

These are kept in one module so the public package namespace
(``data_cleaning.PATH_ZORI`` etc.) remains the single source of truth
that tests can monkeypatch.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

START_DATE = "2018-01-01"

SF_ZIPCODES = [
    "94102", "94103", "94104", "94105", "94107", "94108", "94109",
    "94110", "94111", "94112", "94114", "94115", "94116", "94117",
    "94118", "94121", "94122", "94123", "94124", "94127", "94128",
    "94129", "94130", "94131", "94132", "94133", "94134", "94158",
]

SF_ZIP_SET = frozenset(SF_ZIPCODES)

ZORI_DROP_ZIPS = frozenset({"94104", "94111", "94127", "94128", "94130"})

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

PATH_ZORI = REPO_ROOT / "data" / "raw" / "zillow_observed_rent_index.csv"
PATH_ZHVI = REPO_ROOT / "data" / "raw" / "zillow_home_value_index.csv"
PATH_CENSUS = REPO_ROOT / "data" / "raw" / "census_acs_sf.csv"
PATH_REDFIN = REPO_ROOT / "data" / "raw" / "redfin_median_asking_rent.csv"
PATH_CRIME = REPO_ROOT / "data" / "raw" / "sf_crime_incidents.csv"
PATH_ZIP_POLYGONS = REPO_ROOT / "data" / "raw" / "sf_zip_polygons.json"

DIR_SCRAPED = REPO_ROOT / "data" / "scraped"
LISTINGS_SCRAPE_REQUIRED_COLS = frozenset({"address", "beds_baths"})

OUT_PROCESSED = REPO_ROOT / "data" / "processed"

LISTINGS_RENT_MIN = 500.0
LISTINGS_RENT_MAX = 10000.0

WALKSCORE_COLUMNS = (
    "walk_score",
    "walk_description",
    "transit_score",
    "transit_description",
    "bike_score",
    "bike_description",
)

CENSUS_SENTINEL = -666666666

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

CRIME_FEATURE_COLUMNS = (
    "crime_total_month_zip",
    "crime_violent_month_zip",
    "crime_property_month_zip",
    "crime_total_month_zip_log1p",
)

ZIP_IN_ADDRESS = re.compile(r"\bCA\s+(\d{5})\b", re.IGNORECASE)
FIRST_RENT = re.compile(r"\$\s*([\d,]+)\+?")
FLOORPLAN_RENT_RE = re.compile(
    r"(Studio|\d+\s+Beds\+|\d+\s+Beds|\d+\s+Bed)\s+\$\s*([\d,]+)\+?",
    re.IGNORECASE,
)
SQFT_RE = re.compile(r"([\d,]+)\s*sq\.?\s*ft\.?", re.IGNORECASE)
RAW_SQFT_COLUMNS = ("sqft", "square_feet", "sq_ft", "square_feet_listed")
LISTINGS_OUTPUT_COLUMNS = ["zip_code", "date", "rent_usd", "beds_baths", "sqft", "address", "url"]
