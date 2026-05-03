"""Data cleaning for SF real-estate ML: ZORI, ZHVI, Census, Redfin, listings, crime.

The package preserves the public surface of the previous single-file
``scripts/data_cleaning.py`` so existing call sites (``import data_cleaning as dc``)
keep working without changes. Functions, constants, and the
``build_clean_dataset`` orchestrator are all re-exported below.

Run the pipeline as a module from the ``scripts/`` directory::

    cd scripts && python -m data_cleaning
"""

from __future__ import annotations

from .constants import (
    CENSUS_SENTINEL,
    CRIME_FEATURE_COLUMNS,
    CRIME_PROPERTY_CATEGORIES,
    CRIME_VIOLENT_CATEGORIES,
    DATE_COL_RE,
    DIR_SCRAPED,
    FIRST_RENT,
    FLOORPLAN_RENT_RE,
    LISTINGS_OUTPUT_COLUMNS,
    LISTINGS_RENT_MAX,
    LISTINGS_RENT_MIN,
    LISTINGS_SCRAPE_REQUIRED_COLS,
    OUT_PROCESSED,
    PATH_CENSUS,
    PATH_CRIME,
    PATH_REDFIN,
    PATH_ZHVI,
    PATH_ZIP_POLYGONS,
    PATH_ZORI,
    RAW_SQFT_COLUMNS,
    REPO_ROOT,
    SF_ZIPCODES,
    SF_ZIP_SET,
    SQFT_RE,
    START_DATE,
    WALKSCORE_COLUMNS,
    ZILLOW_ID_COLS,
    ZIP_IN_ADDRESS,
    ZORI_DROP_ZIPS,
)
from .crime import clean_crime, merge_crime_into_listings
from .geo import _assign_zip_by_point, _load_zip_polygons
from .listings import (
    clean_all_scraped_listings,
    clean_listings,
    discover_scraped_listing_csvs,
)
from .panels import clean_census, clean_redfin
from .pipeline import build_clean_dataset
from .utils import (
    _date_columns,
    _extract_sqft_from_text,
    _filter_from_start_date,
    _melt_zillow_wide,
    _parse_floorplans_from_text,
    _strip_money,
    _strip_pct,
    _strip_sqft_from_label,
    parse_date,
    standardize_zip,
)
from .walkscore import enrich_listings_with_walkscore
from .zillow import clean_zhvi, clean_zori

__all__ = [
    # constants
    "CENSUS_SENTINEL",
    "CRIME_FEATURE_COLUMNS",
    "CRIME_PROPERTY_CATEGORIES",
    "CRIME_VIOLENT_CATEGORIES",
    "DATE_COL_RE",
    "DIR_SCRAPED",
    "FIRST_RENT",
    "FLOORPLAN_RENT_RE",
    "LISTINGS_OUTPUT_COLUMNS",
    "LISTINGS_RENT_MAX",
    "LISTINGS_RENT_MIN",
    "LISTINGS_SCRAPE_REQUIRED_COLS",
    "OUT_PROCESSED",
    "PATH_CENSUS",
    "PATH_CRIME",
    "PATH_REDFIN",
    "PATH_ZHVI",
    "PATH_ZIP_POLYGONS",
    "PATH_ZORI",
    "RAW_SQFT_COLUMNS",
    "REPO_ROOT",
    "SF_ZIPCODES",
    "SF_ZIP_SET",
    "SQFT_RE",
    "START_DATE",
    "WALKSCORE_COLUMNS",
    "ZILLOW_ID_COLS",
    "ZIP_IN_ADDRESS",
    "ZORI_DROP_ZIPS",
    # public functions
    "build_clean_dataset",
    "clean_all_scraped_listings",
    "clean_census",
    "clean_crime",
    "clean_listings",
    "clean_redfin",
    "clean_zhvi",
    "clean_zori",
    "discover_scraped_listing_csvs",
    "enrich_listings_with_walkscore",
    "merge_crime_into_listings",
    "parse_date",
    "standardize_zip",
    # underscore helpers retained for tests / cross-script imports
    "_assign_zip_by_point",
    "_date_columns",
    "_extract_sqft_from_text",
    "_filter_from_start_date",
    "_load_zip_polygons",
    "_melt_zillow_wide",
    "_parse_floorplans_from_text",
    "_strip_money",
    "_strip_pct",
    "_strip_sqft_from_label",
]
