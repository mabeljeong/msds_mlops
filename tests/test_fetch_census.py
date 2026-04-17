"""
Integration tests (mocked HTTP) for scripts/fetch_census.py.
"""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts import fetch_census
from scripts.fetch_census import SF_ZIPCODES, VARIABLES, fetch_sf_census


HEADER = [
    "NAME",
    "B25064_001E",  # median_rent
    "B19013_001E",  # median_income
    "B25002_003E",  # vacant_units
    "B01003_001E",  # total_population
    "B25003_002E",  # owner_units
    "B25003_003E",  # renter_units
    "B25034_002E",  # units_built_2020_later
    "B08301_001E",  # total_commuters
    "zip code tabulation area",
]

ROWS = [
    # SF ZIP 94102
    ["ZCTA5 94102", "2500", "90000", "100", "30000", "5000", "10000", "200", "15000", "94102"],
    # SF ZIP 94110
    ["ZCTA5 94110", "3000", "100000", "50", "40000", "8000", "12000", "300", "20000", "94110"],
    # Duplicate of 94110 — should be deduplicated
    ["ZCTA5 94110", "3000", "100000", "50", "40000", "8000", "12000", "300", "20000", "94110"],
    # Non-SF ZIP — should be filtered out
    ["ZCTA5 90210", "5000", "200000", "10", "10000", "3000", "2000", "50", "5000", "90210"],
    # Duplicate of 94102 with sentinel values — exercises sentinel→NaN coercion
    [
        "ZCTA5 94102",
        "-666666666", "-666666666", "-666666666", "-666666666",
        "-666666666", "-666666666", "-666666666", "-666666666",
        "94102",
    ],
]

MOCK_PAYLOAD = [HEADER] + ROWS


class TestCensusApiParsingAndFilters:
    @pytest.fixture
    def census_df(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_PAYLOAD
        mock_resp.raise_for_status.return_value = None

        with patch.object(fetch_census.requests, "get", return_value=mock_resp):
            return fetch_sf_census()

    def test_correct_column_and_row_parsing(self, census_df):
        # Renamed columns from VARIABLES are present; raw codes & NAME are gone.
        for renamed in VARIABLES.values():
            assert renamed in census_df.columns
        for raw_code in VARIABLES.keys():
            assert raw_code not in census_df.columns
        assert "NAME" not in census_df.columns
        assert "zip_code" in census_df.columns
        assert "census_renter_ratio" in census_df.columns
        assert "census_vacancy_rate" in census_df.columns

        # Numeric columns are coerced to numeric dtype (int or float).
        for col in VARIABLES.values():
            assert pd.api.types.is_numeric_dtype(census_df[col])

        # Spot-check parsed values for ZIP 94110.
        row_94110 = census_df.loc[census_df["zip_code"] == "94110"].iloc[0]
        assert row_94110["census_median_rent"] == 3000.0
        assert row_94110["census_median_income"] == 100000.0
        assert row_94110["census_renter_units"] == 12000.0
        # renter_ratio = renter / (owner + renter) = 12000 / 20000 = 0.6
        assert math.isclose(row_94110["census_renter_ratio"], 0.6)
        # vacancy_rate = vacant / (occupied + vacant) = 50 / (20000 + 50)
        assert math.isclose(
            row_94110["census_vacancy_rate"], 50 / (20000 + 50)
        )

    def test_sf_only_zip_filtering(self, census_df):
        zips = set(census_df["zip_code"])
        assert "90210" not in zips
        assert zips.issubset(SF_ZIPCODES)
        assert {"94102", "94110"}.issubset(zips)

    def test_deduplication_of_zip_codes(self, census_df):
        assert census_df["zip_code"].is_unique
        # Only the two unique SF ZIPs should remain after dedup + filtering.
        assert len(census_df) == 2
