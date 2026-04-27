"""Tests for ``scripts/enrich_crime_features.py``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import enrich_crime_features as ecf


def _square_polygon(zip_code: int, lon0: float, lat0: float, *, population: int) -> dict:
    return {
        "zipcode": zip_code,
        "population": population,
        "area": 1.0,
        "contour": [
            [lon0, lat0],
            [lon0 + 0.01, lat0],
            [lon0 + 0.01, lat0 + 0.01],
            [lon0, lat0 + 0.01],
            [lon0, lat0],
        ],
    }


def _write_polygons(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                _square_polygon(94110, -122.42, 37.74, population=1000),
                _square_polygon(94103, -122.41, 37.77, population=2000),
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_build_crime_features_rolls_and_normalizes(tmp_path: Path):
    crime_zip_month = pd.DataFrame(
        {
            "zip_code": ["94110", "94110", "94110", "94103"],
            "month": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-03-01"],
            "crime_total_month_zip": [10, 20, 30, 4],
            "crime_violent_month_zip": [2, 4, 3, 1],
            "crime_property_month_zip": [5, 8, 9, 2],
        }
    )
    crime_path = tmp_path / "crime_zip_month.csv"
    polygons_path = _write_polygons(tmp_path / "polygons.json")
    crime_zip_month.to_csv(crime_path, index=False)

    out = ecf.build_crime_features(crime_path, polygons_path)

    assert set(out.columns) == {"zip_code", "month", "zip_population", *ecf.MERGE_FEATURE_COLUMNS}

    row = out[(out["zip_code"] == "94110") & (out["month"] == pd.Timestamp("2024-03-01"))].iloc[0]
    assert row["crime_total_3mo_avg"] == 20.0
    assert row["crime_total_12mo_avg"] == 20.0
    assert row["crime_violent_3mo_avg"] == 3.0
    assert row["crime_property_3mo_avg"] == (5 + 8 + 9) / 3
    assert row["violent_share"] == 3 / 30
    assert row["crime_trend"] == 0.0
    assert row["crime_rate_per_1k"] == 30.0

    # ZIPs from the polygon file get a complete month grid, with missing crime
    # months filled as zero before rolling windows.
    zero_row = out[(out["zip_code"] == "94103") & (out["month"] == pd.Timestamp("2024-01-01"))].iloc[0]
    assert zero_row["crime_total_month_zip"] == 0
    assert zero_row["violent_share"] == 0.0


def test_build_category_breakdown_top_six_latest_12_months(tmp_path: Path):
    polygons_path = _write_polygons(tmp_path / "polygons.json")
    incidents = pd.DataFrame(
        {
            "incident_date": ["2024-06-15"] * 9 + ["2022-01-01"],
            "incident_category": [
                "Larceny Theft",
                "Larceny Theft",
                "Assault",
                "Burglary",
                "Robbery",
                "Fraud",
                "Vandalism",
                "Drug Offense",
                "Motor Vehicle Theft",
                "Larceny Theft",
            ],
            "latitude": [37.745] * 10,
            "longitude": [-122.415] * 10,
        }
    )
    incidents_path = tmp_path / "incidents.csv"
    incidents.to_csv(incidents_path, index=False)

    out = ecf.build_category_breakdown(incidents_path, polygons_path, top_n=6)

    assert len(out) == 6
    assert set(out.columns) == {"zip_code", "category", "count_12mo", "share_of_total"}
    assert out["zip_code"].unique().tolist() == ["94110"]
    assert out.iloc[0]["category"] == "Larceny Theft"
    assert out.iloc[0]["count_12mo"] == 2
    assert np.isclose(out.iloc[0]["share_of_total"], 2 / 9)
    assert "Drug Offense" in set(out["category"]) or "Motor Vehicle Theft" in set(out["category"])


def test_merge_features_into_listings_preserves_rows_and_writes(tmp_path: Path):
    listings = pd.DataFrame(
        {
            "zip_code": ["94110", "94103", "94110"],
            "date": ["2024-03-20", "2024-03-21", "2024-04-01"],
            "rent_usd": [3500, 3200, 3600],
            "crime_total_month_zip": [999, 999, 999],
        }
    )
    features = pd.DataFrame(
        {
            "zip_code": ["94110", "94103"],
            "month": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "crime_total_month_zip": [30, 4],
            "crime_violent_month_zip": [3, 1],
            "crime_property_month_zip": [9, 2],
            "crime_total_3mo_avg": [20.0, 4.0],
            "crime_total_12mo_avg": [20.0, 4.0],
            "crime_violent_3mo_avg": [3.0, 1.0],
            "crime_violent_12mo_avg": [3.0, 1.0],
            "crime_property_3mo_avg": [7.33, 2.0],
            "crime_property_12mo_avg": [7.33, 2.0],
            "violent_share": [0.1, 0.25],
            "crime_trend": [0.0, 0.0],
            "crime_rate_per_1k": [30.0, 2.0],
        }
    )
    listings_path = tmp_path / "listings_clean.csv"
    output_path = tmp_path / "listings_clean_v2.csv"
    listings.to_csv(listings_path, index=False)

    out = ecf.merge_features_into_listings(listings_path, features=features, output_path=output_path)

    assert len(out) == len(listings)
    assert output_path.is_file()
    assert out.loc[0, "crime_total_month_zip"] == 30
    assert out.loc[0, "crime_total_3mo_avg"] == 20.0
    assert out.loc[1, "crime_rate_per_1k"] == 2.0
    assert out.loc[2, "crime_total_month_zip"] == 0
    assert out.loc[2, "crime_total_3mo_avg"] == 0.0
