"""
Tests for the SF crime data integration.

Covers four levels:

1. ``test_unit_assign_zip_by_point``
       Unit: point-in-polygon ZIP assignment with synthetic squares.
2. ``test_unit_load_zip_polygons_skips_invalid``
       Unit: deck.gl polygon loader rejects degenerate / non-numeric entries.
3. ``test_functional_clean_crime``
       Functional: end-to-end ``clean_crime`` over a synthetic incidents CSV
       and polygon file (no network, no real data).
4. ``test_functional_merge_crime_into_listings``
       Functional: row-count + zero-fill invariants for the listings merge.
5. ``test_functional_fetch_all_incidents_paginates``
       Functional: ``fetch_sf_crime.fetch_all_incidents`` paginates correctly
       and stops on a short page (HTTP layer monkeypatched).

Run from repo root:

    pytest tests/test_crime.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_cleaning as dc
import fetch_sf_crime as fsc


# =============================================================================
# 1. Helpers: synthetic SF-shaped polygons + incidents
# =============================================================================


def _square_polygon(zip_code: int, lon0: float, lat0: float, side: float = 0.01) -> dict:
    """Return one deck.gl-style entry: a closed square ring at (lon0, lat0)."""
    return {
        "zipcode": zip_code,
        "population": 1,
        "area": 1.0,
        "contour": [
            [lon0, lat0],
            [lon0 + side, lat0],
            [lon0 + side, lat0 + side],
            [lon0, lat0 + side],
            [lon0, lat0],
        ],
    }


def _write_polygons_file(path: Path) -> Path:
    """Two non-overlapping squares for ZIP 94110 and ZIP 94103."""
    path.write_text(json.dumps([
        _square_polygon(94110, lon0=-122.42, lat0=37.74),
        _square_polygon(94103, lon0=-122.41, lat0=37.77),
    ]))
    return path


# =============================================================================
# 2. Unit tests
# =============================================================================


def test_unit_assign_zip_by_point(tmp_path: Path):
    """Points inside / outside / on missing-coord rows resolve correctly."""
    polygons_path = _write_polygons_file(tmp_path / "polys.json")
    polygons = dc._load_zip_polygons(polygons_path)
    assert set(polygons.keys()) == {"94110", "94103"}

    lat = pd.Series([37.745, 37.775, 37.745, np.nan, 0.0])
    lon = pd.Series([-122.415, -122.405, -122.30, -122.4, np.nan])

    out = dc._assign_zip_by_point(lat, lon, polygons)
    assert list(out) == ["94110", "94103", pd.NA, pd.NA, pd.NA]
    assert str(out.dtype) == "string"


def test_unit_load_zip_polygons_skips_invalid(tmp_path: Path):
    """Bad / partial entries are dropped silently; valid ones survive."""
    path = tmp_path / "polys.json"
    path.write_text(json.dumps([
        # Valid
        _square_polygon(94110, lon0=-122.42, lat0=37.74),
        # Too few points
        {"zipcode": 94114, "contour": [[-122.4, 37.7], [-122.39, 37.7]]},
        # Non-numeric ZIP
        {"zipcode": "abc", "contour": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]},
        # Missing contour
        {"zipcode": 94115},
    ]))
    polygons = dc._load_zip_polygons(path)
    assert set(polygons.keys()) == {"94110"}


# =============================================================================
# 3. Functional: clean_crime end-to-end on synthetic data
# =============================================================================


def test_functional_clean_crime(tmp_path: Path):
    """
    Synthetic incidents CSV + polygons → ZIP-month aggregate with correct
    counts, violent / property splits, and the log1p companion column.
    """
    polygons_path = _write_polygons_file(tmp_path / "polys.json")

    # 7 incidents:
    #   * 4 in 94110 / 2024-05  (1 Assault, 2 Larceny Theft, 1 Drug Offense)
    #   * 2 in 94103 / 2024-05  (1 Robbery, 1 Burglary)
    #   * 1 outside any polygon (should be dropped)
    incidents = pd.DataFrame(
        {
            "incident_id": [1, 2, 3, 4, 5, 6, 7],
            "incident_datetime": ["2024-05-01T08:00:00.000"] * 7,
            "incident_date": ["2024-05-01T00:00:00.000"] * 7,
            "incident_category": [
                "Assault",
                "Larceny Theft",
                "Larceny Theft",
                "Drug Offense",
                "Robbery",
                "Burglary",
                "Assault",
            ],
            "incident_subcategory": [""] * 7,
            "latitude": [37.745, 37.745, 37.745, 37.745, 37.775, 37.775, 0.0],
            "longitude": [-122.415, -122.415, -122.415, -122.415, -122.405, -122.405, 0.0],
        }
    )
    crime_path = tmp_path / "incidents.csv"
    incidents.to_csv(crime_path, index=False)

    agg = dc.clean_crime(crime_path, polygons_path)
    assert agg is not None
    assert set(agg.columns) == {
        "zip_code",
        "month",
        "crime_total_month_zip",
        "crime_violent_month_zip",
        "crime_property_month_zip",
        "crime_total_month_zip_log1p",
    }
    assert len(agg) == 2
    assert set(agg["zip_code"]) == {"94110", "94103"}
    assert agg["month"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-05-01"]

    by_zip = agg.set_index("zip_code")
    assert by_zip.loc["94110", "crime_total_month_zip"] == 4
    assert by_zip.loc["94110", "crime_violent_month_zip"] == 1   # Assault
    assert by_zip.loc["94110", "crime_property_month_zip"] == 2  # 2 Larceny Theft
    assert by_zip.loc["94103", "crime_total_month_zip"] == 2
    assert by_zip.loc["94103", "crime_violent_month_zip"] == 1   # Robbery
    assert by_zip.loc["94103", "crime_property_month_zip"] == 1  # Burglary

    assert by_zip.loc["94110", "crime_total_month_zip_log1p"] == np.log1p(4)
    assert by_zip.loc["94103", "crime_total_month_zip_log1p"] == np.log1p(2)


def test_functional_clean_crime_handles_missing_files(tmp_path: Path):
    """Missing crime CSV or polygon file -> None (skip), no exception."""
    assert dc.clean_crime(tmp_path / "missing.csv", tmp_path / "missing.json") is None

    incidents = pd.DataFrame(
        {
            "incident_id": [1],
            "incident_datetime": ["2024-05-01T08:00:00.000"],
            "incident_date": ["2024-05-01T00:00:00.000"],
            "incident_category": ["Assault"],
            "incident_subcategory": [""],
            "latitude": [37.745],
            "longitude": [-122.415],
        }
    )
    csv_path = tmp_path / "incidents.csv"
    incidents.to_csv(csv_path, index=False)
    assert dc.clean_crime(csv_path, tmp_path / "missing.json") is None


# =============================================================================
# 4. Functional: merge_crime_into_listings invariants
# =============================================================================


def test_functional_merge_crime_into_listings_preserves_rows():
    """Left-join semantics: row count fixed, missing keys zero-filled."""
    listings = pd.DataFrame(
        {
            "zip_code": ["94110", "94103", "94110", "94115"],
            "date": pd.to_datetime(["2024-05-15", "2024-05-20", "2024-06-10", "2024-05-15"]),
            "rent_usd": [3500.0, 4200.0, 3600.0, 5000.0],
        }
    )
    crime = pd.DataFrame(
        {
            "zip_code": ["94110", "94103"],
            "month": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "crime_total_month_zip": [4, 2],
            "crime_violent_month_zip": [1, 1],
            "crime_property_month_zip": [2, 1],
            "crime_total_month_zip_log1p": [np.log1p(4), np.log1p(2)],
        }
    )

    out = dc.merge_crime_into_listings(listings, crime)
    assert len(out) == len(listings)
    for col in dc.CRIME_FEATURE_COLUMNS:
        assert col in out.columns
        assert out[col].isna().sum() == 0

    by_idx = out.reset_index(drop=True)
    assert by_idx.loc[0, "crime_total_month_zip"] == 4   # 94110, May 2024
    assert by_idx.loc[1, "crime_total_month_zip"] == 2   # 94103, May 2024
    assert by_idx.loc[2, "crime_total_month_zip"] == 0   # 94110 but June (no row in crime)
    assert by_idx.loc[3, "crime_total_month_zip"] == 0   # 94115 (unmatched zip)
    assert by_idx.loc[3, "crime_total_month_zip_log1p"] == 0.0


def test_functional_merge_crime_into_listings_with_no_crime():
    """When crime is None / empty, listings still get zero-filled feature columns."""
    listings = pd.DataFrame(
        {
            "zip_code": ["94110", "94103"],
            "date": pd.to_datetime(["2024-05-15", "2024-05-20"]),
        }
    )
    out_none = dc.merge_crime_into_listings(listings, None)
    out_empty = dc.merge_crime_into_listings(
        listings,
        pd.DataFrame(columns=["zip_code", "month", *dc.CRIME_FEATURE_COLUMNS]),
    )

    for out in (out_none, out_empty):
        assert len(out) == len(listings)
        for col in dc.CRIME_FEATURE_COLUMNS:
            assert col in out.columns
            assert out[col].notna().all()
        assert (out["crime_total_month_zip"] == 0).all()
        assert (out["crime_total_month_zip_log1p"] == 0.0).all()


# =============================================================================
# 5. Functional: fetch_sf_crime pagination
# =============================================================================


def test_functional_fetch_all_incidents_paginates(monkeypatch):
    """
    ``fetch_all_incidents`` should:
      * call ``fetch_page`` repeatedly with increasing offsets
      * stop when the server returns a short / empty page
      * concatenate all returned rows in order
    """
    pages = [
        [{"incident_id": str(i)} for i in range(1, 4)],   # full page
        [{"incident_id": str(i)} for i in range(4, 7)],   # full page
        [{"incident_id": "7"}],                           # short page -> stop
    ]
    calls: list[tuple[int, int]] = []

    def fake_fetch_page(start_date, offset, page_size, **kwargs):
        calls.append((offset, page_size))
        idx = offset // page_size
        return pages[idx] if idx < len(pages) else []

    monkeypatch.setattr(fsc, "fetch_page", fake_fetch_page)

    df = fsc.fetch_all_incidents(start_date="2018-01-01", page_size=3, max_rows=None)
    assert len(df) == 7
    assert df["incident_id"].tolist() == [str(i) for i in range(1, 8)]
    assert calls == [(0, 3), (3, 3), (6, 3)]


def test_functional_fetch_all_incidents_respects_max_rows(monkeypatch):
    """``max_rows`` truncates the accumulated rows and stops paginating."""
    pages = [[{"incident_id": str(i)} for i in range(1, 6)] for _ in range(10)]

    def fake_fetch_page(start_date, offset, page_size, **kwargs):
        idx = offset // page_size
        return pages[idx] if idx < len(pages) else []

    monkeypatch.setattr(fsc, "fetch_page", fake_fetch_page)

    df = fsc.fetch_all_incidents(start_date="2018-01-01", page_size=5, max_rows=7)
    assert len(df) == 7
    assert df["incident_id"].tolist() == ["1", "2", "3", "4", "5", "1", "2"]
