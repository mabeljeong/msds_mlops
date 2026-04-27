"""
Tests for ``scripts/data_cleaning.py``.

Four tests across three levels:

1. ``test_unit_helpers_standardization``     — unit: ZIP normalization + money / pct
                                                 parsing (pure string → numeric helpers).
2. ``test_unit_parse_floorplans_from_text``  — unit: regex extraction for floorplan parsing.
3. ``test_functional_clean_zori``            — functional: end-to-end ZORI cleaning on
                                                 synthetic wide → long DataFrame.
4. ``test_integration_build_clean_dataset``  — integration: full pipeline
                                                 (``build_clean_dataset``) with
                                                 ZORI + ZHVI + listings I/O in tmp repo.

Run from repo root:

    pytest tests/test_data_cleaning.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data_cleaning as dc


# =============================================================================
# 1. Unit tests (pure helpers; no I/O)
# =============================================================================


def test_unit_helpers_standardization():
    """
    Pure string → numeric helpers:
      * ``standardize_zip`` zero-pads valid ZIPs and returns ``NA`` for invalid ones.
      * ``_strip_money`` / ``_strip_pct`` parse currency / percentage strings.
      * ``_extract_sqft_from_text`` pulls ``N sq ft`` values out of free text.
    """
    raw = pd.Series([94110, "94103", 9414, "abc", None, -1, 100_000, 94110.0])
    out = dc.standardize_zip(raw)

    assert str(out.dtype) == "string"
    assert out.iloc[0] == "94110"
    assert out.iloc[1] == "94103"
    assert out.iloc[2] == "09414"
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])
    assert pd.isna(out.iloc[5])
    assert pd.isna(out.iloc[6])
    assert out.iloc[7] == "94110"

    assert dc._strip_money("$3,250") == 3250.0
    assert dc._strip_money("  $1,000.50 ") == 1000.5
    assert dc._strip_money("") is None
    assert dc._strip_money("nan") is None
    assert dc._strip_money(None) is None
    assert dc._strip_money(float("nan")) is None
    assert dc._strip_money("not-a-number") is None

    assert dc._strip_pct("+2.5%") == 2.5
    assert dc._strip_pct("-1.25%") == -1.25
    assert dc._strip_pct("0") == 0.0
    assert dc._strip_pct(None) is None

    assert dc._extract_sqft_from_text("Studio, 942 sq ft, $3,000") == 942.0
    assert dc._extract_sqft_from_text("1 Bed, 1,100 sq. ft.") == 1100.0
    assert dc._extract_sqft_from_text("no sqft here") is None
    assert dc._extract_sqft_from_text(None) is None


def test_unit_parse_floorplans_from_text():
    """``_parse_floorplans_from_text`` extracts every (label, rent) pair."""
    text = "Studio $2,500 / 1 Bed $3,100 / 2 Beds $4,250+ / 3 Beds+ $6,000"
    pairs = dc._parse_floorplans_from_text(text)

    assert pairs == [
        ("Studio", 2500.0),
        ("1 Bed", 3100.0),
        ("2 Beds", 4250.0),
        ("3 Beds+", 6000.0),
    ]
    assert dc._parse_floorplans_from_text("") == []
    assert dc._parse_floorplans_from_text(None) == []
    assert dc._parse_floorplans_from_text("studio no price") == []


# =============================================================================
# 2. Functional test: single cleaner end-to-end with a synthetic CSV
# =============================================================================


def test_functional_clean_zori(tmp_path: Path):
    """
    Write a minimal ZORI-wide CSV, run ``clean_zori``, and check the long output:
      * SF-only ZIPs survive (non-SF ``99999`` dropped)
      * Sparse ZIPs in ``ZORI_DROP_ZIPS`` dropped (``94104``)
      * Dates before ``START_DATE`` filtered (``2017-01-31`` dropped)
      * Schema is ``[zip_code, date, zori_rent, ...]``
    """
    wide = pd.DataFrame(
        {
            "RegionID": [1, 2, 3, 4],
            "RegionName": [94110, 94103, 94104, 99999],
            "SizeRank": [1, 2, 3, 4],
            "City": ["San Francisco"] * 4,
            "Metro": ["SF"] * 4,
            "CountyName": ["SF"] * 4,
            "State": ["CA"] * 4,
            "StateName": ["California"] * 4,
            "2017-01-31": [2000.0, 2100.0, 2200.0, 9999.0],
            "2018-01-31": [2500.0, 2600.0, 2700.0, 9999.0],
            "2020-06-30": [3000.0, 3100.0, 3200.0, 9999.0],
        }
    )
    src = tmp_path / "zori_wide.csv"
    wide.to_csv(src, index=False)

    out = dc.clean_zori(src)

    assert out is not None
    assert list(out.columns[:3]) == ["zip_code", "date", "zori_rent"]

    zips = set(out["zip_code"].dropna().unique())
    assert zips == {"94110", "94103"}

    assert out["date"].min() >= pd.Timestamp(dc.START_DATE)

    row = out[(out["zip_code"] == "94110") & (out["date"] == pd.Timestamp("2018-01-31"))]
    assert len(row) == 1
    assert row["zori_rent"].iloc[0] == 2500.0


# =============================================================================
# 3. Integration test: full ``build_clean_dataset`` with a fake repo layout
# =============================================================================


def _write_zillow_wide(path: Path, value_a: float, value_b: float, *, encoding: str = "utf-8"):
    """Minimal Zillow-wide CSV for two SF ZIPs across two dates after START_DATE."""
    wide = pd.DataFrame(
        {
            "RegionID": [1, 2],
            "RegionName": [94110, 94103],
            "SizeRank": [1, 2],
            "City": ["San Francisco", "San Francisco"],
            "State": ["CA", "CA"],
            "2018-06-30": [value_a, value_a + 50],
            "2020-06-30": [value_b, value_b + 50],
        }
    )
    wide.to_csv(path, index=False, encoding=encoding)


def test_integration_build_clean_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    End-to-end: wire ZORI + ZHVI + scraped listings through ``build_clean_dataset``
    by monkeypatching the module-level paths, then verify:
      * All expected output CSVs are written
      * The returned panel is the inner join of ZORI and ZHVI on (zip_code, date)
      * Listings output contains the rows we seeded, expanded per floorplan
    """
    raw = tmp_path / "raw"
    scraped = tmp_path / "scraped"
    processed = tmp_path / "processed"
    raw.mkdir()
    scraped.mkdir()

    zori_path = raw / "zori_wide.csv"
    zhvi_path = raw / "zhvi_wide.csv"
    _write_zillow_wide(zori_path, 2500.0, 3000.0)
    _write_zillow_wide(zhvi_path, 1_000_000.0, 1_200_000.0, encoding="ISO-8859-1")

    listings_src = pd.DataFrame(
        {
            "address": [
                "123 Market St, San Francisco, CA 94103",
                "500 Valencia St, San Francisco, CA 94110",
            ],
            "beds_baths": ["Studio $2,800", "1 Bed $3,500, 900 sq ft"],
            "pricing": ["Studio $2,800 / 1 Bed $3,400", pd.NA],
            "url": ["https://x/1", "https://x/2"],
            "scraped_at": ["2023-05-01", "2023-05-02"],
        }
    )
    listings_src.to_csv(scraped / "sf_apartments_listings.csv", index=False)

    monkeypatch.setattr(dc, "PATH_ZORI", zori_path)
    monkeypatch.setattr(dc, "PATH_ZHVI", zhvi_path)
    monkeypatch.setattr(dc, "PATH_CENSUS", tmp_path / "does_not_exist_census.csv")
    monkeypatch.setattr(dc, "PATH_REDFIN", tmp_path / "does_not_exist_redfin.csv")
    monkeypatch.setattr(dc, "PATH_CRIME", tmp_path / "does_not_exist_crime.csv")
    monkeypatch.setattr(dc, "PATH_ZIP_POLYGONS", tmp_path / "does_not_exist_polygons.json")
    monkeypatch.setattr(dc, "DIR_SCRAPED", scraped)
    monkeypatch.setattr(dc, "OUT_PROCESSED", processed)

    # Stub WalkScore enrichment so the integration test never hits the network;
    # we only need to verify the score columns are present and persisted.
    def _fake_enrich(listings):
        out = listings.copy()
        out["walk_score"] = 88
        out["walk_description"] = "Very Walkable"
        out["transit_score"] = 70
        out["transit_description"] = "Excellent Transit"
        out["bike_score"] = 65
        out["bike_description"] = "Bikeable"
        return out

    monkeypatch.setattr(dc, "enrich_listings_with_walkscore", _fake_enrich)

    panel = dc.build_clean_dataset()

    assert panel is not None
    assert set(panel.columns) >= {"zip_code", "date", "zori_rent", "zhvi_value"}
    assert set(panel["zip_code"].unique()) == {"94110", "94103"}
    assert set(panel["date"].dt.strftime("%Y-%m-%d")) == {"2018-06-30", "2020-06-30"}
    assert len(panel) == 4

    for name in ("zori_long.csv", "zhvi_long.csv", "listings_clean.csv", "panel_zori_zhvi.csv"):
        assert (processed / name).is_file(), f"expected {name} to be written"
    assert not (processed / "census_clean.csv").exists()
    assert not (processed / "redfin_metro_clean.csv").exists()
    assert not (processed / "crime_zip_month.csv").exists()

    listings_out = pd.read_csv(processed / "listings_clean.csv")
    # Crime features are added even when no crime data is available; zero-filled.
    for col in dc.CRIME_FEATURE_COLUMNS:
        assert col in listings_out.columns, f"missing {col}"
    assert (listings_out["crime_total_month_zip"] == 0).all()
    assert (listings_out["crime_total_month_zip_log1p"] == 0.0).all()
    assert set(listings_out["zip_code"].astype(str).str.zfill(5).unique()) == {"94103", "94110"}
    assert len(listings_out) == 3

    # WalkScore columns are appended (stubbed values from monkeypatched enricher).
    for col in (
        "walk_score",
        "walk_description",
        "transit_score",
        "transit_description",
        "bike_score",
        "bike_description",
    ):
        assert col in listings_out.columns, f"missing {col}"
    assert (listings_out["walk_score"] == 88).all()
    assert (listings_out["transit_score"] == 70).all()

    # Listing at 94110 was parsed from beds_baths with "900 sq ft"; sqft should be 900.
    valencia = listings_out.loc[listings_out["zip_code"].astype(str).str.zfill(5) == "94110"]
    assert len(valencia) == 1
    assert valencia["beds_baths"].iloc[0] == "1 Bed"
    assert valencia["sqft"].iloc[0] == 900.0
    assert valencia["rent_usd"].iloc[0] == 3500.0

    # Listing at 94103 expanded into Studio + 1 Bed from the pricing column.
    market = listings_out.loc[listings_out["zip_code"].astype(str).str.zfill(5) == "94103"]
    assert set(market["beds_baths"]) == {"Studio", "1 Bed"}
    assert set(market["rent_usd"]) == {2800.0, 3400.0}


def test_enrich_listings_validates_complete_coverage(monkeypatch):
    """``enrich_listings_with_walkscore`` raises when the underlying enricher
    leaves nulls in any of the six score columns."""
    listings = pd.DataFrame(
        {
            "zip_code": ["94103", "94110"],
            "address": ["123 Market St", "500 Valencia St"],
            "url": ["u1", "u2"],
        }
    )

    monkeypatch.setenv("WALKSCORE_API_KEY", "test-key")

    def _broken_enrich(df, **_kwargs):
        out = df.copy()
        out["walk_score"] = [88, None]
        out["walk_description"] = ["Very Walkable", None]
        out["transit_score"] = [70, None]
        out["transit_description"] = ["Excellent Transit", None]
        out["bike_score"] = [65, None]
        out["bike_description"] = ["Bikeable", None]
        return out

    import fetch_walkscore as fw

    monkeypatch.setattr(fw, "enrich_walkscore", _broken_enrich)

    with pytest.raises(ValueError, match="WalkScore enrichment incomplete"):
        dc.enrich_listings_with_walkscore(listings)


def test_enrich_listings_raises_on_walkscore_api_error(monkeypatch):
    """Permanent Walk Score API errors (invalid key / quota / IP block) must
    propagate so we never silently write a listings file with empty score
    columns."""
    listings = pd.DataFrame(
        {
            "zip_code": ["94103"],
            "address": ["123 Market St"],
            "url": ["u1"],
        }
    )

    monkeypatch.setenv("WALKSCORE_API_KEY", "test-key")

    import fetch_walkscore as fw

    def _exploding_enrich(df, **_kwargs):
        raise fw.WalkScoreAPIError(fw.WS_STATUS_QUOTA_EXCEEDED, "quota exceeded")

    monkeypatch.setattr(fw, "enrich_walkscore", _exploding_enrich)

    with pytest.raises(fw.WalkScoreAPIError):
        dc.enrich_listings_with_walkscore(listings)


def test_enrich_listings_passes_when_complete(monkeypatch):
    listings = pd.DataFrame(
        {
            "zip_code": ["94103"],
            "address": ["123 Market St"],
            "url": ["u1"],
        }
    )

    monkeypatch.setenv("WALKSCORE_API_KEY", "test-key")

    def _ok_enrich(df, **_kwargs):
        out = df.copy()
        out["walk_score"] = 88
        out["walk_description"] = "Very Walkable"
        out["transit_score"] = 70
        out["transit_description"] = "Excellent Transit"
        out["bike_score"] = 65
        out["bike_description"] = "Bikeable"
        return out

    import fetch_walkscore as fw

    monkeypatch.setattr(fw, "enrich_walkscore", _ok_enrich)

    out = dc.enrich_listings_with_walkscore(listings)
    for col in (
        "walk_score",
        "walk_description",
        "transit_score",
        "transit_description",
        "bike_score",
        "bike_description",
    ):
        assert out[col].notna().all()
