"""End-to-end pipeline: clean each source, write per-source CSVs, return ZORI+ZHVI panel."""

from __future__ import annotations

import pandas as pd


def build_clean_dataset() -> pd.DataFrame | None:
    """
    Load and clean all sources; inner-join ZORI + ZHVI on (zip_code, date).

    Returns the merged panel, or ``None`` if ZORI or ZHVI is missing.

    Path constants and the WalkScore enrichment hook are read from the
    package namespace at call time so tests can monkeypatch
    ``data_cleaning.PATH_ZORI``, ``data_cleaning.OUT_PROCESSED``,
    ``data_cleaning.enrich_listings_with_walkscore``, etc.
    """
    import data_cleaning as _pkg

    out_processed = _pkg.OUT_PROCESSED
    out_processed.mkdir(parents=True, exist_ok=True)

    zori = _pkg.clean_zori(_pkg.PATH_ZORI)
    zhvi = _pkg.clean_zhvi(_pkg.PATH_ZHVI)
    census = _pkg.clean_census(_pkg.PATH_CENSUS)
    redfin = _pkg.clean_redfin(_pkg.PATH_REDFIN)
    crime = _pkg.clean_crime(_pkg.PATH_CRIME, _pkg.PATH_ZIP_POLYGONS)
    listings = _pkg.clean_all_scraped_listings()

    if census is not None:
        census.to_csv(out_processed / "census_clean.csv", index=False)
        print(f"[Write] {out_processed / 'census_clean.csv'}")
    if redfin is not None:
        redfin.to_csv(out_processed / "redfin_metro_clean.csv", index=False)
        print(f"[Write] {out_processed / 'redfin_metro_clean.csv'}")
    if crime is not None:
        crime.to_csv(out_processed / "crime_zip_month.csv", index=False)
        print(f"[Write] {out_processed / 'crime_zip_month.csv'}")
    if listings is not None:
        listings = _pkg.enrich_listings_with_walkscore(listings)
        listings = _pkg.merge_crime_into_listings(listings, crime)
        listings.to_csv(out_processed / "listings_clean.csv", index=False)
        print(f"[Write] {out_processed / 'listings_clean.csv'}")
    if zori is not None:
        zori.to_csv(out_processed / "zori_long.csv", index=False)
        print(f"[Write] {out_processed / 'zori_long.csv'}")
    if zhvi is not None:
        zhvi.to_csv(out_processed / "zhvi_long.csv", index=False)
        print(f"[Write] {out_processed / 'zhvi_long.csv'}")

    if zori is None or zhvi is None:
        print("[Panel] Cannot inner-join ZORI + ZHVI — one or both inputs missing.")
        return None

    z_sub = zori[["zip_code", "date", "zori_rent"]].copy()
    h_sub = zhvi[["zip_code", "date", "zhvi_value"]].copy()
    panel = z_sub.merge(h_sub, on=["zip_code", "date"], how="inner")

    print(f"[Panel] Inner join ZORI + ZHVI on (zip_code, date): {len(panel)} rows")
    panel.to_csv(out_processed / "panel_zori_zhvi.csv", index=False)
    print(f"[Write] {out_processed / 'panel_zori_zhvi.csv'}")
    return panel
