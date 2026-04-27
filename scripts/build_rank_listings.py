#!/usr/bin/env python3
"""Convert demo/scored_listings.csv into a clean JSON the API can serve to the frontend.

Output: demo/listings_for_rank.json
- One record per scored listing
- Includes the full feature payload, actual_rent_usd, address/title/url where available,
  and (lat, lng) using a static SF ZIP centroid table so the frontend can plot markers
  without an external geocoder.

Run: python scripts/build_rank_listings.py
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORED_CSV = ROOT / "demo" / "scored_listings.csv"
CLEAN_LISTINGS_CSV = ROOT / "data" / "scraped" / "sf_apartments_listings_clean.csv"
OUTPUT = ROOT / "demo" / "listings_for_rank.json"

# Approximate SF ZIP centroids (lat, lng). Sufficient for visualization.
SF_ZIP_CENTROIDS: dict[str, tuple[float, float]] = {
    "94102": (37.7793, -122.4192),
    "94103": (37.7726, -122.4099),
    "94104": (37.7918, -122.4019),
    "94105": (37.7898, -122.3942),
    "94107": (37.7665, -122.3958),
    "94108": (37.7929, -122.4082),
    "94109": (37.7929, -122.4194),
    "94110": (37.7506, -122.4148),
    "94111": (37.7989, -122.4020),
    "94112": (37.7204, -122.4427),
    "94114": (37.7587, -122.4329),
    "94115": (37.7857, -122.4378),
    "94116": (37.7449, -122.4863),
    "94117": (37.7710, -122.4422),
    "94118": (37.7811, -122.4612),
    "94121": (37.7768, -122.4955),
    "94122": (37.7600, -122.4822),
    "94123": (37.7997, -122.4370),
    "94124": (37.7321, -122.3858),
    "94127": (37.7359, -122.4592),
    "94129": (37.7989, -122.4662),
    "94130": (37.8230, -122.3697),
    "94131": (37.7456, -122.4413),
    "94132": (37.7211, -122.4754),
    "94133": (37.8009, -122.4109),
    "94134": (37.7186, -122.4108),
    "94158": (37.7702, -122.3879),
    # Daly City ZIPs (SF metro, appear in our scraped data)
    "94014": (37.6826, -122.4541),
    "94015": (37.6794, -122.4801),
}

PRICE_RE = re.compile(r"\$?\s*([\d,]+)")


def _parse_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text or text == "nan":
        return {}
    # scored_listings.csv stores Python-style dict literals
    return ast.literal_eval(text)


def _zip_meta(zip_code: str) -> tuple[float | None, float | None]:
    z = str(zip_code).zfill(5)
    if z in SF_ZIP_CENTROIDS:
        lat, lng = SF_ZIP_CENTROIDS[z]
        return lat, lng
    return None, None


def _lookup_clean_listing(zip_code: str, bedrooms: float | None) -> dict[str, str]:
    """Best-effort title/address/url from scraped listings, matched on zip+beds."""
    if not CLEAN_LISTINGS_CSV.exists():
        return {}
    df = pd.read_csv(CLEAN_LISTINGS_CSV)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    target_zip = str(zip_code).zfill(5)

    candidates = df.loc[df["zip_code"] == target_zip]
    if bedrooms is not None and not pd.isna(bedrooms):
        bed_label = "Studio" if float(bedrooms) == 0 else f"{int(bedrooms)} Bed"
        bed_match = candidates[candidates["beds_baths"].fillna("").str.startswith(bed_label)]
        if not bed_match.empty:
            candidates = bed_match
    if candidates.empty:
        return {}
    row = candidates.iloc[0]
    return {
        "title": str(row.get("title", "")).strip() or None,
        "address": str(row.get("address", "")).strip() or None,
        "url": str(row.get("url", "")).strip() or None,
    }


def main() -> None:
    if not SCORED_CSV.exists():
        raise SystemExit(f"missing {SCORED_CSV} — run scripts/build_demo_examples.py first")

    df = pd.read_csv(SCORED_CSV)
    out: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        payload = _parse_payload(row.get("request_payload"))
        if not payload:
            continue

        zip_code = str(payload.get("zip_code", row.get("zip_code", ""))).zfill(5)
        bedrooms = payload.get("bedrooms", row.get("bedrooms"))
        actual = float(payload.get("actual_rent_usd", row.get("actual_rent_usd", 0.0)))

        lat, lng = _zip_meta(zip_code)
        meta = _lookup_clean_listing(zip_code, bedrooms)

        record = {
            "listing_id": f"L{i:03d}",
            **{k: v for k, v in payload.items() if k != "actual_rent_usd"},
            "zip_code": zip_code,
            "actual_rent_usd": actual,
            "lat": lat,
            "lng": lng,
            "title": meta.get("title"),
            "address": meta.get("address"),
            "url": meta.get("url"),
        }
        out.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {len(out)} listings -> {OUTPUT}")


if __name__ == "__main__":
    main()
