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


def _parse_price(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    m = PRICE_RE.search(str(raw))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _load_scraped() -> pd.DataFrame:
    """Pre-process the scraped listings table once for repeated lookups."""
    if not CLEAN_LISTINGS_CSV.exists():
        return pd.DataFrame(columns=["zip_code", "beds_baths", "title", "address", "url", "_price"])
    df = pd.read_csv(CLEAN_LISTINGS_CSV)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df["_price"] = df["pricing"].map(_parse_price)
    return df


class _ScrapedMatcher:
    """Claim-once matcher: each scraped row is handed out at most once.

    Match priority:
      1) zip + beds + closest unclaimed price within $200
      2) zip + beds + closest unclaimed price (any delta)
      3) zip + closest unclaimed price (any delta)
      4) no match -> empty dict
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.claimed: set[int] = set()

    @staticmethod
    def _bed_label(bedrooms: float | None) -> str | None:
        if bedrooms is None or pd.isna(bedrooms):
            return None
        return "Studio" if float(bedrooms) == 0 else f"{int(float(bedrooms))} Bed"

    def lookup(self, zip_code: str, bedrooms: float | None, actual_rent: float) -> dict[str, str | None]:
        if self.df.empty:
            return {}
        target_zip = str(zip_code).zfill(5)
        bed_prefix = self._bed_label(bedrooms)

        zip_pool = self.df.loc[
            (self.df["zip_code"] == target_zip) & (~self.df.index.isin(self.claimed))
        ]
        if zip_pool.empty:
            return {}

        bed_pool = zip_pool
        if bed_prefix is not None:
            beds_match = zip_pool[zip_pool["beds_baths"].fillna("").str.startswith(bed_prefix)]
            if not beds_match.empty:
                bed_pool = beds_match

        idx = self._closest(bed_pool, actual_rent, tolerance=200.0)
        if idx is None:
            idx = self._closest(bed_pool, actual_rent, tolerance=None)
        if idx is None and bed_pool is not zip_pool:
            idx = self._closest(zip_pool, actual_rent, tolerance=None)
        if idx is None:
            return {}

        self.claimed.add(idx)
        row = self.df.loc[idx]
        return {
            "title": (str(row.get("title", "")).strip() or None),
            "address": (str(row.get("address", "")).strip() or None),
            "url": (str(row.get("url", "")).strip() or None),
        }

    @staticmethod
    def _closest(pool: pd.DataFrame, target: float, tolerance: float | None) -> int | None:
        if pool.empty:
            return None
        priced = pool.dropna(subset=["_price"])
        if not priced.empty:
            deltas = (priced["_price"] - target).abs()
            best = deltas.idxmin()
            if tolerance is None or deltas.loc[best] <= tolerance:
                return int(best)
            return None
        if tolerance is None:
            return int(pool.index[0])
        return None


def main() -> None:
    if not SCORED_CSV.exists():
        raise SystemExit(f"missing {SCORED_CSV} — run scripts/build_demo_examples.py first")

    df = pd.read_csv(SCORED_CSV)
    matcher = _ScrapedMatcher(_load_scraped())
    out: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        payload = _parse_payload(row.get("request_payload"))
        if not payload:
            continue

        zip_code = str(payload.get("zip_code", row.get("zip_code", ""))).zfill(5)
        bedrooms = payload.get("bedrooms", row.get("bedrooms"))
        actual = float(payload.get("actual_rent_usd", row.get("actual_rent_usd", 0.0)))

        lat, lng = _zip_meta(zip_code)
        meta = matcher.lookup(zip_code, bedrooms, actual)

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
