"""WalkScore enrichment for cleaned listings (geocode + WalkScore API)."""

from __future__ import annotations

import os

import pandas as pd

from .constants import REPO_ROOT, WALKSCORE_COLUMNS


def enrich_listings_with_walkscore(listings: pd.DataFrame) -> pd.DataFrame:
    """
    Append WalkScore / Transit / Bike score columns to cleaned listings.

    Geocodes ``address`` to lat/lon (Nominatim), then calls the WalkScore API
    via :mod:`fetch_walkscore`. With ``ensure_complete=True``, missing scores
    are imputed from ZIP/global means so every row has at least a
    ``walk_score`` populated.

    Failure modes:
      * No API key configured -> log and return null-filled columns (allows
        the rest of the pipeline to keep running locally).
      * Permanent API errors (invalid key, quota exceeded, IP blocked) ->
        re-raised as :class:`WalkScoreAPIError` so we never silently write a
        listings file with empty score columns.
      * Other transient errors -> logged with a non-null walk_score check at
        the end (which raises if any row is still missing a walk_score).
    """
    out = listings.copy()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("WALKSCORE_API_KEY"):
        print("[Listings] WALKSCORE_API_KEY not set; skipping WalkScore enrichment.")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    try:
        from fetch_walkscore import WalkScoreAPIError, enrich_walkscore  # local import
    except ImportError as exc:
        print(f"[Listings] WalkScore module unavailable ({exc}); skipping enrichment.")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    print(f"[Listings] Enriching {len(out)} rows with WalkScore (geocode + API)...")
    try:
        enriched = enrich_walkscore(out, ensure_complete=True)
    except WalkScoreAPIError:
        # Invalid key / quota / IP block — never silently write blank scores.
        raise
    except Exception as exc:
        print(f"[Listings] WalkScore enrichment failed: {exc}")
        for col in WALKSCORE_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    n_scored = int(enriched["walk_score"].notna().sum()) if "walk_score" in enriched.columns else 0
    print(f"[Listings] WalkScore: scored {n_scored}/{len(enriched)} rows")

    missing = {col: int(enriched[col].isna().sum()) for col in WALKSCORE_COLUMNS if col in enriched.columns}
    incomplete = {col: n for col, n in missing.items() if n > 0}
    if incomplete:
        raise ValueError(
            "WalkScore enrichment incomplete; nulls remain in: "
            + ", ".join(f"{c}={n}" for c, n in incomplete.items())
        )
    print("[Listings] WalkScore: all six score columns are non-null.")
    return enriched
