"""
Enrich a listings DataFrame with Walk Score, Transit Score, and Bike Score.

Expected input CSV columns:
    address     – street address string (e.g. "39 Bruton St")
    latitude    – optional float; auto-geocoded from ``address`` when missing
    longitude   – optional float; auto-geocoded from ``address`` when missing
    city        – optional; defaults to "San Francisco" if missing
    url         – optional; mined for street address when ``address`` is missing
    zip_code    – optional; used as last-resort geocode fallback (centroid)

Usage (once the apartments.com scrape is available):
    from scripts.fetch_walkscore import enrich_walkscore
    enriched_df = enrich_walkscore(listings_df)

Or standalone:
    python scripts/fetch_walkscore.py --input data/raw/listings.csv \
                                      --output data/processed/listings_walkscore.csv
"""

import os
import random
import re
import time
import argparse
from typing import Optional
from urllib.parse import urlparse

import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Resolved lazily so importing this module never raises when the key is unset
# (e.g. during unrelated tests). ``get_scores`` short-circuits to all-None if
# the key is missing.
API_KEY = os.environ.get("WALKSCORE_API_KEY", "")
WALKSCORE_URL = "https://api.walkscore.com/score"
DEFAULT_CITY = "San Francisco"
DEFAULT_STATE = "CA"

# Walk Score JSON status codes (returned as ``status`` in the response body).
# The API intentionally returns HTTP 200 for several auth/quota errors, so we
# must branch on the JSON status to detect them.
WS_STATUS_OK = 1
WS_STATUS_IN_PROGRESS = 2
WS_STATUS_INVALID_LATLON = 30
WS_STATUS_INTERNAL_ERROR = 31
WS_STATUS_INVALID_KEY = 40
WS_STATUS_QUOTA_EXCEEDED = 41
WS_STATUS_IP_BLOCKED = 42

# Statuses that will never resolve by retrying the same request — surface them
# so the caller can fail loudly rather than write blank score columns.
WS_FATAL_STATUSES = frozenset({
    WS_STATUS_INVALID_KEY,
    WS_STATUS_QUOTA_EXCEEDED,
    WS_STATUS_IP_BLOCKED,
})
# Transient statuses where the same coordinates may yet produce a score.
WS_TRANSIENT_STATUSES = frozenset({
    WS_STATUS_IN_PROGRESS,
    WS_STATUS_INTERNAL_ERROR,
})


class WalkScoreAPIError(RuntimeError):
    """Raised for permanent Walk Score API failures (invalid key, quota, IP block)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"WalkScore API status {status}: {message}")
        self.status = status

# Free, key-less geocoder. Per Nominatim usage policy: identify with a
# User-Agent and rate-limit to ≤1 request/second.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_USER_AGENT = "msds-mlops-rentiq/1.0 (data-cleaning pipeline)"

# Hard fallback when even ZIP geocoding fails — San Francisco city centroid.
SF_CITY_CENTROID = (37.7749, -122.4194)

SCORE_COLUMNS = (
    "walk_score",
    "walk_description",
    "transit_score",
    "transit_description",
    "bike_score",
    "bike_description",
)

_NULL_SCORES = {col: None for col in SCORE_COLUMNS}

# Module-level cache: identical (address, city) keys are geocoded only once
# per process. Populated/read by ``geocode_address``.
_geocode_cache: dict[str, tuple[Optional[float], Optional[float]]] = {}
_score_cache: dict[tuple[str, str, float, float], dict] = {}

# Standard WalkScore description buckets (per walkscore.com/methodology).
_WALK_BUCKETS = (
    (90, "Walker's Paradise"),
    (70, "Very Walkable"),
    (50, "Somewhat Walkable"),
    (25, "Car-Dependent"),
    (0, "Car-Dependent"),
)
_TRANSIT_BUCKETS = (
    (90, "Rider's Paradise"),
    (70, "Excellent Transit"),
    (50, "Good Transit"),
    (25, "Some Transit"),
    (0, "Minimal Transit"),
)
_BIKE_BUCKETS = (
    (90, "Biker's Paradise"),
    (70, "Very Bikeable"),
    (50, "Bikeable"),
    (25, "Somewhat Bikeable"),
    (0, "Somewhat Bikeable"),
)
_BUCKETS_BY_KIND = {"walk": _WALK_BUCKETS, "transit": _TRANSIT_BUCKETS, "bike": _BIKE_BUCKETS}

# Address-recovery regex helpers.
ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
UNIT_AT_RE = re.compile(r"^\s*(?:unit|apt|apartment|suite|ste|#)\s*[\w-]+\s+at\s+(.+)$", re.IGNORECASE)
TRAILING_UNIT_RE = re.compile(r",?\s*(?:unit|apt|apartment|suite|ste|#)\s*[\w-]+\s*$", re.IGNORECASE)

# apartments.com slug suffixes to strip when reconstructing street addresses.
_SLUG_DROP_PREFIX_TOKENS = {
    "san", "francisco", "sf", "ca", "california",
    "unit", "apt", "apartment", "suite", "ste",
}


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def describe_score(score: Optional[float], kind: str) -> Optional[str]:
    """
    Map a numeric score to its standard WalkScore description bucket.

    ``kind`` is one of "walk", "transit", "bike". Returns ``None`` for
    unknown kinds or missing scores so callers can still detect gaps.
    """
    if _is_missing(score):
        return None
    buckets = _BUCKETS_BY_KIND.get(kind)
    if not buckets:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    for threshold, label in buckets:
        if s >= threshold:
            return label
    return buckets[-1][1]


def _slug_to_street(slug: str) -> Optional[str]:
    """
    Convert an apartments.com slug like ``14-isis-st-san-francisco-ca-unit-4``
    to a street-address-ish string ``14 isis st``.

    Strips the trailing ``san-francisco-ca[-unit-X|-id####]`` tokens.
    """
    if not slug:
        return None
    parts = slug.split("-")
    keep: list[str] = []
    for token in parts:
        t = token.lower()
        if t in _SLUG_DROP_PREFIX_TOKENS:
            break
        keep.append(token)
    if not keep:
        return None
    # Filter out apartments.com property hash IDs (alphanumeric blobs).
    keep = [t for t in keep if not (len(t) >= 6 and re.fullmatch(r"[a-z0-9]+", t.lower()))]
    if not keep:
        return None
    return " ".join(keep)


def _address_from_url(url: str) -> Optional[str]:
    """Pull a street-address-ish string out of an apartments.com URL slug."""
    if _is_missing(url) or not str(url).strip():
        return None
    try:
        path = urlparse(str(url)).path
    except Exception:
        return None
    if not path:
        return None
    segments = [seg for seg in path.split("/") if seg]
    for seg in segments:
        if "-" not in seg:
            continue
        if re.fullmatch(r"[a-z0-9]{4,12}", seg.lower()):
            # Pure hash id like "yp192jv" — skip.
            continue
        street = _slug_to_street(seg)
        if street and any(c.isdigit() for c in street):
            return street
    return None


def _normalize_address_text(address: str) -> Optional[str]:
    """Strip ``Unit X at`` prefixes and trailing unit fragments."""
    if _is_missing(address):
        return None
    s = str(address).strip()
    if not s:
        return None
    m = UNIT_AT_RE.match(s)
    if m:
        s = m.group(1).strip()
    s = TRAILING_UNIT_RE.sub("", s).strip(" ,")
    return s or None


def _looks_addressable(text: str) -> bool:
    """Heuristic: a usable street address mentions a number and at least one word."""
    if not text:
        return False
    return bool(re.search(r"\d", text)) and bool(re.search(r"[A-Za-z]", text))


def recover_address(
    address: object = None,
    url: object = None,
    zip_code: object = None,
    city: str = DEFAULT_CITY,
) -> tuple[Optional[str], str]:
    """
    Build the best query address from whatever fields are available.

    Tries, in order:
      1. ``address`` (with unit-prefix normalization) if it contains digits
         and letters.
      2. URL slug parse (apartments.com).
      3. ZIP-centroid query (``"94110, San Francisco, CA"``).

    Returns ``(query_address, source)`` where ``source`` is one of
    ``"address"``, ``"address_normalized"``, ``"url"``, ``"zip_fallback"``,
    ``"none"``. Returns ``(None, "none")`` only if every fallback fails.
    """
    cleaned_city = str(city).strip() if not _is_missing(city) and str(city).strip() else DEFAULT_CITY

    if not _is_missing(address):
        raw = str(address).strip()
        norm = _normalize_address_text(raw)
        if norm and _looks_addressable(norm):
            source = "address" if norm == raw else "address_normalized"
            return norm, source

    if not _is_missing(url):
        from_url = _address_from_url(str(url))
        if from_url and _looks_addressable(from_url):
            return from_url, "url"

    if not _is_missing(zip_code):
        z = str(zip_code).strip()
        if z and z.lower() != "nan":
            try:
                z = str(int(float(z))).zfill(5)
            except ValueError:
                pass
            return f"{z}, {cleaned_city}, {DEFAULT_STATE}", "zip_fallback"

    return None, "none"


def geocode_address(
    address: str,
    city: str = DEFAULT_CITY,
    *,
    delay: float = 1.0,
) -> tuple[Optional[float], Optional[float]]:
    """
    Resolve ``address`` to ``(lat, lon)`` via Nominatim.

    Returns ``(None, None)`` on any error or empty result so callers can
    null-fill downstream score fields without aborting the pipeline.

    Identical inputs are cached for the lifetime of the process.
    """
    if _is_missing(address) or not str(address).strip():
        return (None, None)

    addr = str(address).strip()
    cty = str(city).strip() if not _is_missing(city) else DEFAULT_CITY
    key = f"{addr}|{cty}"
    if key in _geocode_cache:
        return _geocode_cache[key]

    try:
        params = {"q": f"{addr}, {cty}", "format": "json", "limit": 1}
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": GEOCODE_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            result: tuple[Optional[float], Optional[float]] = (None, None)
        else:
            result = (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:
        result = (None, None)

    _geocode_cache[key] = result
    if delay > 0:
        time.sleep(delay)
    return result


def geocode_with_fallback(
    address: Optional[str],
    city: str,
    zip_code: object = None,
    *,
    delay: float = 1.0,
) -> tuple[float, float, str]:
    """
    Always returns coordinates, falling back through:
      ``address`` -> ``zip centroid`` -> ``city centroid``.

    ``source`` ∈ ``{"address", "zip", "city"}`` indicates which path won.
    """
    if address and not _is_missing(address):
        lat, lon = geocode_address(address, city, delay=delay)
        if lat is not None and lon is not None:
            return float(lat), float(lon), "address"

    if not _is_missing(zip_code):
        z = str(zip_code).strip()
        if z and z.lower() != "nan":
            try:
                z = str(int(float(z))).zfill(5)
            except ValueError:
                pass
            zlat, zlon = geocode_address(f"{z}, {city}, {DEFAULT_STATE}", DEFAULT_STATE, delay=delay)
            if zlat is not None and zlon is not None:
                return float(zlat), float(zlon), "zip"

    return SF_CITY_CENTROID[0], SF_CITY_CENTROID[1], "city"


_FATAL_STATUS_MESSAGES = {
    WS_STATUS_INVALID_KEY: (
        "Invalid WALKSCORE_API_KEY. Verify the key in .env matches the one "
        "issued at walkscore.com/professional/api-sign-up.php."
    ),
    WS_STATUS_QUOTA_EXCEEDED: (
        "Daily API quota exceeded. Wait until quota resets or request a "
        "higher tier from Walk Score before re-running enrichment."
    ),
    WS_STATUS_IP_BLOCKED: (
        "Your IP address has been blocked by Walk Score. Contact Walk Score "
        "support to unblock before retrying."
    ),
}


def get_scores(address: str, city: str, lat: float, lon: float) -> dict:
    """
    Fetch walk/transit/bike scores for a single address.

    Returns a dict with keys:
        walk_score, walk_description,
        transit_score, transit_description,
        bike_score, bike_description

    Behavior on failures:
      * Network/HTTP exceptions -> returns all-None so the pipeline keeps moving.
      * JSON ``status`` 1 (success) -> returns parsed scores.
      * JSON ``status`` 2 / 31 (transient) -> returns all-None; the retry layer
        will reissue the same request.
      * JSON ``status`` 30 (invalid lat/lon) -> returns all-None so the caller
        can attempt fallback coordinates.
      * JSON ``status`` 40, 41, 42 -> raises :class:`WalkScoreAPIError` so the
        pipeline fails loudly instead of writing blank score columns.
    """
    cache_key = (str(address), str(city), round(float(lat), 6), round(float(lon), 6))
    if cache_key in _score_cache:
        return dict(_score_cache[cache_key])

    try:
        full_address = f"{address}, {city}"
        params = {
            "format": "json",
            "address": full_address,
            "lat": lat,
            "lon": lon,
            "transit": 1,
            "bike": 1,
            "wsapikey": API_KEY,
        }
        resp = requests.get(WALKSCORE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return dict(_NULL_SCORES)

    status = data.get("status")
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None

    if status_int in WS_FATAL_STATUSES:
        message = _FATAL_STATUS_MESSAGES.get(status_int, "fatal Walk Score error")
        raise WalkScoreAPIError(status_int, message)

    if status_int is not None and status_int != WS_STATUS_OK:
        # Transient (2/31) or coordinate (30) failures: surface as all-None so
        # the retry layer or coordinate fallback can take over.
        return dict(_NULL_SCORES)

    result = {
        "walk_score": data.get("walkscore"),
        "walk_description": data.get("description"),
        "transit_score": data.get("transit", {}).get("score") if isinstance(data.get("transit"), dict) else None,
        "transit_description": data.get("transit", {}).get("description") if isinstance(data.get("transit"), dict) else None,
        "bike_score": data.get("bike", {}).get("score") if isinstance(data.get("bike"), dict) else None,
        "bike_description": data.get("bike", {}).get("description") if isinstance(data.get("bike"), dict) else None,
    }
    _score_cache[cache_key] = result
    return dict(result)


def get_scores_with_retry(
    address: str,
    city: str,
    lat: float,
    lon: float,
    *,
    max_retries: int = 2,
    backoff: float = 0.5,
) -> dict:
    """
    Call :func:`get_scores` up to ``max_retries + 1`` times.

    Retries when ``walk_score`` comes back ``None`` (covers transient network
    errors and Walk Score's transient JSON statuses 2 / 31). Uses bounded
    exponential backoff with jitter.

    :class:`WalkScoreAPIError` (invalid key / quota / IP block) propagates
    immediately — retrying cannot fix those.
    """
    result = get_scores(address, city, lat, lon)
    if not _is_missing(result.get("walk_score")):
        return result
    for attempt in range(1, max_retries + 1):
        delay = backoff * (2 ** (attempt - 1))
        if delay > 0:
            delay += random.uniform(0, backoff)
        if delay > 0:
            time.sleep(delay)
        result = get_scores(address, city, lat, lon)
        if not _is_missing(result.get("walk_score")):
            return result
    return result


def _fill_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill *_description columns from *_score buckets when missing."""
    pairs = (
        ("walk_score", "walk_description", "walk"),
        ("transit_score", "transit_description", "transit"),
        ("bike_score", "bike_description", "bike"),
    )
    for score_col, desc_col, kind in pairs:
        if score_col not in df.columns or desc_col not in df.columns:
            continue
        mask = df[desc_col].isna() & df[score_col].notna()
        if mask.any():
            df.loc[mask, desc_col] = df.loc[mask, score_col].apply(
                lambda v, k=kind: describe_score(v, k)
            )
    return df


def impute_missing_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing score/description fields deterministically.

    Numeric scores are imputed by ``zip_code`` group mean, then global mean
    (rounded to nearest integer). Descriptions are derived from filled
    scores via standard WalkScore buckets.

    Returns the same DataFrame (mutated copy-safe).
    """
    out = df
    has_zip = "zip_code" in out.columns

    for score_col in ("walk_score", "transit_score", "bike_score"):
        if score_col not in out.columns:
            continue
        numeric = pd.to_numeric(out[score_col], errors="coerce")
        out[score_col] = numeric

        if has_zip:
            zip_means = numeric.groupby(out["zip_code"]).transform("mean")
            out[score_col] = out[score_col].fillna(zip_means)

        global_mean = pd.to_numeric(out[score_col], errors="coerce").mean()
        if pd.notna(global_mean):
            out[score_col] = out[score_col].fillna(global_mean)

        if pd.notna(out[score_col]).any():
            out[score_col] = out[score_col].round().astype("Int64")

    out = _fill_descriptions(out)
    return out


def enrich_walkscore(
    df: pd.DataFrame,
    delay: float = 0.25,
    geocode_delay: float = 1.0,
    *,
    ensure_complete: bool = False,
    score_retries: int = 2,
) -> pd.DataFrame:
    """
    Add walk/transit/bike score columns to a listings DataFrame.

    Args:
        df:               DataFrame with at least column ``address`` *or*
                          ``url`` *or* ``zip_code``. Optional columns
                          ``latitude``, ``longitude``, ``city``. The function
                          recovers a query address from whichever fields are
                          present and falls back to ZIP/city centroids when
                          geocoding fails.
        delay:            Seconds to sleep between WalkScore API calls.
        geocode_delay:    Seconds to sleep between geocode calls.
        ensure_complete:  When True, missing transit/bike fields are imputed
                          from ZIP/global aggregates and descriptions are
                          backfilled from score buckets so every row has all
                          six fields populated.
        score_retries:    Bounded retries when WalkScore returns no walk score.

    Returns:
        ``df`` with the six score columns appended (copy returned).
    """
    df = df.copy()
    has_city = "city" in df.columns
    has_lat = "latitude" in df.columns
    has_lon = "longitude" in df.columns
    has_url = "url" in df.columns
    has_zip = "zip_code" in df.columns
    has_address = "address" in df.columns

    if not (has_address or has_url or has_zip):
        for col in SCORE_COLUMNS:
            df[col] = None
        return df

    score_rows: list[dict] = []
    total = len(df)
    api_scored = 0
    for i, row in enumerate(df.itertuples(index=False), start=1):
        address_raw = getattr(row, "address", None) if has_address else None
        url_raw = getattr(row, "url", None) if has_url else None
        zip_raw = getattr(row, "zip_code", None) if has_zip else None

        city = getattr(row, "city", DEFAULT_CITY) if has_city else DEFAULT_CITY
        if _is_missing(city) or not str(city).strip():
            city = DEFAULT_CITY

        query_addr, _src = recover_address(address_raw, url_raw, zip_raw, city)

        lat = getattr(row, "latitude", None) if has_lat else None
        lon = getattr(row, "longitude", None) if has_lon else None
        coord_src = "row" if not (_is_missing(lat) or _is_missing(lon)) else None

        if coord_src is None:
            lat, lon, coord_src = geocode_with_fallback(
                query_addr, city, zip_raw, delay=geocode_delay
            )

        api_addr = query_addr if query_addr else f"{city}"
        scores = get_scores_with_retry(
            api_addr, city, float(lat), float(lon), max_retries=score_retries
        )

        # If the first attempt produced no walk_score and we did not already
        # use the SF city centroid, retry once with the centroid coordinates.
        # The Walk Score API geocoder occasionally rejects perimeter coords
        # (status=30); falling back to a known-good centroid lets us still
        # return a score for the listing's general neighborhood.
        if _is_missing(scores.get("walk_score")) and coord_src != "city":
            fallback_lat, fallback_lon = SF_CITY_CENTROID
            if (round(float(lat), 6), round(float(lon), 6)) != (
                round(fallback_lat, 6),
                round(fallback_lon, 6),
            ):
                fallback_addr = api_addr if query_addr else f"{city}, {DEFAULT_STATE}"
                scores = get_scores_with_retry(
                    fallback_addr,
                    city,
                    float(fallback_lat),
                    float(fallback_lon),
                    max_retries=score_retries,
                )

        if not _is_missing(scores.get("walk_score")):
            api_scored += 1
        score_rows.append(scores)

        if i % 25 == 0:
            print(f"  WalkScore: processed {i}/{total} rows ({api_scored} scored by API)...")

        if delay > 0:
            time.sleep(delay)

    print(f"  WalkScore: API returned scores for {api_scored}/{total} rows")

    scores_df = pd.DataFrame(score_rows, index=df.index, columns=list(SCORE_COLUMNS))
    enriched = pd.concat([df, scores_df], axis=1)

    if ensure_complete:
        enriched = impute_missing_scores(enriched)

    return enriched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich listings with WalkScore data")
    parser.add_argument("--input", required=True, help="Path to input listings CSV")
    parser.add_argument("--output", required=True, help="Path to save enriched CSV")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between WalkScore API calls (default: 0.25)",
    )
    parser.add_argument(
        "--geocode-delay",
        type=float,
        default=1.0,
        help="Seconds between Nominatim geocode calls (default: 1.0)",
    )
    parser.add_argument(
        "--ensure-complete",
        action="store_true",
        help="Impute missing transit/bike fields and bucket-fill descriptions",
    )
    args = parser.parse_args()

    print(f"Loading listings from {args.input}...")
    listings = pd.read_csv(args.input)
    print(f"  {len(listings):,} rows loaded")

    print("Fetching WalkScore data...")
    enriched = enrich_walkscore(
        listings,
        delay=args.delay,
        geocode_delay=args.geocode_delay,
        ensure_complete=args.ensure_complete,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"\nSaved {len(enriched):,} rows → {out_path}")

    scored = enriched["walk_score"].notna().sum()
    print(f"Successfully scored: {scored:,}/{len(enriched):,} rows")
