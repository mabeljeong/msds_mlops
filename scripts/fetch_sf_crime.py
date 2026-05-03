"""
Fetch SFPD crime incidents (dataset ``wg3w-h783``) from the SF Open Data
Socrata endpoint, plus SF ZIP code polygons for spatial joining.

The dataset has 1M+ rows and the Socrata endpoint returns a default of
1,000 rows per request. Pagination is required to retrieve the full
dataset; we use ``$limit`` + ``$offset`` with deterministic ``$order``.

Outputs (under ``data/raw/``):
    sf_crime_incidents.csv    – one row per incident, minimal columns
    sf_zip_polygons.json      – SF ZIP polygons (deck.gl format)

Usage:
    python scripts/fetch_sf_crime.py
    python scripts/fetch_sf_crime.py --start-date 2018-01-01 --page-size 50000
    python scripts/fetch_sf_crime.py --max-rows 5000           # quick smoke test
    python scripts/fetch_sf_crime.py --skip-polygons           # incidents only

Env:
    SF_OPEN_DATA_APP_TOKEN   optional Socrata app token (higher rate limits)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

CRIME_ENDPOINT = "https://data.sfgov.org/resource/wg3w-h783.json"
ZIP_POLYGONS_URL = (
    "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/sf-zipcodes.json"
)

DEFAULT_START_DATE = "2018-01-01"
# SODA 2.1 endpoints have no upper $limit; 50,000 is a safe, fast page size.
DEFAULT_PAGE_SIZE = 50_000

DEFAULT_INCIDENTS_OUT = RAW_DIR / "sf_crime_incidents.csv"
DEFAULT_POLYGONS_OUT = RAW_DIR / "sf_zip_polygons.json"

# Minimal column set: enough for ZIP-level monthly aggregation downstream.
CRIME_FIELDS = (
    "incident_id",
    "incident_datetime",
    "incident_date",
    "incident_category",
    "incident_subcategory",
    "latitude",
    "longitude",
)


def _build_params(start_date: str, offset: int, page_size: int) -> dict[str, str]:
    return {
        "$select": ",".join(CRIME_FIELDS),
        "$where": f"incident_date >= '{start_date}T00:00:00.000'",
        # Stable ordering -> deterministic offset pagination.
        "$order": "incident_id",
        "$limit": str(page_size),
        "$offset": str(offset),
    }


def fetch_page(
    start_date: str,
    offset: int,
    page_size: int,
    *,
    app_token: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch one page of incidents; retry transient errors with backoff."""
    headers: dict[str, str] = {}
    if app_token:
        headers["X-App-Token"] = app_token

    params = _build_params(start_date, offset, page_size)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(CRIME_ENDPOINT, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            sleep_s = min(2 ** attempt, 30)
            print(f"  [retry {attempt}/{max_retries}] offset={offset} failed: {exc}; sleeping {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Crime API failed after {max_retries} retries at offset={offset}") from last_exc


def fetch_all_incidents(
    *,
    start_date: str = DEFAULT_START_DATE,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_rows: int | None = None,
    app_token: str | None = None,
    pause_s: float = 0.0,
) -> pd.DataFrame:
    """
    Page through every record from ``start_date`` onward.

    Stops when:
      * page size mismatches (page < requested -> last page reached)
      * ``max_rows`` is hit (truncate)
      * server returns an empty page

    Returns a DataFrame with ``CRIME_FIELDS`` columns.
    """
    all_rows: list[dict] = []
    offset = 0

    while True:
        rows = fetch_page(start_date, offset, page_size, app_token=app_token)
        n = len(rows)
        all_rows.extend(rows)
        print(f"  offset={offset:>9,}  +{n:>6,}  total={len(all_rows):>9,}")

        if n == 0:
            break
        if max_rows is not None and len(all_rows) >= max_rows:
            all_rows = all_rows[:max_rows]
            break
        if n < page_size:
            break

        offset += page_size
        if pause_s > 0:
            time.sleep(pause_s)

    df = pd.DataFrame(all_rows, columns=list(CRIME_FIELDS))
    return df


def fetch_incidents_to_csv(
    out_path: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_rows: int | None = None,
    app_token: str | None = None,
    pause_s: float = 0.0,
) -> int:
    """
    Stream paginated incidents directly to disk.

    The full endpoint is large enough that accumulating every page in memory can
    fail before the final CSV is written. This writer appends each page to a
    temporary CSV and only replaces ``out_path`` after a complete successful
    pull, preserving the previous output if the API or process fails mid-run.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        try:
            total = max(sum(1 for _ in tmp_path.open(encoding="utf-8")) - 1, 0)
        except OSError:
            total = 0
        wrote_header = total > 0
        offset = total
        print(f"  Resuming partial pull from {tmp_path} at offset={offset:,}", flush=True)
    else:
        offset = 0
        total = 0
        wrote_header = False

    while True:
        rows = fetch_page(start_date, offset, page_size, app_token=app_token)
        n_api = len(rows)

        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            rows = rows[:remaining]

        n_write = len(rows)
        if n_write:
            page = pd.DataFrame(rows, columns=list(CRIME_FIELDS))
            page.to_csv(tmp_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            total += n_write

        print(f"  offset={offset:>9,}  +{n_write:>6,}  total={total:>9,}", flush=True)

        if n_api == 0:
            break
        if max_rows is not None and total >= max_rows:
            break
        if n_api < page_size:
            break

        offset += page_size
        if pause_s > 0:
            time.sleep(pause_s)

    if not wrote_header:
        pd.DataFrame(columns=list(CRIME_FIELDS)).to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
    return total


def fetch_zip_polygons(out_path: Path = DEFAULT_POLYGONS_OUT, *, timeout: float = 60.0) -> Path:
    """Download SF ZIP polygons (deck.gl format) and write JSON to ``out_path``."""
    resp = requests.get(ZIP_POLYGONS_URL, timeout=timeout)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch SF crime incidents + ZIP polygons.")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="ISO date floor (default: 2018-01-01)")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Rows per request (default: 50000)")
    p.add_argument("--max-rows", type=int, default=None, help="Truncate after N rows (for smoke tests)")
    p.add_argument("--output", type=Path, default=DEFAULT_INCIDENTS_OUT, help="Incidents CSV output path")
    p.add_argument("--polygons-output", type=Path, default=DEFAULT_POLYGONS_OUT, help="ZIP polygons JSON output path")
    p.add_argument("--skip-polygons", action="store_true", help="Don't refresh ZIP polygons file")
    p.add_argument("--skip-incidents", action="store_true", help="Don't fetch incidents (polygons only)")
    p.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between pages")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    app_token = os.environ.get("SF_OPEN_DATA_APP_TOKEN") or None
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_polygons:
        print(f"Fetching ZIP polygons → {args.polygons_output}", flush=True)
        fetch_zip_polygons(args.polygons_output)
        print(f"  wrote {args.polygons_output}", flush=True)

    if args.skip_incidents:
        return

    print(
        f"Fetching incidents from {args.start_date} "
        f"(page_size={args.page_size:,}, max_rows={args.max_rows})",
        flush=True,
    )
    n_rows = fetch_incidents_to_csv(
        args.output,
        start_date=args.start_date,
        page_size=args.page_size,
        max_rows=args.max_rows,
        app_token=app_token,
        pause_s=args.pause,
    )

    print(f"\nSaved {n_rows:,} rows → {args.output}", flush=True)
    if n_rows > 0:
        with_coords = pd.read_csv(args.output, usecols=["latitude"])["latitude"].notna().sum()
        print(f"  rows with lat/lon: {with_coords:,}/{n_rows:,}", flush=True)


if __name__ == "__main__":
    main()
