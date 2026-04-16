#!/usr/bin/env python3
"""
Scrape SF apartment listings from Craigslist sfbay.craigslist.org.

Uses requests + BeautifulSoup — no browser or Playwright needed.

Output CSV has the same columns as sf_apartments_listings.csv:
  title, pricing, beds_baths, address, amenities, url, scraped_at

Usage (run from msds_mlops/):
  pip install requests beautifulsoup4 lxml
  # First run — collect up to ~1000 listings (9 pages):
  python scrape/scrape_craigslist_sf.py --max-pages 9

  # Resume / append more without re-scraping duplicates:
  python scrape/scrape_craigslist_sf.py --append
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://sfbay.craigslist.org/search/sfc/apa"
RESULTS_PER_PAGE = 120
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "scraped" / "sf_craigslist_listings.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://sfbay.craigslist.org/",
}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def page_url(offset: int) -> str:
    return f"{BASE_URL}?s={offset}"


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _text(tag) -> str:
    """Return stripped inner text of a BeautifulSoup tag, or empty string."""
    if tag is None:
        return ""
    return re.sub(r"\s+", " ", tag.get_text()).strip()


def _title_from_url(url: str) -> str:
    """
    Build a fallback title from Craigslist slug.
    Example: .../1600-efficiency-studio/7927000169.html -> 1600 efficiency studio
    """
    m = re.search(r"/d/([^/]+)/\d+\.html$", url)
    if not m:
        return ""
    slug = unquote(m.group(1)).replace("-", " ").strip()
    return re.sub(r"\s+", " ", slug)


def _parse_via_selectors(soup: BeautifulSoup, now: str) -> list[dict]:
    """Primary parser: CSS selectors for current and legacy Craigslist HTML."""
    rows: list[dict] = []

    items = soup.select("li.cl-search-result")
    if not items:
        items = soup.select("li.result-row")

    for item in items:
        link = item.select_one("a.posting-title") or item.select_one("a.result-title")
        if not link:
            continue

        url = link.get("href", "").strip()
        if url and not url.startswith("http"):
            url = "https://sfbay.craigslist.org" + url

        # Title — the .label span contains only the title, excluding price/hood
        label = link.select_one(".label") or link.select_one(".result-title")
        title = _text(label) if label else _text(link)

        # Price
        price_tag = (
            item.select_one("span.priceinfo")
            or item.select_one("span.price")
            or link.select_one("span.priceinfo")
            or link.select_one("span.price")
        )
        pricing = _text(price_tag)

        # Beds/baths — shown as "2br / 1ba" or "2br - 800ft²"
        housing_tag = item.select_one("span.housing") or item.select_one(".housing")
        beds_baths = _text(housing_tag)

        # Neighborhood / address
        hood_tag = (
            item.select_one("span.hood")
            or item.select_one(".nearby")
            or item.select_one(".result-hood")
            or link.select_one("span.hood")
        )
        address = _text(hood_tag).strip("() ")

        if not address:
            full_text = _text(link)
            m = re.search(r"\$[\d,]+\s+(.+)$", full_text)
            if m:
                address = m.group(1).strip()

        if not title:
            title = (
                link.get("aria-label", "").strip()
                or link.get("title", "").strip()
                or _title_from_url(url)
            )

        if not url:
            continue

        rows.append(
            {
                "title": title,
                "pricing": pricing,
                "beds_baths": beds_baths,
                "address": address,
                "amenities": "",
                "url": url,
                "scraped_at": now,
            }
        )
    return rows


def _parse_via_links(soup: BeautifulSoup, now: str) -> list[dict]:
    """
    Fallback parser: finds all anchor tags whose href matches the Craigslist
    listing URL pattern. Works regardless of surrounding HTML structure.

    Each anchor's text is typically: "Title$Price Neighborhood"
    e.g. "Sunny 1BR Retreat$3,695 city of san francisco"
    """
    rows: list[dict] = []
    seen_urls: set[str] = set()

    pattern = re.compile(r"https?://sfbay\.craigslist\.org/[a-z]{3}/apa/d/[^\"'>]+\.html")

    for link in soup.find_all("a", href=pattern):
        url = link.get("href", "").strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)

        full_text = _text(link)
        if not full_text:
            continue

        # Split on the first "$N,NNN" price token
        price_m = re.search(r"(\$[\d,]+)", full_text)
        if price_m:
            price_start = price_m.start()
            price_end = price_m.end()
            title = full_text[:price_start].strip()
            pricing = price_m.group(1)
            address = full_text[price_end:].strip()
        else:
            title = full_text
            pricing = ""
            address = ""

        if not title:
            title = _title_from_url(url)
        if not title:
            # Skip malformed anchors instead of writing blank-title records.
            continue

        rows.append(
            {
                "title": title,
                "pricing": pricing,
                "beds_baths": "",
                "address": address,
                "amenities": "",
                "url": url,
                "scraped_at": now,
            }
        )
    return rows


def parse_listings(html: str) -> list[dict]:
    """Parse one search-results page and return a list of row dicts."""
    soup = BeautifulSoup(html, "lxml")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = _parse_via_selectors(soup, now)

    # If CSS selectors returned nothing, fall back to link-scanning
    if not rows:
        rows = _parse_via_links(soup, now)
        if rows:
            print("  (used link-fallback parser)")

    return rows


# ---------------------------------------------------------------------------
# State file (same pattern as scrape_apartments_sf.py)
# ---------------------------------------------------------------------------

def _state_path(out_path: Path) -> Path:
    return out_path.with_name("craigslist_state.json")


def load_state(out_path: Path) -> int | None:
    sp = _state_path(out_path)
    if not sp.exists():
        return None
    try:
        with open(sp, encoding="utf-8") as f:
            return int(json.load(f)["last_offset"])
    except Exception:
        return None


def save_state(out_path: Path, offset: int) -> None:
    sp = _state_path(out_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"last_offset": offset}, f)


# ---------------------------------------------------------------------------
# CSV helpers (same pattern as scrape_apartments_sf.py)
# ---------------------------------------------------------------------------

def load_existing(out_path: Path) -> tuple[set[str], list[dict]]:
    seen: set[str] = set()
    rows: list[dict] = []
    if not out_path.exists():
        return seen, rows
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row.get("url") or ""
            key = u or f"{row.get('title', '')}|{row.get('address', '')}"
            if key not in seen:
                seen.add(key)
                rows.append(row)
    print(f"Loaded {len(rows)} existing rows from {out_path}")
    return seen, rows


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

def scrape(
    out_path: Path,
    max_pages: int,
    start_offset: int,
    delay_sec: float,
    append: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        seen, all_rows = load_existing(out_path)
    else:
        seen: set[str] = set()
        all_rows: list[dict] = []

    session = requests.Session()
    session.headers.update(HEADERS)

    for page_num in range(max_pages):
        offset = start_offset + page_num * RESULTS_PER_PAGE
        url = page_url(offset)
        print(f"Fetching page {page_num + 1} (offset {offset}): {url}")

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request failed: {e}. Stopping.")
            break

        batch = parse_listings(resp.text)
        if not batch:
            print("  No listings found on this page — reached end of results.")
            break

        new_count = 0
        for row in batch:
            key = row.get("url") or f"{row.get('title', '')}|{row.get('address', '')}"
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)
            new_count += 1

        print(f"  +{new_count} new listings (total {len(all_rows)})")
        save_state(out_path, offset)

        if new_count == 0 and page_num > 0:
            print("  No new listings on this page; stopping.")
            break

        if page_num < max_pages - 1:
            time.sleep(delay_sec + random.uniform(0, 1.5))

    fieldnames = ["title", "pricing", "beds_baths", "address", "amenities", "url", "scraped_at"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape Craigslist SF apartment listings to CSV."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=9,
        help="Max pages to fetch (120 listings each, default: 9 = ~1080 listings)",
    )
    ap.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Page offset to start from (0-indexed). Auto-detected from state file when --append is set.",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting. Skips already-seen URLs.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Base seconds between page requests (default: 2.0, jitter added automatically)",
    )
    args = ap.parse_args()

    # Resolve start offset
    if args.start_page is not None:
        start_offset = args.start_page * RESULTS_PER_PAGE
    elif args.append:
        last = load_state(args.out)
        if last is not None:
            start_offset = last + RESULTS_PER_PAGE
            print(f"Resuming from offset {start_offset} (last completed offset: {last})")
        else:
            start_offset = 0
            print("No state file found; starting from offset 0.")
    else:
        start_offset = 0

    scrape(
        out_path=args.out,
        max_pages=args.max_pages,
        start_offset=start_offset,
        delay_sec=args.delay,
        append=args.append,
    )


if __name__ == "__main__":
    main()
