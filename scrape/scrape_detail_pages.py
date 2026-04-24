#!/usr/bin/env python3
"""
Enrich apartments.com listings with amenities from each detail page.

Reads a CSV produced by scrape_apartments_sf.py, visits each listing URL,
extracts amenities and other fields, and writes an enriched CSV.

Supports resume: if the output CSV already exists, URLs already enriched
are skipped so you can Ctrl+C and restart safely.

Usage:
  source .venv/bin/activate
  python scrape_detail_pages.py --input data/scraped/sf_apartments_listings.csv
  python scrape_detail_pages.py --input data/scraped/sf_apartments_listings.csv --chrome --delay 18
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

FIELDNAMES = [
    "title",
    "pricing",
    "beds_baths",
    "address",
    "url",
    "amenities",
]


def dismiss_overlays(page) -> None:
    for role, name in (
        ("button", "Accept"),
        ("button", "I Accept"),
        ("button", "Agree"),
    ):
        try:
            loc = page.get_by_role(role, name=name, exact=False)
            if loc.count() and loc.first.is_visible(timeout=1500):
                loc.first.click(timeout=2000)
                page.wait_for_timeout(500)
        except Exception:
            pass


def human_scroll(page) -> None:
    try:
        for _ in range(random.randint(2, 5)):
            page.mouse.wheel(0, random.randint(250, 600))
            page.wait_for_timeout(random.randint(300, 800))
    except Exception:
        pass


def looks_like_block(page) -> bool:
    u = page.url.lower()
    if any(x in u for x in ("captcha", "access denied", "akamai", "interstitial")):
        return True
    try:
        body = page.locator("body").inner_text(timeout=5000).lower()
    except Exception:
        return False
    markers = (
        "captcha",
        "access denied",
        "unusual traffic",
        "verify you are human",
        "pardon our interruption",
        "are you a robot",
        "akamai",
        "access to this page",
    )
    return any(m in body for m in markers)


def _collect_text(page, selectors: list[str]) -> list[str]:
    """Try multiple selectors; return unique non-empty text items."""
    items: list[str] = []
    seen: set[str] = set()
    for sel in selectors:
        try:
            locs = page.locator(sel)
            for i in range(locs.count()):
                raw = locs.nth(i).inner_text(timeout=2000)
                txt = re.sub(r"\s+", " ", raw).strip()
                if txt and txt.lower() not in seen:
                    seen.add(txt.lower())
                    items.append(txt)
        except Exception:
            continue
    return items


def extract_detail(page) -> dict:
    """Extract amenities from a detail page."""
    amenity_sels = [
        # Primary target from site inspection:
        # text under "Apartment Features" in li.specinfo nodes.
        "section:has-text('Apartment Features') li.specinfo",
        "div:has-text('Apartment Features') li.specinfo",
        "li.specinfo",
        # Fallbacks for occasional layout changes.
        ".amenityLabel",
        ".amenity span",
        ".specInfo.amenity",
        "[class*='amenity'] li",
        ".amenities li",
        ".amenitiesSection li span",
        ".uniqueAmenity",
    ]
    amenities = _collect_text(page, amenity_sels)
    return {"amenities": " | ".join(amenities)}


def merge_listing_and_details(listing: dict, details: dict) -> dict:
    """Return enriched listing where detail-page values override source values."""
    return {**listing, **details}


def warm_up_session(page) -> None:
    """Visit a browse page first to reduce immediate anti-bot challenges."""
    try:
        page.goto("https://www.apartments.com/san-francisco-ca/", wait_until="domcontentloaded")
        dismiss_overlays(page)
        page.wait_for_timeout(random.randint(4000, 8000))
        human_scroll(page)
        page.wait_for_timeout(random.randint(2000, 5000))
    except Exception:
        # Best-effort warmup: continue even if this fails.
        pass


def navigate_with_retries(page, url: str, max_retries: int) -> bool:
    """Navigate to URL and return False only if repeatedly blocked/timed out."""
    for attempt in range(max_retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeout:
            if attempt == max_retries:
                print("  Timeout — skipping.")
                return False
            page.wait_for_timeout(random.randint(2500, 5000))
            continue

        dismiss_overlays(page)
        page.wait_for_timeout(random.randint(1800, 4000))
        human_scroll(page)

        if not looks_like_block(page):
            return True

        if attempt == max_retries:
            print("  Blocked / challenge page detected after retries — skipping.")
            return False

        print("  Challenge detected, backing off and retrying...")
        page.wait_for_timeout(random.randint(7000, 15000))
        warm_up_session(page)
    return False


def load_input(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done_urls(path: Path) -> set[str]:
    """Return URLs already present in the enriched output (for resume)."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["url"] for row in csv.DictReader(f) if row.get("url")}


def append_row(path: Path, row: dict, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def enrich(
    input_path: Path,
    output_path: Path,
    delay_sec: float,
    headless: bool,
    use_chrome: bool,
    max_retries: int,
) -> None:
    listings = load_input(input_path)
    done_urls = load_done_urls(output_path)
    to_scrape = [r for r in listings if r.get("url") and r["url"] not in done_urls]
    print(f"{len(listings)} total listings, {len(done_urls)} already enriched, {len(to_scrape)} remaining.")

    if not to_scrape:
        print("Nothing to do.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with Stealth().use_sync(sync_playwright()) as p:
        launch_kw: dict = {
            "headless": headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
            "ignore_default_args": ["--enable-automation"],
        }
        if use_chrome:
            launch_kw["channel"] = "chrome"

        browser = p.chromium.launch(**launch_kw)
        context = browser.new_context(
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1365, "height": 900},
            geolocation={"latitude": 37.7749, "longitude": -122.4194},
            permissions=["geolocation"],
            color_scheme="light",
        )
        page = context.new_page()
        page.set_default_timeout(60000)
        warm_up_session(page)

        for idx, listing in enumerate(to_scrape, 1):
            url = listing["url"]
            print(f"[{idx}/{len(to_scrape)}] {url}")
            if not navigate_with_retries(page, url, max_retries=max_retries):
                continue

            details = extract_detail(page)
            merged = merge_listing_and_details(listing, details)
            append_row(output_path, merged, write_header=write_header)
            write_header = False
            print(f"  amenities={len(details['amenities'].split(' | ')) if details['amenities'] else 0}")

            if idx < len(to_scrape):
                jitter = random.uniform(0.75, 1.35)
                time.sleep(max(2.0, delay_sec * jitter))

        context.close()
        browser.close()

    final_count = len(load_done_urls(output_path))
    print(f"Enriched CSV now has {final_count} rows at {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Enrich apartments.com listings CSV with amenities from each detail page."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/scraped/sf_apartments_listings.csv"),
        help="Input listings CSV (default: data/scraped/sf_apartments_listings.csv)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/scraped/sf_apartments_listings_with_amenities.csv"),
        help="Output enriched CSV (default: data/scraped/sf_apartments_listings_with_amenities.csv)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=15.0,
        help="Base seconds between detail-page loads (jitter applied).",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window (higher block risk).",
    )
    ap.add_argument(
        "--chrome",
        action="store_true",
        help="Use installed Google Chrome instead of bundled Chromium.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries for blocked/timed-out detail pages (default: 1).",
    )
    args = ap.parse_args()
    enrich(
        input_path=args.input,
        output_path=args.output,
        delay_sec=args.delay,
        headless=args.headless,
        use_chrome=args.chrome,
        max_retries=max(0, args.max_retries),
    )


if __name__ == "__main__":
    main()
