#!/usr/bin/env python3
"""
Scrape rental listing cards from apartments.com for San Francisco, CA.

Check apartments.com Terms of Service and robots.txt before use. Uses slow
page loads and a real browser profile to reduce load; for personal/educational
use unless you have explicit permission.

Usage (venv recommended — PEP 668 on many Linux distros):
  python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
  pip install -r requirements-scraper.txt
  playwright install chromium
  # First run — visible browser + stealth + slow delays:
  python scrape_apartments_sf.py --max-pages 2 --chrome

  # Later sessions — append new rows starting from page 3:
  python scrape_apartments_sf.py --append --start-page 3 --max-pages 2 --chrome

If you still get blocked, apartments.com uses strong bot protection; try longer --delay, same Wi-Fi as normal browsing, or another data source.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from playwright._impl._errors import TargetClosedError
from playwright_stealth import Stealth

BASE = "https://www.apartments.com/san-francisco-ca/"


def listing_url(page_index: int) -> str:
    if page_index <= 1:
        return BASE
    return urljoin(BASE, f"{page_index}/")


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


def text_or_empty(locator) -> str:
    try:
        t = locator.inner_text(timeout=3000)
        return re.sub(r"\s+", " ", t).strip()
    except Exception:
        return ""


def extract_cards(page):
    """Return list of dicts from visible listing cards."""
    cards = page.locator("article.placard")
    n = cards.count()
    if n == 0:
        cards = page.locator("li.mortar-wrapper article, article[data-listingid]")
        n = cards.count()

    rows = []
    for i in range(n):
        card = cards.nth(i)
        link = card.locator("a.property-link").first
        href = ""
        try:
            href = link.get_attribute("href") or ""
        except Exception:
            pass
        if not href:
            try:
                href = card.locator('a[href*="/san-francisco-ca/"]').first.get_attribute("href") or ""
            except Exception:
                pass
        if href and not href.startswith("http"):
            href = urljoin("https://www.apartments.com", href)

        title = text_or_empty(card.locator(".property-title, .placardTitle, a.property-link").first)
        pricing = text_or_empty(card.locator(".property-pricing, .pricingContainer, [class*='rent']").first)
        address = text_or_empty(card.locator(".property-address, .addressContainer, [class*='address']").first)
        beds_baths = text_or_empty(
            card.locator(".property-beds, .beds-baths, [class*='bed']").first
        )
        amenities = text_or_empty(card.locator("p.property-amenities").first)

        if not href and not title:
            continue

        rows.append(
            {
                "title": title,
                "pricing": pricing,
                "beds_baths": beds_baths,
                "address": address,
                "amenities": amenities,
                "url": href,
                "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
        )
    return rows


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
        "unusual traffic",
        "verify you are human",
        "pardon our interruption",
        "are you a robot",
        "access to this page has been denied",
    )
    return any(m in body for m in markers)


def human_scroll(page) -> None:
    """Light scrolling so the session looks less like a cold bot."""
    try:
        for _ in range(random.randint(2, 4)):
            page.mouse.wheel(0, random.randint(200, 500))
            page.wait_for_timeout(random.randint(300, 900))
    except Exception:
        pass


def warm_start(page, do_warm: bool) -> None:
    if not do_warm:
        return
    try:
        page.goto("https://www.apartments.com/", wait_until="domcontentloaded", timeout=60000)
        dismiss_overlays(page)
        human_scroll(page)
        page.wait_for_timeout(random.randint(2000, 4500))
    except Exception:
        pass


def load_existing(out_path: Path) -> tuple[set[str], list[dict]]:
    """Load rows from an existing CSV so we can append without duplicates."""
    seen: set[str] = set()
    rows: list[dict] = []
    if not out_path.exists():
        return seen, rows
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get("url") or ""
            key = u or f"{row.get('title', '')}|{row.get('address', '')}"
            if key not in seen:
                seen.add(key)
                rows.append(row)
    print(f"Loaded {len(rows)} existing rows from {out_path}")
    return seen, rows


def scrape_sf(
    out_path: Path,
    max_pages: int,
    start_page: int,
    delay_sec: float,
    headless: bool,
    use_chrome: bool,
    no_warm: bool,
    append: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        seen, all_rows = load_existing(out_path)
    else:
        seen: set[str] = set()
        all_rows: list[dict] = []

    # Stealth patches new_page / new_context so listings see a less "automated" fingerprint.
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
        try:
            browser = p.chromium.launch(**launch_kw)
        except Exception as e:
            print(
                "Launch failed (if you used --chrome, install Google Chrome or omit --chrome). Error:",
                e,
            )
            raise

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

        warm_start(page, do_warm=not no_warm)

        end_page = start_page + max_pages - 1
        blocked = False
        for pg in range(start_page, end_page + 1):
            url = listing_url(pg)
            print(f"Loading page {pg}: {url}")
            try:
                page.goto(url, wait_until="domcontentloaded")
            except (PlaywrightTimeout, TargetClosedError, Exception) as e:
                print(f"Navigation error on page {pg}: {e}; stopping.")
                blocked = True
                break

            try:
                dismiss_overlays(page)
                page.wait_for_timeout(random.randint(2000, 4000))
                human_scroll(page)
            except (TargetClosedError, Exception):
                print("Page closed unexpectedly (likely a bot challenge). Stopping.")
                blocked = True
                break

            if looks_like_block(page):
                print(
                    "Site is showing a block or challenge (Akamai is aggressive here).\n"
                    "Wait 30–60 min then retry with a higher --delay or run in a fresh terminal."
                )
                blocked = True
                break

            try:
                page.wait_for_selector("article.placard, li.mortar-wrapper", timeout=25000)
            except PlaywrightTimeout:
                print("No listing selectors found; layout may have changed or no results.")
                if pg > start_page:
                    break
            except (TargetClosedError, Exception) as e:
                print(f"Page closed while waiting for listings: {e}; stopping.")
                blocked = True
                break

            try:
                batch = extract_cards(page)
            except (TargetClosedError, Exception) as e:
                print(f"Error extracting cards: {e}; stopping.")
                blocked = True
                break

            new_count = 0
            for row in batch:
                u = row.get("url") or ""
                key = u or f"{row.get('title', '')}|{row.get('address', '')}"
                if key in seen:
                    continue
                seen.add(key)
                all_rows.append(row)
                new_count += 1

            print(f"  +{new_count} new listings (total {len(all_rows)})")
            if pg < end_page and new_count == 0 and pg > start_page:
                print("No new listings on this page; stopping pagination.")
                break

            if pg < end_page:
                jitter = random.uniform(0.75, 1.35)
                time.sleep(max(1.0, delay_sec * jitter))

        if blocked:
            print("Stopped early due to block/error. Wait before retrying.")

        context.close()
        browser.close()

    fieldnames = ["title", "pricing", "beds_baths", "address", "amenities", "url", "scraped_at"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Scrape apartments.com San Francisco, CA listings to CSV.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/sf_apartments_listings.csv"),
        help="Output CSV path (default: data/sf_apartments_listings.csv)",
    )
    ap.add_argument("--max-pages", type=int, default=50, help="Max result pages to fetch (safety cap).")
    ap.add_argument("--start-page", type=int, default=1, help="First search-results page number (use with --append to resume).")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting. Skips already-seen URLs.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Base seconds between page loads (jitter applied). Default higher to reduce blocks.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window (more likely to be blocked on apartments.com).",
    )
    ap.add_argument(
        "--chrome",
        action="store_true",
        help="Use installed Google Chrome instead of bundled Chromium (often fewer flags).",
    )
    ap.add_argument(
        "--no-warm",
        action="store_true",
        help="Skip opening the apartments.com homepage before SF results.",
    )
    args = ap.parse_args()
    scrape_sf(
        out_path=args.out,
        max_pages=args.max_pages,
        start_page=args.start_page,
        delay_sec=args.delay,
        headless=args.headless,
        use_chrome=args.chrome,
        no_warm=args.no_warm,
        append=args.append,
    )


if __name__ == "__main__":
    main()
