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
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from playwright._impl._errors import TargetClosedError
from playwright_stealth import Stealth

BASE = "https://www.apartments.com/san-francisco-ca/"
FIELDNAMES = ["title", "pricing", "beds_baths", "address", "amenities", "url", "scraped_at"]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "scraped" / "sf_apartments_listings.csv"


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


def title_from_href(href: str) -> str:
    """
    Derive a readable title from Apartments.com URL slug when title selector is empty.
    Example: /isle-house-san-francisco-ca/yp192jv/ -> Isle House San Francisco Ca
    """
    if not href:
        return ""
    base = href.split("#", 1)[0]
    m = re.search(r"/([^/]+)/[a-z0-9]+/?$", base, flags=re.IGNORECASE)
    if not m:
        return ""
    slug = unquote(m.group(1)).replace("-", " ").strip()
    if not slug:
        return ""
    return re.sub(r"\s+", " ", slug).title()


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

        if not href:
            continue
        if not title:
            title = title_from_href(href)
        if not title:
            # Skip malformed cards where neither selector nor URL slug yields a title.
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
            if not (row.get("title") or "").strip():
                row["title"] = title_from_href(row.get("url") or "")
            u = row.get("url") or ""
            key = u or f"{row.get('title', '')}|{row.get('address', '')}"
            if key not in seen:
                seen.add(key)
                rows.append(row)
    print(f"Loaded {len(rows)} existing rows from {out_path}")
    return seen, rows


def state_path_for(out_path: Path) -> Path:
    return out_path.with_name("scrape_state.json")


def load_state(out_path: Path) -> int | None:
    """Return the last successfully scraped page number, or None if no state exists."""
    sp = state_path_for(out_path)
    if not sp.exists():
        return None
    try:
        with open(sp, encoding="utf-8") as f:
            data = json.load(f)
        return int(data["last_page"])
    except Exception:
        return None


def save_state(out_path: Path, last_page: int) -> None:
    """Persist the last successfully scraped page number."""
    sp = state_path_for(out_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"last_page": last_page}, f)


def write_rows_csv(out_path: Path, rows: list[dict]) -> None:
    """Atomically write rows so state cannot get ahead of durable data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp_path.replace(out_path)


def backoff_sleep(base_sec: float, attempt: int) -> None:
    wait = max(1.0, base_sec * (2 ** max(0, attempt - 1)) * random.uniform(0.85, 1.25))
    print(f"  Retrying after {wait:.1f}s backoff (attempt {attempt})...")
    time.sleep(wait)


def prompt_manual_challenge(page, page_num: int) -> bool:
    """
    Let the operator solve human verification in the browser, then continue.
    Returns True when the challenge appears resolved.
    """
    print(
        f"Challenge detected on page {page_num}. "
        "Complete verification in the browser, then press Enter to continue.\n"
        "Type 'q' and press Enter to stop this run."
    )
    try:
        choice = input("> ").strip().lower()
    except EOFError:
        return False
    if choice == "q":
        return False
    page.wait_for_timeout(1500)
    return not looks_like_block(page)


def scrape_sf(
    out_path: Path,
    max_pages: int,
    start_page: int,
    delay_sec: float,
    headless: bool,
    use_chrome: bool,
    no_warm: bool,
    append: bool,
    max_retries: int,
    retry_base_sec: float,
    flush_every: int,
    manual_challenge: bool,
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
        pages_since_flush = 0
        last_success_page: int | None = None
        for pg in range(start_page, end_page + 1):
            url = listing_url(pg)
            print(f"Loading page {pg}: {url}")
            page_ok = False
            batch: list[dict] = []
            for attempt in range(1, max_retries + 2):
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_load_state("load", timeout=30000)
                    dismiss_overlays(page)
                    page.wait_for_timeout(random.randint(2000, 4000))
                    human_scroll(page)
                except (PlaywrightTimeout, TargetClosedError, Exception) as e:
                    if attempt <= max_retries:
                        print(f"Navigation/setup error on page {pg}: {e}")
                        backoff_sleep(retry_base_sec, attempt)
                        continue
                    print(f"Navigation/setup failed on page {pg} after retries: {e}; stopping.")
                    blocked = True
                    break

                if looks_like_block(page):
                    if manual_challenge:
                        if prompt_manual_challenge(page, pg):
                            print("Challenge appears resolved; continuing.")
                        else:
                            print("Challenge not resolved. Stopping this run.")
                            blocked = True
                            break
                    else:
                        print(
                            "Site is showing a block or challenge (Akamai is aggressive here).\n"
                            "Wait 30-60 min then retry with a higher --delay or use --manual-challenge."
                        )
                        blocked = True
                        break

                try:
                    page.wait_for_selector("article.placard, li.mortar-wrapper", timeout=25000)
                except PlaywrightTimeout:
                    if attempt <= max_retries:
                        print(f"Listings not ready on page {pg} (attempt {attempt}).")
                        backoff_sleep(retry_base_sec, attempt)
                        continue
                    print("No listing selectors found after retries; layout may have changed or no results.")
                    break
                except (TargetClosedError, Exception) as e:
                    if attempt <= max_retries:
                        print(f"Page wait error on page {pg}: {e}")
                        backoff_sleep(retry_base_sec, attempt)
                        continue
                    print(f"Page closed while waiting for listings after retries: {e}; stopping.")
                    blocked = True
                    break

                try:
                    batch = extract_cards(page)
                except (TargetClosedError, Exception) as e:
                    if attempt <= max_retries:
                        print(f"Card extraction error on page {pg}: {e}")
                        backoff_sleep(retry_base_sec, attempt)
                        continue
                    print(f"Error extracting cards after retries: {e}; stopping.")
                    blocked = True
                    break

                page_ok = True
                break

            if blocked:
                break
            if not page_ok:
                if pg > start_page:
                    print("Stopping pagination due to repeated selector/parse failures.")
                    break
                print("Could not parse the first requested page; stopping.")
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
            last_success_page = pg
            pages_since_flush += 1
            if pages_since_flush >= flush_every:
                write_rows_csv(out_path, all_rows)
                save_state(out_path, pg)
                pages_since_flush = 0
                print(f"  Flushed CSV and state at page {pg}")
            if pg < end_page and new_count == 0 and pg > start_page:
                print("No new listings on this page; stopping pagination.")
                break

            if pg < end_page:
                jitter = random.uniform(0.75, 1.35)
                time.sleep(max(1.0, delay_sec * jitter))

        if last_success_page is not None and pages_since_flush > 0:
            write_rows_csv(out_path, all_rows)
            save_state(out_path, last_success_page)
            print(f"Final flush completed at page {last_success_page}")
        elif not out_path.exists():
            # Keep behavior predictable for first run with no successful pages.
            write_rows_csv(out_path, all_rows)

        if blocked:
            print("Stopped early due to block/error. Wait before retrying.")

        context.close()
        browser.close()

    print(f"Wrote {len(all_rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Scrape apartments.com San Francisco, CA listings to CSV.")
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    ap.add_argument("--max-pages", type=int, default=50, help="Max result pages to fetch (safety cap).")
    ap.add_argument("--start-page", type=int, default=None, help="First search-results page number. When --append is set and this is omitted, auto-detected from the state file.")
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
    ap.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per page for transient navigation/selector/extraction errors (default: 2).",
    )
    ap.add_argument(
        "--retry-base",
        type=float,
        default=4.0,
        help="Base seconds for exponential retry backoff (default: 4.0).",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush CSV/state every N successful pages (default: 1).",
    )
    ap.add_argument(
        "--manual-challenge",
        action="store_true",
        help="Pause on bot challenge so you can solve it in the visible browser, then continue.",
    )
    ap.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow overwrite when --append is not set and output file already exists.",
    )
    args = ap.parse_args()

    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.retry_base <= 0:
        raise SystemExit("--retry-base must be > 0")
    if args.flush_every < 1:
        raise SystemExit("--flush-every must be >= 1")
    if args.out.exists() and not args.append and not args.force_overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing file: {args.out}. "
            "Use --append to continue or --force-overwrite to replace it."
        )

    start_page = args.start_page
    if start_page is None:
        if args.append:
            last = load_state(args.out)
            if last is not None:
                start_page = last + 1
                print(f"Resuming from page {start_page} (last completed page: {last})")
            else:
                start_page = 1
                print("No state file found; starting from page 1.")
        else:
            start_page = 1

    scrape_sf(
        out_path=args.out,
        max_pages=args.max_pages,
        start_page=start_page,
        delay_sec=args.delay,
        headless=args.headless,
        use_chrome=args.chrome,
        no_warm=args.no_warm,
        append=args.append,
        max_retries=args.max_retries,
        retry_base_sec=args.retry_base,
        flush_every=args.flush_every,
        manual_challenge=args.manual_challenge,
    )


if __name__ == "__main__":
    main()
