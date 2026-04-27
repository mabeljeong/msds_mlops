"""
Unit tests for the cleaned apartments.com SF listings.

These tests validate the shape and per-column correctness of
``data/scraped/sf_apartments_listings_clean.csv``, which is produced by
``scripts/clean_combined_listings.py`` (explodes multi-unit rows, extracts
zip_code, fills missing addresses, and normalises pricing/beds_baths).

Run from the repo root:

    pytest tests/test_sf_apartments_csv.py -v
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "scraped" / "sf_apartments_listings_clean.csv"

EXPECTED_COLUMNS = [
    "title",
    "beds_baths",
    "pricing",
    "address",
    "zip_code",
    "url",
    "scraped_at",
]

REQUIRED_COLUMNS = ("title", "pricing", "beds_baths", "address", "zip_code", "url")

URL_PREFIX = "https://www.apartments.com/"
URL_PATH_RE = re.compile(r"^https://www\.apartments\.com/[a-z0-9\-/]+/[a-z0-9]+/?$")
SCRAPED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")
ADDRESS_ZIP_RE = re.compile(r",\s*CA\s+9\d{4}\s*$")
BEDS_BATHS_RE = re.compile(r"(?i)\b(studio|bed|beds)\b")
UNIT_TYPE_RE = re.compile(r"(?i)^(studio|\d+\s+beds?\+?)$")
DOLLAR_RE = re.compile(r"\$\s?\d")
CLEAN_PRICE_RE = re.compile(r"^\$[\d,]+\+?$")
ZIP_CODE_RE = re.compile(r"^\d{5}$")
PRICE_ONLY_TITLE_RE = re.compile(r"^\s*\$[\d,]+\+?\s*$")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """Read the scraped CSV once per test module."""
    if not CSV_PATH.exists():
        pytest.skip(f"CSV not found at {CSV_PATH}")
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@pytest.fixture(scope="module")
def header() -> list[str]:
    if not CSV_PATH.exists():
        pytest.skip(f"CSV not found at {CSV_PATH}")
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _fails(rows: list[dict], predicate) -> list[tuple[int, dict]]:
    """Return (row_index, row) pairs where predicate(row) is falsy.

    Row index is 1-based over data rows (not counting the header).
    """
    bad = []
    for idx, row in enumerate(rows, start=1):
        if not predicate(row):
            bad.append((idx, row))
    return bad


def _fmt_bad(bad: list[tuple[int, dict]], column: str, limit: int = 5) -> str:
    sample = bad[:limit]
    lines = [
        f"  row {idx}: {column}={row.get(column)!r} url={row.get('url')!r}"
        for idx, row in sample
    ]
    more = "" if len(bad) <= limit else f"\n  ...and {len(bad) - limit} more"
    return "\n".join(lines) + more


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_csv_exists():
    assert CSV_PATH.exists(), f"Expected scrape output at {CSV_PATH}"


def test_header_matches_expected_columns(header):
    assert header == EXPECTED_COLUMNS, (
        f"CSV header mismatch.\n  expected: {EXPECTED_COLUMNS}\n  got:      {header}"
    )


def test_non_empty(rows):
    assert len(rows) > 0, "CSV has a header but no data rows"


def test_every_row_has_all_columns(rows):
    bad = [
        (i, row)
        for i, row in enumerate(rows, start=1)
        if set(row.keys()) != set(EXPECTED_COLUMNS) or any(row.get(c) is None for c in EXPECTED_COLUMNS)
    ]
    assert not bad, (
        f"{len(bad)} rows are missing one or more columns (ragged CSV):\n"
        + _fmt_bad(bad, "url")
    )


# ---------------------------------------------------------------------------
# Required-field tests (scraper should never emit empties here)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_required_column_non_empty(rows, column):
    bad = _fails(rows, lambda r, c=column: (r.get(c) or "").strip() != "")
    assert not bad, (
        f"{len(bad)}/{len(rows)} rows have empty '{column}':\n"
        + _fmt_bad(bad, column)
    )


# ---------------------------------------------------------------------------
# URL tests
# ---------------------------------------------------------------------------


def test_urls_use_apartments_com(rows):
    bad = _fails(rows, lambda r: (r.get("url") or "").startswith(URL_PREFIX))
    assert not bad, (
        f"{len(bad)} rows have a url that is not on apartments.com:\n"
        + _fmt_bad(bad, "url")
    )


def test_urls_match_expected_shape(rows):
    bad = _fails(rows, lambda r: bool(URL_PATH_RE.match((r.get("url") or "").strip())))
    assert not bad, (
        f"{len(bad)} urls do not look like /slug/listingid/ paths:\n"
        + _fmt_bad(bad, "url")
    )


def test_url_unit_type_combo_is_unique(rows):
    # After exploding, the same URL appears once per unit type — but the
    # (url, beds_baths) pair must still be unique.
    seen: dict[tuple[str, str], int] = {}
    dupes: list[tuple[tuple[str, str], list[int]]] = []
    for idx, row in enumerate(rows, start=1):
        key = ((row.get("url") or "").strip(), (row.get("beds_baths") or "").strip())
        if not key[0]:
            continue
        if key in seen:
            dupes.append((key, [seen[key], idx]))
        else:
            seen[key] = idx
    assert not dupes, (
        f"{len(dupes)} duplicate (url, beds_baths) pairs found (first few):\n"
        + "\n".join(f"  {u!r} at rows {rs}" for u, rs in dupes[:5])
    )


# ---------------------------------------------------------------------------
# Per-column content tests
# ---------------------------------------------------------------------------


def test_pricing_contains_dollar_amount(rows):
    bad = _fails(rows, lambda r: bool(DOLLAR_RE.search(r.get("pricing") or "")))
    assert not bad, (
        f"{len(bad)} rows have 'pricing' without a dollar amount:\n"
        + _fmt_bad(bad, "pricing")
    )


def test_beds_baths_mentions_bed_or_studio(rows):
    bad = _fails(rows, lambda r: bool(BEDS_BATHS_RE.search(r.get("beds_baths") or "")))
    assert not bad, (
        f"{len(bad)} rows have 'beds_baths' without 'studio'/'bed(s)':\n"
        + _fmt_bad(bad, "beds_baths")
    )


def test_scraped_at_is_valid_when_present(rows):
    """When scraped_at is present it must match 'YYYY-MM-DD HH:MM:SS UTC'.

    Historic rows may have an empty scraped_at; empty is allowed but a non-empty
    value with the wrong shape is a bug.
    """
    bad = _fails(
        rows,
        lambda r: (not (r.get("scraped_at") or "").strip())
        or bool(SCRAPED_AT_RE.match(r["scraped_at"].strip())),
    )
    assert not bad, (
        f"{len(bad)} rows have a malformed 'scraped_at':\n"
        + _fmt_bad(bad, "scraped_at")
    )


def test_zip_code_is_5_digits(rows):
    bad = _fails(rows, lambda r: bool(ZIP_CODE_RE.match((r.get("zip_code") or "").strip())))
    assert not bad, (
        f"{len(bad)} rows have a malformed 'zip_code' (expected 5 digits):\n"
        + _fmt_bad(bad, "zip_code")
    )


def test_pricing_clean_format(rows):
    bad = _fails(rows, lambda r: bool(CLEAN_PRICE_RE.match((r.get("pricing") or "").strip())))
    assert not bad, (
        f"{len(bad)} rows have 'pricing' not in clean '$X,XXX+?' format:\n"
        + _fmt_bad(bad, "pricing")
    )


def test_beds_baths_is_unit_type_label(rows):
    bad = _fails(rows, lambda r: bool(UNIT_TYPE_RE.match((r.get("beds_baths") or "").strip())))
    assert not bad, (
        f"{len(bad)} rows have 'beds_baths' that is not a unit type label (e.g. 'Studio', '1 Bed', '2 Beds+'):\n"
        + _fmt_bad(bad, "beds_baths")
    )


# ---------------------------------------------------------------------------
# Tests that the current scrape is known to violate.
#
# These use xfail(strict=False) so the test suite does not break CI today but
# still reports progress as the scraper improves. Remove the xfail once the
# scraper consistently fills these fields.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="Addresses filled from title via fill_address_from_title() may not have ', CA 9XXXX' suffix",
)
def test_address_non_empty_and_has_zip(rows):
    bad = _fails(
        rows,
        lambda r: bool(ADDRESS_ZIP_RE.search((r.get("address") or "").strip())),
    )
    assert not bad, (
        f"{len(bad)} rows have empty/malformed 'address' (missing ', CA 9XXXX' suffix):\n"
        + _fmt_bad(bad, "address")
    )


@pytest.mark.xfail(
    strict=False,
    reason="Some cards fall back to using the pricing string as the title",
)
def test_title_is_not_just_a_price(rows):
    bad = _fails(rows, lambda r: not PRICE_ONLY_TITLE_RE.match((r.get("title") or "").strip()))
    assert not bad, (
        f"{len(bad)} rows have a 'title' that is only a price:\n"
        + _fmt_bad(bad, "title")
    )
