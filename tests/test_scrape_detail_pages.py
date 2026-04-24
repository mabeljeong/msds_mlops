from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scrape" / "scrape_detail_pages.py"

# The test environment may not have Playwright installed.
# Stub these imports so we can unit-test pure parsing/merge logic.
if "playwright.sync_api" not in sys.modules:
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.TimeoutError = Exception
    sync_api.sync_playwright = lambda: None
    playwright = types.ModuleType("playwright")
    playwright.sync_api = sync_api
    sys.modules["playwright"] = playwright
    sys.modules["playwright.sync_api"] = sync_api

if "playwright_stealth" not in sys.modules:
    stealth_mod = types.ModuleType("playwright_stealth")

    class _DummyStealth:
        def use_sync(self, ctx):
            return ctx

    stealth_mod.Stealth = _DummyStealth
    sys.modules["playwright_stealth"] = stealth_mod

SPEC = importlib.util.spec_from_file_location("scrape_detail_pages", MODULE_PATH)
assert SPEC and SPEC.loader, f"Unable to load module at {MODULE_PATH}"
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

extract_detail = MODULE.extract_detail
merge_listing_and_details = MODULE.merge_listing_and_details


class _FakeNode:
    def __init__(self, text: str):
        self._text = text

    def inner_text(self, timeout: int = 2000) -> str:
        return self._text


class _FakeLocator:
    def __init__(self, texts: list[str]):
        self._texts = texts

    def count(self) -> int:
        return len(self._texts)

    def nth(self, index: int) -> _FakeNode:
        return _FakeNode(self._texts[index])


class _FakePage:
    def __init__(self, selector_to_texts: dict[str, list[str]]):
        self._selector_to_texts = selector_to_texts

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self._selector_to_texts.get(selector, []))


def test_extract_detail_collects_apartment_features_specinfo():
    page = _FakePage(
        {
            "section:has-text('Apartment Features') li.specinfo": [
                "In Unit Laundry",
                "Dishwasher",
            ],
            "div:has-text('Apartment Features') li.specinfo": ["Dishwasher"],
            "li.specinfo": ["In Unit Laundry"],
        }
    )

    details = extract_detail(page)

    assert details["amenities"] == "In Unit Laundry | Dishwasher"


def test_extract_detail_uses_empty_string_when_no_matches():
    page = _FakePage({})
    details = extract_detail(page)
    assert details["amenities"] == ""


def test_merge_listing_and_details_replaces_existing_amenities():
    listing = {
        "title": "Sample Listing",
        "amenities": "Old value",
        "url": "https://www.apartments.com/example/abc123/",
    }
    details = {"amenities": "New value from detail page"}

    merged = merge_listing_and_details(listing, details)

    assert merged["amenities"] == "New value from detail page"
