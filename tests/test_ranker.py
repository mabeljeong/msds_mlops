"""
Unit tests for ``app.ranker`` and ``app.listing_fields``.

The ranker pieces are pure functions that operate on listing/flag dicts, so
these are fast unit tests with no I/O.
"""

from __future__ import annotations

import pytest

from app.listing_fields import actual_rent_usd_from_listing
from app.ranker import (
    COMPONENT_KEYS,
    _normalize_walk_transit,
    _safety_scores,
    composite,
    normalize_weights,
    score_listings,
)

# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_one():
    out = normalize_weights({"safety": 1, "walk": 2, "transit": 1})
    assert pytest.approx(sum(out.values()), abs=1e-9) == 1.0
    assert out["walk"] > out["safety"]


def test_normalize_weights_all_zero_gives_uniform():
    out = normalize_weights({"safety": 0, "walk": 0, "transit": 0})
    for k in COMPONENT_KEYS:
        assert out[k] == pytest.approx(1.0 / len(COMPONENT_KEYS))


def test_normalize_weights_negatives_clamped_to_zero():
    out = normalize_weights({"safety": -5, "walk": 1, "transit": 1})
    assert out["safety"] == pytest.approx(0.0)
    assert out["walk"] == pytest.approx(0.5)
    assert out["transit"] == pytest.approx(0.5)


def test_normalize_weights_missing_keys_default_to_zero():
    out = normalize_weights({"walk": 1.0})
    assert out["walk"] == pytest.approx(1.0)
    assert out["safety"] == pytest.approx(0.0)
    assert out["transit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _normalize_walk_transit
# ---------------------------------------------------------------------------


def test_normalize_walk_transit_clamping():
    assert _normalize_walk_transit(None) == 0.5
    assert _normalize_walk_transit(0) == 0.0
    assert _normalize_walk_transit(50) == 0.5
    assert _normalize_walk_transit(100) == 1.0
    assert _normalize_walk_transit(150) == 1.0  # clipped
    assert _normalize_walk_transit(-10) == 0.0  # clipped


# ---------------------------------------------------------------------------
# _safety_scores
# ---------------------------------------------------------------------------


def test_safety_scores_all_missing_returns_half():
    assert _safety_scores([None, None, None]) == [0.5, 0.5, 0.5]


def test_safety_scores_min_max_invert_crime():
    """Lowest crime → 1.0, highest crime → 0.0; missing → 0.5."""
    out = _safety_scores([1.0, 9.0, None, 5.0])
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert 0.0 < out[3] < 1.0


def test_safety_scores_constant_crime_returns_ones():
    """If every finite crime value is identical we have no spread; treat present values as best."""
    out = _safety_scores([3.0, 3.0, 3.0])
    assert out == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# score_listings
# ---------------------------------------------------------------------------


def test_score_listings_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_listings([{"zip_code": "94103"}], [{}, {}])


def test_score_listings_components_in_unit_interval():
    listings = [
        {"walk_score": 95, "transit_score": 90, "crime_total_month_zip_log1p_latest": 2.0},
        {"walk_score": 40, "transit_score": 40, "crime_total_month_zip_log1p_latest": 8.0},
    ]
    flags = [{}, {}]
    out = score_listings(listings, flags)
    assert len(out) == 2
    for sc in out:
        assert set(sc.component_scores.keys()) == set(COMPONENT_KEYS)
        for v in sc.component_scores.values():
            assert 0.0 <= v <= 1.0
    # listing 0 should beat listing 1 on all three components
    assert out[0].component_scores["safety"] > out[1].component_scores["safety"]
    assert out[0].component_scores["walk"] > out[1].component_scores["walk"]
    assert out[0].component_scores["transit"] > out[1].component_scores["transit"]


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------


def test_composite_dot_product_matches_hand_computed():
    listings = [
        {"walk_score": 100, "transit_score": 100, "crime_total_month_zip_log1p_latest": 0.0},
    ]
    scored = score_listings(listings, [{}])
    weights = normalize_weights({"safety": 1, "walk": 1, "transit": 1})
    expected = sum(weights[k] * scored[0].component_scores[k] for k in COMPONENT_KEYS)
    assert composite(scored[0], weights) == pytest.approx(expected)
    assert 0.0 <= composite(scored[0], weights) <= 1.0


# ---------------------------------------------------------------------------
# actual_rent_usd_from_listing
# ---------------------------------------------------------------------------


def test_actual_rent_prefers_actual_rent_usd_key():
    assert actual_rent_usd_from_listing({"actual_rent_usd": 3000, "rent_usd": 9999}) == 3000.0


def test_actual_rent_falls_back_to_rent_usd():
    assert actual_rent_usd_from_listing({"rent_usd": 2750}) == 2750.0


def test_actual_rent_returns_none_when_missing():
    assert actual_rent_usd_from_listing({"zip_code": "94103"}) is None


def test_actual_rent_returns_none_for_unparseable():
    assert actual_rent_usd_from_listing({"actual_rent_usd": "not-a-number"}) is None


def test_actual_rent_returns_none_for_explicit_none():
    assert actual_rent_usd_from_listing({"actual_rent_usd": None, "rent_usd": None}) is None
