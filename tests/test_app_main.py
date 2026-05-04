"""
Integration tests for the RentIQ FastAPI service.

These tests run against the placeholder model (no MLflow tracking server
required). The placeholder is the offline fallback that ``app.model_loader``
returns whenever ``MLFLOW_MODEL_URI`` is unset, so this exercises the same
load + serve path that ``/health`` reports as ``model_source="placeholder"``.

Run from the repo root:

    pytest tests/test_app_main.py -v
"""

from __future__ import annotations

import os

# Force the placeholder load path *before* importing the app, so the lifespan
# does not attempt to reach a real MLflow tracking server in CI.
os.environ.pop("MLFLOW_MODEL_URI", None)
os.environ.pop("MLFLOW_TRACKING_URI", None)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ranker import COMPONENT_KEYS


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient that runs the FastAPI lifespan (loads the placeholder model)."""
    with TestClient(app) as c:
        yield c


def _minimal_predict_payload(**overrides) -> dict:
    base = {
        "zip_code": "94103",
        "bedrooms": 2,
    }
    base.update(overrides)
    return base


def _full_listing_payload(**overrides) -> dict:
    base = {
        "zip_code": "94103",
        "bedrooms": 2,
        "bathrooms": 1,
        "walk_score": 95,
        "transit_score": 90,
        "census_median_income": 130_000,
        "crime_total_month_zip_log1p_latest": 5.2,
        "zori_baseline": 3450,
        "actual_rent_usd": 3500,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok_with_placeholder(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_source"] == "placeholder"
    assert body["rank_component_keys"] == list(COMPONENT_KEYS)


# ---------------------------------------------------------------------------
# /listings
# ---------------------------------------------------------------------------


def test_listings_returns_count_and_list(client):
    r = client.get("/listings")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body and "listings" in body
    assert isinstance(body["listings"], list)
    assert body["count"] == len(body["listings"])
    assert body["count"] >= 0


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_minimal_payload_returns_finite_rent(client):
    r = client.post("/predict", json=_minimal_predict_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["predicted_rent_usd"], float)
    assert 1500.0 <= body["predicted_rent_usd"] <= 12000.0
    assert body["model_source"] == "placeholder"


def test_predict_rejects_short_zip(client):
    r = client.post("/predict", json=_minimal_predict_payload(zip_code="9410"))
    assert r.status_code == 422


def test_predict_rejects_long_zip(client):
    r = client.post("/predict", json=_minimal_predict_payload(zip_code="941030"))
    assert r.status_code == 422


def test_predict_rejects_missing_required_fields(client):
    r = client.post("/predict", json={"zip_code": "94103"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /flag_overpriced
# ---------------------------------------------------------------------------


def test_flag_overpriced_placeholder_never_flags(client):
    """The placeholder model is explicitly never allowed to assert 'overpriced'."""
    payload = _full_listing_payload(actual_rent_usd=10_000)  # very high
    r = client.post("/flag_overpriced", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["flag_overpriced"] is False
    assert body["flag_reason"] == "placeholder_model_no_overprice_verdict"
    assert isinstance(body["top_shap_contributors"], list)


def test_flag_overpriced_returns_delta_fields(client):
    payload = _full_listing_payload(actual_rent_usd=4000)
    r = client.post("/flag_overpriced", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_rent_usd"], float)
    assert body["delta_usd"] is not None
    assert body["delta_pct"] is not None


def test_flag_overpriced_rejects_zero_rent(client):
    payload = _full_listing_payload(actual_rent_usd=0)
    r = client.post("/flag_overpriced", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /rank
# ---------------------------------------------------------------------------


def test_rank_empty_listings_rejected(client):
    """RankRequest.listings has min_length=1 — Pydantic rejects with 422."""
    r = client.post("/rank", json={"listings": []})
    assert r.status_code in (400, 422)


def test_rank_two_listings_returns_sorted_results(client):
    payload = {
        "listings": [
            _full_listing_payload(
                listing_id="a",
                walk_score=99,
                transit_score=99,
                crime_total_month_zip_log1p_latest=2.0,
                actual_rent_usd=3500,
            ),
            _full_listing_payload(
                listing_id="b",
                zip_code="94110",
                walk_score=40,
                transit_score=40,
                crime_total_month_zip_log1p_latest=8.0,
                actual_rent_usd=3500,
            ),
        ],
        "weights": {"safety": 1.0, "walk": 1.0, "transit": 1.0},
    }
    r = client.post("/rank", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input"] == 2
    assert body["n_returned"] == 2
    assert len(body["results"]) == 2

    # weights must be normalized to sum to 1.0
    weights = body["weights_normalized"]
    assert pytest.approx(sum(weights.values()), abs=1e-3) == 1.0
    assert set(weights.keys()) == set(COMPONENT_KEYS)

    # listing 'a' (high walk/transit, low crime) should outrank 'b'
    ranks = {row["listing_id"]: row["rank"] for row in body["results"]}
    assert ranks["a"] < ranks["b"]

    # composite scores must descend with rank
    composites = [row["composite_score"] for row in body["results"]]
    assert composites == sorted(composites, reverse=True)

    # every result has the canonical component keys
    for row in body["results"]:
        assert set(row["component_scores"].keys()) == set(COMPONENT_KEYS)


def test_rank_top_n_truncates_results(client):
    payload = {
        "listings": [
            _full_listing_payload(listing_id=str(i), actual_rent_usd=3000 + i * 50)
            for i in range(5)
        ],
        "top_n": 2,
    }
    r = client.post("/rank", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input"] == 5
    assert body["n_returned"] == 2
    assert len(body["results"]) == 2


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------


def test_root_returns_spa_or_banner(client):
    """Either the SPA HTML (when web/index.html exists) or the JSON banner."""
    r = client.get("/")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    if "text/html" in ctype:
        assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()
    else:
        body = r.json()
        assert body.get("service") == "RentRadar"
