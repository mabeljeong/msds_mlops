"""
Unit tests for ``app.model_loader``.

These tests focus on the offline placeholder code path (no MLflow, no network).
The real registered MLflow model is exercised separately via integration smoke
tests outside the unit-test suite.
"""

from __future__ import annotations

import pandas as pd

from app import model_loader
from app.model_loader import LoadedModel, _placeholder_predict, load_model


# ---------------------------------------------------------------------------
# _placeholder_predict — pure heuristic
# ---------------------------------------------------------------------------


def _row(**features) -> pd.DataFrame:
    return pd.DataFrame([features])


def test_placeholder_predict_clipped_range():
    """Output is always inside the [1500, 12000] safety clip."""
    out_low = _placeholder_predict(_row(bedrooms=0, zori_baseline=100))
    out_high = _placeholder_predict(_row(bedrooms=5, zori_baseline=20_000))
    assert 1500.0 <= out_low["predicted_rent_usd"] <= 12000.0
    assert 1500.0 <= out_high["predicted_rent_usd"] <= 12000.0


def test_placeholder_predict_monotonic_in_bedrooms():
    """More bedrooms (1 → 4) should weakly increase the predicted rent."""
    base = dict(zori_baseline=3500, bathrooms=1, walk_score=80, transit_score=80)
    p1 = _placeholder_predict(_row(bedrooms=1, **base))["predicted_rent_usd"]
    p4 = _placeholder_predict(_row(bedrooms=4, **base))["predicted_rent_usd"]
    assert p4 > p1


def test_placeholder_predict_returns_no_quantiles():
    """The placeholder bundle does not provide a fair-rent band."""
    out = _placeholder_predict(_row(bedrooms=2, zori_baseline=3500))
    assert out["fair_rent_p25"] is None
    assert out["fair_rent_p75"] is None


def test_placeholder_predict_handles_missing_inputs():
    """Missing/NaN inputs must not raise; we should still get a number back."""
    out = _placeholder_predict(_row(bedrooms=2))
    assert isinstance(out["predicted_rent_usd"], float)
    assert 1500.0 <= out["predicted_rent_usd"] <= 12000.0


# ---------------------------------------------------------------------------
# LoadedModel (placeholder)
# ---------------------------------------------------------------------------


def _placeholder_loaded_model() -> LoadedModel:
    return LoadedModel(
        source="placeholder",
        sklearn_model=None,
        pyfunc_model=None,
        version=None,
        load_message="test",
    )


def test_loaded_model_predict_row_returns_predicted_rent():
    m = _placeholder_loaded_model()
    out = m.predict_row({"zip_code": "94103", "bedrooms": 2, "zori_baseline": 3500})
    assert isinstance(out["predicted_rent_usd"], float)
    assert 1500.0 <= out["predicted_rent_usd"] <= 12000.0


def test_loaded_model_flag_overpriced_placeholder_never_flags():
    """Placeholder must never assert overpriced — even with absurd actual rent."""
    m = _placeholder_loaded_model()
    listing = {
        "zip_code": "94103",
        "bedrooms": 2,
        "actual_rent_usd": 50_000,  # absurdly high
        "zori_baseline": 3500,
    }
    out = m.flag_overpriced(listing)
    assert out["flag_overpriced"] is False
    assert out["flag_reason"] == "placeholder_model_no_overprice_verdict"
    assert out["delta_usd"] is not None
    assert out["delta_pct"] is not None
    assert out["top_shap_contributors"] == []


def test_loaded_model_flag_overpriced_handles_missing_actual_rent():
    """A listing with no actual rent should still return a result with None deltas."""
    m = _placeholder_loaded_model()
    listing = {"zip_code": "94103", "bedrooms": 2, "zori_baseline": 3500}
    out = m.flag_overpriced(listing)
    assert out["flag_overpriced"] is False
    assert out["delta_usd"] is None
    assert out["delta_pct"] is None


# ---------------------------------------------------------------------------
# load_model() — fallback path
# ---------------------------------------------------------------------------


def test_load_model_falls_back_to_placeholder_when_uri_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    m = load_model()
    assert m.source == "placeholder"
    assert m.pyfunc_model is None
    assert m.sklearn_model is not None
    assert m.version is None


def test_load_model_falls_back_when_mlflow_load_raises(monkeypatch):
    """If MLFLOW_MODEL_URI is set but the load call raises, we must still come up."""
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/DoesNotExist/1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # unreachable

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated mlflow failure")

    monkeypatch.setattr(model_loader.mlflow.pyfunc, "load_model", _raise)
    m = load_model()
    assert m.source == "placeholder"
    assert m.pyfunc_model is None
