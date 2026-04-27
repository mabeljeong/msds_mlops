"""Composite scoring + ranking on top of the rent model's flag_overpriced output.

Each listing receives five component scores in [0, 1]:

    - price_fairness : how the asking rent sits relative to the model's p10/p90 band.
                       In-band or below → 1.0. Above p90 → degrades linearly with
                       overage; pegged to 0.0 when the listing is ≥50% above p90.
    - safety         : 1 − batch-normalized crime intensity (lower crime = higher score).
                       Falls back to 0.5 when the crime feature is missing.
    - walk           : walk_score / 100 (or 0.5 if missing).
    - transit        : transit_score / 100 (or 0.5 if missing).
    - affordability  : how the asking rent sits vs the renter's budget. ≤ budget → 1.0.
                       Linearly degrades; pegged to 0.0 at 1.5× budget. If no budget,
                       score relative to the cheapest listing in the batch.

The composite score is the sum of (normalized weight × component), so it is also in
[0, 1] regardless of the absolute weight scale the user provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


COMPONENT_KEYS = ("price_fairness", "safety", "walk", "transit", "affordability")


@dataclass
class ScoredListing:
    """A flag_overpriced result enriched with component scores and the source listing."""

    listing: dict
    flag_result: dict
    component_scores: dict[str, float]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _price_fairness(actual: float, predicted: float, p90: float | None) -> float:
    """Above-p90 listings are penalized; in-band/below = 1.0."""
    if p90 is not None and p90 > 0:
        if actual <= p90:
            return 1.0
        overage_pct = (actual - p90) / p90  # >0
        return float(max(0.0, 1.0 - min(overage_pct / 0.5, 1.0)))

    if predicted <= 0:
        return 0.5
    delta_pct = (actual - predicted) / predicted
    return float(max(0.0, min(1.0, 1.0 - max(0.0, delta_pct))))


def _affordability(actual: float, budget: float | None, batch_min: float) -> float:
    """≤ budget → 1.0; linearly degrades; 0.0 at ≥1.5× budget."""
    if budget is not None and budget > 0:
        if actual <= budget:
            return 1.0
        excess = (actual - budget) / budget
        return float(max(0.0, 1.0 - min(excess / 0.5, 1.0)))

    if batch_min <= 0 or not np.isfinite(batch_min):
        return 0.5
    return float(max(0.0, min(1.0, batch_min / actual)))


def _normalize_walk_transit(value: float | None) -> float:
    if value is None:
        return 0.5
    return float(max(0.0, min(1.0, value / 100.0)))


def _safety_scores(crime_values: list[float | None]) -> list[float]:
    finite = [v for v in crime_values if v is not None and np.isfinite(v)]
    if not finite:
        return [0.5] * len(crime_values)
    cmin, cmax = float(min(finite)), float(max(finite))
    if cmax - cmin < 1e-9:
        return [1.0 if v is not None else 0.5 for v in crime_values]
    out: list[float] = []
    for v in crime_values:
        if v is None or not np.isfinite(v):
            out.append(0.5)
        else:
            out.append(float(1.0 - (v - cmin) / (cmax - cmin)))
    return out


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    raw = {k: max(0.0, float(weights.get(k, 0.0))) for k in COMPONENT_KEYS}
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / len(COMPONENT_KEYS) for k in COMPONENT_KEYS}
    return {k: v / total for k, v in raw.items()}


def score_listings(
    listings: Iterable[Mapping],
    flag_results: Iterable[Mapping],
    *,
    budget_usd: float | None = None,
) -> list[ScoredListing]:
    """Compute component scores for each listing+flag_result pair."""
    listings = list(listings)
    flag_results = list(flag_results)
    if len(listings) != len(flag_results):
        raise ValueError("listings and flag_results must have the same length")

    crime_values = [_safe_float(l.get("crime_total_month_zip_log1p_latest")) for l in listings]
    safety_scores = _safety_scores(crime_values)

    actuals = [_safe_float(l.get("actual_rent_usd")) or 0.0 for l in listings]
    finite_actuals = [a for a in actuals if a > 0]
    batch_min = min(finite_actuals) if finite_actuals else 0.0

    out: list[ScoredListing] = []
    for listing, flag, safety, actual in zip(listings, flag_results, safety_scores, actuals):
        predicted = _safe_float(flag.get("predicted_rent_usd")) or 0.0
        p90 = _safe_float(flag.get("fair_rent_p90"))

        components = {
            "price_fairness": _price_fairness(actual, predicted, p90),
            "safety": float(safety),
            "walk": _normalize_walk_transit(_safe_float(listing.get("walk_score"))),
            "transit": _normalize_walk_transit(_safe_float(listing.get("transit_score"))),
            "affordability": _affordability(actual, budget_usd, batch_min),
        }
        out.append(ScoredListing(listing=dict(listing), flag_result=dict(flag), component_scores=components))
    return out


def composite(scored: ScoredListing, normalized_weights: Mapping[str, float]) -> float:
    return float(
        sum(
            normalized_weights[k] * scored.component_scores[k]
            for k in COMPONENT_KEYS
        )
    )
