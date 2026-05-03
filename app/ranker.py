"""Composite scoring + ranking on top of the rent model's flag_overpriced output.

Each listing receives three component scores in [0, 1]:

    - safety   : 1 − batch-normalized crime intensity (lower crime = higher score).
                 Falls back to 0.5 when the crime feature is missing.
    - walk     : walk_score / 100 (or 0.5 if missing).
    - transit  : transit_score / 100 (or 0.5 if missing).

Price fairness/affordability are no longer part of the composite — budget is now
a strict client-side filter, and the fair-rent band (p25/p75) is surfaced on the
listing card from the model output but does not feed the ranking.

The composite score is the sum of (normalized weight × component), so it is also
in [0, 1] regardless of the absolute weight scale the user provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


COMPONENT_KEYS = ("safety", "walk", "transit")


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
) -> list[ScoredListing]:
    """Compute component scores for each listing+flag_result pair."""
    listings = list(listings)
    flag_results = list(flag_results)
    if len(listings) != len(flag_results):
        raise ValueError("listings and flag_results must have the same length")

    crime_values = [_safe_float(l.get("crime_total_month_zip_log1p_latest")) for l in listings]
    safety_scores = _safety_scores(crime_values)

    out: list[ScoredListing] = []
    for listing, flag, safety in zip(listings, flag_results, safety_scores):
        components = {
            "safety": float(safety),
            "walk": _normalize_walk_transit(_safe_float(listing.get("walk_score"))),
            "transit": _normalize_walk_transit(_safe_float(listing.get("transit_score"))),
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
