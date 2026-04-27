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


def _price_fairness_with_band(actual: float, p10: float, p90: float) -> float:
    """Continuous score in [0, 1] using a calibrated p10/p90 band.

        actual ≤ p10                   -> 1.00 (steal)
        p10 < actual ≤ p90 (in-band)   -> linear 1.00 → 0.50
        actual > p90                   -> linear 0.50 → 0.00 at 1.5× p90
    """
    if actual <= p10:
        return 1.0
    if actual <= p90:
        return float(1.0 - 0.5 * (actual - p10) / (p90 - p10))
    overage_pct = (actual - p90) / p90
    return float(max(0.0, 0.5 - 0.5 * min(overage_pct / 0.5, 1.0)))


def _price_fairness_batch(
    actuals: list[float],
    predicteds: list[float],
    p10s: list[float | None],
    p90s: list[float | None],
) -> list[float]:
    """Score every listing's price fairness, mixing band-based and rank-based fallback.

    When a listing has a calibrated p10/p90 band we use it directly. When the band is
    missing (placeholder model), we fall back to batch-rank normalization on
    ``delta_pct = (actual - predicted) / predicted``: the listing with the lowest
    delta in the batch gets 1.0, the highest gets 0.0, and ties share scores. This
    keeps the score informative even when absolute predictions are biased.
    """
    n = len(actuals)
    out: list[float] = [0.5] * n
    fallback_idx: list[int] = []
    fallback_deltas: list[float] = []

    for i in range(n):
        p10, p90 = p10s[i], p90s[i]
        if p10 is not None and p90 is not None and p90 > p10 > 0:
            out[i] = _price_fairness_with_band(actuals[i], p10, p90)
            continue

        predicted = predicteds[i]
        if predicted > 0:
            fallback_deltas.append((actuals[i] - predicted) / predicted)
            fallback_idx.append(i)

    if not fallback_idx:
        return out

    if len(fallback_idx) == 1:
        # No batch to compare against; use a smooth single-listing curve.
        d = fallback_deltas[0]
        if d <= -0.10:
            score = 1.0
        elif d <= 0.0:
            score = 1.0 - 0.15 * (d + 0.10) / 0.10
        else:
            score = max(0.0, 0.85 - 0.85 * min(d / 0.30, 1.0))
        out[fallback_idx[0]] = float(score)
        return out

    # Batch-rank: lowest delta_pct (cheapest vs prediction) → 1.0
    order = np.argsort(fallback_deltas)
    ranks = np.empty(len(fallback_deltas), dtype=float)
    ranks[order] = np.arange(len(fallback_deltas))
    denom = max(1, len(fallback_deltas) - 1)
    for k, idx in enumerate(fallback_idx):
        out[idx] = float(1.0 - ranks[k] / denom)

    return out


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

    predicteds = [_safe_float(f.get("predicted_rent_usd")) or 0.0 for f in flag_results]
    p10s = [_safe_float(f.get("fair_rent_p10")) for f in flag_results]
    p90s = [_safe_float(f.get("fair_rent_p90")) for f in flag_results]
    fairness_scores = _price_fairness_batch(actuals, predicteds, p10s, p90s)

    out: list[ScoredListing] = []
    for i, (listing, flag, safety, actual) in enumerate(zip(listings, flag_results, safety_scores, actuals)):
        components = {
            "price_fairness": fairness_scores[i],
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
