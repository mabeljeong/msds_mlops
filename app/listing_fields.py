"""Shared accessors for listing payload fields with mixed naming."""

from __future__ import annotations

from typing import Any, Mapping


def actual_rent_usd_from_listing(listing: Mapping[str, Any]) -> float | None:
    """Return the observed listing rent, accepting either ``actual_rent_usd`` or ``rent_usd``.

    Listings come from two sources (model inputs and demo fixtures) that disagree
    on the field name. This single resolver keeps that detail out of call sites.
    """
    for key in ("actual_rent_usd", "rent_usd"):
        if key in listing and listing[key] is not None:
            try:
                return float(listing[key])
            except (TypeError, ValueError):
                return None
    return None
