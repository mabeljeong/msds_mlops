"""Geospatial helpers: load SF ZIP polygons + point-in-polygon ZIP assignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import standardize_zip


def _load_zip_polygons(path: Path | str) -> dict[str, object]:
    """
    Load a deck.gl-format SF ZIP polygons JSON into ``{zip_code: shapely.Polygon}``.

    Input format (``data/raw/sf_zip_polygons.json``)::

        [
            {"zipcode": 94110, "population": ..., "area": ...,
             "contour": [[lon, lat], [lon, lat], ...]},
            ...
        ]
    """
    from shapely.geometry import Polygon  # local: keep package import light

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for entry in raw:
        zip_series = standardize_zip(pd.Series([entry.get("zipcode")]))
        zip_str = zip_series.iloc[0]
        if pd.isna(zip_str):
            continue
        contour = entry.get("contour") or []
        if len(contour) < 4:
            continue
        try:
            poly = Polygon([(float(lon), float(lat)) for lon, lat in contour])
        except (TypeError, ValueError):
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        out[str(zip_str)] = poly
    return out


def _assign_zip_by_point(
    latitude: pd.Series,
    longitude: pd.Series,
    polygons: dict[str, object],
) -> pd.Series:
    """
    Vectorized point-in-polygon ZIP assignment using shapely's STRtree.

    ``latitude`` / ``longitude`` are float Series (NaN allowed). Returns a
    string Series ``zip_code`` aligned with the input index, ``NA`` when no
    polygon contains the point or coordinates are missing.
    """
    import shapely
    from shapely.strtree import STRtree

    lat = pd.to_numeric(latitude, errors="coerce").to_numpy()
    lon = pd.to_numeric(longitude, errors="coerce").to_numpy()
    out = pd.array([pd.NA] * len(lat), dtype="string")

    if not polygons:
        return pd.Series(out, index=latitude.index, name="zip_code")

    zip_codes = list(polygons.keys())
    polys = list(polygons.values())
    tree = STRtree(polys)

    valid_idx = np.where(~(np.isnan(lat) | np.isnan(lon)))[0]
    if len(valid_idx) == 0:
        return pd.Series(out, index=latitude.index, name="zip_code")

    points = shapely.points(lon[valid_idx], lat[valid_idx])
    point_idxs, poly_idxs = tree.query(points, predicate="within")

    # One SF point should match at most one ZIP polygon. If polygon edges ever
    # produce duplicates, keeping the first match is deterministic.
    matched_points: set[int] = set()
    for point_idx, poly_idx in zip(point_idxs, poly_idxs):
        p = int(point_idx)
        if p in matched_points:
            continue
        out[int(valid_idx[p])] = zip_codes[int(poly_idx)]
        matched_points.add(p)

    return pd.Series(out, index=latitude.index, name="zip_code")
