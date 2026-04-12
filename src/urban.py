"""
urban.py
========
Acquire and process the built-up / urban-core polygon for Boulder, CO.

The polygon is used as an interior obstacle in the doubly-connected flow
model: fluid cannot penetrate the urban core, so streamlines deflect around
it.

Data source: OpenStreetMap via osmnx (landuse and building polygons).
Fallback   : hard-coded approximate downtown Boulder polygon in UTM 13N.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# ── Color constants exported for visualization ────────────────────────────
URBAN_BOUNDARY_COLOR = "#5C2D91"   # deep purple
URBAN_FILL_COLOR     = "#EDE7F6"   # lavender tint

# ── Fallback: approximate downtown Boulder in UTM Zone 13N ────────────────
# Pearl Street Mall area / urban core, roughly 2.5 km × 2 km
_DOWNTOWN_UTM = np.array([
    [474800, 4428500],
    [476600, 4428100],
    [477900, 4428600],
    [478200, 4430000],
    [477800, 4431100],
    [476000, 4431400],
    [474500, 4430800],
    [474400, 4429500],
], dtype=float)


def _fallback_urban_polygon() -> Polygon:
    """Hard-coded approximate downtown Boulder polygon in UTM 13N."""
    return Polygon(_DOWNTOWN_UTM)


def get_urban_polygon(
    boulder_polygon_utm: Polygon,
    method: str = "osmnx",
    n_vertices: int = 8,
    margin_fraction: float = 0.04,
    epsg_utm: int = 26913,
) -> Optional[Polygon]:
    """Return a simplified urban-core polygon in UTM coordinates.

    Parameters
    ----------
    boulder_polygon_utm : outer Boulder boundary in UTM (EPSG:26913).
    method              : "osmnx" (download from OSM) or "fallback".
    n_vertices          : target vertex count after simplification (6-10).
    margin_fraction     : shrink from outer boundary by this fraction of
                          the outer polygon's linear scale, to keep inner
                          polygon strictly inside.
    epsg_utm            : EPSG code of the UTM projection.

    Returns
    -------
    Polygon in UTM coordinates, or None on failure.
    """
    poly: Optional[Polygon] = None

    if method == "osmnx":
        poly = _get_from_osmnx(boulder_polygon_utm, epsg_utm)

    if poly is None:
        logger.warning("Using fallback downtown Boulder polygon")
        poly = _fallback_urban_polygon()

    if poly is None or poly.is_empty:
        logger.error("Failed to obtain any urban polygon")
        return None

    # Clip to Boulder boundary
    poly = _clip_inside(poly, boulder_polygon_utm, margin_fraction)
    if poly is None or poly.is_empty:
        logger.error("Urban polygon is empty after clipping to Boulder boundary")
        return None

    # Simplify to target vertex count
    poly = _simplify_to_n(poly, n_vertices)

    n_v = len(poly.exterior.coords) - 1
    logger.info(
        "Urban polygon ready: %d vertices, area %.2f km²",
        n_v, poly.area / 1e6,
    )
    return poly


# ── OSM download ──────────────────────────────────────────────────────────

def _get_from_osmnx(
    boulder_polygon_utm: Polygon,
    epsg_utm: int = 26913,
) -> Optional[Polygon]:
    """Download urban landuse polygons from OSM and return their union."""
    try:
        import osmnx as ox
        from pyproj import Transformer

        # Convert outer polygon to WGS84 for osmnx
        to_wgs84 = Transformer.from_crs(
            f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True
        )
        coords_wgs84 = np.array([
            to_wgs84.transform(x, y)
            for x, y in boulder_polygon_utm.exterior.coords
        ])
        poly_wgs84 = Polygon(coords_wgs84)

        # Use only the high-density commercial/retail core - NOT residential,
        # which spans most of the city and produces an overly large polygon.
        logger.info("Querying OSM for commercial/retail core polygons …")
        tags = {"landuse": ["commercial", "retail", "institutional"]}
        gdf = ox.features_from_polygon(poly_wgs84, tags=tags)

        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(polys) == 0:
            logger.warning("OSM returned no commercial/retail polygons - trying broader tags")
            tags2 = {"landuse": ["commercial", "retail", "industrial", "institutional"]}
            gdf2 = ox.features_from_polygon(poly_wgs84, tags=tags2)
            polys = gdf2[gdf2.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if len(polys) == 0:
                return None

        logger.info("OSM commercial/retail: %d polygons found", len(polys))
        polys = polys.set_crs("EPSG:4326").to_crs(f"EPSG:{epsg_utm}")

        # Union, then take the largest contiguous polygon (not convex hull)
        union = unary_union(polys.geometry.values)
        if isinstance(union, MultiPolygon):
            # Pick the largest piece
            union = max(union.geoms, key=lambda g: g.area)

        return union if isinstance(union, Polygon) else None

    except ImportError:
        logger.warning("osmnx not installed - falling back to hardcoded polygon")
        return None
    except Exception as exc:
        logger.warning("OSM download failed (%s) - falling back", exc)
        return None


# ── Geometry helpers ──────────────────────────────────────────────────────

def _clip_inside(
    inner: Polygon,
    outer: Polygon,
    margin_fraction: float = 0.04,
) -> Optional[Polygon]:
    """Clip inner polygon to outer and shrink by a margin."""
    # Buffer outer inward to create a safety margin
    margin = outer.length * margin_fraction * 0.5
    outer_shrunk = outer.buffer(-margin)

    clipped = inner.intersection(outer_shrunk)
    if clipped.is_empty:
        # Try without shrinkage
        clipped = inner.intersection(outer)

    if clipped.is_empty:
        return None
    if isinstance(clipped, MultiPolygon):
        clipped = max(clipped.geoms, key=lambda g: g.area)
    return clipped if isinstance(clipped, Polygon) else None


def _simplify_to_n(poly: Polygon, n_target: int) -> Polygon:
    """Douglas-Peucker simplification via binary search to hit n_target vertices."""
    # Work with convex hull for a clean simple polygon
    hull = poly.convex_hull
    if not isinstance(hull, Polygon):
        hull = poly

    lo, hi = 0.0, hull.length
    best = hull
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = hull.simplify(mid, preserve_topology=True)
        n = len(s.exterior.coords) - 1
        if n > n_target:
            lo = mid
        elif n < max(4, n_target - 2):
            hi = mid
            best = s
        else:
            best = s
            break

    n_got = len(best.exterior.coords) - 1
    logger.info("Urban polygon simplified: %d vertices (target %d)", n_got, n_target)
    return best


def polygon_to_complex_inner(
    inner_utm: Polygon,
    center: complex,
    scale: float,
) -> np.ndarray:
    """Convert inner polygon to the same normalised complex frame as outer.

    Uses the same center / scale returned by polygon_to_complex() so
    both polygons share the same coordinate system.
    """
    coords = np.array(inner_utm.exterior.coords)[:-1]
    z = coords[:, 0] + 1j * coords[:, 1]
    return (z - center) / scale
