"""
polygon.py
==========
Load the Boulder city boundary from a TIGER/Line shapefile,
reproject to UTM Zone 13N (metres), and simplify to a manageable
number of vertices for the Schwarz–Christoffel solver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_boulder_polygon(
    shapefile_path: str | Path,
    name_field: str = "NAME",
    name_value: str = "Boulder",
) -> Polygon:
    """Load the Boulder city boundary from a TIGER/Line *Place* shapefile.

    Parameters
    ----------
    shapefile_path : path to the ``.shp`` file (or directory containing it).
    name_field : column that holds the place name (default ``NAME``).
    name_value : value to match (default ``Boulder``).

    Returns
    -------
    shapely.geometry.Polygon
        Boulder boundary reprojected to **EPSG:26913** (UTM 13N, metres).
    """
    path = Path(shapefile_path)
    if path.is_dir():
        shps = list(path.glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp file found in {path}")
        path = shps[0]

    logger.info("Loading shapefile: %s", path)
    gdf = gpd.read_file(path)
    matches = gdf[gdf[name_field] == name_value]
    if matches.empty:
        raise ValueError(
            f"No feature with {name_field}='{name_value}' found in {path}.\n"
            f"Available names (first 20): {gdf[name_field].unique()[:20].tolist()}"
        )

    # Reproject to UTM Zone 13N so distances are in metres
    gdf_utm = matches.to_crs(epsg=26913)
    geom = gdf_utm.geometry.values[0]

    # If MultiPolygon, take the largest component
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    logger.info("Loaded Boulder polygon: %d vertices", len(geom.exterior.coords))
    return geom


def simplify_polygon(
    polygon: Polygon,
    tolerance: float = 500.0,
    min_vertices: int = 8,
    max_vertices: int = 20,
) -> Polygon:
    """Simplify *polygon* to a manageable vertex count for SC mapping.

    Uses the Douglas–Peucker algorithm via Shapely.  If the result has
    fewer than *min_vertices* or more than *max_vertices*, the tolerance
    is adjusted automatically.

    Returns
    -------
    shapely.geometry.Polygon
        Simplified polygon (same CRS as input).
    """
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    n = len(simplified.exterior.coords) - 1  # drop closing duplicate

    # Adaptive tolerance adjustment
    lo, hi = tolerance * 0.1, tolerance * 10.0
    for _ in range(30):
        if min_vertices <= n <= max_vertices:
            break
        tol = (lo + hi) / 2.0
        simplified = polygon.simplify(tol, preserve_topology=True)
        n = len(simplified.exterior.coords) - 1
        if n > max_vertices:
            lo = tol
        elif n < min_vertices:
            hi = tol

    logger.info("Simplified polygon: %d vertices (tolerance=%.1f)", n, tolerance)
    return simplified


def polygon_to_complex(polygon: Polygon, normalise: bool = True) -> np.ndarray:
    """Convert a Shapely polygon to an array of complex coordinates.

    Parameters
    ----------
    polygon : Shapely polygon.
    normalise : if True, centre at the origin and scale so that the
        farthest vertex has modulus 1.

    Returns
    -------
    np.ndarray of complex128, shape ``(n,)`` — the vertices in order,
    **without** the closing duplicate.
    """
    coords = np.array(polygon.exterior.coords)[:-1]  # drop closing repeat
    z = coords[:, 0] + 1j * coords[:, 1]

    if normalise:
        z -= z.mean()
        z /= np.abs(z).max()

    return z


def ensure_ccw(z_poly: np.ndarray) -> np.ndarray:
    """Ensure the polygon vertices are ordered counter-clockwise.

    The signed area (shoelace formula) is used: positive ⇒ CCW.
    """
    x, y = z_poly.real, z_poly.imag
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:
        logger.info("Reversing vertex order to CCW")
        z_poly = z_poly[::-1]
    return z_poly


# ---------------------------------------------------------------------------
# Target side-length ratios (used by SC parameter solver)
# ---------------------------------------------------------------------------

def side_length_ratios(z_poly: np.ndarray) -> np.ndarray:
    """Return the ratio of each side length to the last side length.

    Parameters
    ----------
    z_poly : complex vertices, shape ``(n,)``.

    Returns
    -------
    np.ndarray, shape ``(n-1,)`` — ``|sideₖ| / |sideₙ₋₁|`` for k = 0 … n−2.
    """
    sides = np.abs(np.diff(np.append(z_poly, z_poly[0])))
    ratios = sides[:-1] / sides[-1]
    return ratios
