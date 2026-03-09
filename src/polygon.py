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


def load_boulder_polygon(
    shapefile_path: str | Path,
    name_field: str = "NAME",
    name_value: str = "Boulder",
) -> Polygon:
    """Load Boulder city boundary and reproject to UTM 13N (metres)."""
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
            f"No feature with {name_field}='{name_value}' found.\n"
            f"Available (first 20): {gdf[name_field].unique()[:20].tolist()}"
        )

    gdf_utm = matches.to_crs(epsg=26913)
    geom = gdf_utm.geometry.values[0]

    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    logger.info("Loaded Boulder polygon: %d vertices",
                len(geom.exterior.coords) - 1)
    return geom


def simplify_polygon(
    polygon: Polygon,
    tolerance: float = 150.0,
    min_vertices: int = 10,
    max_vertices: int = 20,
) -> Polygon:
    """Simplify polygon via Douglas–Peucker with adaptive tolerance."""
    lo, hi = 1.0, tolerance * 20.0
    tol = tolerance
    simplified = polygon.simplify(tol, preserve_topology=True)
    n = len(simplified.exterior.coords) - 1

    for _ in range(50):
        if min_vertices <= n <= max_vertices:
            break
        if n > max_vertices:
            lo = tol
        else:
            hi = tol
        tol = (lo + hi) / 2.0
        simplified = polygon.simplify(tol, preserve_topology=True)
        n = len(simplified.exterior.coords) - 1

    logger.info("Simplified polygon: %d vertices (tol=%.1f m)", n, tol)
    return simplified


def polygon_to_complex(
    polygon: Polygon,
    normalise: bool = True,
) -> Tuple[np.ndarray, complex, float]:
    """Convert Shapely polygon → complex array.

    Returns (z, center, scale) where z = (raw - center) / scale.
    """
    coords = np.array(polygon.exterior.coords)[:-1]
    z = coords[:, 0] + 1j * coords[:, 1]

    center = 0.0 + 0.0j
    scale = 1.0
    if normalise:
        center = z.mean()
        z -= center
        scale = np.abs(z).max()
        z /= scale

    return z, center, scale


def complex_to_polygon(z_poly: np.ndarray) -> Polygon:
    """Create a Shapely Polygon from complex vertices."""
    coords = [(z.real, z.imag) for z in z_poly]
    coords.append(coords[0])
    return Polygon(coords)


def ensure_ccw(z_poly: np.ndarray) -> np.ndarray:
    """Ensure vertices are counter-clockwise (positive signed area)."""
    x, y = z_poly.real, z_poly.imag
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:
        logger.info("Reversing vertex order to CCW")
        z_poly = z_poly[::-1]
    return z_poly
