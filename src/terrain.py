"""
terrain.py
==========
Query DEM elevation data and build a terrain-informed complex potential.

Approach (Plan B):
    1.  Sample elevation at each polygon vertex via the USGS 3DEP
        Elevation Point Query Service (EPQS).  Falls back to a simple
        linear model of Boulder's west→east slope if the API is
        unreachable.
    2.  Fit a linear plane   e(x,y) = a + b·x + c·y   to the vertex
        elevations → mean terrain gradient  ∇e = (b, c).
    3.  Identify the highest- and lowest-elevation vertices and place a
        source–sink pair (with mirror images) in the upper half-plane ℍ
        near the corresponding pre-vertices.

Complex potential
-----------------
    W(ζ) = U·ζ  +  (Q / 2π) · [ log(ζ − s₊) + log(ζ − s̄₊)
                                − log(ζ − s₋) − log(ζ − s̄₋) ]

where s₊ = ζₖ[i_high] + δj  (source, high elevation)
      s₋ = ζₖ[i_low]  + δj  (sink,   low elevation)
and s̄ denotes the complex conjugate (image below ℝ).

The image terms ensure ψ = 0 on ℝ, preserving the no-penetration
boundary condition on the polygon.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from .sc_solver import SCParameters

logger = logging.getLogger(__name__)

# ── Data container ────────────────────────────────────────────────────────


@dataclass
class TerrainInfo:
    """Holds everything the terrain-aware potential needs."""
    elevations: np.ndarray            # elevation at each vertex (metres)
    grad_xy: Tuple[float, float]      # (de/dx, de/dy)  uphill gradient
    theta_downhill: float             # angle of steepest descent (rad)
    slope_magnitude: float            # |∇e|  (m / m)
    sources: List[Tuple[complex, float]] = field(default_factory=list)
    # each entry is (ζ_location_in_H, Q_strength)


# ── USGS 3DEP Elevation Point Query Service ──────────────────────────────

_EPQS_URL = "https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&wkid=4326"
_UTM_TO_LONLAT = None  # lazy singleton


def _get_transformer(epsg_source: int = 26913) -> Transformer:
    global _UTM_TO_LONLAT
    if _UTM_TO_LONLAT is None:
        _UTM_TO_LONLAT = Transformer.from_crs(
            f"EPSG:{epsg_source}", "EPSG:4326", always_xy=True
        )
    return _UTM_TO_LONLAT


def _query_epqs(lon: float, lat: float, timeout: float = 12.0) -> Optional[float]:
    """Query a single elevation from USGS EPQS.  Returns None on failure."""
    url = _EPQS_URL.format(lon=lon, lat=lat)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "APPM4360-SC-Flow/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            val = float(data["value"])
            if val < -1000:  # -1e6 sentinel for ocean / no data
                return None
            return val
    except Exception as exc:
        logger.debug("EPQS query failed (%.4f, %.4f): %s", lon, lat, exc)
        return None


# ── Dense sampling helpers ────────────────────────────────────────────────


def _sample_boundary_points(
    polygon_utm: Polygon,
    n_per_edge: int = 3,
) -> np.ndarray:
    """Generate extra sample points along each polygon edge.

    Returns array of shape (M, 2) in UTM coordinates.
    Does NOT include the original vertices (those are queried separately).
    """
    coords = np.array(polygon_utm.exterior.coords)  # includes closing point
    pts = []
    for i in range(len(coords) - 1):
        for t in np.linspace(0, 1, n_per_edge + 2)[1:-1]:  # exclude endpoints
            pts.append(coords[i] * (1 - t) + coords[i + 1] * t)
    return np.array(pts) if pts else np.empty((0, 2))


def _sample_interior_points(
    polygon_utm: Polygon,
    n_pts: int = 25,
) -> np.ndarray:
    """Generate a grid of interior sample points inside the polygon."""
    minx, miny, maxx, maxy = polygon_utm.bounds
    side = int(np.ceil(np.sqrt(n_pts * 1.5)))  # oversample to account for exterior
    xs = np.linspace(minx, maxx, side + 2)[1:-1]
    ys = np.linspace(miny, maxy, side + 2)[1:-1]
    pts = []
    for x in xs:
        for y in ys:
            if polygon_utm.contains(Point(x, y)):
                pts.append([x, y])
            if len(pts) >= n_pts:
                break
        if len(pts) >= n_pts:
            break
    return np.array(pts) if pts else np.empty((0, 2))


def _batch_query_elevations(
    coords_utm: np.ndarray,
    epsg_source: int = 26913,
    max_workers: int = 6,
    timeout: float = 15.0,
) -> np.ndarray:
    """Query elevations for an array of UTM points using concurrent threads.

    Returns array of elevations; NaN where the API fails.
    """
    n = len(coords_utm)
    elevations = np.full(n, np.nan)
    if n == 0:
        return elevations

    transformer = _get_transformer(epsg_source)
    lonlats = np.array([transformer.transform(x, y) for x, y in coords_utm])

    def _query_one(idx: int) -> Tuple[int, Optional[float]]:
        lon, lat = lonlats[idx]
        return idx, _query_epqs(lon, lat, timeout=timeout)

    logger.info("Querying USGS 3DEP for %d points (%d concurrent) …",
                n, max_workers)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_query_one, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc="USGS 3DEP"):
            idx, val = fut.result()
            if val is not None:
                elevations[idx] = val
                n_ok += 1

    logger.info("Elevation API: %d / %d points OK (%.0f%%)",
                n_ok, n, 100 * n_ok / max(n, 1))
    return elevations


def get_vertex_elevations(
    polygon_utm: Polygon,
    epsg_source: int = 26913,
) -> np.ndarray:
    """Return elevation (m) at each vertex. NaN where unavailable."""
    coords = np.array(polygon_utm.exterior.coords)[:-1]
    elevations = _batch_query_elevations(coords, epsg_source)

    # Fall back to linear model if too few successes
    if np.sum(np.isfinite(elevations)) < 3:
        logger.warning("Too few API results — using fallback elevation model")
        elevations = _boulder_fallback(coords)

    return elevations


def get_dense_elevations(
    polygon_utm: Polygon,
    n_per_edge: int = 3,
    n_interior: int = 25,
    epsg_source: int = 26913,
    max_workers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query elevations at vertices + boundary interpolation + interior grid.

    Returns
    -------
    all_coords : (M, 2)  all sample points in UTM
    all_elevs  : (M,)    elevations in metres (NaN filled with fallback)
    """
    vertex_coords = np.array(polygon_utm.exterior.coords)[:-1]
    boundary_coords = _sample_boundary_points(polygon_utm, n_per_edge)
    interior_coords = _sample_interior_points(polygon_utm, n_interior)

    all_coords = np.vstack([
        vertex_coords,
        boundary_coords,
        interior_coords,
    ])
    n_v = len(vertex_coords)
    n_b = len(boundary_coords)
    n_i = len(interior_coords)
    logger.info("Elevation sample plan: %d vertices + %d boundary + %d interior = %d total",
                n_v, n_b, n_i, len(all_coords))

    all_elevs = _batch_query_elevations(all_coords, epsg_source, max_workers)

    # Fill NaN with fallback model
    nan_mask = np.isnan(all_elevs)
    if nan_mask.any():
        fb = _boulder_fallback(all_coords[nan_mask])
        all_elevs[nan_mask] = fb
        logger.info("Filled %d NaN elevations with fallback model", nan_mask.sum())

    return all_coords, all_elevs


def _boulder_fallback(coords_utm: np.ndarray) -> np.ndarray:
    """Linear model of Boulder topography (west ≈ 1770 m, east ≈ 1570 m).

    Based on USGS 1/3 arc-second DEM statistics for Boulder.
    """
    x = coords_utm[:, 0]
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min + 1e-6
    elev = 1770.0 - 200.0 * (x - x_min) / span
    logger.info("Fallback elevations: %.0f … %.0f m", elev.min(), elev.max())
    return elev


def compute_terrain_info(
    polygon_utm: Polygon,
    sc_params: SCParameters,
    *,
    delta: float = 0.25,
    Q_scale: float = 0.35,
    n_per_edge: int = 3,
    n_interior: int = 25,
    max_workers: int = 6,
    epsg_source: int = 26913,
) -> TerrainInfo:
    """Full pipeline: dense elevation query → gradient → source/sink.

    Parameters
    ----------
    polygon_utm : simplified polygon in UTM coordinates.
    sc_params   : solved SC parameters (for pre-vertex locations).
    delta       : imaginary lift for source / sink above ℝ.
    Q_scale     : source strength as fraction of free-stream.
    n_per_edge  : extra sample points per polygon edge.
    n_interior  : interior sample points for the plane fit.
    max_workers : concurrent API threads.
    epsg_source : CRS of the UTM polygon.
    """
    # 1. Dense elevation sampling
    all_coords, all_elevs = get_dense_elevations(
        polygon_utm, n_per_edge=n_per_edge, n_interior=n_interior,
        epsg_source=epsg_source, max_workers=max_workers,
    )

    # Extract vertex elevations (first n_v entries)
    vertex_coords = np.array(polygon_utm.exterior.coords)[:-1]
    n_v = len(vertex_coords)
    elevations = all_elevs[:n_v]

    # 2. Linear-plane fit using ALL sample points → better gradient
    x, y = all_coords[:, 0], all_coords[:, 1]
    A = np.column_stack([np.ones_like(x), x, y])
    coeffs, *_ = np.linalg.lstsq(A, all_elevs, rcond=None)
    grad_x, grad_y = coeffs[1], coeffs[2]   # uphill gradient (m / m)

    logger.info("Plane fit using %d sample points (R² quality check):", len(all_elevs))
    predicted = A @ coeffs
    ss_res = np.sum((all_elevs - predicted) ** 2)
    ss_tot = np.sum((all_elevs - all_elevs.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)
    logger.info("  R² = %.4f  (1.0 = perfect linear terrain)", r_squared)

    slope_mag = np.hypot(grad_x, grad_y)
    theta_down = np.arctan2(-grad_y, -grad_x)     # downhill direction

    logger.info("Terrain gradient: (%.5f, %.5f) m/m  |∇e|=%.5f",
                grad_x, grad_y, slope_mag)
    logger.info("Downhill direction: %.1f°  (0° = +x / east)",
                np.degrees(theta_down))

    # 3. Source / sink in ℍ
    zk = sc_params.zk  # pre-vertices on ℝ, same indexing as polygon vertices

    i_high = int(np.argmax(elevations))
    i_low  = int(np.argmin(elevations))
    elev_range = elevations.max() - elevations.min()

    sources: List[Tuple[complex, float]] = []
    if i_high != i_low and elev_range > 1.0:
        Q = Q_scale
        s_plus  = zk[i_high] + delta * 1j
        s_minus = zk[i_low]  + delta * 1j
        sources = [(s_plus, +Q), (s_minus, -Q)]
        logger.info(
            "Source at ζ = %.3f + %.3fj  (v%d, elev %.0f m, Q = +%.3f)",
            s_plus.real, s_plus.imag, i_high, elevations[i_high], Q,
        )
        logger.info(
            "Sink   at ζ = %.3f + %.3fj  (v%d, elev %.0f m, Q = −%.3f)",
            s_minus.real, s_minus.imag, i_low, elevations[i_low], Q,
        )
    else:
        logger.warning("Elevation range (%.1f m) too small — no sources added",
                        elev_range)

    return TerrainInfo(
        elevations=elevations,
        grad_xy=(grad_x, grad_y),
        theta_downhill=theta_down,
        slope_magnitude=slope_mag,
        sources=sources,
    )


# ── Potential function ────────────────────────────────────────────────────


def terrain_potential(
    zeta: complex,
    U: float,
    sources: List[Tuple[complex, float]],
) -> complex:
    r"""Evaluate the terrain-informed complex potential at a single ζ.

        W(ζ) = U·ζ  +  Σⱼ  (Qⱼ / 2π) · [ log(ζ − sⱼ) + log(ζ − s̄ⱼ) ]

    Each source sⱼ in ℍ has an image s̄ⱼ below ℝ so that the stream
    function vanishes on the real axis (no-penetration condition).
    """
    W = U * zeta
    for s, Q in sources:
        d1 = zeta - s
        d2 = zeta - np.conj(s)
        # Regularize to avoid log(0)
        if abs(d1) < 1e-12:
            d1 = 1e-12 + 0j
        if abs(d2) < 1e-12:
            d2 = 1e-12 + 0j
        W += (Q / (2.0 * np.pi)) * (np.log(d1) + np.log(d2))
    return W


def uniform_potential(zeta: complex, U: float = 1.0) -> complex:
    """Original uniform potential: W(ζ) = U·ζ."""
    return U * zeta
