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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

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


# ── Public helpers ────────────────────────────────────────────────────────


def get_vertex_elevations(
    polygon_utm: Polygon,
    epsg_source: int = 26913,
) -> np.ndarray:
    """Return elevation (m) at each vertex. NaN where unavailable."""
    coords = np.array(polygon_utm.exterior.coords)[:-1]
    n = len(coords)
    elevations = np.full(n, np.nan)

    transformer = _get_transformer(epsg_source)

    logger.info("Querying USGS 3DEP for %d vertex elevations …", n)
    n_ok = 0
    for i, (x_utm, y_utm) in enumerate(coords):
        lon, lat = transformer.transform(x_utm, y_utm)
        val = _query_epqs(lon, lat)
        if val is not None:
            elevations[i] = val
            n_ok += 1
            logger.info("  v%d  (%.5f, %.5f) → %.1f m", i, lon, lat, val)
        else:
            logger.warning("  v%d  (%.5f, %.5f) → NO DATA", i, lon, lat)

    logger.info("Elevation API: %d / %d vertices OK", n_ok, n)

    # Fall back to linear model if too few successes
    if n_ok < 3:
        logger.warning("Too few API results — using fallback elevation model")
        elevations = _boulder_fallback(coords)

    return elevations


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
    epsg_source: int = 26913,
) -> TerrainInfo:
    """Full pipeline: query elevation → gradient → source/sink parameters.

    Parameters
    ----------
    polygon_utm : simplified polygon in UTM coordinates.
    sc_params   : solved SC parameters (for pre-vertex locations).
    delta       : imaginary lift for source / sink above ℝ.
    Q_scale     : source strength as fraction of free-stream.
    epsg_source : CRS of the UTM polygon.
    """
    # 1. Elevations
    elevations = get_vertex_elevations(polygon_utm, epsg_source)

    coords = np.array(polygon_utm.exterior.coords)[:-1]

    # Fill any remaining NaN with fallback
    if np.any(np.isnan(elevations)):
        fb = _boulder_fallback(coords)
        mask = np.isnan(elevations)
        elevations[mask] = fb[mask]

    # 2. Linear-plane fit → terrain gradient
    x, y = coords[:, 0], coords[:, 1]
    A = np.column_stack([np.ones_like(x), x, y])
    coeffs, *_ = np.linalg.lstsq(A, elevations, rcond=None)
    grad_x, grad_y = coeffs[1], coeffs[2]   # uphill gradient (m / m)

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
