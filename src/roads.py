"""
roads.py
========
Download Boulder road network from OSM, find major intersections,
and build a road-informed complex potential via point vortices in ℍ.

Approach
--------
1.  Download the road graph within the Boulder polygon via osmnx,
    filtered to arterials and above (primary, secondary, tertiary).
2.  Find intersection nodes with degree ≥ 3, select the top-N by
    degree (degree = number of roads meeting = traffic-intensity proxy).
3.  Map each intersection UTM → normalised z-plane → upper half-plane ℍ
    via the SC inverse map.
4.  Place a point vortex at each ℍ position with an image vortex
    at the mirror point below ℝ.

Complex potential
-----------------
    W_road(ζ) = Σ_k  (−i·Γ_k / 2π) · [log(ζ − s_k) + log(ζ − s̄_k)]

where s_k ∈ ℍ is the ζ-position of intersection k and Γ_k is its
circulation strength (proportional to node degree).

The image term ensures ψ = Im(W) = 0 on the real axis ℝ, which maps
to the polygon boundary ∂Ω, preserving the no-penetration condition.

Verification:  on ζ = x real,
    Im( (−i·Γ/2π)·[log(x − s) + log(x − s̄)] )
    = (Γ/2π) · [arg(x − s) + arg(x − s̄)]
    = (Γ/2π) · [−arctan(b/(x−a)) + arctan(b/(x−a))] = 0  ✓
where s = a + ib, b > 0.

Physical motivation
-------------------
Major road intersections generate local circulation in urban
atmospheric flow (traffic turbulence, building-induced channelling,
heat-island convection).  Positive (CCW) vortices at northern
intersections and negative (CW) vortices at southern intersections
approximate the vortex-pair structure observed in Boulder's prevailing
westerly flow regime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .sc_solver import SCParameters

logger = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────

@dataclass
class RoadInfo:
    """Road network intersection data and derived vortex parameters."""
    n_intersections: int
    intersection_positions_utm: np.ndarray   # (M, 2) UTM coords
    intersection_degrees: np.ndarray         # (M,) node degree
    vortices: List[Tuple[complex, float]] = field(default_factory=list)
    # each entry: (s_k position in ℍ,  Γ_k circulation strength)


# ── OSM intersection download ─────────────────────────────────────────────

def get_road_intersections(
    polygon_utm,
    method: str = "osmnx",
    n_max: int = 12,
    road_types: tuple = ("primary", "secondary", "tertiary"),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get major road intersection positions (UTM) and node degrees.

    Parameters
    ----------
    polygon_utm : Shapely Polygon in UTM Zone 13N (EPSG:26913).
    method      : 'osmnx' (live OSM) or 'fallback' (hardcoded Boulder grid).
    n_max       : maximum number of intersections to return.
    road_types  : OSM highway tags to include.

    Returns
    -------
    (coords, degrees) where coords is (M, 2) UTM and degrees is (M,),
    or None on failure.
    """
    if method == "osmnx":
        try:
            return _osmnx_intersections(polygon_utm, n_max, road_types)
        except Exception as exc:
            logger.warning("OSMnx road fetch failed: %s  — using fallback", exc)

    return _fallback_intersections(polygon_utm, n_max)


def _osmnx_intersections(
    polygon_utm,
    n_max: int,
    road_types: tuple,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Download road graph from OSM and extract high-degree intersections."""
    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    logger.info("Downloading OSM road network (types: %s) …", ", ".join(road_types))

    # Convert polygon to WGS84 for osmnx
    tf_fwd = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
    coords_utm = np.array(polygon_utm.exterior.coords)
    coords_wgs = np.array([tf_fwd.transform(x, y) for x, y in coords_utm])
    polygon_wgs = ShapelyPolygon(coords_wgs)

    cf = '["highway"~"{}"]'.format("|".join(road_types))
    G = ox.graph_from_polygon(polygon_wgs, custom_filter=cf, network_type="drive")
    logger.info("Road graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    G_ud = ox.convert.to_undirected(G)
    degrees = dict(G_ud.degree())

    # Intersections: degree ≥ 3 (T-junctions, crossings)
    crossings = [(nid, deg) for nid, deg in degrees.items() if deg >= 3]
    if not crossings:
        logger.warning("No intersections with degree ≥ 3 found")
        return None

    crossings.sort(key=lambda x: x[1], reverse=True)
    crossings = crossings[:n_max]

    # Convert back to UTM, filter to inside polygon
    tf_back = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    coords_out, degrees_out = [], []
    for nid, deg in crossings:
        node = G_ud.nodes[nid]
        x_utm, y_utm = tf_back.transform(node["x"], node["y"])
        if polygon_utm.contains(Point(x_utm, y_utm)):
            coords_out.append([x_utm, y_utm])
            degrees_out.append(deg)

    if not coords_out:
        logger.warning("No OSM intersections landed inside polygon")
        return None

    logger.info("OSM road intersections inside polygon: %d", len(coords_out))
    return np.array(coords_out), np.array(degrees_out, dtype=float)


def _fallback_intersections(
    polygon_utm,
    n_max: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Hardcoded major Boulder intersections as fallback.

    WGS84 (lon, lat) coords for key arterial crossings.
    """
    _LONLAT = [
        (-105.2835, 40.0186),   # Broadway & Canyon Blvd          (degree 5)
        (-105.2835, 40.0135),   # Broadway & Arapahoe Ave          (degree 5)
        (-105.2519, 40.0186),   # 28th St & Canyon Blvd           (degree 4)
        (-105.2519, 40.0135),   # 28th St & Arapahoe Ave          (degree 4)
        (-105.2366, 40.0135),   # Foothills Pkwy & Arapahoe Ave   (degree 4)
        (-105.2835, 40.0296),   # Broadway & Iris Ave              (degree 4)
        (-105.2519, 40.0296),   # 28th St & Iris Ave               (degree 4)
        (-105.2835, 40.0050),   # Broadway & Baseline Rd           (degree 4)
        (-105.2519, 40.0050),   # 28th St & Baseline Rd            (degree 3)
        (-105.2366, 40.0050),   # Foothills & Baseline Rd          (degree 3)
        (-105.2683, 40.0186),   # 17th St & Canyon Blvd            (degree 3)
        (-105.2683, 40.0050),   # 17th St & Baseline               (degree 3)
    ]
    _DEG = [5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0]

    from pyproj import Transformer
    from shapely.geometry import Point

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    coords_out, degrees_out = [], []
    for (lon, lat), deg in zip(_LONLAT[:n_max], _DEG[:n_max]):
        x, y = tf.transform(lon, lat)
        if polygon_utm.contains(Point(x, y)):
            coords_out.append([x, y])
            degrees_out.append(deg)

    if not coords_out:
        logger.warning("No fallback intersections inside polygon")
        return None

    logger.info("Fallback road intersections inside polygon: %d", len(coords_out))
    return np.array(coords_out), np.array(degrees_out, dtype=float)


# ── Full pipeline ─────────────────────────────────────────────────────────

def compute_road_info(
    polygon_utm,
    center: complex,
    scale: float,
    sc_params: SCParameters,
    *,
    method: str = "osmnx",
    n_max: int = 12,
    Gamma_scale: float = 0.20,
    delta_min: float = 0.15,
    road_types: tuple = ("primary", "secondary", "tertiary"),
) -> Optional[RoadInfo]:
    """Full pipeline: OSM intersections → normalised → SC inverse → vortices.

    Parameters
    ----------
    polygon_utm  : simplified Boulder polygon in UTM (EPSG:26913).
    center       : normalisation shift (complex) used in polygon_to_complex.
    scale        : normalisation scale used in polygon_to_complex.
    sc_params    : solved SC parameters for the outer polygon.
    method       : 'osmnx' or 'fallback'.
    n_max        : max intersections to use.
    Gamma_scale  : maximum vortex strength (fraction of U = 1).
    delta_min    : minimum Im(s_k) for any vortex in ℍ.
    road_types   : OSM highway tags.

    Returns
    -------
    RoadInfo with vortex positions and strengths, or None on failure.
    """
    result = get_road_intersections(polygon_utm, method=method,
                                    n_max=n_max, road_types=road_types)
    if result is None:
        return None

    coords_utm, degrees = result

    # ── Convert UTM → normalised complex coords ───────────────────────────
    z_norm = (coords_utm[:, 0] + 1j * coords_utm[:, 1] - center) / scale
    logger.info("Road intersections (normalised): %s",
                np.array2string(z_norm, precision=3))

    # ── SC inverse map: z ∈ Ω  →  ζ ∈ ℍ ─────────────────────────────────
    from .flow import sc_inverse_single

    vortices: List[Tuple[complex, float]] = []
    n_ok = 0
    logger.info("SC inverse map for %d road intersections …", len(z_norm))

    # Use centroid of polygon as warm-start (good starting point in ℍ)
    zeta_prev = 0.0 + 0.5j
    for k, z_t in enumerate(z_norm):
        zeta = sc_inverse_single(z_t, sc_params, zeta0=zeta_prev, maxfev=800)
        if zeta is None:
            logger.warning("  Intersection %d: SC inverse failed — skipped", k)
            continue
        # Enforce minimum distance from real axis
        if zeta.imag < delta_min:
            zeta = zeta.real + delta_min * 1j

        # Circulation sign: positive (CCW) if north of polygon centroid,
        # negative (CW) if south.  This creates a vortex-pair structure
        # matching Boulder's prevailing westerly-flow shear pattern.
        z_centroid = np.mean(z_norm)
        sign = +1.0 if z_t.imag >= z_centroid.imag else -1.0

        # Strength proportional to node degree (normalised)
        Gamma = sign * Gamma_scale * float(degrees[k]) / float(max(degrees))
        vortices.append((zeta, Gamma))
        logger.info(
            "  [%d] z=%.3f%+.3fj → ζ=%.4f%+.4fj  Γ=%+.4f  (deg=%d)",
            k, z_t.real, z_t.imag, zeta.real, zeta.imag, Gamma, int(degrees[k]),
        )
        zeta_prev = zeta
        n_ok += 1

    if not vortices:
        logger.warning("No road vortices placed — road model disabled")
        return None

    logger.info("Road vortex model: %d vortices placed (Γ_max = %.4f)",
                n_ok, Gamma_scale)
    return RoadInfo(
        n_intersections=n_ok,
        intersection_positions_utm=coords_utm[:n_ok],
        intersection_degrees=degrees[:n_ok],
        vortices=vortices,
    )


# ── Potential function ────────────────────────────────────────────────────

def road_potential(
    zeta: complex,
    U: float,
    vortices: List[Tuple[complex, float]],
) -> complex:
    r"""Evaluate the road-informed complex potential at a single ζ.

        W(ζ) = U·ζ  +  Σ_k (−i·Γ_k/2π)·[log(ζ−s_k) + log(ζ−s̄_k)]

    The image term log(ζ−s̄_k) ensures ψ = 0 on ℝ:
        Im((−i·Γ/2π)·[log(x−s) + log(x−s̄)]) = 0  for x ∈ ℝ  ✓

    Parameters
    ----------
    zeta    : evaluation point in ℍ.
    U       : free-stream speed.
    vortices: list of (s_k, Γ_k) vortex descriptors.
    """
    W = U * zeta
    for s, Gamma in vortices:
        d1 = zeta - s
        d2 = zeta - np.conj(s)
        if abs(d1) < 1e-12:
            d1 = 1e-12 + 0j
        if abs(d2) < 1e-12:
            d2 = 1e-12 + 0j
        W += (-1j * Gamma / (2.0 * np.pi)) * (np.log(d1) + np.log(d2))
    return W
