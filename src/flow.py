"""
flow.py
=======
Compute ψ (stream function) and φ (velocity potential) on a grid
covering the polygon **in normalised coordinates** by numerically
inverting the SC map.

Default complex potential in ℍ:  W(ζ) = U·ζ
Terrain-informed potential:      W(ζ) = U·ζ + source/sink terms

For each grid point z ∈ Ω (normalised coords) we solve f(ζ) = z for ζ ∈ ℍ,
then evaluate W(ζ) to obtain φ = Re(W) and ψ = Im(W).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from tqdm import tqdm

from .sc_solver import SCParameters, sc_map, sc_map_single
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sc_solver_dc import UrbanObstacle

logger = logging.getLogger(__name__)

# Type alias for a potential function:  ζ → W(ζ)
PotentialFn = Callable[[complex], complex]


# ── Parametric streamline / equipotential curves (forward-map, no inverse) ──

def compute_curves_forward(
    params: SCParameters,
    norm_polygon: Polygon,
    n_stream: int = 25,
    n_equip: int = 25,
    n_pts_per_curve: int = 300,
    potential_fn: Optional[PotentialFn] = None,
    U: float = 1.0,
) -> Tuple[list, list]:
    """Compute streamline and equipotential curves via the forward SC map.

    For uniform flow W(ζ) = Uζ:
      - streamlines   are Im(ζ) = const  (horizontal lines in ℍ)
      - equipotentials are Re(ζ) = const  (vertical half-lines in ℍ)

    Each curve is mapped forward z = f(ζ) and clipped to *norm_polygon*.

    Returns
    -------
    stream_curves : list of (x_arr, y_arr) arrays - one per streamline
    equip_curves  : list of (x_arr, y_arr) arrays - one per equipotential
    """
    from shapely.geometry import LineString

    if potential_fn is None:
        from .terrain import uniform_potential
        potential_fn = lambda zeta: uniform_potential(zeta, U)

    prep_poly = prep(norm_polygon)
    # Wide t range: the "infinite" side of the polygon maps to large |t|.
    # Use denser sampling near the origin and sparser at the extremes.
    t_inner = np.linspace(-4.0, 4.0, n_pts_per_curve * 3 // 4)
    t_outer = np.concatenate([np.linspace(-20.0, -4.0, n_pts_per_curve // 8),
                               np.linspace(4.0,  20.0, n_pts_per_curve // 8)])
    t_vals = np.unique(np.concatenate([t_outer, t_inner]))

    # ── Streamlines: Im(ζ) = y₀ for uniform flow ─────────────────────────
    # y₀ range: small y → near boundary; large y → interior / "far" regions.
    # Use log-spacing so we capture near-boundary detail well.
    # Combine near-boundary and far-field levels.
    # Don't go below y=0.04 - very small y gives too-dense boundary lines.
    y_near = np.logspace(-1.4, -0.5, n_stream // 2)   # 0.04 … 0.32
    y_far  = np.logspace(-0.4,  0.8, n_stream - n_stream // 2)  # 0.40 … 6.3
    y_levels = np.unique(np.concatenate([y_near, y_far]))

    stream_curves = []
    logger.info("Computing %d forward-map streamlines …", n_stream)
    for y0 in y_levels:
        zeta_curve = t_vals + 1j * y0
        z_curve = sc_map(zeta_curve, params, n_pts=200)
        x_c, y_c = z_curve.real, z_curve.imag
        # Clip by polygon: keep segments where both endpoints are inside
        xs, ys = [], []
        for k in range(len(x_c) - 1):
            p0 = (x_c[k],   y_c[k])
            p1 = (x_c[k+1], y_c[k+1])
            if prep_poly.contains(Point(p0)) or prep_poly.contains(Point(p1)):
                if not xs:
                    xs.append(x_c[k]); ys.append(y_c[k])
                xs.append(x_c[k+1]); ys.append(y_c[k+1])
            else:
                if xs:
                    stream_curves.append((np.array(xs), np.array(ys)))
                    xs, ys = [], []
        if xs:
            stream_curves.append((np.array(xs), np.array(ys)))

    # ── Equipotentials: Re(ζ) = x₀ for uniform flow ──────────────────────
    # x₀ spans the range of pre-vertices plus some margin.
    zk = params.zk
    x0_levels = np.linspace(zk[0] * 1.3, zk[-1] * 1.3, n_equip)
    y_vals = np.logspace(-1.5, 0.8, n_pts_per_curve)

    equip_curves = []
    logger.info("Computing %d forward-map equipotentials …", n_equip)
    for x0 in x0_levels:
        zeta_curve = x0 + 1j * y_vals
        z_curve = sc_map(zeta_curve, params, n_pts=200)
        x_c, y_c = z_curve.real, z_curve.imag
        xs, ys = [], []
        for k in range(len(x_c) - 1):
            p0 = (x_c[k],   y_c[k])
            p1 = (x_c[k+1], y_c[k+1])
            if prep_poly.contains(Point(p0)) or prep_poly.contains(Point(p1)):
                if not xs:
                    xs.append(x_c[k]); ys.append(y_c[k])
                xs.append(x_c[k+1]); ys.append(y_c[k+1])
            else:
                if xs:
                    equip_curves.append((np.array(xs), np.array(ys)))
                    xs, ys = [], []
        if xs:
            equip_curves.append((np.array(xs), np.array(ys)))

    logger.info("Parametric curves: %d streamlines, %d equipotentials",
                len(stream_curves), len(equip_curves))
    return stream_curves, equip_curves


def sc_inverse_single(
    z_target: complex,
    params: SCParameters,
    zeta0: complex = 0.0 + 0.5j,
    *,
    maxfev: int = 800,
) -> complex | None:
    """Find ζ ∈ ℍ such that f(ζ) = z_target.

    Tries the warm-start guess first, then falls back to a grid of
    initial guesses if needed.  Returns None if all fail.
    """
    def residual(xy):
        zeta = xy[0] + 1j * xy[1]
        fz = sc_map_single(zeta, params, n_pts=250)
        return [fz.real - z_target.real, fz.imag - z_target.imag]

    # Try the warm-start guess first
    guesses = [zeta0]
    # Fallback guesses spread across the upper half-plane
    for rx in np.linspace(-0.8, 0.8, 5):
        for iy in [0.2, 0.6, 1.2]:
            g = rx + 1j * iy
            if abs(g - zeta0) > 0.1:
                guesses.append(g)

    for g in guesses:
        sol, info, ier, msg = optimize.fsolve(
            residual,
            [g.real, g.imag],
            full_output=True,
            maxfev=maxfev,
        )
        if ier == 1:
            zeta = sol[0] + 1j * sol[1]
            if zeta.imag > 1e-10:   # strictly inside ℍ
                return zeta
    return None


def compute_flow_grid(
    norm_polygon: Polygon,
    params: SCParameters,
    n_grid: int = 80,
    U: float = 1.0,
    potential_fn: Optional[PotentialFn] = None,
    zeta_cache: Optional[np.ndarray] = None,
    n_zeta: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ on a grid over *norm_polygon* (normalised coords).

    Uses a **forward-map** approach to avoid the unreliable inverse Newton
    solver.  A dense grid of ζ values in the upper half-plane is mapped
    forward to z-space via f(ζ), potentials are evaluated at those scattered
    points, and the result is interpolated to the regular output grid.

    Parameters
    ----------
    norm_polygon : Shapely polygon in the **normalised** coordinate system.
    params       : solved SC parameters.
    n_grid       : output grid resolution per axis.
    U            : free-stream speed.
    potential_fn : callable ``ζ → W(ζ)``.  Defaults to ``W = Uζ``.
    zeta_cache   : unused (kept for API compatibility).
    n_zeta       : resolution of the ζ sampling grid per axis.

    Returns
    -------
    XX, YY, Psi, Phi, Zeta - all shape ``(n_grid, n_grid)``.
    Psi, Phi, and Zeta are NaN outside the polygon.
    """
    if potential_fn is None:
        from .terrain import uniform_potential
        potential_fn = lambda zeta: uniform_potential(zeta, U)

    # ── 1. Sample ζ in upper half-plane and map forward ──────────────────
    # Use a grid that concentrates near the real axis (where the polygon
    # boundary maps to) and fans out.  Log-spaced y captures both near-
    # boundary detail and far-field behaviour.
    x_zeta = np.linspace(-1.8, 1.8, n_zeta)
    y_zeta = np.logspace(-1.5, 0.7, n_zeta // 2)   # ~0.03 … 5
    X_zeta, Y_zeta = np.meshgrid(x_zeta, y_zeta)
    zeta_flat = (X_zeta + 1j * Y_zeta).ravel()

    logger.info("Forward SC map: evaluating %d ζ points …", len(zeta_flat))
    z_flat = sc_map(zeta_flat, params, n_pts=250)

    # Evaluate potential at each ζ point
    W_flat = np.array([potential_fn(z) for z in zeta_flat], dtype=complex)
    Psi_flat = W_flat.imag
    Phi_flat = W_flat.real

    # ── 2. Build regular z-space output grid ─────────────────────────────
    bounds = norm_polygon.bounds
    pad = 0.04 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] - pad, bounds[2] + pad, n_grid)
    yv = np.linspace(bounds[1] - pad, bounds[3] + pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    # ── 3. Scattered interpolation to regular grid ────────────────────────
    # Only use z points that landed inside or near the polygon bounding box
    bx0, by0, bx1, by1 = bounds
    margin = 0.3
    in_box = (
        (z_flat.real > bx0 - margin) & (z_flat.real < bx1 + margin) &
        (z_flat.imag > by0 - margin) & (z_flat.imag < by1 + margin)
    )
    pts  = np.column_stack([z_flat[in_box].real, z_flat[in_box].imag])
    n_pts = pts.shape[0]
    logger.info("Interpolating from %d scattered z-points …", n_pts)

    if n_pts < 10:
        logger.error("Too few forward-mapped points inside bounding box!")
        Psi = np.full(XX.shape, np.nan)
        Phi = np.full(XX.shape, np.nan)
        Zeta = np.full(XX.shape, np.nan, dtype=complex)
        return XX, YY, Psi, Phi, Zeta

    Psi = griddata(pts, Psi_flat[in_box], (XX, YY), method="linear")
    Phi = griddata(pts, Phi_flat[in_box], (XX, YY), method="linear")
    # Zeta: store Re and Im separately then reassemble
    zr  = griddata(pts, zeta_flat[in_box].real, (XX, YY), method="linear")
    zi  = griddata(pts, zeta_flat[in_box].imag, (XX, YY), method="linear")
    Zeta = zr + 1j * zi

    # ── 4. Mask points outside the polygon ───────────────────────────────
    logger.info("Building interior mask (%d × %d) …", n_grid, n_grid)
    prep_poly = prep(norm_polygon)
    for i in range(n_grid):
        for j in range(n_grid):
            if not prep_poly.contains(Point(XX[i, j], YY[i, j])):
                Psi[i, j]  = np.nan
                Phi[i, j]  = np.nan
                Zeta[i, j] = np.nan

    n_valid = int(np.isfinite(Psi).sum())
    logger.info("Flow grid: %d valid interior points", n_valid)
    return XX, YY, Psi, Phi, Zeta


def compute_flow_grid_urban(
    norm_polygon_outer: Polygon,
    norm_polygon_inner: Polygon,
    params: SCParameters,
    obstacle,
    n_grid: int = 80,
    U: float = 1.0,
    terrain_sources=None,
    n_zeta: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ for the doubly-connected domain Ω_outer \\ Ω_inner.

    Uses the same forward-map + interpolation strategy as compute_flow_grid.
    Points inside the inner polygon (urban core) are masked out after
    interpolation.

    Parameters
    ----------
    norm_polygon_outer : outer (Boulder) boundary in normalised coords.
    norm_polygon_inner : inner (urban core) boundary in normalised coords.
    params             : solved outer SC parameters.
    obstacle           : UrbanObstacle from compute_urban_obstacle().
    n_grid             : grid resolution per axis.
    U                  : free-stream speed.
    terrain_sources    : optional list of (ζ, Q) terrain source/sink pairs.
    n_zeta             : ζ sampling grid resolution per axis.

    Returns
    -------
    XX, YY, Psi, Phi, Zeta - all shape (n_grid, n_grid).
    Points outside outer polygon or inside inner polygon are NaN.
    """
    from .sc_solver_dc import urban_potential, urban_terrain_potential

    if terrain_sources:
        potential_fn = lambda zeta: urban_terrain_potential(
            zeta, U, obstacle, terrain_sources
        )
    else:
        potential_fn = lambda zeta: urban_potential(zeta, U, obstacle)

    # ── Forward map: sample ζ, skip points inside obstacle circle ────────
    x_zeta = np.linspace(-1.8, 1.8, n_zeta)
    y_zeta = np.logspace(-1.5, 0.7, n_zeta // 2)
    X_zeta, Y_zeta = np.meshgrid(x_zeta, y_zeta)
    zeta_flat = (X_zeta + 1j * Y_zeta).ravel()

    # Mask out ζ inside the obstacle circle (no-penetration region)
    not_in_obstacle = np.abs(zeta_flat - obstacle.zeta0) >= obstacle.radius
    zeta_flat = zeta_flat[not_in_obstacle]

    logger.info("Urban forward SC map: evaluating %d ζ points …", len(zeta_flat))
    z_flat = sc_map(zeta_flat, params, n_pts=250)

    W_flat  = np.array([potential_fn(z) for z in zeta_flat], dtype=complex)
    Psi_flat = W_flat.imag
    Phi_flat = W_flat.real

    # ── Build output grid ────────────────────────────────────────────────
    bounds = norm_polygon_outer.bounds
    pad = 0.04 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] - pad, bounds[2] + pad, n_grid)
    yv = np.linspace(bounds[1] - pad, bounds[3] + pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    bx0, by0, bx1, by1 = bounds
    margin = 0.3
    in_box = (
        (z_flat.real > bx0 - margin) & (z_flat.real < bx1 + margin) &
        (z_flat.imag > by0 - margin) & (z_flat.imag < by1 + margin)
    )
    pts = np.column_stack([z_flat[in_box].real, z_flat[in_box].imag])
    logger.info("Urban interpolating from %d scattered z-points …", pts.shape[0])

    Psi  = griddata(pts, Psi_flat[in_box], (XX, YY), method="linear")
    Phi  = griddata(pts, Phi_flat[in_box], (XX, YY), method="linear")
    zr   = griddata(pts, zeta_flat[in_box].real, (XX, YY), method="linear")
    zi   = griddata(pts, zeta_flat[in_box].imag, (XX, YY), method="linear")
    Zeta = zr + 1j * zi

    # ── Mask: outside outer OR inside inner polygon ───────────────────────
    logger.info("Building doubly-connected interior mask (%d × %d) …",
                n_grid, n_grid)
    from shapely.prepared import prep as shapely_prep
    prep_outer = shapely_prep(norm_polygon_outer)
    prep_inner = shapely_prep(norm_polygon_inner)

    for i in range(n_grid):
        for j in range(n_grid):
            pt = Point(XX[i, j], YY[i, j])
            if not prep_outer.contains(pt) or prep_inner.contains(pt):
                Psi[i, j]  = np.nan
                Phi[i, j]  = np.nan
                Zeta[i, j] = np.nan

    n_solved = int(np.isfinite(Psi).sum())
    logger.info("Urban flow grid: %d valid interior points", n_solved)
    return XX, YY, Psi, Phi, Zeta
