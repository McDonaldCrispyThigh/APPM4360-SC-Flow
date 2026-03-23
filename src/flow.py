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
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from tqdm import tqdm

from .sc_solver import SCParameters, sc_map_single
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sc_solver_dc import UrbanObstacle

logger = logging.getLogger(__name__)

# Type alias for a potential function:  ζ → W(ζ)
PotentialFn = Callable[[complex], complex]


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
            if zeta.imag > -1e-10:
                return zeta
    return None


def compute_flow_grid(
    norm_polygon: Polygon,
    params: SCParameters,
    n_grid: int = 80,
    U: float = 1.0,
    potential_fn: Optional[PotentialFn] = None,
    zeta_cache: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ on a grid over *norm_polygon* (normalised coords).

    Parameters
    ----------
    norm_polygon : Shapely polygon in the **normalised** coordinate system
        (centred at 0, max vertex modulus ≈ 1).
    params : solved SC parameters.
    n_grid : grid resolution per axis.
    U : free-stream speed.
    potential_fn : callable ``ζ → W(ζ)``.  If None, uses the uniform
        potential ``W(ζ) = U·ζ``.
    zeta_cache : optional pre-computed ζ grid (from a previous call).
        If provided, the expensive inverse-map step is skipped entirely.

    Returns
    -------
    XX, YY, Psi, Phi, Zeta — all shape ``(n_grid, n_grid)``.
    Psi, Phi, and Zeta are NaN outside the polygon.
    """
    if potential_fn is None:
        from .terrain import uniform_potential
        potential_fn = lambda zeta: uniform_potential(zeta, U)

    bounds = norm_polygon.bounds  # (minx, miny, maxx, maxy)
    pad = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] + pad, bounds[2] - pad, n_grid)
    yv = np.linspace(bounds[1] + pad, bounds[3] - pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    Psi = np.full(XX.shape, np.nan)
    Phi = np.full(XX.shape, np.nan)
    Zeta = np.full(XX.shape, np.nan, dtype=complex)

    # ── Fast path: re-use a cached ζ grid from a previous run ──
    if zeta_cache is not None:
        logger.info("Using cached ζ grid (%d valid points)",
                     int(np.isfinite(zeta_cache).sum()))
        Zeta = zeta_cache.copy()
        valid_mask = np.isfinite(zeta_cache)
        for i in range(n_grid):
            for j in range(n_grid):
                if valid_mask[i, j]:
                    W = potential_fn(Zeta[i, j])
                    Psi[i, j] = W.imag
                    Phi[i, j] = W.real
        n_solved = np.isfinite(Psi).sum()
        logger.info("Evaluated potential at %d cached points", n_solved)
        return XX, YY, Psi, Phi, Zeta

    # Fast interior mask using prepared geometry
    logger.info("Building interior mask (%d × %d) …", n_grid, n_grid)
    prep_poly = prep(norm_polygon)
    mask = np.zeros(XX.shape, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            if prep_poly.contains(Point(XX[i, j], YY[i, j])):
                mask[i, j] = True
    n_inside = mask.sum()
    logger.info("Interior points: %d / %d", n_inside, n_grid * n_grid)

    if n_inside == 0:
        logger.error("No interior points — check polygon/grid coordinates!")
        return XX, YY, Psi, Phi, Zeta

    # Solve inverse map, scanning row-by-row for coherent warm-starts
    logger.info("Inverting SC map for %d interior points …", n_inside)
    zeta_prev = 0.0 + 0.5j

    interior_indices = np.argwhere(mask)
    for idx in tqdm(interior_indices, desc="Inverse SC map"):
        i, j = idx
        z_target = XX[i, j] + 1j * YY[i, j]
        zeta = sc_inverse_single(z_target, params, zeta0=zeta_prev)
        if zeta is not None:
            Zeta[i, j] = zeta
            W = potential_fn(zeta)
            Psi[i, j] = W.imag
            Phi[i, j] = W.real
            zeta_prev = zeta

    n_solved = np.isfinite(Psi).sum()
    logger.info("Inverted %d / %d interior points", n_solved, n_inside)

    return XX, YY, Psi, Phi, Zeta


def compute_flow_grid_urban(
    norm_polygon_outer: Polygon,
    norm_polygon_inner: Polygon,
    params: SCParameters,
    obstacle,
    n_grid: int = 80,
    U: float = 1.0,
    terrain_sources=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ for the doubly-connected domain Ω_outer \\ Ω_inner.

    The urban core (Ω_inner) is treated as a solid obstacle: grid points
    inside it are masked to NaN.  The complex potential uses the circle-
    theorem correction from sc_solver_dc.py to enforce approximate
    no-penetration on the obstacle boundary.

    Parameters
    ----------
    norm_polygon_outer : outer (Boulder) boundary in normalised coords.
    norm_polygon_inner : inner (urban core) boundary in normalised coords.
    params             : solved outer SC parameters.
    obstacle           : UrbanObstacle from compute_urban_obstacle().
    n_grid             : grid resolution per axis.
    U                  : free-stream speed.
    terrain_sources    : optional list of (ζ, Q) terrain source/sink pairs.

    Returns
    -------
    XX, YY, Psi, Phi, Zeta — all shape (n_grid, n_grid).
    Points outside outer polygon or inside inner polygon are NaN.
    """
    from .sc_solver_dc import urban_potential, urban_terrain_potential

    # Build potential function
    if terrain_sources:
        potential_fn = lambda zeta: urban_terrain_potential(
            zeta, U, obstacle, terrain_sources
        )
    else:
        potential_fn = lambda zeta: urban_potential(zeta, U, obstacle)

    bounds = norm_polygon_outer.bounds
    pad = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] + pad, bounds[2] - pad, n_grid)
    yv = np.linspace(bounds[1] + pad, bounds[3] - pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    Psi  = np.full(XX.shape, np.nan)
    Phi  = np.full(XX.shape, np.nan)
    Zeta = np.full(XX.shape, np.nan, dtype=complex)

    # Build doubly-connected interior mask: inside outer, outside inner
    logger.info("Building doubly-connected interior mask (%d × %d) …",
                n_grid, n_grid)
    from shapely.prepared import prep as shapely_prep
    prep_outer = shapely_prep(norm_polygon_outer)
    prep_inner = shapely_prep(norm_polygon_inner)

    mask = np.zeros(XX.shape, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            pt = Point(XX[i, j], YY[i, j])
            if prep_outer.contains(pt) and not prep_inner.contains(pt):
                mask[i, j] = True

    n_inside = mask.sum()
    logger.info("Doubly-connected interior: %d / %d grid points", n_inside, n_grid * n_grid)

    if n_inside == 0:
        logger.error("No interior points — check polygon coordinates!")
        return XX, YY, Psi, Phi, Zeta

    # Invert outer SC map and evaluate potential
    logger.info("Inverting outer SC map for %d doubly-connected points …", n_inside)
    zeta_prev = 0.0 + 0.5j
    interior_indices = np.argwhere(mask)

    for idx in tqdm(interior_indices, desc="Inverse SC (urban)"):
        i, j = idx
        z_target = XX[i, j] + 1j * YY[i, j]

        # Skip points very close to the obstacle circle (avoid singularity)
        zeta = sc_inverse_single(z_target, params, zeta0=zeta_prev, maxfev=800)
        if zeta is None:
            continue

        # Skip if the inverse image lands inside the obstacle circle
        if abs(zeta - obstacle.zeta0) < obstacle.radius * 0.95:
            continue

        Zeta[i, j] = zeta
        W = potential_fn(zeta)
        Psi[i, j] = W.imag
        Phi[i, j] = W.real
        zeta_prev = zeta

    n_solved = int(np.isfinite(Psi).sum())
    logger.info("Urban flow grid: %d / %d points solved", n_solved, n_inside)
    return XX, YY, Psi, Phi, Zeta
