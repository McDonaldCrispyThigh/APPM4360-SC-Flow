"""
flow.py
=======
Compute ψ (stream function) and φ (velocity potential) on a grid
covering the polygon **in normalised coordinates** by numerically
inverting the SC map.

Complex potential in ℍ:  W(ζ) = U·ζ
  →  φ = U·Re(ζ),   ψ = U·Im(ζ)

For each grid point z ∈ Ω (normalised coords) we solve f(ζ) = z for ζ ∈ ℍ.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import optimize
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from tqdm import tqdm

from .sc_solver import SCParameters, sc_map_single

logger = logging.getLogger(__name__)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ on a grid over *norm_polygon* (normalised coords).

    Parameters
    ----------
    norm_polygon : Shapely polygon in the **normalised** coordinate system
        (centred at 0, max vertex modulus ≈ 1).
    params : solved SC parameters.
    n_grid : grid resolution per axis.
    U : free-stream speed.

    Returns
    -------
    XX, YY, Psi, Phi  — all shape ``(n_grid, n_grid)``.
    Psi and Phi are NaN outside the polygon.
    """
    bounds = norm_polygon.bounds  # (minx, miny, maxx, maxy)
    pad = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] + pad, bounds[2] - pad, n_grid)
    yv = np.linspace(bounds[1] + pad, bounds[3] - pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    Psi = np.full(XX.shape, np.nan)
    Phi = np.full(XX.shape, np.nan)

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
        return XX, YY, Psi, Phi

    # Solve inverse map, scanning row-by-row for coherent warm-starts
    logger.info("Inverting SC map for %d interior points …", n_inside)
    zeta_prev = 0.0 + 0.5j

    interior_indices = np.argwhere(mask)
    for idx in tqdm(interior_indices, desc="Inverse SC map"):
        i, j = idx
        z_target = XX[i, j] + 1j * YY[i, j]
        zeta = sc_inverse_single(z_target, params, zeta0=zeta_prev)
        if zeta is not None:
            Psi[i, j] = U * zeta.imag
            Phi[i, j] = U * zeta.real
            zeta_prev = zeta

    n_solved = np.isfinite(Psi).sum()
    logger.info("Inverted %d / %d interior points", n_solved, n_inside)
    return XX, YY, Psi, Phi
