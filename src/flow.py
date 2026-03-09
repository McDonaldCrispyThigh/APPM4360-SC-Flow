"""
flow.py
=======
Compute the stream function ψ and velocity potential φ on a grid
covering the Boulder polygon by numerically inverting the SC map.

The complex potential in the upper half-plane is  W(ζ) = U·ζ ,
so  φ = U·Re(ζ),  ψ = U·Im(ζ).

For each grid point z ∈ Ω we solve  f(ζ) = z  for ζ ∈ ℍ ,
then read off  ψ = Im(ζ)  (with U = 1).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import optimize
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from .sc_solver import SCParameters, sc_map_single

logger = logging.getLogger(__name__)


# ── Inverse SC map (numerical root-finding) ──────────────────────────────

def sc_inverse_single(
    z_target: complex,
    params: SCParameters,
    zeta0: complex = 0.0 + 0.5j,
    *,
    tol: float = 1e-8,
    maxiter: int = 60,
) -> complex | None:
    """Find ζ ∈ ℍ such that f(ζ) = z_target.

    Uses ``scipy.optimize.fsolve`` with the residual  f(ζ) − z_target = 0
    decomposed into real and imaginary parts.

    Returns None if the solver does not converge.
    """

    def residual(xy):
        zeta = xy[0] + 1j * xy[1]
        fz = sc_map_single(zeta, params, n_pts=200)
        return [fz.real - z_target.real, fz.imag - z_target.imag]

    sol, info, ier, msg = optimize.fsolve(
        residual,
        [zeta0.real, zeta0.imag],
        full_output=True,
        maxfev=maxiter * 10,
    )
    if ier == 1:
        zeta = sol[0] + 1j * sol[1]
        # Ensure ζ is in the upper half-plane
        if zeta.imag < -1e-10:
            return None
        return zeta
    return None


# ── Build the stream-function grid ───────────────────────────────────────

def compute_flow_grid(
    simplified_polygon: Polygon,
    params: SCParameters,
    n_grid: int = 80,
    U: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ψ and φ on a rectangular grid that covers *simplified_polygon*.

    Parameters
    ----------
    simplified_polygon : Shapely polygon (UTM or normalised coords).
    params : solved SC parameters.
    n_grid : number of grid points along each axis.
    U : free-stream speed.

    Returns
    -------
    XX, YY : meshgrid arrays, shape ``(n_grid, n_grid)``.
    Psi : stream function, same shape — NaN outside the polygon.
    Phi : velocity potential, same shape — NaN outside the polygon.
    """
    bounds = simplified_polygon.bounds  # (minx, miny, maxx, maxy)
    pad = 0.02 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    xv = np.linspace(bounds[0] - pad, bounds[2] + pad, n_grid)
    yv = np.linspace(bounds[1] - pad, bounds[3] + pad, n_grid)
    XX, YY = np.meshgrid(xv, yv)

    Psi = np.full(XX.shape, np.nan)
    Phi = np.full(XX.shape, np.nan)

    # Pre-compute mask (which grid points are inside the polygon)
    logger.info("Building interior mask (%d × %d grid) …", n_grid, n_grid)
    mask = np.zeros(XX.shape, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            if simplified_polygon.contains(Point(XX[i, j], YY[i, j])):
                mask[i, j] = True
    n_inside = mask.sum()
    logger.info("Interior points: %d / %d", n_inside, n_grid * n_grid)

    # ── Solve inverse map for each interior point ──
    logger.info("Inverting SC map for %d interior points …", n_inside)

    # Use a running "last good ζ" as the initial guess for nearby points
    zeta_prev = 0.0 + 0.5j

    interior_indices = np.argwhere(mask)
    for count, (i, j) in enumerate(tqdm(interior_indices, desc="Inverse SC map")):
        z_target = XX[i, j] + 1j * YY[i, j]
        zeta = sc_inverse_single(z_target, params, zeta0=zeta_prev)
        if zeta is not None:
            Psi[i, j] = U * zeta.imag
            Phi[i, j] = U * zeta.real
            zeta_prev = zeta  # warm-start next solve

    n_solved = np.isfinite(Psi).sum()
    logger.info("Successfully inverted %d / %d interior points", n_solved, n_inside)

    return XX, YY, Psi, Phi
