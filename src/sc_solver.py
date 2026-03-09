"""
sc_solver.py
============
Schwarz–Christoffel parameter problem solver and forward-map evaluator.

The SC map  f : ℍ → Ω  is

    f(ζ) = A + C ∫₀^ζ  ∏ₖ (t − ζₖ)^(αₖ − 1)  dt

where ζₖ ∈ ℝ are pre-images of the polygon vertices on the real axis,
αₖ·π are the interior angles, and A, C are complex constants.

The *parameter problem* is to find the ζₖ so that the image polygon has
the correct side-length ratios.  Three pre-vertices are fixed by the
Möbius normalisation of ℍ; the remaining n − 3 are solved for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.integrate import quad

logger = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────

@dataclass
class SCParameters:
    """All solved SC mapping parameters."""
    zk: np.ndarray        # pre-vertices on ℝ, shape (n,) – the last is +∞ (omitted from integrand)
    alphas: np.ndarray    # interior angles / π, shape (n,)
    betas: np.ndarray     # exponents αₖ − 1, shape (n,)
    A: complex            # translation constant
    C: complex            # scale / rotation constant
    z_poly: np.ndarray    # target polygon vertices (complex)


# ── SC integrand ──────────────────────────────────────────────────────────

def _sc_integrand_real(t: float, zk: np.ndarray, betas: np.ndarray) -> complex:
    """Evaluate ∏ₖ (t − ζₖ)^βₖ  for real t.

    Uses the principal branch for each factor.  When t < ζₖ , the factor
    is |t − ζₖ|^βₖ · exp(i·π·βₖ) so that the integral follows the real
    axis correctly.
    """
    val = 1.0 + 0.0j
    for z, b in zip(zk, betas):
        diff = t - z
        if diff == 0.0:
            return 0.0 + 0.0j
        val *= np.abs(diff) ** b * np.exp(1j * b * (np.pi if diff < 0 else 0.0))
    return val


# ── Integrate along the real axis between two pre-vertices ────────────────

def _integrate_side(zk_a: float, zk_b: float,
                    zk: np.ndarray, betas: np.ndarray,
                    n_pts: int = 800) -> complex:
    """Numerically integrate the SC integrand from *zk_a* to *zk_b*.

    Uses composite Gauss–Legendre quadrature on a real-axis segment,
    handling the near-singular behaviour at the endpoints via a
    change of variable  t = zk_a + (zk_b − zk_a) · u  on Gauss nodes.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_pts)
    # Map from [-1, 1] to [zk_a, zk_b]
    mid = 0.5 * (zk_a + zk_b)
    half = 0.5 * (zk_b - zk_a)
    result = 0.0 + 0.0j
    for node, w in zip(nodes, weights):
        t = mid + half * node
        result += w * _sc_integrand_real(t, zk, betas)
    result *= half
    return result


def _side_lengths(zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute |∫ side| for each consecutive pair of pre-vertices."""
    n = len(zk)
    lengths = np.empty(n)
    for i in range(n):
        j = (i + 1) % n
        # The last side wraps from zk[-1] → +∞ → −∞ → zk[0].
        # We handle this with a large-radius substitution.
        if j == 0:
            lengths[i] = _integrate_infinite_side(zk, betas, i)
        else:
            seg = _integrate_side(zk[i], zk[j], zk, betas)
            lengths[i] = np.abs(seg)
    return lengths


def _integrate_infinite_side(
    zk: np.ndarray, betas: np.ndarray, last_idx: int,
    R: float = 50.0, n_pts: int = 800,
) -> float:
    """Integrate the 'infinite' side: zk[last] → +∞ → −∞ → zk[0].

    We split into two half-lines and use the substitution  t = R·tan(θ)
    to handle the improper integrals, then sum their absolute values.
    """
    # Part 1: zk[last] → +R  (finite segment)
    seg1 = _integrate_side(zk[last_idx], zk[last_idx] + R, zk, betas, n_pts)
    # Part 2: zk[0] - R → zk[0]  (finite segment)
    seg2 = _integrate_side(zk[0] - R, zk[0], zk, betas, n_pts)
    # The tails beyond ±R are approximated; for large R and ∑βₖ = -2
    # the integrand decays as t^{-2} so the tails are small.
    return np.abs(seg1) + np.abs(seg2)


# ── SC parameter problem ─────────────────────────────────────────────────

def solve_parameters(
    z_poly: np.ndarray,
    alphas: np.ndarray,
    *,
    maxiter: int = 400,
    tol: float = 1e-10,
) -> SCParameters:
    """Solve the Schwarz–Christoffel parameter problem.

    The Möbius normalisation fixes three pre-vertices:
        ζ₀ = −1,  ζ₁ = 0,  ζ₂ = 1
    The remaining n − 3 pre-vertices are found by nonlinear least
    squares so that the image side-length ratios match the target polygon.

    Parameters
    ----------
    z_poly : complex vertices of the target polygon, shape ``(n,)``, CCW.
    alphas : interior angles / π, shape ``(n,)``.

    Returns
    -------
    SCParameters
    """
    n = len(z_poly)
    betas = alphas - 1.0

    # ── Target side-length ratios (normalised by last side) ──
    target_sides = np.abs(np.diff(np.append(z_poly, z_poly[0])))
    target_ratios = target_sides[:-1] / target_sides[-1]  # length n-1

    # ── Fixed pre-vertices (Möbius normalisation) ──
    # Fix indices 0, 1, n-1 on the real axis
    fixed_vals = np.array([-1.0, 0.0, 1.0])
    fixed_idx = [0, 1, n - 1]
    free_idx = [i for i in range(n) if i not in fixed_idx]

    # Initial guess: equally spaced in (0, 1) for the free pre-vertices
    n_free = len(free_idx)
    if n_free == 0:
        # Triangle — no free params
        zk = fixed_vals.copy()
    else:
        # Initial: spread between second fixed (0) and last fixed (1)
        x0 = np.linspace(0.1, 0.9, n_free + 2)[1:-1]

        def _residuals(params):
            # Build full pre-vertex array, sorted
            zk = np.empty(n)
            zk[fixed_idx[0]] = fixed_vals[0]
            zk[fixed_idx[1]] = fixed_vals[1]
            zk[fixed_idx[2]] = fixed_vals[2]
            for k, idx in enumerate(free_idx):
                zk[idx] = params[k]
            zk_sorted = np.sort(zk)
            sl = _side_lengths(zk_sorted, betas)
            ratios = sl[:-1] / sl[-1]
            return ratios - target_ratios

        logger.info("Solving SC parameter problem  (n=%d, free=%d) …", n, n_free)
        result = optimize.least_squares(
            _residuals, x0,
            method="lm", max_nfev=maxiter * 20,
            ftol=tol, xtol=tol, gtol=tol,
        )
        if not result.success:
            logger.warning("SC solver did not fully converge: %s", result.message)
        else:
            logger.info("SC solver converged: cost=%.2e", result.cost)

        zk = np.empty(n)
        zk[fixed_idx[0]] = fixed_vals[0]
        zk[fixed_idx[1]] = fixed_vals[1]
        zk[fixed_idx[2]] = fixed_vals[2]
        for k, idx in enumerate(free_idx):
            zk[idx] = result.x[k]

    zk = np.sort(zk)

    # ── Determine A and C ──
    # Map ζ₀ → z_poly[0],  and use the side-vector from ζ₀→ζ₁ to fix C.
    A, C = _solve_AC(zk, betas, z_poly)

    params = SCParameters(
        zk=zk, alphas=alphas, betas=betas,
        A=A, C=C, z_poly=z_poly,
    )
    logger.info("SC parameters solved.  A=%.4g%+.4gj   C=%.4g%+.4gj",
                A.real, A.imag, C.real, C.imag)
    return params


def _solve_AC(zk, betas, z_poly):
    """Determine translation A and scale/rotation C from the solved pre-vertices."""
    # Integrate ζ₀ → ζ₁
    I01 = _integrate_side(zk[0], zk[1], zk, betas)
    # The image side should be  z_poly[1] − z_poly[0]
    delta_z = z_poly[1] - z_poly[0]
    C = delta_z / I01 if abs(I01) > 1e-15 else 1.0 + 0.0j

    # A = z_poly[0] − C · ∫₀^{ζ₀} (but our base-point is 0, and ζ₀ = zk[0])
    # f(ζ₀) = A + C · ∫₀^{ζ₀} integrand dt = z_poly[0]
    I_base = _integrate_side(0.0, zk[0], zk, betas)
    A = z_poly[0] - C * I_base
    return A, C


# ── Forward map evaluation ────────────────────────────────────────────────

def sc_map_single(zeta: complex, params: SCParameters,
                  n_pts: int = 300) -> complex:
    """Evaluate f(ζ) for a single point in ℍ (or on ℝ).

    The integration path goes from 0 to Re(ζ) along the real axis,
    then vertically to ζ.  This avoids crossing the real-axis
    singularities.
    """
    zk = params.zk
    betas = params.betas
    A = params.A
    C = params.C

    # Leg 1: real axis  0 → Re(ζ)
    if abs(zeta.real) > 1e-14:
        I_real = _integrate_side_complex(0.0 + 0.0j, zeta.real + 0.0j,
                                         zk, betas, n_pts)
    else:
        I_real = 0.0 + 0.0j

    # Leg 2: vertical  Re(ζ) → ζ
    if abs(zeta.imag) > 1e-14:
        I_vert = _integrate_side_complex(zeta.real + 0.0j, zeta,
                                         zk, betas, n_pts)
    else:
        I_vert = 0.0 + 0.0j

    return A + C * (I_real + I_vert)


def _sc_integrand_complex(t: complex, zk: np.ndarray, betas: np.ndarray) -> complex:
    """SC integrand for complex t (off the real axis)."""
    val = 1.0 + 0.0j
    for z, b in zip(zk, betas):
        diff = t - z
        if abs(diff) < 1e-15:
            return 0.0 + 0.0j
        # Principal branch of power
        val *= np.exp(b * np.log(diff))
    return val


def _integrate_side_complex(
    za: complex, zb: complex,
    zk: np.ndarray, betas: np.ndarray,
    n_pts: int = 300,
) -> complex:
    """Integrate along the straight-line segment za → zb in ℂ."""
    nodes, weights = np.polynomial.legendre.leggauss(n_pts)
    mid = 0.5 * (za + zb)
    half = 0.5 * (zb - za)
    result = 0.0 + 0.0j
    for node, w in zip(nodes, weights):
        t = mid + half * node
        result += w * _sc_integrand_complex(t, zk, betas)
    result *= half
    return result


def sc_map(
    zeta_arr: np.ndarray,
    params: SCParameters,
    n_pts: int = 300,
) -> np.ndarray:
    """Evaluate the SC forward map on an array of complex points.

    Parameters
    ----------
    zeta_arr : 1-D array of complex points in ℍ.
    params : solved SC parameters.
    n_pts : quadrature order per segment.

    Returns
    -------
    np.ndarray of complex — mapped points in the physical domain Ω.
    """
    return np.array([sc_map_single(z, params, n_pts) for z in zeta_arr])
