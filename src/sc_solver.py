"""
sc_solver.py
============
Schwarz–Christoffel parameter problem solver and forward-map evaluator.

    f(ζ) = A + C ∫₀^ζ  ∏ₖ (t − ζₖ)^(αₖ − 1)  dt

Vectorised with NumPy for performance.  The Möbius normalisation fixes
three pre-vertices; the remaining n − 3 are found by nonlinear least
squares on the side-length ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────

@dataclass
class SCParameters:
    """Solved SC mapping parameters."""
    zk: np.ndarray        # pre-vertices on ℝ, shape (n,)
    alphas: np.ndarray    # interior angles / π
    betas: np.ndarray     # exponents αₖ − 1
    A: complex
    C: complex
    z_poly: np.ndarray    # target polygon vertices (complex, normalised)


# ── Vectorised SC integrand ──────────────────────────────────────────────

def _sc_prod_real(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Evaluate ∏ₖ (t − ζₖ)^βₖ for real t (vectorised over t).

    t : (N,)   zk : (n,)   betas : (n,)   → result : (N,) complex
    """
    diffs = t[:, None] - zk[None, :]            # (N, n)
    signs = np.where(diffs < 0, np.pi, 0.0)     # phase correction
    log_abs = betas[None, :] * np.log(np.abs(diffs) + 1e-300)
    phases  = 1j * betas[None, :] * signs
    return np.prod(np.exp(log_abs + phases), axis=1)


def _sc_prod_complex(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Evaluate ∏ₖ (t − ζₖ)^βₖ for complex t (vectorised over t).

    Uses the principal branch of log.
    """
    diffs = t[:, None] - zk[None, :]            # (N, n) complex
    return np.prod(np.exp(betas[None, :] * np.log(diffs)), axis=1)


# ── Gauss–Legendre integration (vectorised) ──────────────────────────────

_GL_CACHE: dict[int, tuple] = {}

def _gl_nodes(n_pts: int):
    """Cached Gauss–Legendre nodes and weights."""
    if n_pts not in _GL_CACHE:
        _GL_CACHE[n_pts] = np.polynomial.legendre.leggauss(n_pts)
    return _GL_CACHE[n_pts]


def integrate_real(a: float, b: float,
                   zk: np.ndarray, betas: np.ndarray,
                   n_pts: int = 500) -> complex:
    """∫_a^b ∏ₖ (t−ζₖ)^βₖ dt   along the real axis."""
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (a + b)
    half = 0.5 * (b - a)
    t = mid + half * nodes                       # (n_pts,)
    vals = _sc_prod_real(t, zk, betas)           # (n_pts,) complex
    return complex(half * np.dot(weights, vals))


def integrate_complex(za: complex, zb: complex,
                      zk: np.ndarray, betas: np.ndarray,
                      n_pts: int = 400) -> complex:
    """∫_{za}^{zb} ∏ₖ (t−ζₖ)^βₖ dt   along a straight line in ℂ."""
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (za + zb)
    half = 0.5 * (zb - za)
    t = mid + half * nodes                       # (n_pts,) complex
    vals = _sc_prod_complex(t, zk, betas)        # (n_pts,) complex
    return complex(half * np.dot(weights, vals))


# ── Side lengths (for the parameter problem) ─────────────────────────────

def _side_length(zk_a: float, zk_b: float,
                 zk: np.ndarray, betas: np.ndarray) -> float:
    """| ∫_{zk_a}^{zk_b} integrand dt |"""
    return abs(integrate_real(zk_a, zk_b, zk, betas))


def _all_side_lengths(zk: np.ndarray, betas: np.ndarray,
                      R: float = 40.0) -> np.ndarray:
    """Compute side lengths for all n sides including the infinite one."""
    n = len(zk)
    lengths = np.empty(n)
    for i in range(n - 1):
        lengths[i] = _side_length(zk[i], zk[i + 1], zk, betas)
    # Last side: zk[n-1] → +∞ → −∞ → zk[0]
    seg1 = abs(integrate_real(zk[-1], zk[-1] + R, zk, betas))
    seg2 = abs(integrate_real(zk[0] - R, zk[0], zk, betas))
    lengths[n - 1] = seg1 + seg2
    return lengths


# ── SC parameter problem ─────────────────────────────────────────────────

def solve_parameters(
    z_poly: np.ndarray,
    alphas: np.ndarray,
    *,
    maxiter: int = 600,
    tol: float = 1e-10,
) -> SCParameters:
    """Solve the SC parameter problem.

    Möbius normalisation: ζ₀ = −1, ζ₁ = 0, ζ_{n−1} = 1.
    Free parameters: ζ₂, …, ζ_{n−2}  (must satisfy −1 < ζ₀ < ζ₁ < … < ζ_{n−1} = 1).
    """
    n = len(z_poly)
    betas = alphas - 1.0

    # Target side-length ratios
    target_sides = np.abs(np.diff(np.append(z_poly, z_poly[0])))
    target_ratios = target_sides[:-1] / target_sides[-1]

    # Fixed pre-vertices
    fixed_vals = {0: -1.0, 1: 0.0, n - 1: 1.0}
    free_idx = [i for i in range(n) if i not in fixed_vals]
    n_free = len(free_idx)

    if n_free == 0:
        zk = np.array([-1.0, 0.0, 1.0])
    else:
        # Initial guess: equally spaced in (0, 1)
        x0 = np.linspace(0.05, 0.95, n_free + 2)[1:-1]

        def _residuals(params):
            zk = np.empty(n)
            for idx, val in fixed_vals.items():
                zk[idx] = val
            for k, idx in enumerate(free_idx):
                zk[idx] = params[k]
            zk_sorted = np.sort(zk)
            sl = _all_side_lengths(zk_sorted, betas)
            ratios = sl[:-1] / sl[-1]
            return ratios - target_ratios

        logger.info("Solving SC parameters (n=%d, free=%d) …", n, n_free)
        result = optimize.least_squares(
            _residuals, x0,
            method="lm",
            max_nfev=maxiter * 30,
            ftol=tol, xtol=tol, gtol=tol,
        )
        if result.success:
            logger.info("SC solver converged: cost=%.2e", result.cost)
        else:
            logger.warning("SC solver: %s  (cost=%.2e)", result.message, result.cost)

        zk = np.empty(n)
        for idx, val in fixed_vals.items():
            zk[idx] = val
        for k, idx in enumerate(free_idx):
            zk[idx] = result.x[k]

    zk = np.sort(zk)

    # Determine A and C
    A, C = _solve_AC(zk, betas, z_poly)

    params = SCParameters(zk=zk, alphas=alphas, betas=betas,
                          A=A, C=C, z_poly=z_poly)
    logger.info("A = %.6g%+.6gj   C = %.6g%+.6gj", A.real, A.imag, C.real, C.imag)
    return params


def _solve_AC(zk, betas, z_poly):
    """Determine translation A and scale/rotation C.

    Uses the first two mapped vertices to fix C and A.
    Integration stays slightly above the real axis to avoid singularities.
    """
    delta = 0.01j  # tiny lift into ℍ

    # Integrate along a path slightly above the real axis: zk[0]+δ → zk[1]+δ
    I01 = integrate_complex(zk[0] + delta, zk[1] + delta, zk, betas)
    delta_z = z_poly[1] - z_poly[0]
    C = delta_z / I01 if abs(I01) > 1e-15 else 1.0 + 0.0j

    # f(0) = A + C * ∫_0^0 = A, but our base is at ζ=0.
    # Compute f(zk[0]) and match to z_poly[0] to get A.
    # ∫_0^{zk[0]}: path 0→0+δ → zk[0]+δ → zk[0]
    I_base = (integrate_complex(0.0, delta, zk, betas)
              + integrate_complex(delta, zk[0] + delta, zk, betas)
              + integrate_complex(zk[0] + delta, zk[0] + 0.0j, zk, betas))
    A = z_poly[0] - C * I_base
    return A, C


# ── Forward map  f(ζ) ────────────────────────────────────────────────────

# Reference point in upper half-plane (avoids real-axis singularities)
_ZETA_REF = 0.0 + 0.5j
_F_REF_CACHE: dict[int, complex] = {}   # keyed by id(params)


def _f_at_ref(params: SCParameters, n_pts: int = 500) -> complex:
    """Evaluate the SC integral at the reference point _ZETA_REF."""
    key = id(params)
    if key not in _F_REF_CACHE:
        zk, betas = params.zk, params.betas
        # Path: zk[0] → zk[0]+iδ → _ZETA_REF   (stays in ℍ)
        delta = 0.5
        p0 = zk[0] + 0.0j
        p1 = zk[0] + 1j * delta
        p2 = _ZETA_REF
        # But we need ∫_0^{_ZETA_REF}, same base as A/C calibration.
        # Use path: 0→0+iδ → Re(ref)+iδ → ref  (all in ℍ, no singularities)
        I = 0.0 + 0.0j
        I += integrate_complex(0.0 + 0.0j, 0.0 + 1j * delta, zk, betas, n_pts)
        I += integrate_complex(0.0 + 1j * delta, _ZETA_REF.real + 1j * delta,
                               zk, betas, n_pts)
        I += integrate_complex(_ZETA_REF.real + 1j * delta, _ZETA_REF,
                               zk, betas, n_pts)
        _F_REF_CACHE[key] = I
    return _F_REF_CACHE[key]


def sc_map_single(zeta: complex, params: SCParameters, n_pts: int = 400) -> complex:
    """Evaluate f(ζ) for a single point in ℍ.

    Integration path stays in the upper half-plane to avoid
    the real-axis singularities at the pre-vertices.
    Uses a cached reference evaluation at _ZETA_REF for efficiency.
    """
    zk, betas, A, C = params.zk, params.betas, params.A, params.C

    # For points on or very near the real axis, lift slightly
    if abs(zeta.imag) < 1e-12:
        zeta = zeta.real + 1e-10j

    # Integrate ref → ζ via a path in ℍ
    I_ref = _f_at_ref(params, n_pts)

    # Path from _ZETA_REF to ζ: go via an intermediate point at
    # a safe height δ = max(Im(ζ), Im(ref)) / 2 to stay in ℍ
    delta = max(abs(zeta.imag), 0.3)
    mid_y = delta
    p_mid1 = _ZETA_REF.real + 1j * mid_y
    p_mid2 = zeta.real + 1j * mid_y

    I_path = 0.0 + 0.0j
    # _ZETA_REF → p_mid1
    if abs(_ZETA_REF - p_mid1) > 1e-14:
        I_path += integrate_complex(_ZETA_REF, p_mid1, zk, betas, n_pts)
    # p_mid1 → p_mid2 (horizontal)
    if abs(p_mid1 - p_mid2) > 1e-14:
        I_path += integrate_complex(p_mid1, p_mid2, zk, betas, n_pts)
    # p_mid2 → ζ
    if abs(p_mid2 - zeta) > 1e-14:
        I_path += integrate_complex(p_mid2, zeta, zk, betas, n_pts)

    return A + C * (I_ref + I_path)


def sc_map(zeta_arr: np.ndarray, params: SCParameters, n_pts: int = 400) -> np.ndarray:
    """Evaluate the SC forward map on an array of complex points."""
    return np.array([sc_map_single(z, params, n_pts) for z in zeta_arr])
