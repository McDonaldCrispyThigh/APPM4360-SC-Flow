"""
sc_solver.py
============
Schwarz-Christoffel parameter problem solver and forward-map evaluator.

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


# ── Gauss-Legendre integration (vectorised) ──────────────────────────────

_GL_CACHE: dict[int, tuple] = {}

def _gl_nodes(n_pts: int):
    """Cached Gauss-Legendre nodes and weights."""
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
                      R: float = 500.0) -> np.ndarray:
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
        # Softmax parameterisation: p ∈ ℝⁿ_free (unconstrained).
        # Maps to n_free strictly-ordered values in (0, 1) via:
        #   w = [exp(p₀), …, exp(p_{m-1}), 1]   (m+1 weights, last fixed)
        #   gaps = w / sum(w)                     (sum to 1, all positive)
        #   ζ_free[i] = cumsum(gaps)[i]           (strictly increasing in (0,1))
        # Initial p=0 → equal spacing: ζ_free = [1/(m+1), …, m/(m+1)].
        # This gives a smooth, everywhere-differentiable residual (no sort).

        def _softmax_to_zk_free(p: np.ndarray) -> np.ndarray:
            w = np.append(np.exp(p - p.max()), 1.0)   # numerically stable
            w /= w.sum()
            return np.cumsum(w[:n_free])               # n_free values in (0, 1)

        # Side-length-proportional init to reduce crowding.
        # The n_free+1 softmax weights correspond to polygon sides 1..n-2.
        init_sides = target_sides[1:n - 1]          # shape (n_free + 1,)
        ratios = np.maximum(init_sides[:-1] / init_sides[-1], 1e-6)
        x0 = np.log(ratios)     # last weight fixed at 1 (log=0)

        def _residuals(p):
            zk_arr = np.empty(n)
            for idx, val in fixed_vals.items():
                zk_arr[idx] = val
            zk_free = _softmax_to_zk_free(p)
            for k, idx in enumerate(free_idx):
                zk_arr[idx] = zk_free[k]
            sl = _all_side_lengths(zk_arr, betas)
            ratios = sl[:-1] / sl[-1]
            return ratios - target_ratios

        logger.info("Solving SC parameters (n=%d, free=%d) …", n, n_free)
        # p is unconstrained; use Levenberg-Marquardt for fastest convergence.
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
        zk_free = _softmax_to_zk_free(result.x)
        for k, idx in enumerate(free_idx):
            zk[idx] = zk_free[k]

    zk = np.sort(zk)

    # Determine A and C
    A, C = _solve_AC(zk, betas, z_poly)

    params = SCParameters(zk=zk, alphas=alphas, betas=betas,
                          A=A, C=C, z_poly=z_poly)
    logger.info("A = %.6g%+.6gj   C = %.6g%+.6gj", A.real, A.imag, C.real, C.imag)
    return params


def _solve_AC(zk, betas, z_poly):
    """Determine A = f(_ZETA_REF) and scale/rotation C via least squares.

    The SC map is parameterised as
        f(ζ) = A + C ∫_{_ZETA_REF}^{ζ} integrand dt

    where _ZETA_REF = 0.5j lies safely in ℍ away from all branch points.
    We compute the integrals I_k = ∫_{_ZETA_REF}^{zk[k] + ε} for all n
    vertices (ε = 0.05j keeps paths away from branch points), then solve
    the overdetermined linear system
        A + C · I_k = z_poly[k]   for k = 0, …, n-1
    in the real least-squares sense to get A and C that globally minimise
    the vertex error rather than enforcing only two calibration points.
    """
    eps = 0.05j   # safe approach height above each pre-vertex

    def _path_integral(zeta_target):
        """∫_{_ZETA_REF}^{zeta_target} via an L-shaped path in ℍ."""
        mid_y = max(abs(zeta_target.imag), 0.3)
        p1 = _ZETA_REF.real + 1j * mid_y
        p2 = zeta_target.real + 1j * mid_y
        I = 0.0 + 0.0j
        if abs(_ZETA_REF - p1) > 1e-14:
            I += integrate_complex(_ZETA_REF, p1, zk, betas)
        if abs(p1 - p2) > 1e-14:
            I += integrate_complex(p1, p2, zk, betas)
        if abs(p2 - zeta_target) > 1e-14:
            I += integrate_complex(p2, zeta_target, zk, betas)
        return I

    # Use only the two extremal fixed pre-vertices (zk[0] and zk[-1]).
    # Their integrals are computed far from the crowded region near zk[1]=0,
    # so GL quadrature is most accurate there.
    I_first = _path_integral(zk[0]  + eps)
    I_last  = _path_integral(zk[-1] + eps)

    dI = I_last - I_first
    C = (z_poly[-1] - z_poly[0]) / dI if abs(dI) > 1e-15 else 1.0 + 0.0j
    A = z_poly[0] - C * I_first
    return A, C


# ── Forward map  f(ζ) ────────────────────────────────────────────────────

# Reference point in upper half-plane (avoids real-axis singularities).
# A = f(_ZETA_REF) is stored in SCParameters.A; all integrals are relative
# to this base so no branch-point arithmetic is ever needed.
_ZETA_REF = 0.0 + 0.5j


def sc_map_single(zeta: complex, params: SCParameters, n_pts: int = 400) -> complex:
    """Evaluate f(ζ) = A + C ∫_{_ZETA_REF}^{ζ} integrand dt.

    A = f(_ZETA_REF) is stored in params.A.
    Integration path stays in ℍ via an L-shaped route.
    """
    zk, betas, A, C = params.zk, params.betas, params.A, params.C

    # For points on or very near the real axis, lift slightly
    if abs(zeta.imag) < 1e-12:
        zeta = zeta.real + 1e-10j

    # L-shaped path from _ZETA_REF to ζ, staying at height ≥ min(Im(ζ), 0.3)
    delta = max(abs(zeta.imag), 0.3)
    p_mid1 = _ZETA_REF.real + 1j * delta
    p_mid2 = zeta.real       + 1j * delta

    I_path = 0.0 + 0.0j
    if abs(_ZETA_REF - p_mid1) > 1e-14:
        I_path += integrate_complex(_ZETA_REF, p_mid1, zk, betas, n_pts)
    if abs(p_mid1 - p_mid2) > 1e-14:
        I_path += integrate_complex(p_mid1, p_mid2, zk, betas, n_pts)
    if abs(p_mid2 - zeta) > 1e-14:
        I_path += integrate_complex(p_mid2, zeta, zk, betas, n_pts)

    return A + C * I_path


def sc_map(zeta_arr: np.ndarray, params: SCParameters, n_pts: int = 400) -> np.ndarray:
    """Evaluate the SC forward map on an array of complex points."""
    return np.array([sc_map_single(z, params, n_pts) for z in zeta_arr])
