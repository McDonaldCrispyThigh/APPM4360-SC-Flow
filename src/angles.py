"""
angles.py
=========
Interior-angle computation for a polygon given as complex vertices.
Angles are returned in units of π.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def interior_angles_pi(z_poly: np.ndarray) -> np.ndarray:
    """Interior angles of a CCW simple polygon, in units of π.

    For convex vertex 0 < α < 1; reflex vertex 1 < α < 2.
    """
    n = len(z_poly)
    alphas = np.empty(n)

    for k in range(n):
        v_in  = z_poly[k] - z_poly[k - 1]
        v_out = z_poly[(k + 1) % n] - z_poly[k]
        turn = np.angle(v_out / v_in)          # exterior turn
        alphas[k] = 1.0 - turn / np.pi         # interior / π

    return alphas


def verify_angle_sum(alphas: np.ndarray, tol: float = 1e-6) -> bool:
    """Check  Σ αₖ = n − 2."""
    n = len(alphas)
    expected = n - 2
    actual = alphas.sum()
    ok = abs(actual - expected) < tol
    if ok:
        logger.info("Angle sum PASSED: Σαₖ = %.6f ≈ %d (n=%d)", actual, expected, n)
    else:
        logger.warning("Angle sum FAILED: Σαₖ = %.6f ≠ %d (err=%.2e)",
                        actual, expected, abs(actual - expected))
    return ok


def sc_exponents(alphas: np.ndarray) -> np.ndarray:
    """Return βₖ = αₖ − 1  (SC integrand exponents)."""
    return alphas - 1.0
