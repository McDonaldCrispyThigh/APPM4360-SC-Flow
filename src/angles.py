"""
angles.py
=========
Compute interior angles of a polygon given as complex-valued vertices.
The angles are returned in units of π (so a right angle is 0.5).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def interior_angles_pi(z_poly: np.ndarray) -> np.ndarray:
    """Compute interior angles of a *simple, counter-clockwise* polygon.

    Parameters
    ----------
    z_poly : complex vertices, shape ``(n,)``, ordered CCW,
        **without** the closing duplicate.

    Returns
    -------
    alphas : np.ndarray, shape ``(n,)``
        Interior angle at each vertex **divided by π**.
        For a convex vertex 0 < αₖ < 1;  for a reflex vertex 1 < αₖ < 2.
    """
    n = len(z_poly)
    alphas = np.empty(n)

    for k in range(n):
        v_in  = z_poly[k] - z_poly[k - 1]
        v_out = z_poly[(k + 1) % n] - z_poly[k]
        # Signed exterior turn angle in (-π, π]
        turn = np.angle(v_out / v_in)
        # Interior angle in units of π
        alphas[k] = 1.0 - turn / np.pi

    return alphas


def verify_angle_sum(alphas: np.ndarray, tol: float = 1e-6) -> bool:
    """Check that ∑ αₖ = n − 2  (polygon angle-sum identity).

    Parameters
    ----------
    alphas : interior angles in units of π, shape ``(n,)``.
    tol : absolute tolerance.

    Returns
    -------
    bool – True if the identity holds within *tol*.
    """
    n = len(alphas)
    expected = n - 2
    actual = alphas.sum()
    ok = abs(actual - expected) < tol
    if ok:
        logger.info("Angle sum check PASSED: Σαₖ = %.6f ≈ %d  (n=%d)", actual, expected, n)
    else:
        logger.warning(
            "Angle sum check FAILED: Σαₖ = %.6f ≠ %d  (n=%d, err=%.2e)",
            actual, expected, n, abs(actual - expected),
        )
    return ok


def sc_exponents(alphas: np.ndarray) -> np.ndarray:
    """Return the SC integrand exponents βₖ = αₖ − 1.

    These appear in the product  ∏ₖ (t − ζₖ)^βₖ  inside the SC integral.
    """
    return alphas - 1.0
