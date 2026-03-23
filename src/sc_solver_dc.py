"""
sc_solver_dc.py
===============
Doubly-connected flow: urban core modelled as an interior obstacle.

Mathematical approach
---------------------
Step 1.  The outer SC map  f : ℍ → Ω_outer  is already solved by
         sc_solver.py.  In ℍ the outer boundary collapses to ℝ and the
         no-penetration condition ψ = 0 on ℝ is automatic.

Step 2.  We map the inner (urban) polygon boundary into ℍ by applying
         the inverse SC map to each of its vertices, obtaining a closed
         curve  C ⊂ ℍ.

Step 3.  We fit the minimum-enclosing circle of C:
             centre ζ₀ ∈ ℍ,   radius a > 0.

Step 4.  We add the potential for a circular obstacle of radius a
         centred at ζ₀ inside ℍ, using the circle theorem plus a
         method-of-images term so that ψ = 0 is preserved on ℝ:

             W(ζ) = U·ζ  +  U·a²/(ζ − ζ₀)  +  U·a²/(ζ − ζ̄₀)

         On ℝ  the last two terms are complex conjugates → imaginary
         parts cancel → ψ = 0 ✓.
         On |ζ − ζ₀| = a  the term U·a²/(ζ − ζ₀) = U·a·e^{−iθ} is
         purely real → Im(W) ≈ U·Im(ζ₀) + O(a/Im(ζ₀)) = const ✓.

Approximation quality
---------------------
The error in the no-penetration condition on the actual curve C (vs.
the fitted circle) is O(ε), where ε is the maximum relative deviation
of C from the circle.  For a compact, roughly convex urban core the
error is typically small.

The error from the image term (which breaks the circle condition
slightly) is O((a/Im(ζ₀))²), negligible when the obstacle is well
inside ℍ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Polygon

from .sc_solver import SCParameters
from .flow import sc_inverse_single

logger = logging.getLogger(__name__)


# ── Data container ─────────────────────────────────────────────────────────

@dataclass
class UrbanObstacle:
    """Circular obstacle in ℍ representing the urban core.

    Attributes
    ----------
    zeta0  : centre of the enclosing circle in ℍ (Im > 0).
    radius : radius a of the enclosing circle.
    mapped_boundary : array of ζ values on the mapped inner boundary,
                      shape (n₂,).  Useful for diagnostics.
    inner_polygon_norm : inner polygon in the normalised physical frame
                         (same coords as the rest of the flow grid).
    """
    zeta0: complex
    radius: float
    mapped_boundary: np.ndarray
    inner_polygon_norm: Polygon


# ── Main solver ────────────────────────────────────────────────────────────

def compute_urban_obstacle(
    inner_polygon_norm: Polygon,
    outer_params: SCParameters,
    *,
    n_boundary_pts: int = 24,
    radius_margin: float = 1.15,
) -> Optional[UrbanObstacle]:
    """Map the inner polygon into ℍ and fit a bounding circle.

    Parameters
    ----------
    inner_polygon_norm : urban-core polygon in the *normalised* coordinate
                         frame (same frame as outer SC solver).
    outer_params       : solved outer SC parameters.
    n_boundary_pts     : number of boundary points to map into ℍ.
                         More points → better circle fit.
    radius_margin      : multiply the fitted radius by this factor to
                         guarantee the circle strictly encloses C.

    Returns
    -------
    UrbanObstacle, or None if inverse mapping fails for too many points.
    """
    # Sample points on the inner polygon boundary
    boundary_pts = _sample_boundary(inner_polygon_norm, n_boundary_pts)
    logger.info("Mapping %d inner boundary points to ℍ …", len(boundary_pts))

    mapped = []
    n_fail = 0
    zeta_prev = 0.0 + 0.5j
    for z in boundary_pts:
        zeta = sc_inverse_single(z, outer_params, zeta0=zeta_prev, maxfev=600)
        if zeta is not None and zeta.imag > 1e-6:
            mapped.append(zeta)
            zeta_prev = zeta
        else:
            n_fail += 1

    if n_fail > len(boundary_pts) // 2:
        logger.error(
            "Too many inverse-map failures (%d / %d) — cannot fit obstacle",
            n_fail, len(boundary_pts),
        )
        return None

    logger.info(
        "Inner boundary in ℍ: %d / %d points OK  (%d failed)",
        len(mapped), len(boundary_pts), n_fail,
    )

    mapped_arr = np.array(mapped)
    zeta0, radius = _minimum_enclosing_circle(mapped_arr)
    radius *= radius_margin

    if zeta0.imag <= radius:
        logger.warning(
            "Obstacle circle (ζ₀=%.3f+%.3fj, a=%.3f) touches or crosses ℝ — "
            "lifting centre upward",
            zeta0.real, zeta0.imag, radius,
        )
        # Lift centre so Im(ζ₀) > radius (obstacle entirely in ℍ)
        zeta0 = zeta0.real + (radius + 0.05) * 1j

    logger.info(
        "Urban obstacle circle: ζ₀ = %.4f + %.4fj,  a = %.4f",
        zeta0.real, zeta0.imag, radius,
    )
    logger.info(
        "Separation Im(ζ₀)/a = %.2f  (error ~ (a/Im)² ≈ %.1e)",
        zeta0.imag / radius,
        (radius / zeta0.imag) ** 2,
    )

    return UrbanObstacle(
        zeta0=zeta0,
        radius=radius,
        mapped_boundary=mapped_arr,
        inner_polygon_norm=inner_polygon_norm,
    )


# ── Potential function ─────────────────────────────────────────────────────

def urban_potential(
    zeta: complex,
    U: float,
    obstacle: UrbanObstacle,
) -> complex:
    r"""Complex potential including the urban-core obstacle.

        W(ζ) = U·ζ  +  U·a²/(ζ − ζ₀)  +  U·a²/(ζ − ζ̄₀)

    The first term is uniform flow; the second enforces no-penetration on
    the circle |ζ − ζ₀| = a (circle theorem); the third is its image below
    ℝ, restoring ψ = 0 on ℝ.

    Parameters
    ----------
    zeta     : evaluation point in ℍ.
    U        : free-stream speed.
    obstacle : solved UrbanObstacle.
    """
    z0 = obstacle.zeta0
    a  = obstacle.radius
    a2 = a * a

    d1 = zeta - z0
    d2 = zeta - np.conj(z0)

    # Regularise to avoid division by zero (should not happen for ζ ∈ ℍ \ obstacle)
    if abs(d1) < 1e-12:
        d1 = 1e-12 + 0j
    if abs(d2) < 1e-12:
        d2 = 1e-12 + 0j

    return U * zeta + U * a2 / d1 + U * a2 / d2


def urban_terrain_potential(
    zeta: complex,
    U: float,
    obstacle: UrbanObstacle,
    terrain_sources,
) -> complex:
    """Combined urban obstacle + terrain correction potential.

    W(ζ) = urban_potential(ζ) + Σₖ terrain source/sink terms
    """
    from .terrain import terrain_potential
    # Get terrain part (without U·ζ) by subtracting uniform flow
    W_terrain_full = terrain_potential(zeta, U, terrain_sources)
    W_uniform = U * zeta
    terrain_correction = W_terrain_full - W_uniform

    return urban_potential(zeta, U, obstacle) + terrain_correction


# ── Geometry helpers ───────────────────────────────────────────────────────

def _sample_boundary(poly: Polygon, n_pts: int) -> list[complex]:
    """Return n_pts complex points evenly spaced along the polygon boundary."""
    coords = np.array(poly.exterior.coords)  # includes closing point
    # Compute cumulative arc length
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    sample_lens = np.linspace(0, total, n_pts, endpoint=False)
    pts = []
    for s in sample_lens:
        # Find which segment
        idx = np.searchsorted(cumlen, s, side="right") - 1
        idx = min(idx, len(coords) - 2)
        t = (s - cumlen[idx]) / max(seg_lengths[idx], 1e-15)
        p = coords[idx] * (1 - t) + coords[idx + 1] * t
        pts.append(p[0] + 1j * p[1])
    return pts


def _minimum_enclosing_circle(
    points: np.ndarray,
) -> tuple[complex, float]:
    """Compute an enclosing circle for a set of complex points.

    Uses the centroid as centre and max distance as radius.
    This is not the minimum enclosing circle in the strict sense
    but is simple and sufficient for our application.
    """
    cx = points.real.mean()
    cy = points.imag.mean()
    centre = cx + 1j * cy
    radius = float(np.max(np.abs(points - centre)))
    return centre, radius
