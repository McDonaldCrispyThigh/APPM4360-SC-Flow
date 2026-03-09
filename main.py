#!/usr/bin/env python
"""
main.py — Full pipeline for the SC conformal-mapping fluid-flow project.

Usage
-----
    python main.py                         # default: use shapefile in data/raw/
    python main.py --shapefile path/to/file.shp
    python main.py --demo                  # use a built-in demo polygon (no shapefile needed)

Steps executed
--------------
1. Load & simplify the Boulder city boundary polygon.
2. Compute interior angles and verify the angle-sum identity.
3. Solve the SC parameter problem (pre-vertex locations).
4. Compute the stream-function / velocity-potential grid.
5. Generate publication-quality figures → figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from src.polygon import (
    load_boulder_polygon,
    simplify_polygon,
    polygon_to_complex,
    ensure_ccw,
    side_length_ratios,
)
from src.angles import interior_angles_pi, verify_angle_sum, sc_exponents
from src.sc_solver import solve_parameters, sc_map, SCParameters
from src.flow import compute_flow_grid
from src.visualization import (
    plot_polygon_comparison,
    plot_streamlines,
    plot_equipotentials,
    plot_combined,
)

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Demo polygon (no shapefile needed) ────────────────────────────────────

def _make_demo_polygon():
    """Return a Shapely polygon roughly resembling a simplified Boulder shape.

    This is an irregular hexagon useful for testing without the shapefile.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    # Approximate Boulder-like shape (normalised coordinates)
    coords = [
        (-0.8, -0.4),
        (-0.3, -0.9),
        ( 0.5, -0.7),
        ( 0.9,  0.1),
        ( 0.4,  0.8),
        (-0.6,  0.6),
    ]
    return ShapelyPolygon(coords)


# ── Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    shapefile: str | None = None,
    demo: bool = False,
    n_grid: int = 80,
    tolerance: float = 500.0,
    max_vertices: int = 14,
):
    """Execute the full SC + fluid-flow pipeline."""

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — Polygon
    # ════════════════════════════════════════════════════════════════════
    if demo:
        logger.info("=== DEMO MODE — using built-in hexagonal polygon ===")
        from shapely.geometry import Polygon as ShapelyPolygon
        simplified = _make_demo_polygon()
        original = simplified  # same for demo
    else:
        if shapefile is None:
            # Default path
            shapefile = Path("data/raw")
        logger.info("Loading Boulder polygon from %s …", shapefile)
        original = load_boulder_polygon(shapefile)
        simplified = simplify_polygon(
            original, tolerance=tolerance, max_vertices=max_vertices,
        )

    z_poly = polygon_to_complex(simplified, normalise=True)
    z_poly = ensure_ccw(z_poly)
    n = len(z_poly)
    logger.info("Polygon: %d vertices", n)

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — Interior angles
    # ════════════════════════════════════════════════════════════════════
    alphas = interior_angles_pi(z_poly)
    betas = sc_exponents(alphas)
    logger.info("Interior angles (×π): %s", np.array2string(alphas, precision=4))
    if not verify_angle_sum(alphas):
        logger.error("CRITICAL: angle sum ≠ n−2.  Check vertex order / polygon.")
        sys.exit(1)

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — SC parameter problem
    # ════════════════════════════════════════════════════════════════════
    logger.info("Solving SC parameter problem …")
    params = solve_parameters(z_poly, alphas)
    logger.info("Pre-vertices ζₖ: %s", np.array2string(params.zk, precision=6))

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — Quick forward-map sanity check
    # ════════════════════════════════════════════════════════════════════
    logger.info("Sanity check: mapping pre-vertices back to polygon …")
    mapped = sc_map(params.zk + 0.0j, params)
    for k in range(n):
        err = abs(mapped[k] - z_poly[k])
        logger.info("  vertex %d: target=%.4f%+.4fj  mapped=%.4f%+.4fj  err=%.2e",
                     k, z_poly[k].real, z_poly[k].imag,
                     mapped[k].real, mapped[k].imag, err)

    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — Flow grid (inverse map)
    # ════════════════════════════════════════════════════════════════════
    logger.info("Computing flow grid (%d × %d) …", n_grid, n_grid)
    XX, YY, Psi, Phi = compute_flow_grid(simplified, params, n_grid=n_grid)

    solved = np.isfinite(Psi).sum()
    total = (XX.shape[0] * XX.shape[1])
    logger.info("Flow grid complete: %d / %d points solved", solved, total)

    # ════════════════════════════════════════════════════════════════════
    # STEP 6 — Figures
    # ════════════════════════════════════════════════════════════════════
    logger.info("Generating figures …")
    plot_polygon_comparison(original, simplified)
    plot_streamlines(XX, YY, Psi, simplified)
    plot_equipotentials(XX, YY, Phi, simplified)
    plot_combined(XX, YY, Psi, Phi, simplified)

    logger.info("✓  Pipeline finished.  Figures saved to figures/")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Schwarz–Christoffel conformal mapping — ideal fluid flow in Boulder polygon"
    )
    parser.add_argument(
        "--shapefile", type=str, default=None,
        help="Path to the TIGER/Line .shp file (or directory containing it).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with a built-in demo polygon (no shapefile required).",
    )
    parser.add_argument(
        "--grid", type=int, default=80,
        help="Grid resolution along each axis (default: 80).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=500.0,
        help="Douglas–Peucker simplification tolerance in metres (default: 500).",
    )
    parser.add_argument(
        "--max-vertices", type=int, default=14,
        help="Maximum number of polygon vertices after simplification (default: 14).",
    )
    args = parser.parse_args()

    run_pipeline(
        shapefile=args.shapefile,
        demo=args.demo,
        n_grid=args.grid,
        tolerance=args.tolerance,
        max_vertices=args.max_vertices,
    )


if __name__ == "__main__":
    main()
