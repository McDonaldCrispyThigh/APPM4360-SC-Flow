#!/usr/bin/env python
"""
main.py — Full pipeline for the SC conformal-mapping fluid-flow project.

Usage
-----
    python main.py --shapefile data/raw/tl_2025_08_place
    python main.py --demo              # built-in hexagon, no shapefile needed
    python main.py --demo --grid 40    # quick test
    python main.py --shapefile data/raw/tl_2025_08_place --terrain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.polygon import (
    load_boulder_polygon,
    simplify_polygon,
    polygon_to_complex,
    complex_to_polygon,
    ensure_ccw,
)
from src.angles import interior_angles_pi, verify_angle_sum, sc_exponents
from src.sc_solver import solve_parameters, sc_map, SCParameters
from src.flow import compute_flow_grid
from src.visualization import (
    plot_polygon_comparison,
    plot_streamlines,
    plot_equipotentials,
    plot_combined,
    plot_terrain_combined,
    plot_flow_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _make_demo_polygon():
    from shapely.geometry import Polygon as ShapelyPolygon
    coords = [
        (-0.8, -0.4), (-0.3, -0.9), (0.5, -0.7),
        (0.9, 0.1),   (0.4, 0.8),   (-0.6, 0.6),
    ]
    return ShapelyPolygon(coords)


def run_pipeline(
    shapefile=None,
    demo: bool = False,
    n_grid: int = 80,
    tolerance: float = 150.0,
    min_vertices: int = 10,
    max_vertices: int = 20,
    terrain: bool = False,
):
    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Load & simplify polygon
    # ══════════════════════════════════════════════════════════════════
    if demo:
        logger.info("=== DEMO MODE ===")
        simplified_utm = _make_demo_polygon()
        original_utm   = simplified_utm
    else:
        if shapefile is None:
            shapefile = Path("data/raw")
        original_utm = load_boulder_polygon(shapefile)
        simplified_utm = simplify_polygon(
            original_utm,
            tolerance=tolerance,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
        )

    # ── Convert to normalised complex coords ──
    z_poly, center, scale = polygon_to_complex(simplified_utm, normalise=True)
    z_poly = ensure_ccw(z_poly)
    n = len(z_poly)
    logger.info("Polygon: %d vertices  (center=%.1f%+.1fj, scale=%.1f)",
                n, center.real, center.imag, scale)

    # ── Build a Shapely polygon in normalised coords ──
    norm_polygon = complex_to_polygon(z_poly)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — Interior angles
    # ══════════════════════════════════════════════════════════════════
    alphas = interior_angles_pi(z_poly)
    logger.info("αₖ (×π): %s", np.array2string(alphas, precision=4))
    if not verify_angle_sum(alphas):
        logger.error("Angle-sum check failed — aborting.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — SC parameter problem
    # ══════════════════════════════════════════════════════════════════
    params = solve_parameters(z_poly, alphas)
    logger.info("ζₖ = %s", np.array2string(params.zk, precision=6))

    # ── Sanity check: map pre-vertices back ──
    logger.info("Forward-map sanity check …")
    mapped = sc_map(params.zk + 0.0j, params)
    max_err = 0.0
    for k in range(n):
        err = abs(mapped[k] - z_poly[k])
        max_err = max(max_err, err)
        logger.info("  v%d: target=%.4f%+.4fj  mapped=%.4f%+.4fj  err=%.2e",
                     k, z_poly[k].real, z_poly[k].imag,
                     mapped[k].real, mapped[k].imag, err)
    logger.info("Max vertex error: %.2e", max_err)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Flow grid (in normalised coords)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Computing uniform flow grid (%d × %d) …", n_grid, n_grid)
    XX, YY, Psi, Phi, Zeta = compute_flow_grid(norm_polygon, params, n_grid=n_grid)

    n_sol = np.isfinite(Psi).sum()
    logger.info("Flow grid: %d points with valid ψ/φ", n_sol)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4b — Terrain-informed flow (optional)
    # ══════════════════════════════════════════════════════════════════
    terrain_info = None
    Psi_t = Phi_t = None

    if terrain and not demo:
        from src.terrain import compute_terrain_info, terrain_potential

        logger.info("── Terrain mode enabled ──")
        terrain_info = compute_terrain_info(simplified_utm, params)

        # Build the terrain potential function
        sources = terrain_info.sources
        pot_fn = lambda zeta: terrain_potential(zeta, U=1.0, sources=sources)

        logger.info("Computing terrain flow grid (using cached ζ) …")
        _, _, Psi_t, Phi_t, _ = compute_flow_grid(
            norm_polygon, params, n_grid=n_grid,
            potential_fn=pot_fn, zeta_cache=Zeta,
        )
        n_t = np.isfinite(Psi_t).sum()
        logger.info("Terrain flow grid: %d points with valid ψ/φ", n_t)
    elif terrain and demo:
        logger.warning("--terrain requires a real shapefile; skipped in demo mode")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — Figures
    # ══════════════════════════════════════════════════════════════════
    logger.info("Generating figures …")

    # Fig 1: UTM coords (original vs simplified)
    plot_polygon_comparison(original_utm, simplified_utm)

    # Figs 2-4: normalised coords (uniform flow)
    plot_streamlines(XX, YY, Psi, norm_polygon)
    plot_equipotentials(XX, YY, Phi, norm_polygon)
    plot_combined(XX, YY, Psi, Phi, norm_polygon)

    # Figs 5-6: terrain (if computed)
    if Psi_t is not None:
        plot_terrain_combined(
            XX, YY, Psi_t, Phi_t, norm_polygon,
            terrain_info=terrain_info, z_poly=z_poly,
        )
        plot_flow_comparison(
            XX, YY, Psi, Phi, Psi_t, Phi_t, norm_polygon,
        )

    logger.info("Done — figures saved to figures/")


def main():
    p = argparse.ArgumentParser(description="SC conformal mapping – ideal flow")
    p.add_argument("--shapefile", type=str, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--terrain", action="store_true",
                   help="Enable DEM terrain-informed flow (requires shapefile)")
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--tolerance", type=float, default=150.0)
    p.add_argument("--min-vertices", type=int, default=10)
    p.add_argument("--max-vertices", type=int, default=20)
    a = p.parse_args()

    run_pipeline(
        shapefile=a.shapefile, demo=a.demo, n_grid=a.grid,
        tolerance=a.tolerance, min_vertices=a.min_vertices,
        max_vertices=a.max_vertices, terrain=a.terrain,
    )


if __name__ == "__main__":
    main()
