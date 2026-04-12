#!/usr/bin/env python
"""
main.py - Full pipeline for the SC conformal-mapping fluid-flow project.

Usage
-----
    python main.py --shapefile data/raw/tl_2025_08_place
    python main.py --demo              # built-in hexagon, no shapefile needed
    python main.py --demo --grid 40    # quick test
    python main.py --shapefile data/raw/tl_2025_08_place --terrain
    python main.py --shapefile data/raw/tl_2025_08_place --urban
    python main.py --shapefile data/raw/tl_2025_08_place --roads
    python main.py --shapefile data/raw/tl_2025_08_place --terrain --urban --roads
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
    smooth_extreme_angles,
)
from src.angles import interior_angles_pi, verify_angle_sum, sc_exponents
from src.sc_solver import solve_parameters, sc_map, SCParameters
from src.flow import compute_flow_grid, compute_flow_grid_urban, compute_curves_forward
from src.visualization import (
    plot_polygon_comparison,
    plot_streamlines,
    plot_equipotentials,
    plot_combined,
    plot_terrain_combined,
    plot_flow_comparison,
    plot_urban_flow,
    plot_road_flow,
    plot_three_way_comparison,
    plot_four_way_comparison,
    URBAN_STREAM,
    URBAN_EQUIP,
    ROAD_STREAM,
    ROAD_EQUIP,
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
    tolerance: float = 1500.0,
    min_vertices: int = 12,
    max_vertices: int = 16,
    terrain: bool = False,
    urban: bool = False,
    roads: bool = False,
    urban_method: str = "osmnx",
    urban_vertices: int = 8,
    road_method: str = "osmnx",
    road_n_max: int = 12,
    n_per_edge: int = 3,
    n_interior: int = 25,
    max_workers: int = 6,
):
    # ══════════════════════════════════════════════════════════════════
    # STEP 1 - Load & simplify polygon
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
    z_poly = smooth_extreme_angles(z_poly, alpha_min=0.35, alpha_max=1.75, min_vertices=10)
    n = len(z_poly)
    logger.info("Polygon: %d vertices  (center=%.1f%+.1fj, scale=%.1f)",
                n, center.real, center.imag, scale)

    # ── Build Shapely polygons in normalised coords AND UTM ──
    norm_polygon = complex_to_polygon(z_poly)

    # Rebuild UTM polygon from the angle-filtered vertices so that
    # compute_terrain_info sees exactly the same n vertices as sc_params.zk
    from shapely.geometry import Polygon as ShapelyPolygon
    z_utm = z_poly * scale + center
    simplified_utm = ShapelyPolygon([(z.real, z.imag) for z in z_utm])

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 - Interior angles
    # ══════════════════════════════════════════════════════════════════
    alphas = interior_angles_pi(z_poly)
    logger.info("αₖ (×π): %s", np.array2string(alphas, precision=4))
    if not verify_angle_sum(alphas):
        logger.error("Angle-sum check failed - aborting.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 - SC parameter problem
    # ══════════════════════════════════════════════════════════════════
    params = solve_parameters(z_poly, alphas)
    logger.info("ζₖ = %s", np.array2string(params.zk, precision=6))

    # ── Sanity check: map near-pre-vertices (slightly above real axis) ──
    # Evaluating sc_map exactly on the real axis triggers branch-point
    # singularities in the integrand; use a small lift to get accurate values.
    logger.info("Forward-map sanity check …")
    mapped = sc_map(params.zk + 1e-3j, params)
    max_err = 0.0
    for k in range(n):
        err = abs(mapped[k] - z_poly[k])
        max_err = max(max_err, err)
        logger.info("  v%d: target=%.4f%+.4fj  mapped=%.4f%+.4fj  err=%.2e",
                     k, z_poly[k].real, z_poly[k].imag,
                     mapped[k].real, mapped[k].imag, err)
    logger.info("Max vertex error: %.2e", max_err)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 - Flow grid (in normalised coords)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Computing uniform flow grid (%d × %d) …", n_grid, n_grid)
    XX, YY, Psi, Phi, Zeta = compute_flow_grid(norm_polygon, params, n_grid=n_grid)

    n_sol = np.isfinite(Psi).sum()
    logger.info("Flow grid: %d points with valid ψ/φ", n_sol)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4b - Terrain-informed flow (optional)
    # ══════════════════════════════════════════════════════════════════
    terrain_info = None
    Psi_t = Phi_t = None

    if terrain and not demo:
        from src.terrain import compute_terrain_info, terrain_potential

        logger.info("── Terrain mode enabled ──")
        terrain_info = compute_terrain_info(
            simplified_utm, params,
            n_per_edge=n_per_edge,
            n_interior=n_interior,
            max_workers=max_workers,
        )

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
    # STEP 4c - Urban-obstacle doubly-connected flow (optional)
    # ══════════════════════════════════════════════════════════════════
    urban_obstacle  = None
    norm_poly_inner = None
    Psi_u = Phi_u   = None

    if urban and not demo:
        from src.urban import get_urban_polygon, polygon_to_complex_inner
        from src.sc_solver_dc import compute_urban_obstacle
        from shapely.geometry import Polygon as ShapelyPolygon

        logger.info("── Urban mode enabled ──")
        urban_poly_utm = get_urban_polygon(
            simplified_utm,
            method=urban_method,
            n_vertices=urban_vertices,
        )

        if urban_poly_utm is None:
            logger.error("Could not obtain urban polygon - skipping urban mode")
        else:
            # Convert inner polygon to normalised frame
            z_inner = polygon_to_complex_inner(urban_poly_utm, center, scale)
            norm_poly_inner = ShapelyPolygon(
                [(z.real, z.imag) for z in z_inner]
            )

            # Fit circular obstacle in ℍ
            urban_obstacle = compute_urban_obstacle(
                norm_poly_inner, params, n_boundary_pts=24,
            )

            if urban_obstacle is None:
                logger.error("Urban obstacle fitting failed - skipping")
            else:
                # Build potential (with or without terrain correction)
                terrain_sources = terrain_info.sources if terrain_info else None

                logger.info("Computing urban doubly-connected flow grid …")
                XX_u, YY_u, Psi_u, Phi_u, _ = compute_flow_grid_urban(
                    norm_polygon, norm_poly_inner, params,
                    urban_obstacle,
                    n_grid=n_grid,
                    terrain_sources=terrain_sources,
                )
                n_u = int(np.isfinite(Psi_u).sum())
                logger.info("Urban flow grid: %d points with valid ψ/φ", n_u)
    elif urban and demo:
        logger.warning("--urban requires a real shapefile; skipped in demo mode")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4d - Road-vortex flow (optional)
    # ══════════════════════════════════════════════════════════════════
    road_info  = None
    Psi_r = Phi_r = None

    if roads and not demo:
        from src.roads import compute_road_info

        logger.info("── Road mode enabled ──")
        road_info = compute_road_info(
            simplified_utm,
            center=center,
            scale=scale,
            sc_params=params,
            method=road_method,
            n_max=road_n_max,
        )

        if road_info is None:
            logger.error("Road vortex computation failed - skipping road model")
        else:
            # Build road potential (terrain correction optional)
            terrain_sources = terrain_info.sources if terrain_info else None
            if terrain_sources:
                from src.sc_solver_dc import road_terrain_potential
                road_pot = lambda zeta: road_terrain_potential(
                    zeta, U=1.0,
                    terrain_sources=terrain_sources,
                    road_vortices=road_info.vortices,
                )
            else:
                from src.roads import road_potential
                road_pot = lambda zeta: road_potential(zeta, U=1.0,
                                                       vortices=road_info.vortices)

            logger.info("Computing road-vortex flow grid …")
            _, _, Psi_r, Phi_r, _ = compute_flow_grid(
                norm_polygon, params, n_grid=n_grid,
                potential_fn=road_pot, zeta_cache=Zeta,
            )
            n_r = int(np.isfinite(Psi_r).sum())
            logger.info("Road flow grid: %d points with valid ψ/φ", n_r)
    elif roads and demo:
        logger.warning("--roads requires a real shapefile; skipped in demo mode")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 - Figures
    # ══════════════════════════════════════════════════════════════════
    logger.info("Generating figures …")

    # Fig 1: UTM coords (original vs simplified)
    plot_polygon_comparison(original_utm, simplified_utm)

    # Compute exact parametric streamlines / equipotentials via forward map
    stream_curves, equip_curves = compute_curves_forward(
        params, norm_polygon, n_stream=28, n_equip=28,
    )

    # Figs 2-4: normalised coords (uniform flow)
    plot_streamlines(XX, YY, Psi, norm_polygon, stream_curves=stream_curves)
    plot_equipotentials(XX, YY, Phi, norm_polygon, equip_curves=equip_curves)
    plot_combined(XX, YY, Psi, Phi, norm_polygon,
                  stream_curves=stream_curves, equip_curves=equip_curves)

    # Figs 5-6: terrain (if computed)
    if Psi_t is not None:
        plot_terrain_combined(
            XX, YY, Psi_t, Phi_t, norm_polygon,
            terrain_info=terrain_info, z_poly=z_poly,
        )
        plot_flow_comparison(
            XX, YY, Psi, Phi, Psi_t, Phi_t, norm_polygon,
        )

    # Figs 7-8: urban obstacle (if computed)
    if Psi_u is not None:
        # Pass uniform-flow ψ range so the urban panel uses the same contour
        # levels as the baseline, suppressing the dipole spike near ζ₀.
        psi_ref = (float(np.nanpercentile(Psi, 5)),
                   float(np.nanpercentile(Psi, 95))) if Psi is not None else None
        plot_urban_flow(
            XX_u, YY_u, Psi_u, Phi_u, norm_polygon,
            norm_polygon_inner=norm_poly_inner,
            obstacle=urban_obstacle,
            psi_ref_range=psi_ref,
        )
        # Three-way comparison only when both terrain and urban were run
        if Psi_t is not None:
            plot_three_way_comparison(
                XX, YY, Psi, Psi_t, Psi_u,
                norm_polygon, norm_poly_inner,
            )
        else:
            from src.visualization import plot_flow_comparison as _pfc
            _pfc(
                XX, YY, Psi, Phi, Psi_u, Phi_u, norm_polygon,
                filename="fig8_urban_vs_uniform.png",
                title_right=r"Urban-Obstacle Flow  (doubly-connected)",
                suptitle="Flow Comparison - Uniform vs. Urban Obstacle",
                stream_color_right=URBAN_STREAM,
                equip_color_right=URBAN_EQUIP,
            )

    # Figs 9-10: road-vortex flow (if computed)
    if Psi_r is not None:
        plot_road_flow(
            XX, YY, Psi_r, Phi_r, norm_polygon,
            road_info=road_info,
            norm_center=center,
            norm_scale=scale,
        )
        # Side-by-side: uniform vs road-vortex
        from src.visualization import plot_flow_comparison as _pfc
        _pfc(
            XX, YY, Psi, Phi, Psi_r, Phi_r, norm_polygon,
            filename="fig10_road_vs_uniform.png",
            title_right=r"Road-Vortex Flow  (OSM intersections)",
            suptitle="Flow Comparison - Uniform vs. Road-Vortex",
            stream_color_right=ROAD_STREAM,
            equip_color_right=ROAD_EQUIP,
        )
        # Four-way comparison when all three enhancements were run
        if Psi_t is not None and Psi_u is not None:
            plot_four_way_comparison(
                XX, YY, Psi, Psi_t, Psi_u, Psi_r,
                norm_polygon, norm_poly_inner,
            )

    logger.info("Done - figures saved to figures/")


def main():
    p = argparse.ArgumentParser(description="SC conformal mapping - ideal flow")
    p.add_argument("--shapefile", type=str, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--terrain", action="store_true",
                   help="Enable DEM terrain-informed flow (requires shapefile)")
    p.add_argument("--urban", action="store_true",
                   help="Enable doubly-connected urban-obstacle flow (requires shapefile)")
    p.add_argument("--urban-method", type=str, default="osmnx",
                   choices=["osmnx", "fallback"],
                   help="Data source for urban polygon (default: osmnx)")
    p.add_argument("--urban-vertices", type=int, default=8,
                   help="Target vertex count for simplified urban polygon")
    p.add_argument("--roads", action="store_true",
                   help="Enable road-vortex flow from OSM intersection data (requires shapefile)")
    p.add_argument("--road-method", type=str, default="osmnx",
                   choices=["osmnx", "fallback"],
                   help="Data source for road intersections (default: osmnx)")
    p.add_argument("--road-n-max", type=int, default=12,
                   help="Maximum number of road intersections to use as vortices")
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--tolerance", type=float, default=1500.0)
    p.add_argument("--min-vertices", type=int, default=12)
    p.add_argument("--max-vertices", type=int, default=16)
    p.add_argument("--n-per-edge", type=int, default=3,
                   help="Extra elevation sample points per polygon edge")
    p.add_argument("--n-interior", type=int, default=25,
                   help="Interior elevation sample points")
    p.add_argument("--max-workers", type=int, default=6,
                   help="Concurrent API query threads")
    a = p.parse_args()

    run_pipeline(
        shapefile=a.shapefile, demo=a.demo, n_grid=a.grid,
        tolerance=a.tolerance, min_vertices=a.min_vertices,
        max_vertices=a.max_vertices, terrain=a.terrain,
        urban=a.urban, urban_method=a.urban_method,
        urban_vertices=a.urban_vertices,
        roads=a.roads, road_method=a.road_method,
        road_n_max=a.road_n_max,
        n_per_edge=a.n_per_edge, n_interior=a.n_interior,
        max_workers=a.max_workers,
    )


if __name__ == "__main__":
    main()
