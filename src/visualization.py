"""
visualization.py
================
Publication-quality figures for the SC fluid-flow project.

Figure 1 - Original vs. simplified polygon (UTM coords).
Figure 2 - Streamlines  ψ = const  (normalised coords).
Figure 3 - Equipotential lines  φ = const  (normalised coords).
Figure 4 - Combined overlay of streamlines + equipotentials.
Figure 5 - Terrain-informed streamlines (when --terrain is used).
Figure 6 - Side-by-side: uniform flow vs. terrain flow.
Figure 7 - Urban-obstacle doubly-connected flow (when --urban is used).
Figure 8 - Three-way comparison: uniform / terrain / urban.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

BOUNDARY_COLOR = "#1A3A6B"
FILL_COLOR     = "#F0F4FA"
STREAM_COLOR   = "#2563A8"
EQUIP_COLOR    = "#B04020"
TERRAIN_STREAM = "#1B7340"   # green for terrain streamlines
TERRAIN_EQUIP  = "#9B5B00"   # amber for terrain equipotentials
URBAN_BOUNDARY = "#5C2D91"   # deep purple for urban core boundary
URBAN_FILL     = "#EDE7F6"   # lavender for urban core fill
URBAN_STREAM   = "#1A5276"   # dark blue for urban-obstacle streamlines
URBAN_EQUIP    = "#7B241C"   # dark red for urban-obstacle equipotentials
FIG_DIR        = Path("figures")


def _ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure 1 ──────────────────────────────────────────────────────────────

def plot_polygon_comparison(
    original: Polygon,
    simplified: Polygon,
    save: bool = True,
    filename: str = "fig1_polygon_comparison.png",
) -> plt.Figure:
    """Side-by-side: original vs. simplified polygon (in UTM coords)."""
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for ax, geom, title in zip(
        axes,
        [original, simplified],
        [
            f"Original ({len(original.exterior.coords)-1} vertices)",
            f"Simplified ({len(simplified.exterior.coords)-1} vertices)",
        ],
    ):
        x, y = geom.exterior.xy
        ax.fill(x, y, color=FILL_COLOR)
        ax.plot(x, y, color=BOUNDARY_COLOR, lw=1.5)
        ax.scatter(list(x)[:-1], list(y)[:-1], c=BOUNDARY_COLOR, s=18, zorder=3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=8)

    fig.suptitle("Boulder City Boundary - Original vs. Simplified",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


# ── helper: draw boundary in normalised coords ───────────────────────────

def _draw_boundary(ax, norm_polygon: Polygon):
    x, y = norm_polygon.exterior.xy
    ax.fill(x, y, color=FILL_COLOR, zorder=0)
    ax.plot(x, y, color=BOUNDARY_COLOR, lw=2.0, zorder=5)


def _draw_urban(ax, inner_polygon: Optional[Polygon]):
    """Draw the urban-core obstacle (filled + outlined)."""
    if inner_polygon is None:
        return
    x, y = inner_polygon.exterior.xy
    ax.fill(x, y, color=URBAN_FILL, zorder=4)
    ax.plot(x, y, color=URBAN_BOUNDARY, lw=1.8, zorder=6)
    # Label
    cx = inner_polygon.centroid.x
    cy = inner_polygon.centroid.y
    ax.text(cx, cy, "urban\ncore", ha="center", va="center",
            fontsize=7, color=URBAN_BOUNDARY, fontweight="bold", zorder=7)


# ── helpers for parametric curves ─────────────────────────────────────────

def _draw_curves(ax, curves, color, lw=0.9):
    """Plot a list of (x_arr, y_arr) parametric curves."""
    for xc, yc in curves:
        if len(xc) > 1:
            ax.plot(xc, yc, color=color, linewidth=lw, zorder=3)


# ── Figure 2 ──────────────────────────────────────────────────────────────

def plot_streamlines(
    XX, YY, Psi,
    norm_polygon: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig2_streamlines.png",
    stream_curves=None,
) -> plt.Figure:
    """Streamline plot.

    If *stream_curves* is provided (list of (x, y) arrays from the forward
    SC map), they are plotted directly as parametric curves - this gives
    artifact-free results without any grid interpolation.  Otherwise falls
    back to a contour plot of *Psi*.
    """
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    _draw_boundary(ax, norm_polygon)

    if stream_curves is not None:
        _draw_curves(ax, stream_curves, STREAM_COLOR, lw=0.9)
    else:
        valid = np.isfinite(Psi)
        if valid.any():
            lo, hi = np.nanmin(Psi), np.nanmax(Psi)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, Psi, levels=levels,
                       colors=STREAM_COLOR, linewidths=0.9, zorder=3)
        else:
            logger.warning("No valid ψ data for streamlines.")

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Streamlines (ψ = const) - Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


# ── Figure 3 ──────────────────────────────────────────────────────────────

def plot_equipotentials(
    XX, YY, Phi,
    norm_polygon: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig3_equipotentials.png",
    equip_curves=None,
) -> plt.Figure:
    """Equipotential line plot.

    Accepts optional *equip_curves* (forward-map parametric curves) for
    artifact-free rendering; falls back to contour plot otherwise.
    """
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    _draw_boundary(ax, norm_polygon)

    if equip_curves is not None:
        _draw_curves(ax, equip_curves, EQUIP_COLOR, lw=0.9)
    else:
        valid = np.isfinite(Phi)
        if valid.any():
            lo, hi = np.nanmin(Phi), np.nanmax(Phi)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, Phi, levels=levels,
                       colors=EQUIP_COLOR, linewidths=0.9, zorder=3)
        else:
            logger.warning("No valid φ data for equipotentials.")

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Equipotential Lines (φ = const) - Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


# ── Figure 4 ──────────────────────────────────────────────────────────────

def plot_combined(
    XX, YY, Psi, Phi,
    norm_polygon: Polygon,
    n_levels: int = 25,
    save: bool = True,
    filename: str = "fig4_combined.png",
    stream_curves=None,
    equip_curves=None,
) -> plt.Figure:
    """Overlay streamlines (blue) + equipotentials (red)."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    _draw_boundary(ax, norm_polygon)

    if stream_curves is not None:
        _draw_curves(ax, stream_curves, STREAM_COLOR, lw=0.7)
    else:
        valid = np.isfinite(Psi)
        if valid.any():
            lo, hi = np.nanmin(Psi), np.nanmax(Psi)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, Psi, levels=levels,
                       colors=STREAM_COLOR, linewidths=0.7, zorder=3)

    if equip_curves is not None:
        _draw_curves(ax, equip_curves, EQUIP_COLOR, lw=0.7)
    else:
        valid = np.isfinite(Phi)
        if valid.any():
            lo, hi = np.nanmin(Phi), np.nanmax(Phi)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, Phi, levels=levels,
                       colors=EQUIP_COLOR, linewidths=0.7, zorder=3)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Streamlines & Equipotentials - Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Terrain-informed figures (Figures 5 & 6)
# ══════════════════════════════════════════════════════════════════════════

def plot_terrain_combined(
    XX, YY, Psi_t, Phi_t,
    norm_polygon: Polygon,
    terrain_info=None,
    z_poly: Optional[np.ndarray] = None,
    n_levels: int = 25,
    save: bool = True,
    filename: str = "fig5_terrain_flow.png",
) -> plt.Figure:
    """Terrain-informed streamlines (green) + equipotentials (amber).

    Annotates with source/sink markers and gradient arrow.
    """
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    _draw_boundary(ax, norm_polygon)

    for data, color in [
        (Psi_t, TERRAIN_STREAM),
        (Phi_t, TERRAIN_EQUIP),
    ]:
        valid = np.isfinite(data)
        if valid.any():
            lo, hi = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, data, levels=levels,
                       colors=color, linewidths=0.7, zorder=3)

    # Mark high / low elevation vertices and gradient arrow
    if terrain_info is not None and z_poly is not None:
        elevations = terrain_info.elevations
        i_high = int(np.argmax(elevations))
        i_low  = int(np.argmin(elevations))
        ax.plot(z_poly[i_high].real, z_poly[i_high].imag, "r^",
                ms=12, zorder=10, label=f"High ({elevations[i_high]:.0f} m)")
        ax.plot(z_poly[i_low].real,  z_poly[i_low].imag,  "bv",
                ms=12, zorder=10, label=f"Low ({elevations[i_low]:.0f} m)")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

        # Gradient arrow at centre
        valid_psi = np.isfinite(Psi_t)
        if valid_psi.any():
            cx = np.mean(XX[valid_psi])
            cy = np.mean(YY[valid_psi])
            theta = terrain_info.theta_downhill
            arr_len = 0.18
            ax.annotate(
                "",
                xy=(cx + arr_len * np.cos(theta),
                    cy + arr_len * np.sin(theta)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.0),
                zorder=10,
            )
            ax.text(cx, cy - 0.12, "downhill", ha="center",
                    fontsize=9, color="#333333")

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Terrain-Informed Flow - Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


def plot_flow_comparison(
    XX, YY,
    Psi_left, Phi_left,
    Psi_right, Phi_right,
    norm_polygon: Polygon,
    n_levels: int = 22,
    save: bool = True,
    filename: str = "fig6_flow_comparison.png",
    title_left: str = r"Uniform Flow  $W = U\zeta$",
    title_right: str = r"Terrain-Informed Flow  $W = U\zeta + $ sources",
    suptitle: str = "Flow Comparison - Uniform vs. Terrain-Informed",
    stream_color_right: str = TERRAIN_STREAM,
    equip_color_right: str = TERRAIN_EQUIP,
) -> plt.Figure:
    """Side-by-side flow comparison.

    Parameters
    ----------
    title_left / title_right : panel titles (override for non-terrain comparisons).
    suptitle                 : figure-level title.
    stream_color_right       : streamline colour for the right panel.
    equip_color_right        : equipotential colour for the right panel.
    """
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

    titles = [title_left, title_right]
    psi_list = [Psi_left, Psi_right]
    phi_list = [Phi_left, Phi_right]
    stream_colors = [STREAM_COLOR, stream_color_right]
    equip_colors  = [EQUIP_COLOR,  equip_color_right]

    for ax, psi, phi, sc, ec, title in zip(
        axes, psi_list, phi_list, stream_colors, equip_colors, titles,
    ):
        _draw_boundary(ax, norm_polygon)
        for data, color in [(psi, sc), (phi, ec)]:
            valid = np.isfinite(data)
            if valid.any():
                lo, hi = np.nanmin(data), np.nanmax(data)
                levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
                ax.contour(XX, YY, data, levels=levels,
                           colors=color, linewidths=0.6, zorder=3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.suptitle(suptitle,
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Urban-obstacle figures (Figures 7 & 8)
# ══════════════════════════════════════════════════════════════════════════

def plot_urban_flow(
    XX, YY, Psi_u, Phi_u,
    norm_polygon_outer: Polygon,
    norm_polygon_inner: Optional[Polygon] = None,
    obstacle=None,
    n_levels: int = 28,
    save: bool = True,
    filename: str = "fig7_urban_flow.png",
) -> plt.Figure:
    """Streamlines + equipotentials for the doubly-connected domain.

    The urban core is drawn as a filled purple obstacle.  Streamlines
    visibly deflect around it, demonstrating the no-penetration condition.
    If *obstacle* is provided, the mapped boundary in ℍ is annotated as
    diagnostic metadata.
    """
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    _draw_boundary(ax, norm_polygon_outer)
    _draw_urban(ax, norm_polygon_inner)

    for data, color, lw in [
        (Psi_u, URBAN_STREAM, 0.9),
        (Phi_u, URBAN_EQUIP,  0.6),
    ]:
        valid = np.isfinite(data)
        if valid.any():
            lo, hi = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, data, levels=levels,
                       colors=color, linewidths=lw, zorder=3)

    # Annotate circle approximation in ℍ → physical coords (informational)
    if obstacle is not None and norm_polygon_inner is not None:
        cx = norm_polygon_inner.centroid.x
        cy = norm_polygon_inner.centroid.y
        circle = plt.Circle(
            (cx, cy),
            radius=0.02,
            fill=False,
            linestyle=":",
            edgecolor=URBAN_BOUNDARY,
            linewidth=0.8,
            zorder=8,
        )
        ax.add_patch(circle)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(
        "Doubly-Connected Flow - Urban Core as Interior Obstacle\n"
        r"$W(\zeta) = U\zeta + Ua^2/(\zeta-\zeta_0) + Ua^2/(\zeta-\bar\zeta_0)$",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig


def plot_three_way_comparison(
    XX, YY,
    Psi_uniform,
    Psi_terrain,
    Psi_urban,
    norm_polygon_outer: Polygon,
    norm_polygon_inner: Optional[Polygon] = None,
    n_levels: int = 22,
    save: bool = True,
    filename: str = "fig8_three_way_comparison.png",
) -> plt.Figure:
    """Three-panel comparison: uniform / terrain-corrected / urban obstacle.

    This is the key summary figure for the project report, showing how
    each successive modelling enhancement changes the streamline pattern.
    """
    _ensure_fig_dir()
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)

    configs = [
        (Psi_uniform,  STREAM_COLOR,   r"(a) Uniform  $W=U\zeta$",           None),
        (Psi_terrain,  TERRAIN_STREAM, r"(b) Terrain-corrected  (+RBF sources)", None),
        (Psi_urban,    URBAN_STREAM,   r"(c) Urban obstacle  (doubly-connected)", norm_polygon_inner),
    ]

    for ax, (psi, color, title, inner) in zip(axes, configs):
        _draw_boundary(ax, norm_polygon_outer)
        if inner is not None:
            _draw_urban(ax, inner)

        valid = np.isfinite(psi)
        if valid.any():
            lo, hi = np.nanmin(psi), np.nanmax(psi)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, psi, levels=levels,
                       colors=color, linewidths=0.8, zorder=3)

        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Streamline Comparison - Boulder City Polygon\n"
        "Schwarz-Christoffel Conformal Mapping with Progressive Physical Enhancements",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        pdf_name = Path(filename).with_suffix(".pdf").name
        fig.savefig(FIG_DIR / pdf_name, bbox_inches="tight")
        logger.info("Saved %s + %s", FIG_DIR / filename, FIG_DIR / pdf_name)
    return fig
