"""
visualization.py
================
Publication-quality figures for the SC fluid-flow project.

Figure 1 — Original vs. simplified polygon (UTM coords).
Figure 2 — Streamlines  ψ = const  (normalised coords).
Figure 3 — Equipotential lines  φ = const  (normalised coords).
Figure 4 — Combined overlay of streamlines + equipotentials.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

BOUNDARY_COLOR = "#1A3A6B"
FILL_COLOR     = "#F0F4FA"
STREAM_COLOR   = "#2563A8"
EQUIP_COLOR    = "#B04020"
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

    fig.suptitle("Boulder City Boundary — Original vs. Simplified",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── helper: draw boundary in normalised coords ───────────────────────────

def _draw_boundary(ax, norm_polygon: Polygon):
    x, y = norm_polygon.exterior.xy
    ax.fill(x, y, color=FILL_COLOR, zorder=0)
    ax.plot(x, y, color=BOUNDARY_COLOR, lw=2.0, zorder=5)


# ── Figure 2 ──────────────────────────────────────────────────────────────

def plot_streamlines(
    XX, YY, Psi,
    norm_polygon: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig2_streamlines.png",
) -> plt.Figure:
    """Contour plot of the stream function ψ."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    _draw_boundary(ax, norm_polygon)

    valid = np.isfinite(Psi)
    if valid.any():
        lo, hi = np.nanmin(Psi), np.nanmax(Psi)
        levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
        ax.contour(XX, YY, Psi, levels=levels,
                   colors=STREAM_COLOR, linewidths=0.9, zorder=3)
    else:
        logger.warning("No valid ψ data for streamlines.")

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Streamlines (ψ = const) — Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── Figure 3 ──────────────────────────────────────────────────────────────

def plot_equipotentials(
    XX, YY, Phi,
    norm_polygon: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig3_equipotentials.png",
) -> plt.Figure:
    """Contour plot of the velocity potential φ."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    _draw_boundary(ax, norm_polygon)

    valid = np.isfinite(Phi)
    if valid.any():
        lo, hi = np.nanmin(Phi), np.nanmax(Phi)
        levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
        ax.contour(XX, YY, Phi, levels=levels,
                   colors=EQUIP_COLOR, linewidths=0.9, zorder=3)
    else:
        logger.warning("No valid φ data for equipotentials.")

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Equipotential Lines (φ = const) — Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── Figure 4 ──────────────────────────────────────────────────────────────

def plot_combined(
    XX, YY, Psi, Phi,
    norm_polygon: Polygon,
    n_levels: int = 25,
    save: bool = True,
    filename: str = "fig4_combined.png",
) -> plt.Figure:
    """Overlay streamlines (blue) + equipotentials (red)."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    _draw_boundary(ax, norm_polygon)

    for data, color in [(Psi, STREAM_COLOR), (Phi, EQUIP_COLOR)]:
        valid = np.isfinite(data)
        if valid.any():
            lo, hi = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
            ax.contour(XX, YY, data, levels=levels,
                       colors=color, linewidths=0.7, zorder=3)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Streamlines & Equipotentials — Boulder Polygon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig
