"""
visualization.py
================
Publication-quality figures for the SC fluid-flow project.

Produces three figures:
    1) Simplified vs. original polygon comparison.
    2) Streamlines (ψ = const) overlaid on the Boulder boundary.
    3) Equipotential lines (φ = const) overlaid on the Boulder boundary.
    4) Combined streamlines + equipotential contours.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────

BOUNDARY_COLOR = "#1A3A6B"
FILL_COLOR = "#F0F4FA"
STREAM_COLOR = "#2563A8"
EQUIP_COLOR = "#B04020"
FIG_DIR = Path("figures")


def _ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure 1: Polygon comparison ─────────────────────────────────────────

def plot_polygon_comparison(
    original: Polygon,
    simplified: Polygon,
    save: bool = True,
    filename: str = "fig1_polygon_comparison.png",
) -> plt.Figure:
    """Side-by-side plot: original high-res boundary vs. simplified polygon."""
    _ensure_fig_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for ax, geom, title, n_verts in zip(
        axes,
        [original, simplified],
        ["Original Boundary", "Simplified Polygon"],
        [
            len(original.exterior.coords) - 1,
            len(simplified.exterior.coords) - 1,
        ],
    ):
        x, y = geom.exterior.xy
        ax.fill(x, y, color=FILL_COLOR, zorder=0)
        ax.plot(x, y, color=BOUNDARY_COLOR, lw=1.5, zorder=1)
        ax.scatter(x[:-1], y[:-1], c=BOUNDARY_COLOR, s=15, zorder=2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  ({n_verts} vertices)", fontsize=12)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Boulder City Boundary — Original vs. Simplified",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── Figure 2: Streamlines ────────────────────────────────────────────────

def plot_streamlines(
    XX: np.ndarray,
    YY: np.ndarray,
    Psi: np.ndarray,
    simplified: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig2_streamlines.png",
) -> plt.Figure:
    """Contour plot of the stream function ψ inside the Boulder polygon."""
    _ensure_fig_dir()

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    # Boundary
    x_b, y_b = simplified.exterior.xy
    ax.fill(x_b, y_b, color=FILL_COLOR, zorder=0)
    ax.plot(x_b, y_b, color=BOUNDARY_COLOR, lw=2.0, zorder=4)

    # Streamlines
    valid = np.isfinite(Psi)
    if valid.any():
        psi_min, psi_max = np.nanmin(Psi), np.nanmax(Psi)
        levels = np.linspace(psi_min, psi_max, n_levels + 2)[1:-1]
        cs = ax.contour(
            XX, YY, Psi,
            levels=levels,
            colors=STREAM_COLOR,
            linewidths=0.9,
            zorder=3,
        )
    else:
        logger.warning("No valid ψ values — cannot draw streamlines.")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Ideal Flow Streamlines — Boulder City Polygon",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── Figure 3: Equipotential lines ────────────────────────────────────────

def plot_equipotentials(
    XX: np.ndarray,
    YY: np.ndarray,
    Phi: np.ndarray,
    simplified: Polygon,
    n_levels: int = 30,
    save: bool = True,
    filename: str = "fig3_equipotentials.png",
) -> plt.Figure:
    """Contour plot of the velocity potential φ inside the polygon."""
    _ensure_fig_dir()

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    x_b, y_b = simplified.exterior.xy
    ax.fill(x_b, y_b, color=FILL_COLOR, zorder=0)
    ax.plot(x_b, y_b, color=BOUNDARY_COLOR, lw=2.0, zorder=4)

    valid = np.isfinite(Phi)
    if valid.any():
        phi_min, phi_max = np.nanmin(Phi), np.nanmax(Phi)
        levels = np.linspace(phi_min, phi_max, n_levels + 2)[1:-1]
        ax.contour(
            XX, YY, Phi,
            levels=levels,
            colors=EQUIP_COLOR,
            linewidths=0.9,
            zorder=3,
        )
    else:
        logger.warning("No valid φ values — cannot draw equipotentials.")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Equipotential Lines — Boulder City Polygon",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig


# ── Figure 4 (optional): Combined ────────────────────────────────────────

def plot_combined(
    XX: np.ndarray,
    YY: np.ndarray,
    Psi: np.ndarray,
    Phi: np.ndarray,
    simplified: Polygon,
    n_levels: int = 25,
    save: bool = True,
    filename: str = "fig4_combined.png",
) -> plt.Figure:
    """Overlay streamlines (blue) and equipotentials (red) on the polygon."""
    _ensure_fig_dir()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    x_b, y_b = simplified.exterior.xy
    ax.fill(x_b, y_b, color=FILL_COLOR, zorder=0)
    ax.plot(x_b, y_b, color=BOUNDARY_COLOR, lw=2.0, zorder=4)

    for data, color, label in [
        (Psi, STREAM_COLOR, "Streamlines (ψ)"),
        (Phi, EQUIP_COLOR, "Equipotentials (φ)"),
    ]:
        valid = np.isfinite(data)
        if valid.any():
            d_min, d_max = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(d_min, d_max, n_levels + 2)[1:-1]
            ax.contour(
                XX, YY, data,
                levels=levels,
                colors=color,
                linewidths=0.7,
                zorder=3,
            )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Streamlines & Equipotentials — Boulder City Polygon",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        logger.info("Saved %s", FIG_DIR / filename)
    return fig
