# Ideal Fluid Flow in a Polygonal Domain via the Schwarz–Christoffel Transformation

**APPM 4360 — Complex Variables and Applications, Spring 2026**
Congyuan Zheng, Sophia Arany, Alexander Ingalls
University of Colorado Boulder, Department of Applied Mathematics

---

## Overview

This project applies the **Schwarz–Christoffel (SC) conformal mapping** to simulate
steady, irrotational, incompressible fluid flow inside the Boulder, Colorado city
boundary polygon.  Three progressively richer physical models are implemented:

| Model | Potential W(ζ) | Physical meaning |
|-------|---------------|-----------------|
| Uniform flow | Uζ | Ideal parallel flow with no boundary effects |
| Terrain-corrected | Uζ + Σₖ qₖ log(ζ−sₖ) | Slope-driven source/sink terms derived from real DEM data |
| Urban obstacle | Uζ + Ua²/(ζ−ζ₀) + Ua²/(ζ̄−ζ̄₀) | Circle-theorem obstacle representing the downtown commercial core |

The SC map f : ℍ → Ω sends the upper half-plane ℍ to the Boulder polygon Ω,
converting these analytically tractable potentials in ℍ into physically meaningful
streamlines and equipotentials in the actual city boundary.

---

## Mathematical Framework

### Schwarz–Christoffel Mapping

For a polygon with n vertices z₀, …, z_{n−1} and interior angles α₀π, …, α_{n−1}π,
the SC map from the upper half-plane ℍ to Ω is:

```
f(ζ) = A + C ∫_{ζ₀}^{ζ}  ∏ₖ (t − ζₖ)^{αₖ−1}  dt
```

The **pre-vertices** ζₖ ∈ ℝ are the unknowns.  By Möbius normalisation, three are fixed
(ζ₀ = −1, ζ₁ = 0, ζ_{n−1} = 1); the remaining n−3 are found by a nonlinear
least-squares solve (Levenberg–Marquardt, `scipy.optimize.least_squares`) that matches
the edge-length ratios of the mapped polygon to those of the target.

### Terrain Correction (RBF source/sink)

Elevation data from USGS 3DEP (81 sample points) is fitted with a **thin-plate-spline
RBF surface** (scipy `RBFInterpolator`).  The gradient ∇e is computed at each polygon
vertex by centred finite differences.  Each vertex k contributes one singularity in ℍ:

```
W_terrain(ζ) = Uζ + Σₖ  qₖ/(2π) · [log(ζ − sₖ) − log(ζ − s̄ₖ)]
```

where qₖ = Q_scale · (∂e/∂x · cosθ + ∂e/∂y · sinθ) / max|∇e| is the source strength
projected onto the free-stream direction θ, and the image term −log(ζ − s̄ₖ) enforces
ψ = 0 on ℝ (no-penetration on the outer boundary).

### Urban Obstacle (Circle Theorem)

The downtown commercial core is treated as an **impenetrable interior obstacle**,
making the domain doubly connected.  The inner polygon is mapped to ℍ via the inverse
SC map, then approximated by the minimum enclosing circle (centre ζ₀, radius a).
By the **Milne-Thomson circle theorem**, the obstacle is introduced without disturbing
the far-field flow:

```
W_urban(ζ) = Uζ + Ua²/(ζ − ζ₀) + Ua²/(ζ̄ − ζ̄₀)
```

The image term Ua²/(ζ̄ − ζ̄₀) is the reflection below ℝ, preserving ψ = 0 on the real
axis.  Accuracy of the circle approximation is O((a / Im ζ₀)²); the run below achieves
Im(ζ₀)/a = 5.36, giving an error of ~3.5 %.

---

## Pipeline and Key Metrics (last full run)

```
python main.py --shapefile data/raw/tl_2025_08_place \
               --terrain --urban --grid 80 \
               --min-vertices 12 --max-vertices 16
```

| Stage | Metric | Value |
|-------|--------|-------|
| Polygon simplification | Vertices: original → simplified | 1935 → 14 |
| | Douglas–Peucker tolerance | 1575 m |
| SC parameter solve | Residual cost (Levenberg–Marquardt) | 3.64 × 10⁻² |
| | Max vertex mapping error |max‖f(ζₖ) − zₖ‖ ≈ 0.18 (normalised) |
| Elevation sampling | USGS 3DEP points queried / returned | 81 / 81 (100 %) |
| RBF terrain fit | Thin-plate-spline R² | 1.0000 |
| Terrain sources | Vertex source/sink distribution | 4 sources (+), 10 sinks (−) |
| | Strength range |q|  | 3 × 10⁻⁴ – 0.35 |
| Urban polygon | OSM polygons (commercial/retail) | 73 raw → 6-vertex hull, 0.13 km² |
| Urban obstacle | Circle centre ζ₀ (normalised ℍ) | 0.765 + 0.302 i |
| | Circle radius a | 0.056 |
| | Separation ratio Im(ζ₀)/a | 5.36 |
| | Circle-theorem approximation error | ~(a/Im ζ₀)² ≈ 3.5 % |
| Urban flow grid | Valid points (inside Ω, outside obstacle) | 3631 / 4118 |

**SC cost** measures the least-squares residual of the parameter problem — how well the
pre-vertex set reproduces the target edge-length ratios.  A value of 3.64 × 10⁻² is
acceptable for a 14-vertex polygon that includes a near-reflex angle of 1.848π (333°);
this sharp re-entrant corner concentrates the SC integrand and is the dominant source of
error.

**Im(ζ₀)/a** is the ratio of the obstacle's height above the real axis to its radius.
The circle-theorem accuracy scales as (a/Im ζ₀)², so this ratio must be >> 1. At 5.36
the approximation error is ~3.5 %, well within acceptable range for a first-order
physical model.

---

## Results

All figures are saved to [`figures/`](figures/).

### Fig 1 — Boulder City Boundary: Original vs. Simplified

[`figures/fig1_polygon_comparison.png`](figures/fig1_polygon_comparison.png)

The left panel shows the raw TIGER/Line shapefile polygon (1935 vertices, UTM Zone 13N
coordinates).  The right panel shows the 14-vertex Douglas–Peucker simplification used
as the SC domain Ω.  The simplification preserves the macro shape while reducing the
SC parameter problem to a tractable size.  Both panels share the same UTM axis scale
(metres).

### Fig 2 — Streamlines (ψ = const)

[`figures/fig2_streamlines.png`](figures/fig2_streamlines.png)

Curves of constant stream function ψ inside Ω under the uniform potential W = Uζ.
Streamlines enter from the upper boundary (corresponding to ζ → +∞ in ℍ) and wrap
around a stagnation point in the lower-left region — an artefact of the polygon geometry
rather than any physical source.  The density of lines is proportional to local flow
speed.

### Fig 3 — Equipotential Lines (φ = const)

[`figures/fig3_equipotentials.png`](figures/fig3_equipotentials.png)

Curves of constant velocity potential φ.  By the Cauchy–Riemann equations, equipotentials
are everywhere orthogonal to streamlines.  The lines crowd near re-entrant corners
(especially the 1.848π corner in the lower-left), reflecting the elevated velocity that
conformal mappings produce at such singularities.

### Fig 4 — Combined Streamlines & Equipotentials

[`figures/fig4_combined.png`](figures/fig4_combined.png)

Overlay of Figs 2 and 3 (blue streamlines, orange equipotentials).  The orthogonality of
the two families is the key visual confirmation that the SC inverse map is working
correctly: ∇ψ · ∇φ ≈ 0 everywhere in the interior.  The two families together form the
**conformal grid** — the image of a standard rectangular grid in ℍ.

### Fig 5 — Terrain-Informed Flow

[`figures/fig5_terrain_flow.png`](figures/fig5_terrain_flow.png)

Streamlines under the terrain-corrected potential W = Uζ + Σₖ qₖ log(ζ−sₖ).  The red
marker (top-left, elevation ~2063 m) is the highest vertex and acts as the dominant
**source** of the terrain correction; the blue marker (bottom, ~1568 m) is the lowest
and acts as the dominant **sink**.  The black arrow shows the mean downhill direction
(18.9° from east) estimated from the RBF surface gradient.  The green oval marks the
urban core region for geographic context.  Compared to the uniform flow, the streamlines
are noticeably deflected by the elevation-driven terms.

### Fig 6 — Flow Comparison: Uniform vs. Terrain-Corrected

[`figures/fig6_flow_comparison.png`](figures/fig6_flow_comparison.png)

Side-by-side of W = Uζ (left) and W = Uζ + terrain sources (right) at identical contour
levels.  The terrain correction bends streamlines toward lower-elevation regions,
consistent with orographic flow deflection: fluid "channels" along the slope from the
Flatirons toward the plains.

### Fig 7 — Doubly-Connected Flow: Urban Core as Interior Obstacle

[`figures/fig7_urban_flow.png`](figures/fig7_urban_flow.png)

Streamlines under the circle-theorem potential
W(ζ) = Uζ + Ua²/(ζ−ζ₀) + Ua²/(ζ̄−ζ̄₀).  The purple rectangle marks the
downtown Boulder commercial core (0.13 km², 6-vertex OSM polygon) modelled as an
impenetrable obstacle.  Streamlines deflect around it — the doubly-connected topology
means no streamline can pass through the interior of the obstacle.  The stagnation
pattern near the obstacle is analogous to ideal flow around a cylinder mapped to the
polygon domain.

### Fig 8 — Three-Way Comparison

[`figures/fig8_three_way_comparison.png`](figures/fig8_three_way_comparison.png)

Summary figure showing the three physical models at the same contour levels:

- **(a) Uniform** W = Uζ — baseline SC conformal flow, no physical corrections.
- **(b) Terrain-corrected** W = Uζ + RBF sources — slope-driven redistribution from
  81 USGS elevation samples; 14 distributed source/sink singularities.
- **(c) Urban obstacle** W + circle theorem — doubly-connected domain with the
  commercial core as an interior no-penetration boundary.

Each panel uses the same polygon boundary and the same SC map; only the potential in ℍ
changes, illustrating how conformal mapping cleanly separates the domain geometry
(handled once by the SC map) from the physical model (swapped in via W).

---

## Project Structure

```
Project/
├── data/
│   └── raw/                  ← Place TIGER/Line shapefile here
├── src/
│   ├── __init__.py
│   ├── polygon.py            ← Load & simplify Boulder polygon
│   ├── angles.py             ← Interior-angle computation
│   ├── sc_solver.py          ← SC parameter problem & forward map
│   ├── flow.py               ← Inverse map & stream-function grid
│   ├── terrain.py            ← DEM elevation (RBF) + per-vertex sources
│   ├── urban.py              ← Urban-core polygon (OSM) & coordinate conversion
│   ├── sc_solver_dc.py       ← Circle-theorem obstacle in ℍ
│   └── visualization.py      ← Publication-quality figures
├── figures/                  ← Generated output (see Results above)
├── main.py                   ← Full pipeline CLI
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv ComplexEnv
ComplexEnv\Scripts\activate          # Windows
# source ComplexEnv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TIGER/Line shapefile
#    https://www.census.gov/cgi-bin/geo/shapefiles/index.php
#    Year=2025, Layer Type="Places", State="Colorado"
#    Unzip into data/raw/

# 4. Run full pipeline (terrain + urban obstacle)
python main.py --shapefile data/raw/tl_2025_08_place --terrain --urban

# Quick demo (no shapefile needed)
python main.py --demo
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shapefile PATH` | — | Path to TIGER/Line shapefile directory |
| `--demo` | off | Run with built-in hexagon, no shapefile needed |
| `--terrain` | off | Enable DEM terrain-corrected flow |
| `--urban` | off | Enable urban-obstacle doubly-connected flow |
| `--urban-method` | osmnx | `osmnx` (download from OSM) or `fallback` (hardcoded polygon) |
| `--urban-vertices` | 8 | Target vertex count for simplified urban polygon |
| `--grid N` | 80 | Flow-grid resolution (N×N) |
| `--min-vertices` | 18 | Minimum vertices after polygon simplification |
| `--max-vertices` | 30 | Maximum vertices after polygon simplification |

---

## Key References

- [DT09] Driscoll & Trefethen, *Schwarz–Christoffel Mapping*, Cambridge, 2009.
- [AF03] Ablowitz & Fokas, *Complex Variables*, Cambridge, 2003.
- [MT68] Milne-Thomson, *Theoretical Hydrodynamics*, 5th ed., Macmillan, 1968. (Circle theorem)
- [BO99] Bender & Orszag, *Advanced Mathematical Methods*, Springer, 1999.
- US Census Bureau, TIGER/Line Shapefiles, 2025.
- USGS 3DEP Elevation Point Query Service, https://epqs.nationalmap.gov/v1/
- OpenStreetMap contributors, via `osmnx` (Boeing, 2017).
