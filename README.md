# Ideal Fluid Flow via the Schwarz-Christoffel Transformation

**Complex Variables and Applications - Spring 2026**
Congyuan Zheng · Sophia Arany · Alexander Ingalls
University of Colorado Boulder, Department of Applied Mathematics

---

## Overview

This project applies the **Schwarz-Christoffel (SC) conformal mapping** to simulate
steady, irrotational, incompressible fluid flow inside the Boulder, Colorado city
boundary polygon. Three progressively richer physical models are implemented:

| Model | Potential $W(\zeta)$ | Physical meaning |
|-------|----------------------|-----------------|
| Uniform flow | $U\zeta$ | Ideal parallel flow, no boundary corrections |
| Terrain-corrected | $U\zeta + \sum_k q_k \log(\zeta - s_k)$ | Slope-driven sources/sinks from real DEM data |
| Urban obstacle | $U\zeta + \frac{Ua^2}{\zeta - \zeta_0} + \frac{Ua^2}{\bar\zeta - \bar\zeta_0}$ | Circle-theorem obstacle for the downtown commercial core |

The SC map $f : \mathbb{H} \to \Omega$ sends the upper half-plane to the Boulder polygon,
converting analytically tractable potentials in $\mathbb{H}$ into streamlines and
equipotentials in the physical domain.

---

## Mathematical Framework

### Schwarz-Christoffel Mapping

For a polygon with $n$ vertices $z_0, \ldots, z_{n-1}$ and interior angles
$\alpha_0\pi, \ldots, \alpha_{n-1}\pi$, the SC map from $\mathbb{H}$ to $\Omega$ is

$$f(\zeta) = A + C \int_{\zeta_0}^{\zeta} \prod_{k} (t - \zeta_k)^{\alpha_k - 1} \, dt$$

The **pre-vertices** $\zeta_k \in \mathbb{R}$ are the unknowns. By Möbius normalisation
three are fixed ($\zeta_0 = -1$, $\zeta_1 = 0$, $\zeta_{n-1} = 1$); the remaining $n-3$
are found by a nonlinear least-squares solve (Levenberg-Marquardt) that matches the
edge-length ratios of the mapped polygon to those of the target.

### Terrain Correction (RBF source/sink)

Elevation data from USGS 3DEP (81 sample points) is fitted with a thin-plate-spline
RBF surface (`scipy.interpolate.RBFInterpolator`). The gradient $\nabla e$ is computed
at each polygon vertex by centred finite differences. Each vertex $k$ contributes one
singularity in $\mathbb{H}$:

$$W_\text{terrain}(\zeta) = U\zeta + \sum_k \frac{q_k}{2\pi}
  \Bigl[\log(\zeta - s_k) + \log(\zeta - \bar s_k)\Bigr]$$

where $q_k = Q \cdot (\partial_x e \cos\theta + \partial_y e \sin\theta) / \max|\nabla e|$
is the source strength projected onto the free-stream direction $\theta$, and the image
term $+\log(\zeta - \bar s_k)$ (method of images) enforces $\psi = 0$ on $\mathbb{R}$
because $\operatorname{Im}[\log(\zeta-s_k) + \log(\zeta-\bar s_k)] = 0$ for $\zeta \in \mathbb{R}$.

### Urban Obstacle (Circle Theorem)

The downtown commercial core is treated as an impenetrable interior obstacle, making the
domain doubly connected. The inner polygon is mapped to $\mathbb{H}$ via the inverse SC
map, then approximated by the minimum enclosing circle (centre $\zeta_0$, radius $a$).
By the **Milne-Thomson circle theorem**:

$$W_\text{urban}(\zeta) = U\zeta + \frac{Ua^2}{\zeta - \zeta_0} + \frac{Ua^2}{\bar\zeta - \bar\zeta_0}$$

The image term $Ua^2 / (\bar\zeta - \bar\zeta_0)$ reflects the dipole below $\mathbb{R}$,
preserving $\psi = 0$ on the real axis. Accuracy is $O\!\left((a/\text{Im}\,\zeta_0)^2\right)$;
the run below achieves $\text{Im}(\zeta_0)/a = 5.36$, giving ~3.5 % error.

---

## Pipeline and Key Metrics

```bash
python main.py --shapefile data/raw/tl_2025_08_place \
               --terrain --urban --grid 80
```

| Stage | Metric | Value |
|-------|--------|-------|
| Polygon simplification | Vertices: original → D-P simplified | 1935 → 21 |
| | Douglas-Peucker tolerance | 862.5 m |
| Near-cusp smoothing | After `smooth_extreme_angles(alpha=0.5pi)` | 21 → 5 |
| SC parameter solve | Levenberg-Marquardt residual cost | 4.79 × 10⁻² |
| | Pre-vertex positions | [-1, 0, 0.203, 0.601, 1] |
| Elevation sampling | USGS 3DEP points queried / returned | 81 / 81 (100 %) |
| RBF terrain fit | Thin-plate-spline R² | 1.0000 |
| Terrain sources | Vertex source / sink distribution | 4 sources, 10 sinks |
| | Strength range $\lvert q\rvert$ | 3 × 10⁻⁴ - 0.35 |
| Urban polygon | OSM polygons (commercial/retail) | 73 raw → 6-vertex, 0.13 km² |
| Urban obstacle | Circle centre $\zeta_0$ (normalised) | $0.765 + 0.302i$ |
| | Circle radius $a$ | 0.056 |
| | Separation $\text{Im}(\zeta_0)/a$ | 5.36 |
| | Circle-theorem error estimate | ~3.5 % |

---

## Results

### Fig 1 - Boulder City Boundary: Original vs. Simplified

![Boulder boundary original vs simplified](figures/fig1_polygon_comparison.png)

Raw TIGER/Line polygon (1935 vertices, UTM Zone 13N) vs. the 21-vertex
Douglas-Peucker simplification (862.5 m tolerance). Near-cusp vertices
($\alpha < 0.5\pi$) are subsequently removed by `smooth_extreme_angles`,
leaving a 5-vertex polygon used as the SC domain $\Omega$.

### Fig 2 - Streamlines ($\psi = \text{const}$)

![Streamlines uniform flow](figures/fig2_streamlines.png)

Level curves of the stream function $\psi$ under uniform flow $W = U\zeta$.
Curves are computed via the **forward SC map**: horizontal lines $\mathrm{Im}(\zeta) = y_0$
in $\mathbb{H}$ are mapped forward to the polygon, producing exact smooth arcs with no
interpolation artifacts. Flow enters from both sides of the upper boundary and converges
toward a stagnation point at the top vertex - the conformal image of $\zeta \to +\infty$.

### Fig 3 - Equipotential Lines ($\varphi = \text{const}$)

![Equipotentials uniform flow](figures/fig3_equipotentials.png)

Level curves of the velocity potential $\varphi$. By the Cauchy-Riemann equations,
equipotentials are everywhere orthogonal to streamlines. Vertical half-lines
$\mathrm{Re}(\zeta) = x_0$ in $\mathbb{H}$ map forward to the radial-like curves visible here.

### Fig 4 - Combined Streamlines & Equipotentials

![Combined flow uniform](figures/fig4_combined.png)

Overlay of Figs 2 and 3 (blue streamlines, red equipotentials). The two families
form the conformal grid - the image of a rectangular grid in $\mathbb{H}$ under $f$.
Orthogonality throughout the interior confirms the map is conformal.

### Fig 5 - Terrain-Informed Flow

![Terrain-informed flow](figures/fig5_terrain_flow.png)

Streamlines under $W = U\zeta + \sum_k q_k \log(\zeta - s_k)$. The red marker
(top-left, ~2063 m) is the dominant source; the blue marker (bottom, ~1568 m) is the
dominant sink. The black arrow shows the mean downhill direction (18.9° from east)
from the RBF surface. Flow channels toward lower-elevation regions, consistent with
orographic deflection from the Flatirons toward the plains.

### Fig 6 - Uniform vs. Terrain-Corrected

![Flow comparison uniform vs terrain](figures/fig6_flow_comparison.png)

Side-by-side at identical contour levels. The terrain correction bends streamlines
toward lower elevation; the shift is largest in the western portion of the domain
where the elevation gradient is steepest.

### Fig 7 - Urban Core as Interior Obstacle

![Urban obstacle doubly-connected flow](figures/fig7_urban_flow.png)

Circle-theorem potential $W = U\zeta + Ua^2/(\zeta - \zeta_0) + Ua^2/(\bar\zeta - \bar\zeta_0)$.
The purple rectangle marks the downtown commercial core (0.13 km², 6-vertex OSM
polygon). Streamlines deflect around it - no path penetrates the obstacle interior.
The stagnation pattern mirrors ideal flow around a cylinder, mapped to the polygon
domain via $f$.

### Fig 8 - Three-Way Comparison

![Three-way streamline comparison](figures/fig8_three_way_comparison.png)

All three models at the same contour levels:
**(a) Uniform** - baseline SC flow.
**(b) Terrain-corrected** - distributed source/sink terms from 81 USGS elevation samples.
**(c) Urban obstacle** - doubly-connected domain with commercial core as no-penetration boundary.

Each panel uses the same SC map; only the potential $W$ in $\mathbb{H}$ changes - demonstrating how the conformal mapping cleanly decouples domain geometry from
the physical model.

---

## Project Structure

```
.
├── data/
│   └── raw/                  ← TIGER/Line shapefile (Boulder, CO)
├── src/
│   ├── polygon.py            ← Load & simplify Boulder polygon
│   ├── angles.py             ← Interior-angle computation
│   ├── sc_solver.py          ← SC parameter problem & forward map
│   ├── flow.py               ← Forward SC map, parametric curves & flow grid
│   ├── terrain.py            ← DEM elevation (RBF) + per-vertex sources
│   ├── urban.py              ← Urban-core polygon (OSM) & coordinate conversion
│   ├── sc_solver_dc.py       ← Circle-theorem obstacle in H
│   └── visualization.py      ← Figure generation
├── figures/                  ← Generated output (see above)
├── main.py                   ← Full pipeline CLI
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Run full pipeline
python main.py --shapefile data/raw/tl_2025_08_place --terrain --urban

# Quick demo (no shapefile needed)
python main.py --demo
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shapefile PATH` | - | TIGER/Line shapefile directory |
| `--demo` | off | Built-in hexagon domain, no data needed |
| `--terrain` | off | Terrain-corrected flow (USGS DEM) |
| `--urban` | off | Urban-obstacle doubly-connected flow |
| `--urban-method` | `osmnx` | `osmnx` (live OSM) or `fallback` (hardcoded polygon) |
| `--grid N` | 80 | Flow-grid resolution (N × N) |
| `--min-vertices` | 18 | Min vertices after Douglas-Peucker simplification |
| `--max-vertices` | 30 | Max vertices after Douglas-Peucker simplification |

Near-cusp vertices ($\alpha < 0.5\pi$) are always removed automatically after simplification
by `smooth_extreme_angles` regardless of the `--min-vertices` / `--max-vertices` settings.

---

## References

- Driscoll & Trefethen, *Schwarz-Christoffel Mapping*, Cambridge, 2009.
- Ablowitz & Fokas, *Complex Variables*, Cambridge, 2003.
- Milne-Thomson, *Theoretical Hydrodynamics*, 5th ed., Macmillan, 1968.
- US Census Bureau, TIGER/Line Shapefiles, 2025.
- USGS 3DEP Elevation Point Query Service - https://epqs.nationalmap.gov/v1/
- Boeing, G. (2017). OSMnx: New methods for acquiring and analysing street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
