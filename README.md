# Ideal Fluid Flow in a Polygonal Domain via the Schwarz–Christoffel Transformation

**APPM 4360 — Complex Variables and Applications, Spring 2026**  
Congyuan Zheng, Sophia Arany, Alexander Ingalls  
University of Colorado Boulder, Department of Applied Mathematics

## Overview

This project applies the **Schwarz–Christoffel (SC) conformal mapping** to simulate
steady, irrotational, incompressible fluid flow inside the Boulder, Colorado city
boundary polygon. The pipeline:

1. Loads and simplifies the Boulder city boundary from TIGER/Line shapefiles.
2. Computes interior angles of the simplified polygon.
3. Solves the SC parameter problem (pre-vertex locations on the real axis).
4. Evaluates the forward SC map  f : ℍ → Ω.
5. Numerically inverts the map to compute streamlines in the physical domain.
6. Produces publication-quality figures.

## Project Structure

```
Project/
├── data/
│   └── raw/               ← Place TIGER/Line shapefile here
├── src/
│   ├── __init__.py
│   ├── polygon.py         ← Load & simplify Boulder polygon
│   ├── angles.py          ← Interior-angle computation
│   ├── sc_solver.py       ← SC parameter problem & forward map
│   ├── flow.py            ← Inverse map & stream-function grid
│   └── visualization.py   ← Publication-quality figures
├── figures/               ← Generated output figures
├── main.py                ← Run the full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv ComplexEnv
# Windows
ComplexEnv\Scripts\activate
# macOS / Linux
source ComplexEnv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the TIGER/Line shapefile
#    Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
#    Select Year=2023, Layer Type="Places", State="Colorado"
#    Download and unzip into  data/raw/

# 4. Run the pipeline
python main.py
```

## Key References

- [AF03] Ablowitz & Fokas, *Complex Variables*, Cambridge, 2003.
- [DT09] Driscoll & Trefethen, *Schwarz–Christoffel Mapping*, Cambridge, 2009.
- [BO99] Bender & Orszag, *Advanced Mathematical Methods*, Springer, 1999.
- US Census Bureau, TIGER/Line Shapefiles, 2023.
