"""
Schwarz–Christoffel Conformal Mapping for Ideal Fluid Flow
in the Boulder City Polygon.

Modules
-------
polygon        – Load & simplify the Boulder city boundary.
angles         – Compute interior angles of the polygon.
sc_solver      – Solve the SC parameter problem & evaluate the forward map.
flow           – Inverse map & stream-function grid computation.
terrain        – DEM elevation (RBF surface) + per-vertex source/sink potential.
urban          – Urban-core polygon acquisition (OSM) and coordinate conversion.
sc_solver_dc   – Doubly-connected flow: circle-theorem obstacle potential in ℍ.
visualization  – Publication-quality figure generation.
"""
