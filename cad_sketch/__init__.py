"""
cad_sketch — 2D CAD Sketch Tool
================================
A Python module that implements interactive 2D geometry creation and
geometric constraint solving, with matplotlib-based visualization and a
menu-driven CLI.

Package layout
--------------
geometry.py    — Point, Line, Polyline, Circle, Arc, Rectangle, Polygon, Spline, Slot
constraints.py — Coincident, Horizontal, Vertical, Parallel, Perpendicular, Tangent, Concentric
solver.py      — iterative constraint solver
sketch.py      — Sketch class (entity/constraint manager + solver integration)
visualizer.py  — matplotlib renderer
main.py        — CLI entry point
"""

from .geometry import (
    Point,
    Line,
    Polyline,
    Circle,
    Arc,
    Rectangle,
    Polygon,
    Spline,
    Slot,
)
from .constraints import (
    Coincident,
    Horizontal,
    Vertical,
    Parallel,
    Perpendicular,
    Tangent,
    Concentric,
)
from .solver import ConstraintSolver
from .sketch import Sketch
from .visualizer import visualize

__all__ = [
    "Point",
    "Line",
    "Polyline",
    "Circle",
    "Arc",
    "Rectangle",
    "Polygon",
    "Spline",
    "Slot",
    "Coincident",
    "Horizontal",
    "Vertical",
    "Parallel",
    "Perpendicular",
    "Tangent",
    "Concentric",
    "ConstraintSolver",
    "Sketch",
    "visualize",
]
