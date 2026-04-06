"""
visualizer.py — matplotlib-based rendering of a Sketch
=======================================================
Renders all geometry entities in a :class:`~cad_sketch.sketch.Sketch` on a
``matplotlib`` axes object.

Supported entity types
----------------------
* :class:`~cad_sketch.geometry.Point`      — marker
* :class:`~cad_sketch.geometry.Line`       — straight segment
* :class:`~cad_sketch.geometry.Polyline`   — connected line segments
* :class:`~cad_sketch.geometry.Circle`     — full circle
* :class:`~cad_sketch.geometry.Arc`        — partial circle
* :class:`~cad_sketch.geometry.Rectangle`  — 4-line closed shape
* :class:`~cad_sketch.geometry.Polygon`    — N-sided closed shape
* :class:`~cad_sketch.geometry.Spline`     — smooth sampled curve
* :class:`~cad_sketch.geometry.Slot`       — composite (lines + arcs)
"""

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sketch import Sketch

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

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


# ---------------------------------------------------------------------------
# Color / style constants
# ---------------------------------------------------------------------------
_COLOR_ENTITY      = "#1f77b4"   # default blue
_COLOR_CONSTRUCTION = "#aaaaaa"  # light grey for construction geometry
_COLOR_POINT       = "#d62728"   # red for points
_COLOR_CONSTR_PT   = "#bbbbbb"   # grey for construction points
_LINEWIDTH         = 1.5
_LINEWIDTH_CONSTR  = 0.8
_LINESTYLE_CONSTR  = "--"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def visualize(
    sketch: "Sketch",
    title: str = "CAD Sketch",
    show: bool = True,
    ax: Optional[object] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Render all entities in *sketch* using matplotlib.

    Parameters
    ----------
    sketch   : Sketch  — the sketch to render.
    title    : str     — figure title.
    show     : bool    — call ``plt.show()`` after drawing (set *False* when
               embedding in a GUI loop or saving to file only).
    ax       : matplotlib Axes, optional — draw on an existing axes.  When
               *None* a new figure is created.
    filename : str, optional — if provided, save the figure to this path
               (e.g. ``"sketch.png"``) before showing.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[visualizer] matplotlib is not installed. "
              "Install it with:  pip install matplotlib")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw each entity
    for eid, entity in sketch.entities.items():
        _draw_entity(ax, entity)

    # Auto-scale
    ax.autoscale_view()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Per-entity drawing helpers
# ---------------------------------------------------------------------------
def _draw_entity(ax, entity) -> None:
    """Dispatch to the correct drawing function for *entity*."""
    if isinstance(entity, Slot):
        _draw_slot(ax, entity)
    elif isinstance(entity, Rectangle):
        _draw_rectangle(ax, entity)
    elif isinstance(entity, Polygon):
        _draw_polygon(ax, entity)
    elif isinstance(entity, Polyline):
        _draw_polyline(ax, entity)
    elif isinstance(entity, Spline):
        _draw_spline(ax, entity)
    elif isinstance(entity, Arc):
        _draw_arc(ax, entity)
    elif isinstance(entity, Circle):
        _draw_circle(ax, entity)
    elif isinstance(entity, Line):
        _draw_line(ax, entity)
    elif isinstance(entity, Point):
        _draw_point(ax, entity)
    # Unknown entity types are silently skipped


def _color_and_style(construction: bool):
    """Return (color, linewidth, linestyle) based on construction flag."""
    if construction:
        return _COLOR_CONSTRUCTION, _LINEWIDTH_CONSTR, _LINESTYLE_CONSTR
    return _COLOR_ENTITY, _LINEWIDTH, "-"


def _draw_point(ax, pt: Point) -> None:
    """Draw a single point as a marker."""
    color = _COLOR_CONSTR_PT if pt.construction else _COLOR_POINT
    marker = "x" if pt.construction else "+"
    ax.plot(pt.x, pt.y, marker=marker, color=color,
            markersize=6, zorder=5)
    ax.annotate(
        pt.id,
        (pt.x, pt.y),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=6,
        color=color,
        zorder=6,
    )


def _draw_line(ax, line: Line) -> None:
    """Draw a line segment."""
    color, lw, ls = _color_and_style(line.construction)
    ax.plot(
        [line.start.x, line.end.x],
        [line.start.y, line.end.y],
        color=color, linewidth=lw, linestyle=ls, zorder=3,
    )
    # Label at midpoint
    mx, my = line.midpoint()
    ax.annotate(
        line.id,
        (mx, my),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=color,
        zorder=4,
    )


def _draw_polyline(ax, pl: Polyline) -> None:
    """Draw a polyline as connected segments."""
    xs = [p.x for p in pl.points]
    ys = [p.y for p in pl.points]
    if pl.closed:
        xs.append(pl.points[0].x)
        ys.append(pl.points[0].y)
    ax.plot(xs, ys, color=_COLOR_ENTITY, linewidth=_LINEWIDTH, zorder=3)
    # Label near start
    ax.annotate(
        pl.id,
        (pl.points[0].x, pl.points[0].y),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        zorder=4,
    )


def _draw_circle(ax, circle: Circle) -> None:
    """Draw a full circle using a matplotlib patch."""
    patch = mpatches.Circle(
        (circle.center.x, circle.center.y),
        radius=circle.radius,
        fill=False,
        edgecolor=_COLOR_ENTITY,
        linewidth=_LINEWIDTH,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.annotate(
        circle.id,
        (circle.center.x, circle.center.y + circle.radius),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        zorder=4,
    )


def _draw_arc(ax, arc: Arc) -> None:
    """Draw an arc as a partial circle using a matplotlib wedge/arc patch."""
    # matplotlib Arc patch uses the same angle convention (degrees, CCW from +X)
    # We need to handle wrap-around: if end < start, span crosses 360°.
    start = arc.start_angle
    end = arc.end_angle
    if end <= start:
        end += 360.0
    span = end - start

    patch = mpatches.Arc(
        (arc.center.x, arc.center.y),
        width=2 * arc.radius,
        height=2 * arc.radius,
        angle=0.0,
        theta1=start,
        theta2=end,
        color=_COLOR_ENTITY,
        linewidth=_LINEWIDTH,
        zorder=3,
    )
    ax.add_patch(patch)
    # Label at mid-angle
    mid_angle = math.radians(start + span / 2.0)
    lx = arc.center.x + arc.radius * math.cos(mid_angle)
    ly = arc.center.y + arc.radius * math.sin(mid_angle)
    ax.annotate(
        arc.id,
        (lx, ly),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        zorder=4,
    )


def _draw_rectangle(ax, rect: Rectangle) -> None:
    """Draw a rectangle as its four line segments."""
    xs = [c.x for c in rect.corners] + [rect.corners[0].x]
    ys = [c.y for c in rect.corners] + [rect.corners[0].y]
    ax.plot(xs, ys, color=_COLOR_ENTITY, linewidth=_LINEWIDTH, zorder=3)
    cx = (rect.corners[0].x + rect.corners[2].x) / 2.0
    cy = (rect.corners[0].y + rect.corners[2].y) / 2.0
    ax.annotate(
        rect.id,
        (cx, cy),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        ha="center",
        zorder=4,
    )


def _draw_polygon(ax, poly: Polygon) -> None:
    """Draw a closed polygon."""
    xs = [v.x for v in poly.vertices] + [poly.vertices[0].x]
    ys = [v.y for v in poly.vertices] + [poly.vertices[0].y]
    ax.plot(xs, ys, color=_COLOR_ENTITY, linewidth=_LINEWIDTH, zorder=3)
    cx = sum(v.x for v in poly.vertices) / len(poly.vertices)
    cy = sum(v.y for v in poly.vertices) / len(poly.vertices)
    ax.annotate(
        poly.id,
        (cx, cy),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        ha="center",
        zorder=4,
    )


def _draw_spline(ax, spline: Spline) -> None:
    """Draw a spline as a smooth sampled polyline."""
    pts = spline.evaluate()
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=_COLOR_ENTITY, linewidth=_LINEWIDTH, zorder=3)
    # Draw control points as small markers
    for cp in spline.control_points:
        ax.plot(cp.x, cp.y, "o", color=_COLOR_POINT, markersize=4,
                alpha=0.5, zorder=4)
    ax.annotate(
        spline.id,
        (xs[0], ys[0]),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
        color=_COLOR_ENTITY,
        zorder=5,
    )


def _draw_slot(ax, slot: Slot) -> None:
    """Draw a slot as its composite line + arc sub-entities."""
    for ln in slot.lines:
        _draw_line(ax, ln)
    for arc in slot.arcs:
        _draw_arc(ax, arc)
    # Label at midpoint between the two centers
    mx = (slot.c1.x + slot.c2.x) / 2.0
    my = (slot.c1.y + slot.c2.y) / 2.0
    ax.annotate(
        slot.id,
        (mx, my),
        textcoords="offset points",
        xytext=(2, 4),
        fontsize=6,
        color=_COLOR_ENTITY,
        ha="center",
        zorder=4,
    )
