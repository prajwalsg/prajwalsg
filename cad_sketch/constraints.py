"""
constraints.py — Geometric constraint definitions
==================================================
Each constraint is a lightweight object that knows which entities it applies
to and how to express the error (residual) and partial derivatives (Jacobian
rows) that the :class:`~cad_sketch.solver.ConstraintSolver` needs.

Constraints operate on the free variables of :class:`~cad_sketch.geometry.Point`
objects exposed by a :class:`~cad_sketch.sketch.Sketch`.

Supported constraints
---------------------
Coincident    — two points share the same (x, y).
Horizontal    — a line's two endpoints share the same y value.
Vertical      — a line's two endpoints share the same x value.
Parallel      — two lines have the same direction vector.
Perpendicular — the dot product of two line direction vectors is zero.
Tangent       — a line (or arc) is tangent to a circle.
Concentric    — two circles/arcs share the same center point.
"""

import math
from typing import List, Tuple

from .geometry import Line, Circle, Arc, Point

# Type alias used by the solver
# A residual is a list of scalar values that the solver tries to drive to 0.
Residuals = List[float]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Constraint:
    """
    Abstract base class for all geometric constraints.

    Subclasses must implement :meth:`residuals`.
    """

    def __init__(self) -> None:
        self.id: str = ""  # assigned by the Sketch

    def residuals(self) -> Residuals:
        """
        Return a list of scalar residual values.

        Each residual represents the "error" for one equation in the
        constraint system.  The solver will iterate until all residuals are
        close to zero (within tolerance).
        """
        raise NotImplementedError

    def apply(self) -> None:
        """
        Directly adjust entity coordinates to satisfy the constraint
        (used as a pre-conditioning step before the full solver pass).

        Default implementation calls :meth:`residuals` and takes a gradient
        step; subclasses may override for an exact analytical correction.
        """
        # Default: no direct application — let the solver handle it.
        pass

    def description(self) -> str:
        """Return a short human-readable description of the constraint."""
        return repr(self)


# ---------------------------------------------------------------------------
# Coincident
# ---------------------------------------------------------------------------
class Coincident(Constraint):
    """
    Forces two :class:`~cad_sketch.geometry.Point` objects to occupy the same
    position: ``p1.x == p2.x`` and ``p1.y == p2.y``.

    The solver will move both points toward their average position.

    Parameters
    ----------
    p1, p2 : Point
        The two points that must coincide.
    """

    def __init__(self, p1: Point, p2: Point) -> None:
        super().__init__()
        self.p1: Point = p1
        self.p2: Point = p2

    def residuals(self) -> Residuals:
        """Return [p1.x - p2.x, p1.y - p2.y]."""
        return [self.p1.x - self.p2.x, self.p1.y - self.p2.y]

    def apply(self) -> None:
        """Snap both points to their midpoint."""
        mx = (self.p1.x + self.p2.x) / 2.0
        my = (self.p1.y + self.p2.y) / 2.0
        self.p1.x = self.p2.x = mx
        self.p1.y = self.p2.y = my

    def description(self) -> str:
        return f"Coincident({self.p1.id}, {self.p2.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Horizontal
# ---------------------------------------------------------------------------
class Horizontal(Constraint):
    """
    Forces a :class:`~cad_sketch.geometry.Line` to be horizontal by requiring
    its two endpoints to share the same ``y`` value.

    Parameters
    ----------
    line : Line
    """

    def __init__(self, line: Line) -> None:
        super().__init__()
        self.line: Line = line

    def residuals(self) -> Residuals:
        """Return [start.y - end.y]."""
        return [self.line.start.y - self.line.end.y]

    def apply(self) -> None:
        """Snap both endpoints to the average y value."""
        avg_y = (self.line.start.y + self.line.end.y) / 2.0
        self.line.start.y = avg_y
        self.line.end.y = avg_y

    def description(self) -> str:
        return f"Horizontal({self.line.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Vertical
# ---------------------------------------------------------------------------
class Vertical(Constraint):
    """
    Forces a :class:`~cad_sketch.geometry.Line` to be vertical by requiring
    its two endpoints to share the same ``x`` value.

    Parameters
    ----------
    line : Line
    """

    def __init__(self, line: Line) -> None:
        super().__init__()
        self.line: Line = line

    def residuals(self) -> Residuals:
        """Return [start.x - end.x]."""
        return [self.line.start.x - self.line.end.x]

    def apply(self) -> None:
        """Snap both endpoints to the average x value."""
        avg_x = (self.line.start.x + self.line.end.x) / 2.0
        self.line.start.x = avg_x
        self.line.end.x = avg_x

    def description(self) -> str:
        return f"Vertical({self.line.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------
class Parallel(Constraint):
    """
    Ensures two :class:`~cad_sketch.geometry.Line` objects are parallel.

    Uses the cross-product residual:
    ``(dx1*dy2 - dy1*dx2) == 0``

    where ``(dx1, dy1)`` and ``(dx2, dy2)`` are the direction vectors of the
    two lines.

    Parameters
    ----------
    line1, line2 : Line
    """

    def __init__(self, line1: Line, line2: Line) -> None:
        super().__init__()
        self.line1: Line = line1
        self.line2: Line = line2

    def residuals(self) -> Residuals:
        """Return [cross product of the two direction vectors]."""
        dx1 = self.line1.end.x - self.line1.start.x
        dy1 = self.line1.end.y - self.line1.start.y
        dx2 = self.line2.end.x - self.line2.start.x
        dy2 = self.line2.end.y - self.line2.start.y
        return [dx1 * dy2 - dy1 * dx2]

    def apply(self) -> None:
        """
        Rotate line2's endpoint so that its direction matches line1's direction
        while preserving line2's length.
        """
        dx1 = self.line1.end.x - self.line1.start.x
        dy1 = self.line1.end.y - self.line1.start.y
        len1 = math.hypot(dx1, dy1)
        if len1 < 1e-12:
            return
        # Unit direction of line1
        ux, uy = dx1 / len1, dy1 / len1
        # Length of line2
        len2 = math.hypot(
            self.line2.end.x - self.line2.start.x,
            self.line2.end.y - self.line2.start.y,
        )
        self.line2.end.x = self.line2.start.x + ux * len2
        self.line2.end.y = self.line2.start.y + uy * len2

    def description(self) -> str:
        return f"Parallel({self.line1.id}, {self.line2.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Perpendicular
# ---------------------------------------------------------------------------
class Perpendicular(Constraint):
    """
    Ensures two :class:`~cad_sketch.geometry.Line` objects are perpendicular.

    Uses the dot-product residual:
    ``(dx1*dx2 + dy1*dy2) == 0``

    Parameters
    ----------
    line1, line2 : Line
    """

    def __init__(self, line1: Line, line2: Line) -> None:
        super().__init__()
        self.line1: Line = line1
        self.line2: Line = line2

    def residuals(self) -> Residuals:
        """Return [dot product of the two direction vectors]."""
        dx1 = self.line1.end.x - self.line1.start.x
        dy1 = self.line1.end.y - self.line1.start.y
        dx2 = self.line2.end.x - self.line2.start.x
        dy2 = self.line2.end.y - self.line2.start.y
        return [dx1 * dx2 + dy1 * dy2]

    def apply(self) -> None:
        """
        Rotate line2's endpoint 90° relative to line1's direction, preserving
        line2's length.
        """
        dx1 = self.line1.end.x - self.line1.start.x
        dy1 = self.line1.end.y - self.line1.start.y
        len1 = math.hypot(dx1, dy1)
        if len1 < 1e-12:
            return
        # Perpendicular direction: (-dy1, dx1)
        ux, uy = -dy1 / len1, dx1 / len1
        len2 = math.hypot(
            self.line2.end.x - self.line2.start.x,
            self.line2.end.y - self.line2.start.y,
        )
        self.line2.end.x = self.line2.start.x + ux * len2
        self.line2.end.y = self.line2.start.y + uy * len2

    def description(self) -> str:
        return f"Perpendicular({self.line1.id}, {self.line2.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Tangent
# ---------------------------------------------------------------------------
class Tangent(Constraint):
    """
    Ensures a :class:`~cad_sketch.geometry.Line` is tangent to a
    :class:`~cad_sketch.geometry.Circle` (touches at exactly one point).

    Residual: ``|distance(circle.center, line)| - circle.radius == 0``

    where "distance" is the perpendicular distance from the center to the
    infinite line containing the segment.

    Parameters
    ----------
    line   : Line
    circle : Circle
    """

    def __init__(self, line: Line, circle: Circle) -> None:
        super().__init__()
        self.line: Line = line
        self.circle: Circle = circle

    def _dist_point_to_line(self) -> float:
        """Perpendicular distance from circle center to the line."""
        x0, y0 = self.circle.center.x, self.circle.center.y
        x1, y1 = self.line.start.x, self.line.start.y
        x2, y2 = self.line.end.x, self.line.end.y
        dx, dy = x2 - x1, y2 - y1
        denom = math.hypot(dx, dy)
        if denom < 1e-12:
            return math.hypot(x0 - x1, y0 - y1)
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom

    def residuals(self) -> Residuals:
        """Return [distance(center, line) - radius]."""
        return [self._dist_point_to_line() - self.circle.radius]

    def apply(self) -> None:
        """
        Translate the line so that the perpendicular distance from the circle
        center equals the radius.  The line direction is preserved; only the
        offset (perpendicular translation) is adjusted.
        """
        x1, y1 = self.line.start.x, self.line.start.y
        x2, y2 = self.line.end.x, self.line.end.y
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            return

        # Unit normal (perpendicular to line direction)
        nx, ny = -dy / length, dx / length

        # Current signed distance
        dist = self._dist_point_to_line()
        diff = dist - self.circle.radius
        if abs(diff) < 1e-10:
            return

        # Determine the sign: is center "above" or "below" the line?
        x0, y0 = self.circle.center.x, self.circle.center.y
        sign = 1.0 if (nx * (x0 - x1) + ny * (y0 - y1)) > 0 else -1.0

        # Translate line endpoints
        shift = diff * sign
        self.line.start.x += nx * shift
        self.line.start.y += ny * shift
        self.line.end.x += nx * shift
        self.line.end.y += ny * shift

    def description(self) -> str:
        return f"Tangent({self.line.id}, {self.circle.id})"

    def __repr__(self) -> str:
        return self.description()


# ---------------------------------------------------------------------------
# Concentric
# ---------------------------------------------------------------------------
class Concentric(Constraint):
    """
    Forces two circles or arcs to share the same center point.

    Works with any pair of objects that have a ``.center`` attribute (i.e.,
    :class:`~cad_sketch.geometry.Circle` or :class:`~cad_sketch.geometry.Arc`).

    Parameters
    ----------
    entity1, entity2 : Circle or Arc
    """

    def __init__(self, entity1, entity2) -> None:
        super().__init__()
        if not (hasattr(entity1, "center") and hasattr(entity2, "center")):
            raise TypeError(
                "Concentric constraint requires two Circle or Arc objects."
            )
        self.entity1 = entity1
        self.entity2 = entity2

    def residuals(self) -> Residuals:
        """Return [cx1 - cx2, cy1 - cy2]."""
        c1 = self.entity1.center
        c2 = self.entity2.center
        return [c1.x - c2.x, c1.y - c2.y]

    def apply(self) -> None:
        """Move both centers to their midpoint."""
        c1 = self.entity1.center
        c2 = self.entity2.center
        mx = (c1.x + c2.x) / 2.0
        my = (c1.y + c2.y) / 2.0
        c1.x = c2.x = mx
        c1.y = c2.y = my

    def description(self) -> str:
        return f"Concentric({self.entity1.id}, {self.entity2.id})"

    def __repr__(self) -> str:
        return self.description()
