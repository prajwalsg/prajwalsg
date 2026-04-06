"""
geometry.py — 2D CAD Sketch geometry entities
==============================================
All geometry classes store coordinates as plain Python floats so that the
constraint solver can manipulate them in-place without requiring any special
data types.

Classes
-------
Point       — (x, y) — base building block for all geometry.
Line        — segment defined by two Point objects (start, end).
Polyline    — series of connected Line segments.
Circle      — center Point + radius.
Arc         — partial circle; stored as center + start_angle + end_angle + radius.
Rectangle   — 4-line closed shape defined by two corner Points.
Polygon     — N-sided closed shape defined by a list of Points.
Spline      — smooth curve through / near a list of control Points (scipy CubicSpline).
Slot        — two center Points + width → two parallel lines + two semicircular arcs.
"""

import math
import itertools
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# ID counter — every entity gets a unique string ID like "P1", "L3", etc.
# ---------------------------------------------------------------------------
_counters: dict = {}


def _next_id(prefix: str) -> str:
    """Return the next sequential ID for the given entity-type prefix."""
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{prefix}{_counters[prefix]}"


def reset_id_counters() -> None:
    """Reset all entity ID counters (useful for tests)."""
    _counters.clear()


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------
class Point:
    """
    A 2-D point with coordinates (x, y).

    Attributes
    ----------
    id : str
        Unique identifier, e.g. ``"P1"``.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    construction : bool
        When *True* the point is a reference/construction entity and is drawn
        as a faint marker rather than a solid entity.
    """

    def __init__(self, x: float, y: float, construction: bool = False) -> None:
        self.id: str = _next_id("P")
        self.x: float = float(x)
        self.y: float = float(y)
        self.construction: bool = construction

    # ------------------------------------------------------------------
    def distance_to(self, other: "Point") -> float:
        """Return the Euclidean distance to *other*."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[float, float]:
        """Return ``(x, y)`` as a plain tuple."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Point({self.id}: {self.x:.4g}, {self.y:.4g})"


# ---------------------------------------------------------------------------
# Line
# ---------------------------------------------------------------------------
class Line:
    """
    A straight line segment defined by two :class:`Point` objects.

    Attributes
    ----------
    id : str
        Unique identifier, e.g. ``"L1"``.
    start : Point
    end   : Point
    construction : bool
    """

    def __init__(
        self, start: Point, end: Point, construction: bool = False
    ) -> None:
        self.id: str = _next_id("L")
        self.start: Point = start
        self.end: Point = end
        self.construction: bool = construction

    # ------------------------------------------------------------------
    def length(self) -> float:
        """Return the length of the line segment."""
        return self.start.distance_to(self.end)

    def direction(self) -> Tuple[float, float]:
        """Return a unit direction vector ``(dx, dy)`` from start to end."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = math.hypot(dx, dy)
        if length < 1e-12:
            return (0.0, 0.0)
        return (dx / length, dy / length)

    def midpoint(self) -> Tuple[float, float]:
        """Return the midpoint as ``(x, y)``."""
        return (
            (self.start.x + self.end.x) / 2.0,
            (self.start.y + self.end.y) / 2.0,
        )

    def __repr__(self) -> str:
        return (
            f"Line({self.id}: {self.start.as_tuple()} → {self.end.as_tuple()})"
        )


# ---------------------------------------------------------------------------
# Polyline
# ---------------------------------------------------------------------------
class Polyline:
    """
    A series of connected :class:`Line` segments treated as one entity.

    Parameters
    ----------
    points : list of Point
        At least two points that define the polyline vertices.  A
        :class:`Line` segment is created for every consecutive pair.
    closed : bool
        When *True* an extra segment connecting the last point back to the
        first is added, forming a closed shape.
    """

    def __init__(self, points: List[Point], closed: bool = False) -> None:
        if len(points) < 2:
            raise ValueError("Polyline requires at least two points.")
        self.id: str = _next_id("PL")
        self.points: List[Point] = points
        self.closed: bool = closed
        self.segments: List[Line] = self._build_segments()

    def _build_segments(self) -> List[Line]:
        segs: List[Line] = []
        for a, b in zip(self.points, self.points[1:]):
            segs.append(Line(a, b))
        if self.closed:
            segs.append(Line(self.points[-1], self.points[0]))
        return segs

    def total_length(self) -> float:
        """Return the sum of segment lengths."""
        return sum(s.length() for s in self.segments)

    def __repr__(self) -> str:
        pts = ", ".join(str(p.as_tuple()) for p in self.points)
        return f"Polyline({self.id}: [{pts}], closed={self.closed})"


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------
class Circle:
    """
    A full circle defined by a center :class:`Point` and a radius.

    Attributes
    ----------
    id     : str
    center : Point
    radius : float
    """

    def __init__(self, center: Point, radius: float) -> None:
        if radius <= 0:
            raise ValueError("Circle radius must be positive.")
        self.id: str = _next_id("C")
        self.center: Point = center
        self.radius: float = float(radius)

    def area(self) -> float:
        """Return the area of the circle."""
        return math.pi * self.radius ** 2

    def circumference(self) -> float:
        """Return the circumference of the circle."""
        return 2.0 * math.pi * self.radius

    def __repr__(self) -> str:
        return (
            f"Circle({self.id}: center={self.center.as_tuple()}, "
            f"r={self.radius:.4g})"
        )


# ---------------------------------------------------------------------------
# Arc
# ---------------------------------------------------------------------------
class Arc:
    """
    A partial circle (arc).

    Can be constructed either:

    * **Center form** — ``center``, ``radius``, ``start_angle``, ``end_angle``
      (angles in **degrees**, measured counter-clockwise from the positive
      X-axis).
    * **Three-point form** — ``p1`` (start), ``p2`` (mid), ``p3`` (end).
      The center and radius are computed automatically.

    Attributes
    ----------
    id          : str
    center      : Point
    radius      : float
    start_angle : float  (degrees)
    end_angle   : float  (degrees)
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _from_three_points(
        p1: Point, p2: Point, p3: Point
    ) -> Tuple[Point, float, float, float]:
        """
        Given three points on an arc return ``(center, radius, start_angle,
        end_angle)``.
        """
        ax, ay = p1.x, p1.y
        bx, by = p2.x, p2.y
        cx_, cy_ = p3.x, p3.y

        d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-12:
            raise ValueError("Three points are collinear — cannot define an arc.")

        ux = (
            (ax ** 2 + ay ** 2) * (by - cy_)
            + (bx ** 2 + by ** 2) * (cy_ - ay)
            + (cx_ ** 2 + cy_ ** 2) * (ay - by)
        ) / d
        uy = (
            (ax ** 2 + ay ** 2) * (cx_ - bx)
            + (bx ** 2 + by ** 2) * (ax - cx_)
            + (cx_ ** 2 + cy_ ** 2) * (bx - ax)
        ) / d

        center = Point(ux, uy)
        radius = math.hypot(ax - ux, ay - uy)
        start_angle = math.degrees(math.atan2(ay - uy, ax - ux)) % 360.0
        end_angle = math.degrees(math.atan2(cy_ - uy, cx_ - ux)) % 360.0
        return center, radius, start_angle, end_angle

    # ------------------------------------------------------------------
    def __init__(
        self,
        center: Optional[Point] = None,
        radius: float = 1.0,
        start_angle: float = 0.0,
        end_angle: float = 180.0,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
    ) -> None:
        """
        Create an Arc either from ``(center, radius, start_angle, end_angle)``
        or from three points ``(p1, p2, p3)``.
        """
        if p1 is not None and p2 is not None and p3 is not None:
            center, radius, start_angle, end_angle = self._from_three_points(
                p1, p2, p3
            )
        elif center is None:
            raise ValueError(
                "Provide either a center point or three arc points (p1, p2, p3)."
            )

        self.id: str = _next_id("A")
        self.center: Point = center
        self.radius: float = float(radius)
        self.start_angle: float = float(start_angle) % 360.0
        self.end_angle: float = float(end_angle) % 360.0

    # ------------------------------------------------------------------
    def start_point(self) -> Tuple[float, float]:
        """Return the arc start point as ``(x, y)``."""
        a = math.radians(self.start_angle)
        return (
            self.center.x + self.radius * math.cos(a),
            self.center.y + self.radius * math.sin(a),
        )

    def end_point(self) -> Tuple[float, float]:
        """Return the arc end point as ``(x, y)``."""
        a = math.radians(self.end_angle)
        return (
            self.center.x + self.radius * math.cos(a),
            self.center.y + self.radius * math.sin(a),
        )

    def arc_length(self) -> float:
        """Return the arc length."""
        span = (self.end_angle - self.start_angle) % 360.0
        return self.radius * math.radians(span)

    def __repr__(self) -> str:
        return (
            f"Arc({self.id}: center={self.center.as_tuple()}, "
            f"r={self.radius:.4g}, "
            f"{self.start_angle:.2f}°→{self.end_angle:.2f}°)"
        )


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------
class Rectangle:
    """
    A closed rectangle defined either by two corner :class:`Point` objects or
    by a center point plus width and height.

    Internally stored as four :class:`Line` segments.

    Attributes
    ----------
    id       : str
    corners  : list of four Point objects (bottom-left, bottom-right,
               top-right, top-left)
    segments : list of four Line objects
    """

    def __init__(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        center: Optional[Point] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        p1, p2 : Point
            Bottom-left and top-right corners of the rectangle (either this
            pair *or* the ``center/width/height`` trio must be provided).
        center : Point
            Center point (used together with *width* and *height*).
        width, height : float
            Full width and height when using center form.
        """
        if p1 is not None and p2 is not None:
            x0, y0 = min(p1.x, p2.x), min(p1.y, p2.y)
            x1, y1 = max(p1.x, p2.x), max(p1.y, p2.y)
        elif center is not None and width is not None and height is not None:
            x0 = center.x - width / 2.0
            y0 = center.y - height / 2.0
            x1 = center.x + width / 2.0
            y1 = center.y + height / 2.0
        else:
            raise ValueError(
                "Provide either (p1, p2) or (center, width, height)."
            )

        self.id: str = _next_id("R")
        bl = Point(x0, y0)
        br = Point(x1, y0)
        tr = Point(x1, y1)
        tl = Point(x0, y1)
        self.corners: List[Point] = [bl, br, tr, tl]
        self.segments: List[Line] = [
            Line(bl, br),
            Line(br, tr),
            Line(tr, tl),
            Line(tl, bl),
        ]

    def width(self) -> float:
        """Return the width of the rectangle."""
        return abs(self.corners[1].x - self.corners[0].x)

    def height_val(self) -> float:
        """Return the height of the rectangle."""
        return abs(self.corners[2].y - self.corners[0].y)

    def area(self) -> float:
        """Return the area of the rectangle."""
        return self.width() * self.height_val()

    def __repr__(self) -> str:
        bl = self.corners[0].as_tuple()
        tr = self.corners[2].as_tuple()
        return f"Rectangle({self.id}: BL={bl}, TR={tr})"


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------
class Polygon:
    """
    An N-sided closed polygon defined by a list of :class:`Point` vertices.

    Internally stores N :class:`Line` segments.

    Attributes
    ----------
    id       : str
    vertices : list of Point
    segments : list of Line
    """

    def __init__(self, vertices: List[Point]) -> None:
        if len(vertices) < 3:
            raise ValueError("Polygon requires at least three vertices.")
        self.id: str = _next_id("PG")
        self.vertices: List[Point] = vertices
        self.segments: List[Line] = [
            Line(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices))
        ]

    def perimeter(self) -> float:
        """Return the perimeter of the polygon."""
        return sum(s.length() for s in self.segments)

    def area(self) -> float:
        """Return the signed area using the shoelace formula."""
        n = len(self.vertices)
        s = sum(
            self.vertices[i].x * self.vertices[(i + 1) % n].y
            - self.vertices[(i + 1) % n].x * self.vertices[i].y
            for i in range(n)
        )
        return abs(s) / 2.0

    def __repr__(self) -> str:
        pts = [v.as_tuple() for v in self.vertices]
        return f"Polygon({self.id}: {len(self.vertices)} vertices, pts={pts})"


# ---------------------------------------------------------------------------
# Spline
# ---------------------------------------------------------------------------
class Spline:
    """
    A smooth curve through (or near) a list of control :class:`Point` objects.

    Uses ``scipy.interpolate.CubicSpline`` when available; falls back to a
    simple Catmull-Rom implementation otherwise.

    Attributes
    ----------
    id             : str
    control_points : list of Point
    num_samples    : int
        Number of sample points used when evaluating / plotting the curve.
    """

    def __init__(
        self, control_points: List[Point], num_samples: int = 100
    ) -> None:
        if len(control_points) < 2:
            raise ValueError("Spline requires at least two control points.")
        self.id: str = _next_id("SP")
        self.control_points: List[Point] = control_points
        self.num_samples: int = num_samples

    # ------------------------------------------------------------------
    def evaluate(self) -> List[Tuple[float, float]]:
        """
        Return a list of ``(x, y)`` sample points along the spline.

        Uses ``scipy`` if available, otherwise uses Catmull-Rom interpolation.
        """
        xs = [p.x for p in self.control_points]
        ys = [p.y for p in self.control_points]

        try:
            from scipy.interpolate import CubicSpline
            import numpy as np

            t = list(range(len(xs)))
            cs_x = CubicSpline(t, xs)
            cs_y = CubicSpline(t, ys)
            t_fine = np.linspace(0, len(xs) - 1, self.num_samples)
            return list(zip(cs_x(t_fine).tolist(), cs_y(t_fine).tolist()))
        except ImportError:
            return self._catmull_rom(xs, ys)

    def _catmull_rom(
        self, xs: List[float], ys: List[float]
    ) -> List[Tuple[float, float]]:
        """Basic Catmull-Rom spline fallback (no scipy dependency)."""
        points = list(zip(xs, ys))
        # Extend with phantom endpoints
        pts = [points[0]] + points + [points[-1]]
        result: List[Tuple[float, float]] = []
        segments = len(pts) - 3
        spp = max(1, self.num_samples // max(segments, 1))

        for i in range(1, len(pts) - 2):
            p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
            for j in range(spp):
                t = j / spp
                t2, t3 = t * t, t * t * t
                x = 0.5 * (
                    2 * p1[0]
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    2 * p1[1]
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                result.append((x, y))
        result.append(points[-1])
        return result

    def __repr__(self) -> str:
        return (
            f"Spline({self.id}: {len(self.control_points)} control points)"
        )


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------
class Slot:
    """
    A slot shape defined by two center :class:`Point` objects and a *width*.

    Geometry: two parallel lines (top and bottom edges) connecting the two
    ends, plus two semicircular :class:`Arc` objects at each end.

    Attributes
    ----------
    id       : str
    c1       : Point — center of the first (left) rounded end
    c2       : Point — center of the second (right) rounded end
    width    : float — full width of the slot (= 2 × radius of end arcs)
    lines    : list of two Line objects
    arcs     : list of two Arc objects
    """

    def __init__(self, c1: Point, c2: Point, width: float) -> None:
        if width <= 0:
            raise ValueError("Slot width must be positive.")
        self.id: str = _next_id("SL")
        self.c1: Point = c1
        self.c2: Point = c2
        self.width: float = float(width)
        self.lines: List[Line] = []
        self.arcs: List[Arc] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        """Compute the four sub-entities that make up the slot."""
        r = self.width / 2.0
        dx = self.c2.x - self.c1.x
        dy = self.c2.y - self.c1.y
        length = math.hypot(dx, dy)

        # Angle of the slot axis (degrees)
        angle_deg = math.degrees(math.atan2(dy, dx))

        if length < 1e-12:
            # Degenerate case: the two centers coincide — emit a single circle
            self.arcs.append(Arc(center=self.c1, radius=r,
                                 start_angle=0.0, end_angle=360.0))
            return

        # Perpendicular unit vector
        px = -dy / length
        py = dx / length

        # Four corner points
        p1 = Point(self.c1.x + px * r, self.c1.y + py * r)
        p2 = Point(self.c2.x + px * r, self.c2.y + py * r)
        p3 = Point(self.c2.x - px * r, self.c2.y - py * r)
        p4 = Point(self.c1.x - px * r, self.c1.y - py * r)

        self.lines.append(Line(p1, p2))
        self.lines.append(Line(p3, p4))

        # Semi-circle at c2 (facing away from c1)
        a_start2 = angle_deg - 90.0
        a_end2 = angle_deg + 90.0
        self.arcs.append(
            Arc(center=self.c2, radius=r,
                start_angle=a_start2 % 360.0, end_angle=a_end2 % 360.0)
        )

        # Semi-circle at c1 (facing away from c2)
        a_start1 = (angle_deg + 90.0) % 360.0
        a_end1 = (angle_deg + 270.0) % 360.0
        self.arcs.append(
            Arc(center=self.c1, radius=r,
                start_angle=a_start1, end_angle=a_end1)
        )

    def __repr__(self) -> str:
        return (
            f"Slot({self.id}: c1={self.c1.as_tuple()}, "
            f"c2={self.c2.as_tuple()}, width={self.width:.4g})"
        )
