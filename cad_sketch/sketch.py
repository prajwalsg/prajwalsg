"""
sketch.py — Sketch class
=========================
The ``Sketch`` class is the central manager that:

* Stores all geometry entities (Point, Line, Circle, …) in an ordered dict
  keyed by their entity ID.
* Stores all constraints keyed by an auto-assigned constraint ID (``"CON1"``,
  ``"CON2"``, …).
* Calls the :class:`~cad_sketch.solver.ConstraintSolver` automatically after
  every ``add_*`` or ``add_constraint`` call.
* Provides convenience factory methods so callers don't have to import and
  wire up individual classes.
* Supports JSON export and human-readable ``view()`` output.
"""

import json
import math
from typing import Any, Dict, List, Optional, Union

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
    Constraint,
    Coincident,
    Horizontal,
    Vertical,
    Parallel,
    Perpendicular,
    Tangent,
    Concentric,
)
from .solver import ConstraintSolver


class Sketch:
    """
    A 2-D parametric sketch that holds geometry entities and constraints.

    Attributes
    ----------
    entities : dict
        ``{entity_id: entity_object}`` — all geometry in the sketch.
    constraints : dict
        ``{constraint_id: constraint_object}`` — all active constraints.
    solver : ConstraintSolver
        The solver instance used to resolve constraints after each change.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        max_solver_iterations: int = 200,
        solver_tolerance: float = 1e-6,
    ) -> None:
        self.entities: Dict[str, Any] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.solver = ConstraintSolver(
            max_iterations=max_solver_iterations,
            tolerance=solver_tolerance,
        )
        self._constraint_counter: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_entity(self, entity: Any) -> Any:
        """Register an entity and run the solver."""
        self.entities[entity.id] = entity
        self._run_solver()
        return entity

    def _next_constraint_id(self) -> str:
        self._constraint_counter += 1
        return f"CON{self._constraint_counter}"

    def _run_solver(self) -> bool:
        """Run the constraint solver and return convergence status."""
        return self.solver.solve(list(self.constraints.values()))

    # ------------------------------------------------------------------
    # Geometry factory methods
    # ------------------------------------------------------------------
    def add_point(self, x: float, y: float, construction: bool = False) -> Point:
        """
        Add a :class:`~cad_sketch.geometry.Point` to the sketch.

        Parameters
        ----------
        x, y         : float   — coordinates.
        construction : bool    — mark as construction/reference geometry.

        Returns
        -------
        Point
        """
        p = Point(x, y, construction=construction)
        return self._add_entity(p)

    def add_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        construction: bool = False,
    ) -> Line:
        """
        Add a :class:`~cad_sketch.geometry.Line` to the sketch.

        The start and end :class:`~cad_sketch.geometry.Point` objects are
        created automatically and also registered as entities.

        Parameters
        ----------
        x1, y1 : float — start coordinates.
        x2, y2 : float — end coordinates.
        construction : bool

        Returns
        -------
        Line
        """
        start = self.add_point(x1, y1)
        end = self.add_point(x2, y2)
        line = Line(start, end, construction=construction)
        return self._add_entity(line)

    def add_polyline(
        self, coords: List[tuple], closed: bool = False,
        construction: bool = False,
    ) -> Polyline:
        """
        Add a :class:`~cad_sketch.geometry.Polyline` to the sketch.

        Parameters
        ----------
        coords       : list of (x, y) tuples
        closed       : bool — connect last point back to first
        construction : bool — mark as reference geometry

        Returns
        -------
        Polyline
        """
        points = [self.add_point(x, y) for x, y in coords]
        pl = Polyline(points, closed=closed, construction=construction)
        # Register the internal segments' points (already registered above)
        return self._add_entity(pl)

    def add_circle(self, cx: float, cy: float, radius: float) -> Circle:
        """
        Add a :class:`~cad_sketch.geometry.Circle` to the sketch.

        Parameters
        ----------
        cx, cy : float — center coordinates.
        radius : float — radius (must be > 0).

        Returns
        -------
        Circle
        """
        center = self.add_point(cx, cy)
        c = Circle(center, radius)
        return self._add_entity(c)

    def add_arc_center(
        self,
        cx: float,
        cy: float,
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> Arc:
        """
        Add a :class:`~cad_sketch.geometry.Arc` (center form) to the sketch.

        Parameters
        ----------
        cx, cy      : float — center coordinates.
        radius      : float — radius.
        start_angle : float — start angle in degrees (CCW from +X axis).
        end_angle   : float — end angle in degrees.

        Returns
        -------
        Arc
        """
        center = self.add_point(cx, cy)
        a = Arc(center=center, radius=radius,
                start_angle=start_angle, end_angle=end_angle)
        return self._add_entity(a)

    def add_arc_3points(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
    ) -> Arc:
        """
        Add a :class:`~cad_sketch.geometry.Arc` (three-point form) to the sketch.

        Parameters
        ----------
        x1, y1 : float — start point.
        x2, y2 : float — mid point.
        x3, y3 : float — end point.

        Returns
        -------
        Arc
        """
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        p3 = Point(x3, y3)
        a = Arc(p1=p1, p2=p2, p3=p3)
        # Register the computed center as a construction point
        self.entities[a.center.id] = a.center
        return self._add_entity(a)

    def add_rectangle_corners(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Rectangle:
        """
        Add a :class:`~cad_sketch.geometry.Rectangle` (two-corner form).

        Parameters
        ----------
        x1, y1 : float — bottom-left corner.
        x2, y2 : float — top-right corner.

        Returns
        -------
        Rectangle
        """
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        r = Rectangle(p1=p1, p2=p2)
        for corner in r.corners:
            self.entities[corner.id] = corner
        for seg in r.segments:
            self.entities[seg.id] = seg
        return self._add_entity(r)

    def add_rectangle_center(
        self, cx: float, cy: float, width: float, height: float
    ) -> Rectangle:
        """
        Add a :class:`~cad_sketch.geometry.Rectangle` (center form).

        Parameters
        ----------
        cx, cy  : float — center coordinates.
        width   : float — total width.
        height  : float — total height.

        Returns
        -------
        Rectangle
        """
        center = Point(cx, cy)
        r = Rectangle(center=center, width=width, height=height)
        for corner in r.corners:
            self.entities[corner.id] = corner
        for seg in r.segments:
            self.entities[seg.id] = seg
        return self._add_entity(r)

    def add_polygon(self, coords: List[tuple]) -> Polygon:
        """
        Add a :class:`~cad_sketch.geometry.Polygon` to the sketch.

        Parameters
        ----------
        coords : list of (x, y) tuples — polygon vertices.

        Returns
        -------
        Polygon
        """
        vertices = [Point(x, y) for x, y in coords]
        for v in vertices:
            self.entities[v.id] = v
        pg = Polygon(vertices)
        for seg in pg.segments:
            self.entities[seg.id] = seg
        return self._add_entity(pg)

    def add_spline(
        self, coords: List[tuple], num_samples: int = 100
    ) -> Spline:
        """
        Add a :class:`~cad_sketch.geometry.Spline` to the sketch.

        Parameters
        ----------
        coords      : list of (x, y) control-point tuples.
        num_samples : int — evaluation resolution.

        Returns
        -------
        Spline
        """
        pts = [self.add_point(x, y) for x, y in coords]
        sp = Spline(pts, num_samples=num_samples)
        return self._add_entity(sp)

    def add_slot(
        self,
        cx1: float, cy1: float,
        cx2: float, cy2: float,
        width: float,
    ) -> Slot:
        """
        Add a :class:`~cad_sketch.geometry.Slot` to the sketch.

        Parameters
        ----------
        cx1, cy1 : float — center of the first rounded end.
        cx2, cy2 : float — center of the second rounded end.
        width    : float — slot width.

        Returns
        -------
        Slot
        """
        c1 = self.add_point(cx1, cy1)
        c2 = self.add_point(cx2, cy2)
        sl = Slot(c1, c2, width)
        for ln in sl.lines:
            self.entities[ln.start.id] = ln.start
            self.entities[ln.end.id] = ln.end
            self.entities[ln.id] = ln
        for arc in sl.arcs:
            self.entities[arc.center.id] = arc.center
            self.entities[arc.id] = arc
        return self._add_entity(sl)

    # ------------------------------------------------------------------
    # Constraint factory methods
    # ------------------------------------------------------------------
    def _register_constraint(self, c: Constraint) -> Constraint:
        """Assign an ID, register, solve, and return the constraint."""
        c.id = self._next_constraint_id()
        self.constraints[c.id] = c
        self._run_solver()
        return c

    def add_constraint_coincident(
        self, p1: Point, p2: Point
    ) -> Coincident:
        """Add a :class:`~cad_sketch.constraints.Coincident` constraint."""
        return self._register_constraint(Coincident(p1, p2))

    def add_constraint_horizontal(self, line: Line) -> Horizontal:
        """Add a :class:`~cad_sketch.constraints.Horizontal` constraint."""
        return self._register_constraint(Horizontal(line))

    def add_constraint_vertical(self, line: Line) -> Vertical:
        """Add a :class:`~cad_sketch.constraints.Vertical` constraint."""
        return self._register_constraint(Vertical(line))

    def add_constraint_parallel(
        self, line1: Line, line2: Line
    ) -> Parallel:
        """Add a :class:`~cad_sketch.constraints.Parallel` constraint."""
        return self._register_constraint(Parallel(line1, line2))

    def add_constraint_perpendicular(
        self, line1: Line, line2: Line
    ) -> Perpendicular:
        """Add a :class:`~cad_sketch.constraints.Perpendicular` constraint."""
        return self._register_constraint(Perpendicular(line1, line2))

    def add_constraint_tangent(
        self, line: Line, circle: Circle
    ) -> Tangent:
        """Add a :class:`~cad_sketch.constraints.Tangent` constraint."""
        return self._register_constraint(Tangent(line, circle))

    def add_constraint_concentric(self, e1, e2) -> Concentric:
        """Add a :class:`~cad_sketch.constraints.Concentric` constraint."""
        return self._register_constraint(Concentric(e1, e2))

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_entity(self, entity_id: str) -> bool:
        """
        Remove an entity from the sketch by ID.

        Also removes any constraints that reference this entity (detected by
        a simple ID-match on the constraint's description string).

        Parameters
        ----------
        entity_id : str

        Returns
        -------
        bool — *True* if an entity was found and removed.
        """
        if entity_id not in self.entities:
            return False
        del self.entities[entity_id]
        # Remove constraints that mention this entity
        to_remove = [
            cid
            for cid, c in self.constraints.items()
            if entity_id in c.description()
        ]
        for cid in to_remove:
            del self.constraints[cid]
        self._run_solver()
        return True

    def delete_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint from the sketch by ID.

        Parameters
        ----------
        constraint_id : str

        Returns
        -------
        bool
        """
        if constraint_id not in self.constraints:
            return False
        del self.constraints[constraint_id]
        self._run_solver()
        return True

    def clear(self) -> None:
        """Remove all entities and constraints from the sketch."""
        self.entities.clear()
        self.constraints.clear()

    # ------------------------------------------------------------------
    # View / display
    # ------------------------------------------------------------------
    def view(self) -> str:
        """
        Return a formatted multi-line string listing all entities and
        constraints currently in the sketch.
        """
        lines: List[str] = ["=== Sketch Entities ==="]
        if not self.entities:
            lines.append("  (empty)")
        else:
            for eid, ent in self.entities.items():
                lines.append(f"  {eid}: {ent}")
        lines.append("")
        lines.append("=== Constraints ===")
        if not self.constraints:
            lines.append("  (none)")
        else:
            for cid, con in self.constraints.items():
                residuals = con.residuals()
                if residuals:
                    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
                else:
                    rms = 0.0
                lines.append(
                    f"  {cid}: {con.description()}  (rms={rms:.3e})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the sketch to a plain Python dictionary (JSON-compatible).

        Returns
        -------
        dict with keys ``"entities"`` and ``"constraints"``.
        """

        def _serialize(entity: Any) -> Dict[str, Any]:
            t = type(entity).__name__
            d: Dict[str, Any] = {"type": t, "id": entity.id}
            if isinstance(entity, Point):
                d.update({"x": entity.x, "y": entity.y,
                           "construction": entity.construction})
            elif isinstance(entity, Line):
                d.update({
                    "start": entity.start.as_tuple(),
                    "end": entity.end.as_tuple(),
                    "construction": entity.construction,
                })
            elif isinstance(entity, Polyline):
                d.update({
                    "points": [p.as_tuple() for p in entity.points],
                    "closed": entity.closed,
                })
            elif isinstance(entity, Circle):
                d.update({
                    "center": entity.center.as_tuple(),
                    "radius": entity.radius,
                })
            elif isinstance(entity, Arc):
                d.update({
                    "center": entity.center.as_tuple(),
                    "radius": entity.radius,
                    "start_angle": entity.start_angle,
                    "end_angle": entity.end_angle,
                })
            elif isinstance(entity, Rectangle):
                d.update({
                    "corners": [c.as_tuple() for c in entity.corners],
                })
            elif isinstance(entity, Polygon):
                d.update({
                    "vertices": [v.as_tuple() for v in entity.vertices],
                })
            elif isinstance(entity, Spline):
                d.update({
                    "control_points": [p.as_tuple()
                                       for p in entity.control_points],
                    "num_samples": entity.num_samples,
                })
            elif isinstance(entity, Slot):
                d.update({
                    "c1": entity.c1.as_tuple(),
                    "c2": entity.c2.as_tuple(),
                    "width": entity.width,
                })
            return d

        return {
            "entities": {eid: _serialize(e)
                         for eid, e in self.entities.items()},
            "constraints": {
                cid: {"id": cid, "description": c.description()}
                for cid, c in self.constraints.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the sketch to a JSON string.

        Parameters
        ----------
        indent : int — JSON indentation width.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=indent)
