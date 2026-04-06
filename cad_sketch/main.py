"""
main.py — Menu-driven CLI entry point for the 2D CAD Sketch Tool
=================================================================
Run this module directly to start the interactive sketcher::

    python -m cad_sketch.main
    # or, from the repo root:
    python cad_sketch/main.py

Main menu options
-----------------
1. Add Geometry       — choose type, enter parameters
2. Apply Constraint   — choose type, pick entities by ID
3. View Sketch        — print all entities and constraints
4. Visualize          — render with matplotlib
5. Delete Entity      — remove by ID
6. Delete Constraint  — remove by ID
7. Clear Sketch       — remove everything
8. Export to JSON     — print JSON or save to file
0. Exit
"""

import sys
import json
import os

# Allow running directly as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_sketch.sketch import Sketch
from cad_sketch.visualizer import visualize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _input(prompt: str) -> str:
    """Wrapper around input() that gracefully handles EOF (e.g., piped input)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _prompt_float(prompt: str) -> float:
    """Prompt the user for a float value, re-asking on invalid input."""
    while True:
        raw = _input(prompt)
        try:
            return float(raw)
        except ValueError:
            print(f"  Invalid number: '{raw}'. Please enter a numeric value.")


def _prompt_xy(label: str = "point") -> tuple:
    """Prompt the user for 'x y' coordinates and return (x, y)."""
    while True:
        raw = _input(f"  Enter {label} (x y): ")
        parts = raw.split()
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        print("  Please enter two numbers separated by a space, e.g. '10 20'.")


def _pick_entity(sketch: Sketch, prompt: str, entity_type=None):
    """
    Prompt the user to pick an entity by ID.

    Parameters
    ----------
    sketch      : Sketch
    prompt      : str
    entity_type : type or None — if given, filter to only matching types.

    Returns
    -------
    entity object or None if not found / cancelled.
    """
    while True:
        eid = _input(prompt)
        if eid == "":
            return None
        if eid not in sketch.entities:
            print(f"  No entity with ID '{eid}'. Available IDs:")
            for k, v in sketch.entities.items():
                if entity_type is None or isinstance(v, entity_type):
                    print(f"    {k}: {v}")
            continue
        ent = sketch.entities[eid]
        if entity_type is not None and not isinstance(ent, entity_type):
            print(
                f"  Entity '{eid}' is of type {type(ent).__name__}, "
                f"but expected {entity_type.__name__}."
            )
            continue
        return ent


# ---------------------------------------------------------------------------
# Add-geometry sub-menu
# ---------------------------------------------------------------------------
def _menu_add_geometry(sketch: Sketch) -> None:
    """Interactive sub-menu for adding geometry."""
    from cad_sketch.geometry import Line, Circle, Arc

    print(
        "\n  Geometry types:\n"
        "    a. Point\n"
        "    b. Line\n"
        "    c. Circle\n"
        "    d. Arc (center form)\n"
        "    e. Arc (3-point form)\n"
        "    f. Rectangle (corner form)\n"
        "    g. Rectangle (center form)\n"
        "    h. Polygon\n"
        "    i. Spline\n"
        "    j. Slot\n"
        "    k. Polyline\n"
        "    0. Back\n"
    )
    choice = _input("  Choose geometry type: ").lower()

    if choice == "0":
        return

    elif choice == "a":
        x, y = _prompt_xy("the point")
        p = sketch.add_point(x, y)
        print(f"  Point added with ID: {p.id}")

    elif choice == "b":
        x1, y1 = _prompt_xy("start point")
        x2, y2 = _prompt_xy("end point")
        ln = sketch.add_line(x1, y1, x2, y2)
        print(f"  Line added with ID: {ln.id}")

    elif choice == "c":
        cx, cy = _prompt_xy("center point")
        r = _prompt_float("  Enter radius: ")
        c = sketch.add_circle(cx, cy, r)
        print(f"  Circle added with ID: {c.id}")

    elif choice == "d":
        cx, cy = _prompt_xy("center point")
        r = _prompt_float("  Enter radius: ")
        sa = _prompt_float("  Enter start angle (degrees, 0=+X, CCW): ")
        ea = _prompt_float("  Enter end angle   (degrees): ")
        a = sketch.add_arc_center(cx, cy, r, sa, ea)
        print(f"  Arc added with ID: {a.id}")

    elif choice == "e":
        x1, y1 = _prompt_xy("start point (p1)")
        x2, y2 = _prompt_xy("mid point   (p2)")
        x3, y3 = _prompt_xy("end point   (p3)")
        a = sketch.add_arc_3points(x1, y1, x2, y2, x3, y3)
        print(f"  Arc added with ID: {a.id}")

    elif choice == "f":
        x1, y1 = _prompt_xy("bottom-left corner")
        x2, y2 = _prompt_xy("top-right corner")
        r = sketch.add_rectangle_corners(x1, y1, x2, y2)
        print(f"  Rectangle added with ID: {r.id}")

    elif choice == "g":
        cx, cy = _prompt_xy("center point")
        w = _prompt_float("  Enter width: ")
        h = _prompt_float("  Enter height: ")
        r = sketch.add_rectangle_center(cx, cy, w, h)
        print(f"  Rectangle added with ID: {r.id}")

    elif choice == "h":
        n = int(_prompt_float("  Enter number of vertices: "))
        coords = []
        for i in range(n):
            x, y = _prompt_xy(f"vertex {i + 1}")
            coords.append((x, y))
        pg = sketch.add_polygon(coords)
        print(f"  Polygon added with ID: {pg.id}")

    elif choice == "i":
        n = int(_prompt_float("  Enter number of control points: "))
        coords = []
        for i in range(n):
            x, y = _prompt_xy(f"control point {i + 1}")
            coords.append((x, y))
        sp = sketch.add_spline(coords)
        print(f"  Spline added with ID: {sp.id}")

    elif choice == "j":
        cx1, cy1 = _prompt_xy("center of first end")
        cx2, cy2 = _prompt_xy("center of second end")
        w = _prompt_float("  Enter slot width: ")
        sl = sketch.add_slot(cx1, cy1, cx2, cy2, w)
        print(f"  Slot added with ID: {sl.id}")

    elif choice == "k":
        n = int(_prompt_float("  Enter number of points: "))
        coords = []
        for i in range(n):
            x, y = _prompt_xy(f"point {i + 1}")
            coords.append((x, y))
        closed_str = _input("  Closed polyline? (y/n): ").lower()
        closed = closed_str == "y"
        pl = sketch.add_polyline(coords, closed=closed)
        print(f"  Polyline added with ID: {pl.id}")

    else:
        print(f"  Unknown choice '{choice}'.")


# ---------------------------------------------------------------------------
# Apply-constraint sub-menu
# ---------------------------------------------------------------------------
def _menu_apply_constraint(sketch: Sketch) -> None:
    """Interactive sub-menu for applying geometric constraints."""
    from cad_sketch.geometry import Line, Circle, Arc, Point

    print(
        "\n  Constraint types:\n"
        "    a. Coincident   (two points)\n"
        "    b. Horizontal   (one line)\n"
        "    c. Vertical     (one line)\n"
        "    d. Parallel     (two lines)\n"
        "    e. Perpendicular(two lines)\n"
        "    f. Tangent      (line + circle)\n"
        "    g. Concentric   (two circles/arcs)\n"
        "    0. Back\n"
    )
    choice = _input("  Choose constraint type: ").lower()

    if choice == "0":
        return

    elif choice == "a":
        p1 = _pick_entity(sketch, "  Enter ID of first point: ", Point)
        p2 = _pick_entity(sketch, "  Enter ID of second point: ", Point)
        if p1 and p2:
            c = sketch.add_constraint_coincident(p1, p2)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "b":
        ln = _pick_entity(sketch, "  Enter line ID: ", Line)
        if ln:
            c = sketch.add_constraint_horizontal(ln)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "c":
        ln = _pick_entity(sketch, "  Enter line ID: ", Line)
        if ln:
            c = sketch.add_constraint_vertical(ln)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "d":
        ln1 = _pick_entity(sketch, "  Enter first line ID: ", Line)
        ln2 = _pick_entity(sketch, "  Enter second line ID: ", Line)
        if ln1 and ln2:
            c = sketch.add_constraint_parallel(ln1, ln2)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "e":
        ln1 = _pick_entity(sketch, "  Enter first line ID: ", Line)
        ln2 = _pick_entity(sketch, "  Enter second line ID: ", Line)
        if ln1 and ln2:
            c = sketch.add_constraint_perpendicular(ln1, ln2)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "f":
        ln = _pick_entity(sketch, "  Enter line ID: ", Line)
        circ = _pick_entity(sketch, "  Enter circle ID: ", Circle)
        if ln and circ:
            c = sketch.add_constraint_tangent(ln, circ)
            print(f"  Constraint added with ID: {c.id}")

    elif choice == "g":
        e1 = _pick_entity(sketch, "  Enter first circle/arc ID: ")
        e2 = _pick_entity(sketch, "  Enter second circle/arc ID: ")
        if e1 and e2:
            try:
                c = sketch.add_constraint_concentric(e1, e2)
                print(f"  Constraint added with ID: {c.id}")
            except TypeError as exc:
                print(f"  Error: {exc}")

    else:
        print(f"  Unknown choice '{choice}'.")


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------
def run_cli() -> None:
    """Start the interactive CLI."""
    sketch = Sketch()

    print("=" * 50)
    print("       2D CAD Sketch Tool — Python Edition")
    print("=" * 50)

    while True:
        print(
            "\n=== Main Menu ===\n"
            "  1. Add Geometry\n"
            "  2. Apply Constraint\n"
            "  3. View Sketch\n"
            "  4. Visualize\n"
            "  5. Delete Entity\n"
            "  6. Delete Constraint\n"
            "  7. Clear Sketch\n"
            "  8. Export to JSON\n"
            "  0. Exit\n"
        )
        choice = _input("Choose: ").strip()

        if choice == "0":
            print("Goodbye.")
            break

        elif choice == "1":
            _menu_add_geometry(sketch)

        elif choice == "2":
            _menu_apply_constraint(sketch)

        elif choice == "3":
            print(sketch.view())

        elif choice == "4":
            try:
                visualize(sketch, show=True)
            except Exception as exc:
                print(f"  Visualization error: {exc}")

        elif choice == "5":
            eid = _input("  Enter entity ID to delete: ")
            if sketch.delete_entity(eid):
                print(f"  Entity '{eid}' deleted.")
            else:
                print(f"  No entity with ID '{eid}'.")

        elif choice == "6":
            cid = _input("  Enter constraint ID to delete: ")
            if sketch.delete_constraint(cid):
                print(f"  Constraint '{cid}' deleted.")
            else:
                print(f"  No constraint with ID '{cid}'.")

        elif choice == "7":
            sketch.clear()
            print("  Sketch cleared.")

        elif choice == "8":
            dest = _input(
                "  Enter filename to save (leave empty to print to console): "
            )
            json_str = sketch.to_json(indent=2)
            if dest:
                try:
                    with open(dest, "w", encoding="utf-8") as fh:
                        fh.write(json_str)
                    print(f"  Sketch exported to '{dest}'.")
                except OSError as exc:
                    print(f"  Could not write file: {exc}")
            else:
                print(json_str)

        else:
            print(f"  Unknown option '{choice}'.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_cli()
