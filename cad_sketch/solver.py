"""
solver.py — Geometric constraint solver
========================================
Implements an iterative solver that drives all constraint residuals toward
zero.  The approach is a simple **penalty / gradient-descent** method:

1. Collect all active constraints.
2. For each constraint, call ``constraint.apply()`` which makes a direct
   analytical correction to the geometry.
3. Repeat until the total squared residual is below the tolerance or the
   maximum number of iterations is reached.

The ``apply()`` method on each constraint is an *analytical, exact* correction
for that single constraint — e.g. ``Horizontal.apply()`` immediately sets both
endpoints to the average y.  Because each correction may invalidate another
constraint, we iterate until convergence.

This is mathematically equivalent to a Gauss-Seidel / block-coordinate-
descent approach over the constraint equations.

For a production-grade solver (over-constrained or under-constrained
detection, Newton-Raphson with a full Jacobian, etc.) a library such as
`scipy.optimize.fsolve` could replace this inner loop — the residual
interface is already compatible.
"""

import math
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .constraints import Constraint


class ConstraintSolver:
    """
    Iterative constraint solver.

    Parameters
    ----------
    max_iterations : int
        Maximum number of Gauss-Seidel sweeps before giving up.
    tolerance : float
        Convergence threshold: stop when the root-mean-square (RMS) of all
        residuals falls below this value.
    """

    def __init__(
        self, max_iterations: int = 200, tolerance: float = 1e-6
    ) -> None:
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance
        self._last_rms: float = 0.0
        self._iterations_used: int = 0

    # ------------------------------------------------------------------
    def solve(self, constraints: List["Constraint"]) -> bool:
        """
        Run the iterative solver on *constraints*.

        Parameters
        ----------
        constraints : list of Constraint
            All active constraints in the sketch.

        Returns
        -------
        bool
            ``True`` if the solver converged within the tolerance,
            ``False`` if the maximum number of iterations was reached.
        """
        if not constraints:
            self._last_rms = 0.0
            self._iterations_used = 0
            return True

        for iteration in range(self.max_iterations):
            # One Gauss-Seidel sweep: apply each constraint in sequence
            for c in constraints:
                c.apply()

            # Evaluate convergence
            rms = self._compute_rms(constraints)
            if rms < self.tolerance:
                self._last_rms = rms
                self._iterations_used = iteration + 1
                return True

        self._last_rms = self._compute_rms(constraints)
        self._iterations_used = self.max_iterations
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rms(constraints: List["Constraint"]) -> float:
        """Return the root-mean-square of all constraint residuals."""
        total = 0.0
        count = 0
        for c in constraints:
            for r in c.residuals():
                total += r * r
                count += 1
        if count == 0:
            return 0.0
        return math.sqrt(total / count)

    # ------------------------------------------------------------------
    @property
    def last_rms(self) -> float:
        """RMS residual from the most recent :meth:`solve` call."""
        return self._last_rms

    @property
    def iterations_used(self) -> int:
        """Number of iterations used in the most recent :meth:`solve` call."""
        return self._iterations_used
