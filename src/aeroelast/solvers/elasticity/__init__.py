"""elasticity — structural solvers organized by physics and integration scheme.

Public names
------------
StaticLinearSolver      K·u = F  (KSP CG + BoomerAMG)
StaticNonlinearSolver   R(u) = 0  (SNES Newton-Raphson via Rust)
DynamicNewmarkSolver    M·ü + C·u̇ + K·u = F(t)  (Newmark-β)
"""

from .dynamic_newmark import DynamicNewmarkSolver
from .static_linear import StaticLinearSolver
from .static_nonlinear import StaticNonlinearSolver

__all__ = [
    "StaticLinearSolver",
    "StaticNonlinearSolver",
    "DynamicNewmarkSolver",
]
