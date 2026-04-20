"""Compatibility shim — do not add new code here.

All solver implementations live in ``aeroelast.solvers.elasticity``.
This module re-exports the old names so existing imports keep working.
"""

from aeroelast.solvers.elasticity.dynamic_newmark import DynamicNewmarkSolver as LinearDynamicSolver
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver as LinearStaticSolver
from aeroelast.solvers.elasticity.static_nonlinear import StaticNonlinearSolver as NonlinearStaticSolver

__all__ = [
    "LinearStaticSolver",
    "LinearDynamicSolver",
    "NonlinearStaticSolver",
]
