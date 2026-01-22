"""
FSI (Fluid-Structure Interaction) solvers module.

This module provides solvers for FSI problems using preCICE coupling.
"""

from .base import Adapter, ForceClipper, SolverState
from .corotational import (
    ConstantOmega,
    CoordinateTransforms,
    FunctionOmega,
    InertialForcesCalculator,
    OmegaProvider,
    TableOmega,
)
from .linear_dynamic import LinearDynamicFSISolver
from .rotor import LinearDynamicFSIRotorSolver
from .runner import FSIRunner, run_from_yaml

__all__ = [
    # Base classes
    "Adapter",
    "ForceClipper",
    "SolverState",
    # Solvers
    "LinearDynamicFSISolver",
    "LinearDynamicFSIRotorSolver",
    # Co-rotational utilities
    "CoordinateTransforms",
    "InertialForcesCalculator",
    "OmegaProvider",
    "ConstantOmega",
    "TableOmega",
    "FunctionOmega",
    # Runner
    "FSIRunner",
    "run_from_yaml",
]
