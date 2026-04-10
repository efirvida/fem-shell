"""
FSI (Fluid-Structure Interaction) solvers module.

This module provides solvers for FSI problems using preCICE coupling.
Imports are deferred so that modules requiring preCICE (base, corotational,
linear_dynamic, rotor) are only loaded when actually accessed.
"""

__all__ = [
    # Base classes
    "Adapter",
    "ForceClipper",
    "NewmarkCoefficients",
    "SolverState",
    # Solvers
    "LinearDynamicFSISolver",
    "LinearDynamicFSIRotorSolver",
    # Co-rotational utilities
    "CoordinateTransforms",
    "InertialForcesCalculator",
    "OmegaProvider",
    "ComputedOmega",
    "ConstantOmega",
    "TableOmega",
    "FunctionOmega",
    # Runner
    "FSIRunner",
    "run_from_yaml",
]

_LAZY_IMPORTS = {
    # base
    "Adapter": ".base",
    "ForceClipper": ".base",
    "NewmarkCoefficients": ".base",
    "SolverState": ".base",
    # corotational
    "ComputedOmega": ".corotational",
    "ConstantOmega": ".corotational",
    "CoordinateTransforms": ".corotational",
    "FunctionOmega": ".corotational",
    "InertialForcesCalculator": ".corotational",
    "OmegaProvider": ".corotational",
    "TableOmega": ".corotational",
    # solvers
    "LinearDynamicFSISolver": ".linear_dynamic",
    "LinearDynamicFSIRotorSolver": ".rotor",
    # runner
    "FSIRunner": ".runner",
    "run_from_yaml": ".runner",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
