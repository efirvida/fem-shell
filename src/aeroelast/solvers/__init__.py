from aeroelast.postprocess.stress_recovery import (
    StrainResult,
    StressLocation,
    StressRecovery,
    StressResult,
    StressType,
)

_linear_names = []
try:
    # Import PETSc-backed structural solvers only when the dependency is
    # available, so importing the solvers package does not break CLI preview/view.
    from .elasticity import DynamicNewmarkSolver, StaticLinearSolver, StaticNonlinearSolver
    from .modal import ModalSolver

    # Legacy aliases kept for backward compatibility
    LinearStaticSolver = StaticLinearSolver
    LinearDynamicSolver = DynamicNewmarkSolver
    NonlinearStaticSolver = StaticNonlinearSolver

    _linear_names = [
        "StaticLinearSolver",
        "StaticNonlinearSolver",
        "DynamicNewmarkSolver",
        "LinearStaticSolver",
        "LinearDynamicSolver",
        "NonlinearStaticSolver",
        "ModalSolver",
    ]
except ImportError:
    # PETSc not available
    pass

try:
    # FSI classes remain optional because preCICE is not part of the minimal
    # installation used for config inspection and mesh preprocessing.
    from .fsi import (
        Adapter,
        ConstantOmega,
        CoordinateTransforms,
        ForceClipper,
        FSIRunner,
        FunctionOmega,
        InertialForcesCalculator,
        LinearDynamicFSIRotorSolver,
        LinearDynamicFSISolver,
        OmegaProvider,
        TableOmega,
        run_from_yaml,
    )
except ImportError:
    # preCICE not available
    pass

__all__ = [
    "StrainResult",
    "StressLocation",
    "StressRecovery",
    "StressResult",
    "StressType",
] + _linear_names
