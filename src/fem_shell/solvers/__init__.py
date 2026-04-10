from fem_shell.postprocess.stress_recovery import (
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
    from .linear import LinearDynamicSolver, LinearStaticSolver
    from .modal import ModalSolver

    _linear_names = ["LinearDynamicSolver", "LinearStaticSolver", "ModalSolver"]
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
