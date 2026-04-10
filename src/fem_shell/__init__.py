"""
fem-shell: Finite Element Method for shell, solid, and plane elements.

Provides mesh handling, materials, boundary conditions, constitutive models,
element formulations, solvers (static, dynamic, FSI with preCICE), and a
YAML-driven CLI for running simulations.
"""

__version__ = "0.1.0"

from .core.config import FSISimulationConfig
from .core.material import IsotropicMaterial, OrthotropicMaterial

# ---------------------------------------------------------------------------
# Core: mesh, materials, boundary conditions, configuration
# ---------------------------------------------------------------------------
from .core.mesh import (  # Model; Entities; Generators; I/O; Utilities
    BladeMesh,
    BoxSurfaceMesh,
    BoxVolumeMesh,
    ElementSet,
    ElementType,
    MeshElement,
    MeshModel,
    MultiFlapMesh,
    Node,
    NodeSet,
    RotorMesh,
    SquareShapeMesh,
    check_mesh_quality,
    load_mesh,
    verify_solid_element_orientations,
    write_mesh,
)

_bc_names = []
try:
    # Keep top-level imports tolerant to missing PETSc so lightweight CLI paths
    # like `--preview` can import fem_shell without pulling solver backends.
    from .core.bc import BodyForce, DirichletCondition

    _bc_names = ["BodyForce", "DirichletCondition"]
except ImportError:
    # PETSc not available — boundary-condition manager helpers disabled
    pass

# ---------------------------------------------------------------------------
# Constitutive models (failure criteria)
# ---------------------------------------------------------------------------
from .constitutive import (
    FailureMode,
    FailureResult,
    evaluate_ply_failure,
    hashin_failure_indices,
    max_stress_failure_index,
    tsai_wu_failure_index,
)

# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------
from .elements import ElementFactory, ElementFamily, FemElement

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
_solver_names = []
try:
    # Linear/static/modal solvers depend on PETSc. Import them lazily here so
    # mesh/config utilities remain usable in environments without petsc4py.
    from .solvers import LinearDynamicSolver, LinearStaticSolver, ModalSolver

    _solver_names = ["LinearStaticSolver", "LinearDynamicSolver", "ModalSolver"]
except ImportError:
    # PETSc not available — structural solvers disabled
    pass

_fsi_names = []
try:
    # FSI support is optional because it depends on preCICE and related native
    # libraries that are not required for preview/validation workflows.
    from .solvers.fsi import (
        FSIRunner,
        LinearDynamicFSIRotorSolver,
        LinearDynamicFSISolver,
        run_from_yaml,
    )

    _fsi_names = [
        "LinearDynamicFSISolver",
        "LinearDynamicFSIRotorSolver",
        "FSIRunner",
        "run_from_yaml",
    ]
except ImportError:
    # preCICE not available — FSI solvers disabled
    pass

# ---------------------------------------------------------------------------
# Stress recovery / post-processing
# ---------------------------------------------------------------------------
from .postprocess.stress_recovery import StrainResult, StressRecovery, StressResult

__all__ = (
    [
        # Meta
        "__version__",
        # Mesh
        "MeshModel",
        "Node",
        "MeshElement",
        "NodeSet",
        "ElementSet",
        "ElementType",
        "SquareShapeMesh",
        "BoxSurfaceMesh",
        "BoxVolumeMesh",
        "MultiFlapMesh",
        "BladeMesh",
        "RotorMesh",
        "load_mesh",
        "write_mesh",
        "check_mesh_quality",
        "verify_solid_element_orientations",
        # Materials
        "IsotropicMaterial",
        "OrthotropicMaterial",
        # Boundary conditions
        # Configuration
        "FSISimulationConfig",
        # Elements
        "ElementFactory",
        "ElementFamily",
        "FemElement",
        # Constitutive
        "FailureMode",
        "FailureResult",
        "tsai_wu_failure_index",
        "hashin_failure_indices",
        "max_stress_failure_index",
        "evaluate_ply_failure",
        # Solvers
        # Post-processing
        "StressRecovery",
        "StressResult",
        "StrainResult",
    ]
    + _bc_names
    + _solver_names
    + _fsi_names
)
