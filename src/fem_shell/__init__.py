"""
fem-shell: Finite Element Method for shell, solid, and plane elements.

Provides mesh handling, materials, boundary conditions, constitutive models,
element formulations, solvers (static, dynamic, FSI with preCICE), and a
YAML-driven CLI for running simulations.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Core: mesh, materials, boundary conditions, configuration
# ---------------------------------------------------------------------------
from .core.mesh import (
    # Model
    MeshModel,
    # Entities
    Node,
    MeshElement,
    NodeSet,
    ElementSet,
    ElementType,
    # Generators
    SquareShapeMesh,
    BoxSurfaceMesh,
    BoxVolumeMesh,
    MultiFlapMesh,
    BladeMesh,
    RotorMesh,
    # I/O
    load_mesh,
    write_mesh,
    # Utilities
    check_mesh_quality,
    verify_solid_element_orientations,
)
from .core.material import IsotropicMaterial, OrthotropicMaterial
from .core.bc import BodyForce, DirichletCondition
from .core.config import FSISimulationConfig

# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------
from .elements import ElementFactory, ElementFamily, FemElement

# ---------------------------------------------------------------------------
# Constitutive models (failure criteria)
# ---------------------------------------------------------------------------
from .constitutive import (
    FailureMode,
    FailureResult,
    tsai_wu_failure_index,
    hashin_failure_indices,
    max_stress_failure_index,
    evaluate_ply_failure,
)

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
from .solvers import LinearStaticSolver, LinearDynamicSolver, ModalSolver

_fsi_names = []
try:
    from .solvers.fsi import (
        LinearDynamicFSISolver,
        LinearDynamicFSIRotorSolver,
        FSIRunner,
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
from .postprocess.stress_recovery import StressRecovery, StressResult, StrainResult

__all__ = [
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
    "BodyForce",
    "DirichletCondition",
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
    "LinearStaticSolver",
    "LinearDynamicSolver",
    "ModalSolver",
    # Post-processing
    "StressRecovery",
    "StressResult",
    "StrainResult",
] + _fsi_names
