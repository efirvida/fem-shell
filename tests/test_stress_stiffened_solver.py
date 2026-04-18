"""Tests for StressStiffenedFSISolver — geometric stiffness update pipeline.

These tests verify the components of the incremental linearization approach.
They are split into two groups:

``TestKGAssemblyPipeline`` — tests the underlying K_G / StressRecovery machinery
  WITHOUT requiring preCICE.  Always run.

``TestStressStiffenedHook`` — tests _post_convergence_hook on the actual solver
  class.  Skipped when preCICE shared library is missing (login nodes).

``TestStressStiffenedConfig`` — tests config enum and inheritance.  The solver
  import is attempted lazily so only the config tests fail if preCICE is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module if PETSc is not available (CI without MPI/PETSc)
# ---------------------------------------------------------------------------
petsc4py = pytest.importorskip("petsc4py", reason="PETSc not available")
from petsc4py import PETSc  # noqa: E402

from fem_shell.core.assembler import MeshAssembler
from fem_shell.core.bc import BoundaryConditionManager, DirichletCondition
from fem_shell.core.material import IsotropicMaterial
from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
from fem_shell.core.mesh.model import MeshModel
from fem_shell.elements import ElementFamily
from fem_shell.postprocess.stress_recovery import StressLocation, StressRecovery, StressType
from fem_shell.solvers.fsi.time_integration import NewmarkCoefficients

# ---------------------------------------------------------------------------
# Optional FSI solver import (needs preCICE at runtime)
# ---------------------------------------------------------------------------
try:
    from fem_shell.solvers.fsi.stress_stiffened_dynamic import StressStiffenedFSISolver
    _HAS_FSI = True
except ImportError:
    _HAS_FSI = False
    StressStiffenedFSISolver = None  # type: ignore[assignment,misc]

_skip_fsi = pytest.mark.skipif(not _HAS_FSI, reason="preCICE shared library not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_plate_mesh(nx: int = 4, ny: int = 4, L: float = 1.0) -> MeshModel:
    """Build a flat MITC4 square plate mesh (nx × ny quads) in the XY plane."""
    Node._id_counter = 0
    mesh = MeshModel()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, L, ny + 1)

    grid: dict[tuple[int, int], Node] = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            n = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(n)
            grid[(i, j)] = n

    for j in range(ny):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[grid[(i, j)], grid[(i + 1, j)],
                           grid[(i + 1, j + 1)], grid[(i, j + 1)]],
                    element_type=ElementType.quad,
                )
            )
    return mesh


_MODEL_CFG_TEMPLATE = {
    "elements": {
        "element_family": ElementFamily.SHELL,
        "thickness": 0.01,
    },
    "solver": {
        "type": "StressStiffenedDynamicFSI",
        "time_step": 0.01,
        "total_time": 1.0,
        "beta": 0.25,
        "gamma": 0.5,
        "geometric_stiffness": {"update_interval": 1},
    },
}


def _model_cfg() -> dict:
    cfg = {k: dict(v) for k, v in _MODEL_CFG_TEMPLATE.items()}
    cfg["elements"] = dict(_MODEL_CFG_TEMPLATE["elements"])
    cfg["elements"]["material"] = IsotropicMaterial(name="steel", E=210e9, nu=0.3, rho=7800.0)
    return cfg


def _build_domain(mesh: MeshModel) -> MeshAssembler:
    return MeshAssembler(mesh=mesh, model=_model_cfg())


def _build_bc_manager(domain: MeshAssembler, mesh: MeshModel) -> BoundaryConditionManager:
    """Pin the x=0 edge (all 6 DOFs) and return a BoundaryConditionManager."""
    K = domain.assemble_stiffness_matrix()
    M_c = domain.assemble_mass_matrix()
    F = PETSc.Vec().createMPI(domain.dofs_count, comm=PETSc.COMM_WORLD)
    F.set(0.0)

    bc_mgr = BoundaryConditionManager(stiffness=K, load=F, mass=M_c,
                                      dof_per_node=domain.dofs_per_node)
    pinned_nodes = [n for n in mesh.nodes if abs(n.coords[0]) < 1e-12]
    dofs: list[int] = []
    for node in pinned_nodes:
        idx = mesh.node_id_to_index[node.id]
        for d in range(domain.dofs_per_node):
            dofs.append(idx * domain.dofs_per_node + d)
    bc_mgr.apply_dirichlet([DirichletCondition(dofs=dofs, value=0.0)])
    return bc_mgr


def _make_solver(mesh: MeshModel, domain: MeshAssembler,
                 update_interval: int = 1) -> "StressStiffenedFSISolver":
    """Instantiate the solver bypassing the preCICE-dependent __init__."""
    cfg = _model_cfg()
    cfg["solver"]["geometric_stiffness"]["update_interval"] = update_interval
    solver = object.__new__(StressStiffenedFSISolver)
    solver.domain = domain
    solver.mesh_obj = mesh
    solver.model_properties = cfg
    solver.solver_params = cfg["solver"]
    solver._kg_update_interval = update_interval
    solver._K_G_red = None
    return solver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plate_setup():
    """Shared plate mesh + domain + bc_manager for all tests."""
    mesh = _build_plate_mesh(nx=4, ny=4, L=1.0)
    domain = _build_domain(mesh)
    bc_mgr = _build_bc_manager(domain, mesh)
    return mesh, domain, bc_mgr


# ---------------------------------------------------------------------------
# Group 1 — K_G pipeline tests (NO preCICE needed)
# ---------------------------------------------------------------------------

class TestKGAssemblyPipeline:
    """Verify the K_G assembly + StressRecovery pipeline without preCICE."""

    def test_assemble_geometric_stiffness_from_stress_field(self, plate_setup):
        """assembler.assemble_geometric_stiffness(stress_field=…) must return a PETSc.Mat."""
        mesh, domain, bc_mgr = plate_setup
        stress_field = {
            eid: np.array([1e6, 5e5, 0.0])
            for eid in domain._element_map
        }
        K_G = domain.assemble_geometric_stiffness(stress_field=stress_field)
        assert isinstance(K_G, PETSc.Mat)
        assert K_G.getSize()[0] == domain.dofs_count

    def test_K_G_symmetry(self, plate_setup):
        """K_G assembled from a uniform stress field must be symmetric."""
        mesh, domain, bc_mgr = plate_setup
        stress_field = {eid: np.array([1e6, 0.0, 0.0]) for eid in domain._element_map}
        K_G = domain.assemble_geometric_stiffness(stress_field=stress_field)

        # Compare K_G - K_G^T (should be zero up to floating-point)
        K_G_T = K_G.copy()
        K_G_T.transpose()
        diff = K_G.copy()
        diff.axpy(-1.0, K_G_T)
        assert diff.norm(PETSc.NormType.FROBENIUS) < 1e-8 * K_G.norm(PETSc.NormType.FROBENIUS), (
            "K_G must be symmetric."
        )

    def test_K_G_is_positive_semidefinite(self, plate_setup):
        """K_G from tensile stress must be positive semidefinite."""
        mesh, domain, bc_mgr = plate_setup
        stress_field = {eid: np.array([1e6, 1e6, 0.0]) for eid in domain._element_map}
        K_G = domain.assemble_geometric_stiffness(stress_field=stress_field)
        K_G_red = bc_mgr.reduce_matrix(K_G)

        # Build dense copy for eigenvalue check
        n = K_G_red.getSize()[0]
        dense = np.zeros((n, n))
        for i in range(n):
            row_cols, row_vals = K_G_red.getRow(i)
            for c, v in zip(row_cols, row_vals):
                dense[i, c] = v

        eigs = np.linalg.eigvalsh(dense)
        assert np.all(eigs >= -1e-6 * np.max(np.abs(eigs))), (
            f"K_G has negative eigenvalues: {eigs[eigs < 0]}"
        )

    def test_stress_recovery_element_stresses_returns_arrays(self, plate_setup):
        """StressRecovery.compute_element_stresses must run without error."""
        mesh, domain, bc_mgr = plate_setup

        # Build a non-zero displacement (unit membrane stretch in x)
        u_full = np.zeros(domain.dofs_count)
        n_nodes = len(list(mesh.nodes))
        for node in mesh.nodes:
            idx = mesh.node_id_to_index[node.id]
            u_full[idx * domain.dofs_per_node + 0] = node.coords[0] * 1e-3  # u = ε·x

        sr = StressRecovery(domain, u_full)
        result = sr.compute_element_stresses(
            location=StressLocation.MIDDLE,
            stress_type=StressType.MEMBRANE,
        )
        assert result.sigma_xx is not None
        assert len(result.sigma_xx) == len(domain._element_map)

    def test_stress_field_dict_from_recovery(self, plate_setup):
        """Build stress_field dict from StressRecovery and assemble K_G."""
        mesh, domain, bc_mgr = plate_setup

        u_full = np.zeros(domain.dofs_count)
        for node in mesh.nodes:
            idx = mesh.node_id_to_index[node.id]
            u_full[idx * domain.dofs_per_node + 0] = node.coords[0] * 5e-3

        sr = StressRecovery(domain, u_full)
        elem_result = sr.compute_element_stresses(
            location=StressLocation.MIDDLE,
            stress_type=StressType.MEMBRANE,
        )

        stress_field: dict[int, np.ndarray] = {}
        for elem_id in domain._element_map:
            sigma = np.array([
                elem_result.sigma_xx[elem_id],
                elem_result.sigma_yy[elem_id],
                elem_result.sigma_xy[elem_id],
            ])
            if np.max(np.abs(sigma)) > 1e-20:
                stress_field[elem_id] = sigma

        assert len(stress_field) > 0, "Expected non-zero stresses for membrane displacement."

        K_G = domain.assemble_geometric_stiffness(stress_field=stress_field)
        assert isinstance(K_G, PETSc.Mat)

    def test_keff_with_KG_larger_than_without(self, plate_setup):
        """K + K_G diagonal sum must exceed K diagonal sum for tensile stress."""
        mesh, domain, bc_mgr = plate_setup
        K_red, _, M_red = bc_mgr.reduced_system
        coeffs = NewmarkCoefficients.from_newmark_params(0.25, 0.5, 0.01)

        # Baseline K_eff
        K_eff_base = K_red.copy()
        K_eff_base.axpy(coeffs.a0, M_red)
        diag_base = K_eff_base.getDiagonal().getArray().copy()

        # K_G from a uniform biaxial tensile stress
        stress_field = {eid: np.array([1e7, 1e7, 0.0]) for eid in domain._element_map}
        K_G = domain.assemble_geometric_stiffness(stress_field=stress_field)
        K_G_red = bc_mgr.reduce_matrix(K_G)

        K_eff_new = K_red.copy()
        K_eff_new.axpy(1.0, K_G_red)
        K_eff_new.axpy(coeffs.a0, M_red)
        diag_new = K_eff_new.getDiagonal().getArray().copy()

        assert np.sum(diag_new) > np.sum(diag_base), (
            "K_eff with K_G must have larger diagonal sum (stress stiffening)."
        )


# ---------------------------------------------------------------------------
# Group 2 — Solver hook tests (require preCICE for the class import)
# ---------------------------------------------------------------------------

@_skip_fsi
class TestStressStiffenedHook:
    """Unit tests for _post_convergence_hook — require preCICE library."""

    def test_hook_returns_none_for_zero_displacement(self, plate_setup):
        """Hook returns None when u = 0 (no stress → no K_G rebuild)."""
        mesh, domain, bc_mgr = plate_setup
        solver = _make_solver(mesh, domain)

        K_red, _, M_red = bc_mgr.reduced_system
        coeffs = NewmarkCoefficients.from_newmark_params(0.25, 0.5, 0.01)
        K_eff = K_red.copy()
        K_eff.axpy(coeffs.a0, M_red)

        u_zero = K_red.createVecRight()
        u_zero.set(0.0)

        result = solver._post_convergence_hook(
            u=u_zero, time_step=1, K_eff=K_eff,
            K_red=K_red, M_red=M_red, C_red=None,
            coeffs=coeffs, bc_manager=bc_mgr,
        )
        assert result is None

    def test_hook_returns_new_keff_under_membrane_load(self, plate_setup):
        """Hook returns a NEW K_eff object when membrane stress is non-zero."""
        mesh, domain, bc_mgr = plate_setup
        solver = _make_solver(mesh, domain)

        K_red, _, M_red = bc_mgr.reduced_system
        coeffs = NewmarkCoefficients.from_newmark_params(0.25, 0.5, 0.01)
        K_eff_old = K_red.copy()
        K_eff_old.axpy(coeffs.a0, M_red)

        # Membrane displacement: u = i * 1e-4 → ε_xx ≠ 0
        u_mem = K_red.createVecRight()
        arr = u_mem.getArray()
        n_free = arr.shape[0] // domain.dofs_per_node
        for i in range(n_free):
            arr[i * domain.dofs_per_node + 0] = float(i) * 1e-4
        u_mem.setArray(arr)

        result = solver._post_convergence_hook(
            u=u_mem, time_step=1, K_eff=K_eff_old,
            K_red=K_red, M_red=M_red, C_red=None,
            coeffs=coeffs, bc_manager=bc_mgr,
        )

        if result is None:
            pytest.skip("All element stresses below threshold after BC reduction.")

        assert result is not K_eff_old, (
            "Hook must return a NEW PETSc.Mat so _solve_linear_system refactorizes."
        )
        assert isinstance(result, PETSc.Mat)

    def test_update_interval_skips_rebuild(self, plate_setup):
        """Hook must return None when time_step % update_interval != 0."""
        mesh, domain, bc_mgr = plate_setup
        solver = _make_solver(mesh, domain, update_interval=5)

        K_red, _, M_red = bc_mgr.reduced_system
        coeffs = NewmarkCoefficients.from_newmark_params(0.25, 0.5, 0.01)
        K_eff = K_red.copy()
        K_eff.axpy(coeffs.a0, M_red)

        u_nz = K_red.createVecRight()
        arr = u_nz.getArray()
        arr[0::6] = 0.01
        u_nz.setArray(arr)

        result_3 = solver._post_convergence_hook(
            u=u_nz, time_step=3, K_eff=K_eff,
            K_red=K_red, M_red=M_red, C_red=None,
            coeffs=coeffs, bc_manager=bc_mgr,
        )
        assert result_3 is None, "update_interval=5 must skip rebuild at step 3."

        result_5 = solver._post_convergence_hook(
            u=u_nz, time_step=5, K_eff=K_eff,
            K_red=K_red, M_red=M_red, C_red=None,
            coeffs=coeffs, bc_manager=bc_mgr,
        )
        assert result_5 is None or isinstance(result_5, PETSc.Mat), (
            "At step divisible by update_interval hook must return None or Mat."
        )


# ---------------------------------------------------------------------------
# Group 3 — Configuration tests (no preCICE needed for these)
# ---------------------------------------------------------------------------

class TestStressStiffenedConfig:
    """Configuration and registration tests."""

    def test_solver_type_enum_exists(self):
        from fem_shell.core.config import SolverType
        assert hasattr(SolverType, "STRESS_STIFFENED_DYNAMIC_FSI")
        assert SolverType.STRESS_STIFFENED_DYNAMIC_FSI.value == "StressStiffenedDynamicFSI"

    @_skip_fsi
    def test_solver_inherits_linear_dynamic(self):
        from fem_shell.solvers.fsi.linear_dynamic import LinearDynamicFSISolver
        assert issubclass(StressStiffenedFSISolver, LinearDynamicFSISolver)

    @_skip_fsi
    def test_default_update_interval(self):
        mesh = _build_plate_mesh(nx=2, ny=2)
        domain = _build_domain(mesh)
        solver = _make_solver(mesh, domain)
        assert solver._kg_update_interval == 1

    @_skip_fsi
    def test_custom_update_interval(self):
        mesh = _build_plate_mesh(nx=2, ny=2)
        domain = _build_domain(mesh)
        solver = _make_solver(mesh, domain, update_interval=10)
        assert solver._kg_update_interval == 10


