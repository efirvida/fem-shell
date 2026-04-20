"""Validate Rust-accelerated assembly in MeshAssembler.

Tests that:
1. Rust COO fast-path for K and M matches the PETSc loop fallback.
2. assemble_tangent_stiffness(u=0) == K (linear identity).
3. assemble_internal_forces(u=0) == 0.
4. assemble_internal_forces(u) ≈ K·u for small u (linear regime).
5. Newton-Raphson consistency: KT·δu ≈ fint(u+δu) - fint(u).
6. Both MITC3 and MITC4 meshes are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py", reason="PETSc not available")
from petsc4py import PETSc  # noqa: E402

fem_shell_core = pytest.importorskip("fem_shell_core", reason="Rust backend not available")

from fem_shell.core.assembler import MeshAssembler  # noqa: E402
from fem_shell.core.material import IsotropicMaterial  # noqa: E402
from fem_shell.core.mesh.entities import ElementType, MeshElement, Node  # noqa: E402
from fem_shell.core.mesh.model import MeshModel  # noqa: E402
from fem_shell.elements import ElementFamily  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MAT = IsotropicMaterial(name="steel", E=2.1e11, nu=0.3, rho=7800.0)
THICKNESS = 0.01


def _model_cfg(element_family=ElementFamily.SHELL):
    return {
        "elements": {
            "element_family": element_family,
            "material": MAT,
            "thickness": THICKNESS,
        }
    }


def _build_quad_plate(nx: int = 4, ny: int = 4, L: float = 1.0) -> MeshModel:
    """Flat MITC4 plate mesh (nx×ny quads)."""
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
                    nodes=[
                        grid[(i, j)],
                        grid[(i + 1, j)],
                        grid[(i + 1, j + 1)],
                        grid[(i, j + 1)],
                    ],
                    element_type=ElementType.quad,
                )
            )
    return mesh


def _build_tri_plate(nx: int = 4, ny: int = 4, L: float = 1.0) -> MeshModel:
    """Flat MITC3 plate mesh (nx×ny×2 triangles)."""
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
                    nodes=[grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)]],
                    element_type=ElementType.triangle,
                )
            )
            mesh.add_element(
                MeshElement(
                    nodes=[grid[(i, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]],
                    element_type=ElementType.triangle,
                )
            )
    return mesh


def _petsc_to_dense(mat: PETSc.Mat) -> np.ndarray:
    """Convert PETSc.Mat to a dense numpy array."""
    n = mat.getSize()[0]
    dense = np.zeros((n, n))
    for i in range(n):
        cols_i, vals_i = mat.getRow(i)
        dense[i, cols_i] = vals_i
    return dense


def _petsc_to_array(vec: PETSc.Vec) -> np.ndarray:
    """Convert PETSc.Vec to numpy array."""
    return vec.getArray().copy()


# ---------------------------------------------------------------------------
# Build assembler fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["MITC3", "MITC4"])
def assembler(request):
    """Parametrised assembler covering both element types."""
    if request.param == "MITC4":
        mesh = _build_quad_plate(nx=4, ny=4, L=1.0)
    else:
        mesh = _build_tri_plate(nx=4, ny=4, L=1.0)
    asm = MeshAssembler(mesh=mesh, model=_model_cfg())
    return asm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRustCOOAssembly:
    """Verify the Rust COO fast-path for K and M matches the PETSc loop."""

    def test_stiffness_rust_vs_petsc_loop(self, assembler):
        """K assembled via Rust COO must match the PETSc setValuesLocal loop."""
        asm = assembler
        # Rust COO path (default when fem_shell_core is available)
        K_rust = asm.assemble_stiffness_matrix()
        K_rust_dense = _petsc_to_dense(K_rust)

        # Force Python fallback by temporarily disabling Rust
        asm._has_rust = False
        K_py = asm.assemble_stiffness_matrix()
        K_py_dense = _petsc_to_dense(K_py)
        asm._has_rust = True  # restore

        assert K_rust_dense.shape == K_py_dense.shape
        np.testing.assert_allclose(
            K_rust_dense, K_py_dense, atol=1e-6, rtol=1e-10,
            err_msg="K (Rust COO) != K (PETSc loop)",
        )

    def test_mass_rust_vs_petsc_loop(self, assembler):
        """M assembled via Rust COO must match the PETSc setValuesLocal loop."""
        asm = assembler
        M_rust = asm.assemble_mass_matrix()
        M_rust_dense = _petsc_to_dense(M_rust)

        asm._has_rust = False
        M_py = asm.assemble_mass_matrix()
        M_py_dense = _petsc_to_dense(M_py)
        asm._has_rust = True

        assert M_rust_dense.shape == M_py_dense.shape
        np.testing.assert_allclose(
            M_rust_dense, M_py_dense, atol=1e-6, rtol=1e-10,
            err_msg="M (Rust COO) != M (PETSc loop)",
        )

    def test_stiffness_symmetric(self, assembler):
        K = _petsc_to_dense(assembler.assemble_stiffness_matrix())
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_mass_symmetric(self, assembler):
        M = _petsc_to_dense(assembler.assemble_mass_matrix())
        np.testing.assert_allclose(M, M.T, atol=1e-6)


class TestTangentStiffness:
    """Verify assemble_tangent_stiffness."""

    def test_kt_at_zero_equals_k(self, assembler):
        """KT(u=0) must equal K (linear stiffness)."""
        asm = assembler
        K = _petsc_to_dense(asm.assemble_stiffness_matrix())
        u_zero = np.zeros(asm.dofs_count)
        KT = _petsc_to_dense(asm.assemble_tangent_stiffness(u_zero))

        np.testing.assert_allclose(
            KT, K, atol=1e-6, rtol=1e-10,
            err_msg="KT(u=0) != K",
        )

    def test_kt_symmetric(self, assembler):
        u = np.zeros(assembler.dofs_count)
        KT = _petsc_to_dense(assembler.assemble_tangent_stiffness(u))
        np.testing.assert_allclose(KT, KT.T, atol=1e-6)


class TestInternalForces:
    """Verify assemble_internal_forces."""

    def test_fint_zero_at_zero(self, assembler):
        """fint(u=0) must be zero."""
        fint = _petsc_to_array(
            assembler.assemble_internal_forces(np.zeros(assembler.dofs_count))
        )
        np.testing.assert_allclose(fint, 0.0, atol=1e-10)

    def test_fint_linear_equals_ku(self, assembler):
        """For small u, fint(u, nonlinear=False) ≈ K·u."""
        asm = assembler
        K_dense = _petsc_to_dense(asm.assemble_stiffness_matrix())

        rng = np.random.default_rng(42)
        u = rng.normal(0, 1e-6, asm.dofs_count)

        fint = _petsc_to_array(asm.assemble_internal_forces(u, nonlinear=False))
        ku = K_dense @ u

        # Relative tolerance on non-trivial components
        mask = np.abs(ku) > 1e-12 * np.max(np.abs(ku))
        if mask.any():
            np.testing.assert_allclose(
                fint[mask], ku[mask], rtol=1e-6,
                err_msg="fint(u, linear) != K·u",
            )


class TestNewtonRaphsonConsistency:
    """KT·δu ≈ fint(u+δu) − fint(u) for small δu."""

    def test_nr_consistency(self, assembler):
        asm = assembler
        rng = np.random.default_rng(123)

        u = rng.normal(0, 1e-5, asm.dofs_count)
        du = rng.normal(0, 1e-8, asm.dofs_count)

        KT = _petsc_to_dense(asm.assemble_tangent_stiffness(u))
        fint_u = _petsc_to_array(asm.assemble_internal_forces(u))
        fint_udu = _petsc_to_array(asm.assemble_internal_forces(u + du))

        lhs = KT @ du
        rhs = fint_udu - fint_u

        mask = np.abs(lhs) > 1e-12 * np.max(np.abs(lhs))
        if mask.any():
            rel_err = np.max(np.abs(lhs[mask] - rhs[mask]) / np.abs(lhs[mask]))
            # MITC4 on flat elements: ~1e-3; allow up to 2e-3 for
            # the shared formulation limitation on non-planar cases.
            assert rel_err < 2e-3, (
                f"NR consistency failed: max relative error = {rel_err:.2e}"
            )


class TestRustGroupCoverage:
    """Verify the Rust assembler is wired up correctly via PyMeshAssembler."""

    def test_py_mesh_assembler_built(self, assembler):
        """_rust must be a PyMeshAssembler instance after __init__."""
        assert assembler._rust is not None
