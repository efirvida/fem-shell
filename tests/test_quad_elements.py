"""Test suite for QUAD plane-stress elements via Rust PyMeshAssembler.

Tests stiffness/mass matrix properties (symmetry, rigid body modes, PSD)
and rigid body mode verification for QUAD4, QUAD8, QUAD9.

No Python element class imports — Rust assembler is the sole source of truth.
"""

from __future__ import annotations

import numpy as np
import pytest

_aeroelast = pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler  # noqa: E402


# =============================================================================
# Helper
# =============================================================================

def _build_dense(asm: PyMeshAssembler, which: str) -> np.ndarray:
    """Assemble K or M into a dense matrix from COO triplets."""
    if which == "K":
        rows, cols, vals = asm.assemble_k()
    else:
        rows, cols, vals = asm.assemble_m()
    n = asm.dofs_count
    mat = np.zeros((n, n))
    for r, c, v in zip(rows, cols, vals):
        mat[r, c] += v
    return mat


def _make_assembler(nodes_3d: np.ndarray, connectivity: list,
                    elem_type_code: int, mat_dict: dict) -> PyMeshAssembler:
    return PyMeshAssembler(
        node_coords=nodes_3d,
        connectivity=[connectivity],
        elem_types=[elem_type_code],
        materials=[mat_dict],
    )


# =============================================================================
# Node coordinates — in the XY plane (z=0), but passed as 3D
# =============================================================================

def _quad4_nodes_3d():
    """Unit square QUAD4 in XY plane."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


def _quad8_nodes_3d():
    """QUAD8 unit square with midside nodes."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
    ])


def _quad9_nodes_3d():
    """QUAD9 unit square with midside + center nodes."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ])


# =============================================================================
# Element configurations
# =============================================================================

STEEL_PLANE = {"type": "plane_stress", "e": 210e9, "nu": 0.3, "rho": 7800.0, "thickness": 0.01}

QUAD_CONFIGS = {
    "Quad4": {
        "nodes_fn": _quad4_nodes_3d,
        "code": 104,
        "n_nodes": 4,
        "n_dofs": 8,   # 2 DOFs per node (plane stress)
        "rigid_modes": 3,
    },
    "Quad8": {
        "nodes_fn": _quad8_nodes_3d,
        "code": 108,
        "n_nodes": 8,
        "n_dofs": 16,
        "rigid_modes": 3,
    },
    "Quad9": {
        "nodes_fn": _quad9_nodes_3d,
        "code": 109,
        "n_nodes": 9,
        "n_dofs": 18,
        "rigid_modes": 3,
    },
}


@pytest.fixture(params=list(QUAD_CONFIGS.keys()))
def quad_setup(request):
    """Parametrized fixture for QUAD element types."""
    name = request.param
    cfg = QUAD_CONFIGS[name]
    nodes = cfg["nodes_fn"]()
    conn = list(range(cfg["n_nodes"]))
    asm = _make_assembler(nodes, conn, cfg["code"], STEEL_PLANE)
    return {"name": name, "cfg": cfg, "nodes": nodes, "asm": asm}


# =============================================================================
# Stiffness matrix tests
# =============================================================================


class TestStiffnessMatrix:
    """K: dimensions, symmetry, 3 rigid body modes, PSD."""

    def test_dimensions(self, quad_setup):
        K = _build_dense(quad_setup["asm"], "K")
        n = quad_setup["cfg"]["n_dofs"]
        assert K.shape == (n, n), f"{quad_setup['name']}: expected ({n},{n}), got {K.shape}"

    def test_symmetry(self, quad_setup):
        K = _build_dense(quad_setup["asm"], "K")
        np.testing.assert_allclose(K, K.T, atol=1e-6,
                                   err_msg=f"{quad_setup['name']}: K not symmetric")

    def test_rigid_body_modes(self, quad_setup):
        """Plane stress K has 3 rigid body modes (tx, ty, rz)."""
        K = _build_dense(quad_setup["asm"], "K")
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))
        max_eig = eigvals[-1]
        threshold = max_eig * 1e-10 if max_eig > 0 else 1e-10
        zero_count = int(np.sum(eigvals < threshold))
        expected = quad_setup["cfg"]["rigid_modes"]
        assert zero_count == expected, (
            f"{quad_setup['name']}: expected {expected} rigid modes, "
            f"found {zero_count}. Smallest 5 eigvals: {eigvals[:5]}"
        )

    def test_positive_semi_definite(self, quad_setup):
        K = _build_dense(quad_setup["asm"], "K")
        eigvals = np.linalg.eigvalsh(K)
        threshold = -np.max(np.abs(eigvals)) * 1e-10
        neg = eigvals[eigvals < threshold]
        assert len(neg) == 0, f"{quad_setup['name']}: negative eigenvalues: {neg}"


# =============================================================================
# Mass matrix tests
# =============================================================================


class TestMassMatrix:
    """M: symmetry, positive semi-definite."""

    def test_symmetry(self, quad_setup):
        M = _build_dense(quad_setup["asm"], "M")
        np.testing.assert_allclose(M, M.T, atol=1e-10,
                                   err_msg=f"{quad_setup['name']}: M not symmetric")

    def test_positive_semi_definite(self, quad_setup):
        M = _build_dense(quad_setup["asm"], "M")
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > -1e-10), (
            f"{quad_setup['name']}: M has negative eigenvalue: {eigvals.min()}"
        )


# =============================================================================
# Rigid body mode verification using node coordinates
# =============================================================================


class TestRigidBodyModes:
    """K @ rigid_mode ≈ 0 for all 3 rigid modes."""

    def test_rigid_modes_kx_ky_rz(self, quad_setup):
        """All three plane rigid body modes should be in the null space of K."""
        name = quad_setup["name"]
        K = _build_dense(quad_setup["asm"], "K")
        nodes = quad_setup["nodes"]  # (n_nodes, 3)
        n_nodes = quad_setup["cfg"]["n_nodes"]

        # Plane DOFs: [ux0, uy0, ux1, uy1, ...]
        # Translation X: all ux=1, uy=0
        tx = np.zeros(n_nodes * 2)
        tx[0::2] = 1.0

        # Translation Y: all ux=0, uy=1
        ty = np.zeros(n_nodes * 2)
        ty[1::2] = 1.0

        # Rotation about Z: ux = -y, uy = x
        rz = np.zeros(n_nodes * 2)
        for i in range(n_nodes):
            x, y = nodes[i, 0], nodes[i, 1]
            rz[2 * i]     = -y
            rz[2 * i + 1] =  x

        K_norm = np.linalg.norm(K, ord="fro")
        for label, mode in [("tx", tx), ("ty", ty), ("rz", rz)]:
            residual = K @ mode
            rel_err = np.max(np.abs(residual)) / K_norm if K_norm > 0 else np.max(np.abs(residual))
            assert rel_err < 1e-10, (
                f"{name}: rigid mode {label} not in null space. "
                f"Relative error: {rel_err:.4e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
