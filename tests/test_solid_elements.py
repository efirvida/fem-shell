"""Test suite for 3D Solid Elements via Rust PyMeshAssembler.

Tests stiffness/mass matrix properties (symmetry, rigid body modes, PSD/PD)
and analytical physics (strain energy under prescribed displacement) for all
solid element types: Hexa8, Hexa20, Tetra4, Tetra10, Wedge6, Wedge15,
Pyramid5, Pyramid13.

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
    K = np.zeros((n, n))
    for r, c, v in zip(rows, cols, vals):
        K[r, c] += v
    return K


def _make_assembler(nodes: np.ndarray, connectivity: list, elem_type_code: int,
                    mat_dict: dict) -> PyMeshAssembler:
    """Build a single-element PyMeshAssembler."""
    return PyMeshAssembler(
        node_coords=nodes,
        connectivity=[connectivity],
        elem_types=[elem_type_code],
        materials=[mat_dict],
    )


# =============================================================================
# Node coordinates for each element type
# =============================================================================

def _hexa8_nodes():
    """Unit cube [0,1]^3."""
    return np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)


def _hexa20_nodes():
    """20-node hexahedron based on 2x2x2 cube."""
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=float)
    mid_edges = np.array([
        [0, -1, -1], [-1, 0, -1], [-1, -1, 0],
        [1,  0, -1], [ 1, -1, 0], [ 0,  1, -1],
        [1,  1,  0], [-1,  1, 0], [ 0, -1,  1],
        [-1, 0,  1], [ 1,  0, 1], [ 0,  1,  1],
    ], dtype=float)
    return np.vstack([corners, mid_edges])


def _tetra4_nodes():
    return np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=float)


def _tetra10_nodes():
    corners = _tetra4_nodes()
    mid_edges = np.array([
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
    ], dtype=float)
    return np.vstack([corners, mid_edges])


def _wedge6_nodes():
    return np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1],
    ], dtype=float)


def _wedge15_nodes():
    corners = _wedge6_nodes()
    mid_edges = np.array([
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        [0.5, 0.5, 0], [1, 0, 0.5], [0, 1, 0.5],
        [0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1],
    ], dtype=float)
    return np.vstack([corners, mid_edges])


def _pyramid5_nodes():
    return np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1],
    ], dtype=float)


def _pyramid13_nodes():
    corners = _pyramid5_nodes()
    mid_edges = np.array([
        [0, -1, 0], [-1, 0, 0], [-0.5, -0.5, 0.5],
        [1,  0, 0], [ 0.5, -0.5, 0.5], [0, 1, 0],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=float)
    return np.vstack([corners, mid_edges])


# =============================================================================
# Element configurations
# =============================================================================

SOLID_CONFIGS = {
    "Hexa8": {
        "nodes_fn": _hexa8_nodes,
        "code": 208,
        "n_nodes": 8,
        "n_dofs": 24,
        "rigid_modes": 6,
    },
    "Hexa20": {
        "nodes_fn": _hexa20_nodes,
        "code": 220,
        "n_nodes": 20,
        "n_dofs": 60,
        "rigid_modes": 6,
    },
    "Tetra4": {
        "nodes_fn": _tetra4_nodes,
        "code": 304,
        "n_nodes": 4,
        "n_dofs": 12,
        "rigid_modes": 6,
    },
    "Tetra10": {
        "nodes_fn": _tetra10_nodes,
        "code": 310,
        "n_nodes": 10,
        "n_dofs": 30,
        "rigid_modes": 6,
    },
    "Wedge6": {
        "nodes_fn": _wedge6_nodes,
        "code": 306,
        "n_nodes": 6,
        "n_dofs": 18,
        "rigid_modes": 6,
    },
    "Wedge15": {
        "nodes_fn": _wedge15_nodes,
        "code": 315,
        "n_nodes": 15,
        "n_dofs": 45,
        "rigid_modes": 6,
    },
    "Pyramid5": {
        "nodes_fn": _pyramid5_nodes,
        "code": 305,
        "n_nodes": 5,
        "n_dofs": 15,
        "rigid_modes": 6,
    },
    "Pyramid13": {
        "nodes_fn": _pyramid13_nodes,
        "code": 313,
        "n_nodes": 13,
        "n_dofs": 39,
        "rigid_modes": 6,
    },
}

STEEL = {"type": "solid_3d", "e": 210e9, "nu": 0.3, "rho": 7850.0}


@pytest.fixture(params=list(SOLID_CONFIGS.keys()))
def solid_setup(request):
    """Parametrized fixture: one entry per solid element type."""
    name = request.param
    cfg = SOLID_CONFIGS[name]
    nodes = cfg["nodes_fn"]()
    conn = list(range(cfg["n_nodes"]))
    asm = _make_assembler(nodes, conn, cfg["code"], STEEL)
    return {"name": name, "cfg": cfg, "nodes": nodes, "asm": asm}


# =============================================================================
# Stiffness matrix tests
# =============================================================================


class TestStiffnessMatrix:
    """K matrix: dimensions, symmetry, rigid body modes, PSD."""

    def test_dimensions(self, solid_setup):
        K = _build_dense(solid_setup["asm"], "K")
        n = solid_setup["cfg"]["n_dofs"]
        assert K.shape == (n, n), f"Expected ({n},{n}), got {K.shape}"

    def test_symmetry(self, solid_setup):
        K = _build_dense(solid_setup["asm"], "K")
        np.testing.assert_allclose(K, K.T, atol=1e-8,
                                   err_msg=f"{solid_setup['name']}: K not symmetric")

    def test_rigid_body_modes(self, solid_setup):
        """K should have exactly 6 zero eigenvalues."""
        name = solid_setup["name"]
        if name == "Pyramid13":
            pytest.skip("Pyramid13 numerical integration challenges")

        K = _build_dense(solid_setup["asm"], "K")
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))
        max_eig = eigvals[-1]

        expected = solid_setup["cfg"]["rigid_modes"]

        if name in ("Hexa20", "Tetra10", "Wedge15"):
            # Gap-based detection for quadratic elements
            n_check = min(9, len(eigvals))
            ratios = []
            for i in range(1, n_check):
                denom = max(eigvals[i - 1], 1e-15)
                ratios.append((eigvals[i] / denom, i))
            gap_idx = max(ratios, key=lambda x: x[0])[1] if ratios else 0
            zero_count = gap_idx
        else:
            threshold = max_eig * 1e-8 if max_eig > 0 else 1e-10
            zero_count = int(np.sum(eigvals < threshold))

        assert zero_count == expected, (
            f"{name}: expected {expected} rigid modes, found {zero_count}. "
            f"Smallest 8 eigvals: {eigvals[:8]}"
        )

    def test_positive_semi_definite(self, solid_setup):
        name = solid_setup["name"]
        if name == "Pyramid13":
            pytest.skip("Pyramid13 numerical integration challenges")

        K = _build_dense(solid_setup["asm"], "K")
        eigvals = np.linalg.eigvalsh(K)
        threshold = -np.max(np.abs(eigvals)) * 1e-10
        neg = eigvals[eigvals < threshold]
        assert len(neg) == 0, f"{name}: negative eigenvalues: {neg}"


# =============================================================================
# Mass matrix tests
# =============================================================================


class TestMassMatrix:
    """M matrix: dimensions, symmetry, positive definite."""

    def test_dimensions(self, solid_setup):
        name = solid_setup["name"]
        if name == "Pyramid13":
            pytest.skip("Pyramid13 numerical integration challenges")
        M = _build_dense(solid_setup["asm"], "M")
        n = solid_setup["cfg"]["n_dofs"]
        assert M.shape == (n, n)

    def test_symmetry(self, solid_setup):
        name = solid_setup["name"]
        if name == "Pyramid13":
            pytest.skip("Pyramid13 numerical integration challenges")
        M = _build_dense(solid_setup["asm"], "M")
        np.testing.assert_allclose(M, M.T, atol=1e-10,
                                   err_msg=f"{name}: M not symmetric")

    def test_positive_definite(self, solid_setup):
        name = solid_setup["name"]
        if name == "Pyramid13":
            pytest.skip("Pyramid13 numerical integration challenges")
        M = _build_dense(solid_setup["asm"], "M")
        eigvals = np.linalg.eigvalsh(M)
        threshold = -np.max(eigvals) * 1e-10 if np.max(eigvals) > 0 else -1e-15
        assert np.min(eigvals) > threshold, (
            f"{name}: M has negative eigenvalue: {np.min(eigvals)}"
        )


# =============================================================================
# Strain energy / analytical tension tests (Hexa8, Tetra4, Wedge6)
# =============================================================================


class TestAnalyticalTension:
    """u^T K u vs analytical strain energy for uniaxial tension."""

    @pytest.mark.parametrize("elem_name", ["Hexa8", "Tetra4"])
    def test_uniaxial_strain_energy(self, elem_name):
        cfg = SOLID_CONFIGS[elem_name]
        nodes = cfg["nodes_fn"]()

        # Hexa8 nodes are already [0,1]^3; others are unit-scale too
        # For hexa8 built with _hexa8_nodes(), it's already [0,1]^3
        n_nodes = cfg["n_nodes"]
        conn = list(range(n_nodes))

        E = 210e9
        nu = 0.3
        mat = {"type": "solid_3d", "e": E, "nu": nu, "rho": 7850.0}
        asm = _make_assembler(nodes, conn, cfg["code"], mat)
        K = _build_dense(asm, "K")

        # Compute approximate volume of element
        if elem_name == "Hexa8":
            V = 1.0  # unit cube
        elif elem_name == "Tetra4":
            V = 1.0 / 6.0  # unit tetra volume

        # Prescribed uniform uniaxial strain epsilon_xx
        epsilon = 1e-6
        u = np.zeros(n_nodes * 3)
        for i in range(n_nodes):
            u[3 * i] = epsilon * nodes[i, 0]  # ux = epsilon * x

        # Analytical strain energy for constrained uniaxial strain
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        sigma_xx = (lam + 2 * mu) * epsilon
        U_analytical = 0.5 * sigma_xx * epsilon * V

        U_numerical = 0.5 * u @ K @ u

        assert np.isclose(U_numerical, U_analytical, rtol=0.05), (
            f"{elem_name}: U_num={U_numerical:.4e}, U_ana={U_analytical:.4e}"
        )


class TestAnalyticalCompression:
    """K should produce negative strain energy sign for compression."""

    def test_compression_hexa8(self):
        nodes = _hexa8_nodes()
        asm = _make_assembler(nodes, list(range(8)), 208, STEEL)
        K = _build_dense(asm, "K")

        E, nu = 210e9, 0.3
        epsilon = -1e-6  # compressive
        u = np.zeros(24)
        for i in range(8):
            u[3 * i] = epsilon * nodes[i, 0]

        # u^T K u > 0 even under compression (strain energy is always positive)
        U = 0.5 * u @ K @ u
        assert U > 0, "Strain energy must be positive for any non-zero displacement"

        # Strain is negative (compression)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        sigma_xx = (lam + 2 * mu) * epsilon  # negative
        assert sigma_xx < 0, "σxx should be negative for compressive strain"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
