"""
Tests for Rust composite element integration.

Validates composite shell elements (MITC3, MITC4) using the Rust batch
functions and PyMeshAssembler. All tests use physical sanity checks only —
no Python MITC3Composite / MITC4Composite class imports.
"""

import numpy as np
import pytest

from aeroelast.core.material import OrthotropicMaterial
from aeroelast.core.laminate import (
    Ply,
    Laminate,
    create_laminate_from_angles,
)

# Try importing Rust backend
try:
    import _aeroelast as fsc
    from _aeroelast import PyMeshAssembler

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="_aeroelast not available")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def carbon_epoxy():
    return OrthotropicMaterial(
        name="Carbon/Epoxy",
        E=(181e9, 10.3e9, 10.3e9),
        G=(7.17e9, 3.78e9, 7.17e9),
        nu=(0.28, 0.28, 0.28),
        rho=1600,
    )


@pytest.fixture
def glass_epoxy():
    return OrthotropicMaterial(
        name="E-Glass/Epoxy",
        E=(38.6e9, 8.27e9, 8.27e9),
        G=(4.14e9, 3.0e9, 4.14e9),
        nu=(0.26, 0.26, 0.26),
        rho=1800,
    )


@pytest.fixture
def ply_thickness():
    return 0.125e-3


@pytest.fixture
def tri_coords():
    """Right-angle triangle element in the XY plane."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


@pytest.fixture
def quad_coords():
    """1x1 square element in the XY plane."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


@pytest.fixture
def tri_coords_3d():
    """Triangular element tilted in 3D space."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.3],
        [0.5, 0.8, 0.1],
    ])


@pytest.fixture
def quad_coords_3d():
    """Quadrilateral element tilted in 3D space."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.2],
        [1.1, 1.0, 0.3],
        [0.0, 0.9, 0.1],
    ])


# =============================================================================
# Helpers: call Rust batch functions
# =============================================================================


def _rust_ke_mitc3(coords, laminate):
    """Compute MITC3 composite stiffness via Rust batch function."""
    n = 1
    coords_flat = coords.ravel()[np.newaxis, :]  # (1, 9)
    cm = laminate.A.ravel()[np.newaxis, :]        # (1, 9)
    cb = laminate.D.ravel()[np.newaxis, :]
    cs = laminate.Cs.ravel()[np.newaxis, :]
    h = laminate.total_thickness
    a_trace = np.trace(laminate.A)
    e_equiv = a_trace / (3.0 * h)

    ke_flat = fsc.batch_ke_mitc3_composite(
        coords_flat, cm, cb, cs,
        np.array([h]), np.array([e_equiv]),
    )
    return np.asarray(ke_flat).reshape(18, 18)


def _rust_me_mitc3(coords, laminate):
    """Compute MITC3 composite mass via Rust batch function."""
    coords_flat = coords.ravel()[np.newaxis, :]
    mpa = sum(p.material.rho * p.thickness for p in laminate.plies)
    ri = sum(
        p.material.rho * (p.z_top ** 3 - p.z_bottom ** 3) / 3
        for p in laminate.plies
    )
    me_flat = fsc.batch_me_mitc3_composite(
        coords_flat, np.array([mpa]), np.array([ri]),
    )
    return np.asarray(me_flat).reshape(18, 18)


def _rust_ke_mitc4(coords, laminate):
    """Compute MITC4 composite stiffness via Rust batch function."""
    coords_flat = coords.ravel()[np.newaxis, :]
    cm = laminate.A.ravel()[np.newaxis, :]
    cb = laminate.D.ravel()[np.newaxis, :]
    cs = laminate.Cs.ravel()[np.newaxis, :]
    h = laminate.total_thickness
    a_trace = np.trace(laminate.A)
    e_equiv = a_trace / (3.0 * h)

    ke_flat = fsc.batch_ke_mitc4_composite(
        coords_flat, cm, cb, cs,
        np.array([h]), np.array([e_equiv]),
    )
    return np.asarray(ke_flat).reshape(24, 24)


def _rust_me_mitc4(coords, laminate):
    """Compute MITC4 composite mass via Rust batch function."""
    coords_flat = coords.ravel()[np.newaxis, :]
    mpa = sum(p.material.rho * p.thickness for p in laminate.plies)
    ri = sum(
        p.material.rho * (p.z_top ** 3 - p.z_bottom ** 3) / 3
        for p in laminate.plies
    )
    me_flat = fsc.batch_me_mitc4_composite(
        coords_flat, np.array([mpa]), np.array([ri]),
    )
    return np.asarray(me_flat).reshape(24, 24)


def _dense_from_asm(asm, which: str) -> np.ndarray:
    if which == "K":
        rows, cols, vals = asm.assemble_k()
    else:
        rows, cols, vals = asm.assemble_m()
    n = asm.dofs_count
    mat = np.zeros((n, n))
    for r, c, v in zip(rows, cols, vals):
        mat[r, c] += v
    return mat


def _laminate_to_mat_dict(laminate) -> dict:
    """Convert laminate to composite material dict for PyMeshAssembler."""
    h = laminate.total_thickness
    a_trace = np.trace(laminate.A)
    e_equiv = a_trace / (3.0 * h)
    mpa = sum(p.material.rho * p.thickness for p in laminate.plies)
    ri = sum(
        p.material.rho * (p.z_top ** 3 - p.z_bottom ** 3) / 3
        for p in laminate.plies
    )
    return {
        "type": "composite",
        "cm": laminate.A.ravel().tolist(),
        "cb": laminate.D.ravel().tolist(),
        "cs": laminate.Cs.ravel().tolist(),
        "thickness": h,
        "e_equiv": e_equiv,
        "mass_per_area": mpa,
        "rotational_inertia": ri,
    }


# =============================================================================
# Tests: Batch functions — MITC3 composite
# =============================================================================


class TestMITC3BatchSanity:
    """Physical sanity checks for MITC3 composite batch functions."""

    def test_ke_shape_and_symmetry(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc3(tri_coords, laminate)
        assert ke.shape == (18, 18), f"Expected (18,18), got {ke.shape}"
        np.testing.assert_allclose(ke, ke.T, atol=1e-6, err_msg="K not symmetric")

    def test_ke_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc3(tri_coords, laminate)
        eigvals = np.linalg.eigvalsh(ke)
        assert np.all(eigvals >= -1e-6 * max(eigvals)), (
            f"Negative eigenvalue: {eigvals.min():.4e}"
        )

    def test_me_shape_and_symmetry(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        me = _rust_me_mitc3(tri_coords, laminate)
        assert me.shape == (18, 18)
        np.testing.assert_allclose(me, me.T, atol=1e-10, err_msg="M not symmetric")

    def test_me_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        me = _rust_me_mitc3(tri_coords, laminate)
        eigvals = np.linalg.eigvalsh(me)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.4e}"

    def test_unidirectional_0_symmetry(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0])
        ke = _rust_ke_mitc3(tri_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_quasi_isotropic_symmetry(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 90, 90, -45, 45, 0]
        )
        ke = _rust_ke_mitc3(tri_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_3d_element_symmetry(self, carbon_epoxy, ply_thickness, tri_coords_3d):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc3(tri_coords_3d, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_glass_epoxy_symmetry(self, glass_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(glass_epoxy, ply_thickness, [0, 45, -45, 0])
        ke = _rust_ke_mitc3(tri_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)


# =============================================================================
# Tests: Batch functions — MITC4 composite
# =============================================================================


class TestMITC4BatchSanity:
    """Physical sanity checks for MITC4 composite batch functions."""

    def test_ke_shape_and_symmetry(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc4(quad_coords, laminate)
        assert ke.shape == (24, 24)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_ke_positive_semidefinite(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc4(quad_coords, laminate)
        eigvals = np.linalg.eigvalsh(ke)
        assert np.all(eigvals >= -1e-6 * max(eigvals)), (
            f"Negative eigenvalue: {eigvals.min():.4e}"
        )

    def test_me_shape_and_symmetry(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        me = _rust_me_mitc4(quad_coords, laminate)
        assert me.shape == (24, 24)
        np.testing.assert_allclose(me, me.T, atol=1e-10)

    def test_me_positive_semidefinite(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        me = _rust_me_mitc4(quad_coords, laminate)
        eigvals = np.linalg.eigvalsh(me)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.4e}"

    def test_unidirectional_45_symmetry(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [45])
        ke = _rust_ke_mitc4(quad_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_quasi_isotropic_symmetry(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 90, 90, -45, 45, 0]
        )
        ke = _rust_ke_mitc4(quad_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_3d_element_symmetry(self, carbon_epoxy, ply_thickness, quad_coords_3d):
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        ke = _rust_ke_mitc4(quad_coords_3d, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)

    def test_glass_epoxy_angle_ply_symmetry(self, glass_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(glass_epoxy, ply_thickness, [45, -45, -45, 45])
        ke = _rust_ke_mitc4(quad_coords, laminate)
        np.testing.assert_allclose(ke, ke.T, atol=1e-6)


# =============================================================================
# Tests: Batch processing (multiple elements) — sanity only
# =============================================================================


class TestBatchComposite:
    """Batch processing sanity: symmetry and PSD for multiple elements."""

    def test_batch_ke_mitc3_multiple(self, carbon_epoxy, ply_thickness):
        """Multiple MITC3 elements — all K must be symmetric and PSD."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        coords_list = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float),
            np.array([[0, 0, 0], [1, 0, 0.5], [0.5, 1, 0.2]], dtype=float),
        ]
        n = len(coords_list)
        coords_batch = np.array([c.ravel() for c in coords_list])
        cm = np.tile(laminate.A.ravel(), (n, 1))
        cb = np.tile(laminate.D.ravel(), (n, 1))
        cs = np.tile(laminate.Cs.ravel(), (n, 1))
        h = laminate.total_thickness
        e_equiv = np.trace(laminate.A) / (3.0 * h)

        ke_flat = fsc.batch_ke_mitc3_composite(
            coords_batch, cm, cb, cs,
            np.full(n, h), np.full(n, e_equiv),
        )
        ke_batch = np.asarray(ke_flat).reshape(n, 18, 18)

        for i, ke in enumerate(ke_batch):
            np.testing.assert_allclose(ke, ke.T, atol=1e-6,
                                       err_msg=f"Element {i}: K not symmetric")
            eigvals = np.linalg.eigvalsh(ke)
            assert np.all(eigvals >= -1e-6 * max(eigvals)), (
                f"Element {i}: negative eigenvalue {eigvals.min():.4e}"
            )

    def test_batch_ke_mitc4_multiple(self, carbon_epoxy, ply_thickness):
        """Multiple MITC4 elements — all K must be symmetric and PSD."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 45, -45, 0])
        coords_list = [
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=float),
            np.array([[0, 0, 0], [1, 0.1, 0.2], [1.1, 1, 0.3], [0, 0.9, 0.1]], dtype=float),
        ]
        n = len(coords_list)
        coords_batch = np.array([c.ravel() for c in coords_list])
        cm = np.tile(laminate.A.ravel(), (n, 1))
        cb = np.tile(laminate.D.ravel(), (n, 1))
        cs = np.tile(laminate.Cs.ravel(), (n, 1))
        h = laminate.total_thickness
        e_equiv = np.trace(laminate.A) / (3.0 * h)

        ke_flat = fsc.batch_ke_mitc4_composite(
            coords_batch, cm, cb, cs,
            np.full(n, h), np.full(n, e_equiv),
        )
        ke_batch = np.asarray(ke_flat).reshape(n, 24, 24)

        for i, ke in enumerate(ke_batch):
            np.testing.assert_allclose(ke, ke.T, atol=1e-6,
                                       err_msg=f"Element {i}: K not symmetric")
            eigvals = np.linalg.eigvalsh(ke)
            assert np.all(eigvals >= -1e-6 * max(eigvals)), (
                f"Element {i}: negative eigenvalue {eigvals.min():.4e}"
            )


# =============================================================================
# Tests: Physical sanity checks
# =============================================================================


class TestCompositeSanity:
    """Physical sanity checks for composite elements (Rust-only)."""

    def test_ke_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        """Stiffness eigenvalues should be non-negative."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        rust_ke = _rust_ke_mitc3(tri_coords, laminate)
        eigvals = np.linalg.eigvalsh(rust_ke)
        assert np.all(eigvals >= -1e-6 * max(eigvals)), \
            f"Negative eigenvalue: {eigvals.min():.4e}"

    def test_me_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        """Mass eigenvalues should be non-negative."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        rust_me = _rust_me_mitc3(tri_coords, laminate)
        eigvals = np.linalg.eigvalsh(rust_me)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.4e}"

    def test_stiffer_fiber_direction(self, carbon_epoxy, ply_thickness, quad_coords):
        """[0] laminate should be stiffer in x than [90]."""
        lam_0 = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0])
        lam_90 = create_laminate_from_angles(carbon_epoxy, ply_thickness, [90])

        ke_0 = _rust_ke_mitc4(quad_coords, lam_0)
        ke_90 = _rust_ke_mitc4(quad_coords, lam_90)

        # x-translation DOF (index 0) should be stiffer for 0° fibers
        assert ke_0[0, 0] > ke_90[0, 0], "0° should be stiffer in x"

    def test_thicker_laminate_stiffer(self, carbon_epoxy, tri_coords):
        """More plies = stiffer element."""
        thin_lam = create_laminate_from_angles(carbon_epoxy, 0.1e-3, [0, 90])
        thick_lam = create_laminate_from_angles(carbon_epoxy, 0.5e-3, [0, 90])

        ke_thin = _rust_ke_mitc3(tri_coords, thin_lam)
        ke_thick = _rust_ke_mitc3(tri_coords, thick_lam)

        assert np.linalg.norm(ke_thick) > np.linalg.norm(ke_thin), \
            "Thicker laminate should be stiffer"

    def test_assembler_composite_mitc3_symmetry(
        self, carbon_epoxy, ply_thickness, tri_coords
    ):
        """PyMeshAssembler with composite material gives symmetric K/M."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 90, 90, 0])
        mat_dict = _laminate_to_mat_dict(laminate)

        asm = PyMeshAssembler(
            node_coords=tri_coords,
            connectivity=[[0, 1, 2]],
            elem_types=[3],   # MITC3 = 3
            materials=[mat_dict],
        )
        K = _dense_from_asm(asm, "K")
        M = _dense_from_asm(asm, "M")

        np.testing.assert_allclose(K, K.T, atol=1e-6, err_msg="K not symmetric")
        np.testing.assert_allclose(M, M.T, atol=1e-10, err_msg="M not symmetric")

    def test_assembler_composite_mitc4_symmetry(
        self, carbon_epoxy, ply_thickness, quad_coords
    ):
        """PyMeshAssembler with composite material gives symmetric K/M (MITC4)."""
        laminate = create_laminate_from_angles(carbon_epoxy, ply_thickness, [0, 45, -45, 0])
        mat_dict = _laminate_to_mat_dict(laminate)

        asm = PyMeshAssembler(
            node_coords=quad_coords,
            connectivity=[[0, 1, 2, 3]],
            elem_types=[44],   # MITC4Composite = 44
            materials=[mat_dict],
        )
        K = _dense_from_asm(asm, "K")
        M = _dense_from_asm(asm, "M")

        np.testing.assert_allclose(K, K.T, atol=1e-6, err_msg="K not symmetric")
        np.testing.assert_allclose(M, M.T, atol=1e-10, err_msg="M not symmetric")
