"""
Tests for Rust composite element integration.

Validates that Rust batch composite functions produce the same K and M
matrices as the Python MITC3Composite / MITC4Composite elements.
"""

import numpy as np
import pytest

from fem_shell.core.material import OrthotropicMaterial
from fem_shell.core.laminate import (
    Ply,
    Laminate,
    create_laminate_from_angles,
)
from fem_shell.elements.MITC3_composite import MITC3Composite
from fem_shell.elements.MITC4_composite import MITC4Composite

# Try importing Rust backend
try:
    import fem_shell_core as fsc

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="fem_shell_core not available")


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
# Helper: call Rust and compare to Python
# =============================================================================


def _rust_ke_mitc3(coords, laminate):
    """Compute MITC3 composite stiffness via Rust."""
    n = 1
    coords_flat = coords.ravel()[np.newaxis, :]  # (1, 9)
    cm = laminate.A.ravel()[np.newaxis, :]  # (1, 9)
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
    """Compute MITC3 composite mass via Rust."""
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
    """Compute MITC4 composite stiffness via Rust."""
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
    """Compute MITC4 composite mass via Rust."""
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


# =============================================================================
# Tests: MITC3 Composite — K stiffness
# =============================================================================


class TestMITC3CompositeKe:
    """Test MITC3 composite stiffness Rust vs Python."""

    def test_symmetric_crossply(self, carbon_epoxy, ply_thickness, tri_coords):
        """[0/90/90/0] symmetric cross-ply laminate."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc3(tri_coords, laminate)

        assert rust_ke.shape == (18, 18)
        assert np.allclose(rust_ke, rust_ke.T, atol=1e-6), "K not symmetric"
        # Allow some tolerance for drilling stiffness differences
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e} too large"

    def test_unidirectional_0(self, carbon_epoxy, ply_thickness, tri_coords):
        """Single 0° ply."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc3(tri_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_quasi_isotropic(self, carbon_epoxy, ply_thickness, tri_coords):
        """[0/45/-45/90]s quasi-isotropic."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 90, 90, -45, 45, 0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc3(tri_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_3d_element(self, carbon_epoxy, ply_thickness, tri_coords_3d):
        """Element in 3D space (not aligned with XY plane)."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC3Composite(tri_coords_3d, (1, 2, 3), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc3(tri_coords_3d, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_glass_epoxy(self, glass_epoxy, ply_thickness, tri_coords):
        """Glass/Epoxy material."""
        laminate = create_laminate_from_angles(
            glass_epoxy, ply_thickness, [0, 45, -45, 0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc3(tri_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"


# =============================================================================
# Tests: MITC3 Composite — M mass
# =============================================================================


class TestMITC3CompositeMe:
    """Test MITC3 composite mass Rust vs Python."""

    def test_symmetric_crossply(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_me = py_elem.M

        rust_me = _rust_me_mitc3(tri_coords, laminate)

        assert rust_me.shape == (18, 18)
        assert np.allclose(rust_me, rust_me.T, atol=1e-10), "M not symmetric"
        rel_err = np.linalg.norm(rust_me - py_me) / np.linalg.norm(py_me)
        assert rel_err < 1e-10, f"Relative error {rel_err:.4e}"

    def test_quasi_isotropic(self, carbon_epoxy, ply_thickness, tri_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 90, 90, -45, 45, 0]
        )
        py_elem = MITC3Composite(tri_coords, (1, 2, 3), laminate)
        py_me = py_elem.M

        rust_me = _rust_me_mitc3(tri_coords, laminate)
        rel_err = np.linalg.norm(rust_me - py_me) / np.linalg.norm(py_me)
        assert rel_err < 1e-10, f"Relative error {rel_err:.4e}"

    def test_3d_element(self, carbon_epoxy, ply_thickness, tri_coords_3d):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC3Composite(tri_coords_3d, (1, 2, 3), laminate)
        py_me = py_elem.M

        rust_me = _rust_me_mitc3(tri_coords_3d, laminate)
        rel_err = np.linalg.norm(rust_me - py_me) / np.linalg.norm(py_me)
        assert rel_err < 1e-10, f"Relative error {rel_err:.4e}"


# =============================================================================
# Tests: MITC4 Composite — K stiffness
# =============================================================================


class TestMITC4CompositeKe:
    """Test MITC4 composite stiffness Rust vs Python."""

    def test_symmetric_crossply(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC4Composite(quad_coords, (1, 2, 3, 4), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc4(quad_coords, laminate)

        assert rust_ke.shape == (24, 24)
        assert np.allclose(rust_ke, rust_ke.T, atol=1e-6), "K not symmetric"
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_unidirectional_45(self, carbon_epoxy, ply_thickness, quad_coords):
        """Single 45° ply — anisotropic in-plane behavior."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [45]
        )
        py_elem = MITC4Composite(quad_coords, (1, 2, 3, 4), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc4(quad_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_quasi_isotropic(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 90, 90, -45, 45, 0]
        )
        py_elem = MITC4Composite(quad_coords, (1, 2, 3, 4), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc4(quad_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_3d_element(self, carbon_epoxy, ply_thickness, quad_coords_3d):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC4Composite(quad_coords_3d, (1, 2, 3, 4), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc4(quad_coords_3d, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"

    def test_glass_epoxy_angle_ply(self, glass_epoxy, ply_thickness, quad_coords):
        """[+45/-45/-45/+45] angle-ply glass/epoxy."""
        laminate = create_laminate_from_angles(
            glass_epoxy, ply_thickness, [45, -45, -45, 45]
        )
        py_elem = MITC4Composite(quad_coords, (1, 2, 3, 4), laminate)
        py_ke = py_elem.K

        rust_ke = _rust_ke_mitc4(quad_coords, laminate)
        rel_err = np.linalg.norm(rust_ke - py_ke) / np.linalg.norm(py_ke)
        assert rel_err < 0.02, f"Relative error {rel_err:.4e}"


# =============================================================================
# Tests: MITC4 Composite — M mass
# =============================================================================


class TestMITC4CompositeMe:
    """Test MITC4 composite mass Rust vs Python."""

    def test_symmetric_crossply(self, carbon_epoxy, ply_thickness, quad_coords):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC4Composite(quad_coords, (1, 2, 3, 4), laminate)
        py_me = py_elem.M

        rust_me = _rust_me_mitc4(quad_coords, laminate)

        assert rust_me.shape == (24, 24)
        assert np.allclose(rust_me, rust_me.T, atol=1e-10), "M not symmetric"
        rel_err = np.linalg.norm(rust_me - py_me) / np.linalg.norm(py_me)
        assert rel_err < 1e-10, f"Relative error {rel_err:.4e}"

    def test_3d_element(self, carbon_epoxy, ply_thickness, quad_coords_3d):
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        py_elem = MITC4Composite(quad_coords_3d, (1, 2, 3, 4), laminate)
        py_me = py_elem.M

        rust_me = _rust_me_mitc4(quad_coords_3d, laminate)
        rel_err = np.linalg.norm(rust_me - py_me) / np.linalg.norm(py_me)
        assert rel_err < 1e-10, f"Relative error {rel_err:.4e}"


# =============================================================================
# Tests: Batch processing (multiple elements)
# =============================================================================


class TestBatchComposite:
    """Test batch processing with multiple composite elements."""

    def test_batch_ke_mitc3_multiple(self, carbon_epoxy, ply_thickness):
        """Multiple MITC3 elements with same laminate."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
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
        a_trace = np.trace(laminate.A)
        e_equiv = a_trace / (3.0 * h)

        ke_flat = fsc.batch_ke_mitc3_composite(
            coords_batch, cm, cb, cs,
            np.full(n, h), np.full(n, e_equiv),
        )
        ke_batch = np.asarray(ke_flat).reshape(n, 18, 18)

        for i, c in enumerate(coords_list):
            py_elem = MITC3Composite(c, (1, 2, 3), laminate)
            rel_err = np.linalg.norm(ke_batch[i] - py_elem.K) / np.linalg.norm(py_elem.K)
            assert rel_err < 0.02, f"Element {i}: relative error {rel_err:.4e}"

    def test_batch_ke_mitc4_multiple(self, carbon_epoxy, ply_thickness):
        """Multiple MITC4 elements with same laminate."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 45, -45, 0]
        )
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
        a_trace = np.trace(laminate.A)
        e_equiv = a_trace / (3.0 * h)

        ke_flat = fsc.batch_ke_mitc4_composite(
            coords_batch, cm, cb, cs,
            np.full(n, h), np.full(n, e_equiv),
        )
        ke_batch = np.asarray(ke_flat).reshape(n, 24, 24)

        for i, c in enumerate(coords_list):
            py_elem = MITC4Composite(c, (1, 2, 3, 4), laminate)
            rel_err = np.linalg.norm(ke_batch[i] - py_elem.K) / np.linalg.norm(py_elem.K)
            assert rel_err < 0.02, f"Element {i}: relative error {rel_err:.4e}"


# =============================================================================
# Tests: Physical sanity checks
# =============================================================================


class TestCompositeSanity:
    """Physical sanity checks for composite elements."""

    def test_ke_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        """Stiffness eigenvalues should be non-negative."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
        rust_ke = _rust_ke_mitc3(tri_coords, laminate)
        eigvals = np.linalg.eigvalsh(rust_ke)
        assert np.all(eigvals >= -1e-6 * max(eigvals)), \
            f"Negative eigenvalue: {eigvals.min():.4e}"

    def test_me_positive_semidefinite(self, carbon_epoxy, ply_thickness, tri_coords):
        """Mass eigenvalues should be non-negative."""
        laminate = create_laminate_from_angles(
            carbon_epoxy, ply_thickness, [0, 90, 90, 0]
        )
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
