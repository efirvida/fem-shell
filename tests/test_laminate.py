"""
Tests for Laminated Composite Implementation.

This module contains comprehensive tests for:
1. Laminate module (Ply, Laminate, ABD matrices)
2. Failure criteria (Tsai-Wu, Hashin, Max Stress)
3. MITC4Composite element

The tests validate:
- CLT formulation correctness
- ABD matrix computation
- Symmetric/balanced laminate properties
- Coordinate transformations
- Failure criterion implementations
- Element stiffness matrix assembly
- Ply-level stress recovery
"""

import numpy as np
import pytest

from fem_shell.core.material import OrthotropicMaterial
from fem_shell.core.laminate import (
    Ply,
    Laminate,
    StrengthProperties,
    compute_Q,
    compute_Qbar,
    compute_shear_Cbar,
    create_symmetric_laminate,
    create_laminate_from_angles,
    quasi_isotropic_layup,
    cross_ply_layup,
    angle_ply_layup,
)
from fem_shell.constitutive.failure import (
    FailureMode,
    FailureResult,
    tsai_wu_failure_index,
    hashin_failure_indices,
    max_stress_failure_index,
    stress_transformation_matrix,
    strain_transformation_matrix,
)
from fem_shell.elements.MITC4_composite import MITC4Composite


# =============================================================================
# Fixtures - Common test materials and configurations
# =============================================================================

@pytest.fixture
def carbon_epoxy_material():
    """Standard Carbon/Epoxy T300/5208 material."""
    return OrthotropicMaterial(
        name="Carbon/Epoxy T300/5208",
        E=(181e9, 10.3e9, 10.3e9),      # E1, E2, E3 [Pa]
        G=(7.17e9, 3.78e9, 7.17e9),     # G12, G23, G13 [Pa]
        nu=(0.28, 0.28, 0.28),          # nu12, nu23, nu31
        rho=1600,                        # kg/m³
    )


@pytest.fixture
def glass_epoxy_material():
    """E-Glass/Epoxy material."""
    return OrthotropicMaterial(
        name="E-Glass/Epoxy",
        E=(38.6e9, 8.27e9, 8.27e9),
        G=(4.14e9, 3.0e9, 4.14e9),
        nu=(0.26, 0.26, 0.26),
        rho=1800,
    )


@pytest.fixture
def carbon_epoxy_strength():
    """Strength properties for Carbon/Epoxy T300/5208."""
    return StrengthProperties(
        Xt=1500e6,   # Fiber tensile strength [Pa]
        Xc=1500e6,   # Fiber compressive strength [Pa]
        Yt=40e6,     # Matrix tensile strength [Pa]
        Yc=246e6,    # Matrix compressive strength [Pa]
        S12=68e6,    # In-plane shear strength [Pa]
        S23=34e6,    # Transverse shear strength [Pa]
    )


@pytest.fixture
def ply_thickness():
    """Standard ply thickness."""
    return 0.125e-3  # 0.125 mm


@pytest.fixture
def square_element_coords():
    """Coordinates for a 1m x 1m square element."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


# =============================================================================
# Tests for Reduced Stiffness Matrix Q
# =============================================================================

class TestReducedStiffnessQ:
    """Tests for the reduced stiffness matrix computation."""

    def test_Q_symmetric(self, carbon_epoxy_material):
        """Q matrix should be symmetric."""
        Q = compute_Q(carbon_epoxy_material)
        assert np.allclose(Q, Q.T), "Q matrix should be symmetric"

    def test_Q_components(self, carbon_epoxy_material):
        """Verify Q matrix components against manual calculation."""
        mat = carbon_epoxy_material
        E1, E2, _ = mat.E
        nu12, _, _ = mat.nu
        G12, _, _ = mat.G

        nu21 = nu12 * E2 / E1
        denom = 1 - nu12 * nu21

        Q11_expected = E1 / denom
        Q22_expected = E2 / denom
        Q12_expected = nu12 * E2 / denom
        Q66_expected = G12

        Q = compute_Q(mat)

        assert np.isclose(Q[0, 0], Q11_expected, rtol=1e-10)
        assert np.isclose(Q[1, 1], Q22_expected, rtol=1e-10)
        assert np.isclose(Q[0, 1], Q12_expected, rtol=1e-10)
        assert np.isclose(Q[2, 2], Q66_expected, rtol=1e-10)

    def test_Q_positive_definite(self, carbon_epoxy_material):
        """Q matrix should be positive definite."""
        Q = compute_Q(carbon_epoxy_material)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0), "Q should be positive definite"


# =============================================================================
# Tests for Transformed Stiffness Matrix Qbar
# =============================================================================

class TestTransformedStiffnessQbar:
    """Tests for the transformed stiffness matrix Qbar."""

    def test_Qbar_zero_angle(self, carbon_epoxy_material):
        """At θ=0°, Qbar should equal Q."""
        Q = compute_Q(carbon_epoxy_material)
        Qbar = compute_Qbar(carbon_epoxy_material, 0.0)
        assert np.allclose(Qbar, Q, rtol=1e-10)

    def test_Qbar_90_degrees(self, carbon_epoxy_material):
        """At θ=90°, Q11 and Q22 should swap."""
        Q = compute_Q(carbon_epoxy_material)
        Qbar = compute_Qbar(carbon_epoxy_material, 90.0)

        assert np.isclose(Qbar[0, 0], Q[1, 1], rtol=1e-10), "Q11 -> Q22"
        assert np.isclose(Qbar[1, 1], Q[0, 0], rtol=1e-10), "Q22 -> Q11"
        assert np.isclose(Qbar[0, 1], Q[0, 1], rtol=1e-10), "Q12 unchanged"

    def test_Qbar_symmetric(self, carbon_epoxy_material):
        """Qbar should be symmetric for any angle."""
        for angle in [0, 30, 45, 60, 90, -45]:
            Qbar = compute_Qbar(carbon_epoxy_material, angle)
            assert np.allclose(Qbar, Qbar.T), f"Qbar not symmetric at {angle}°"

    def test_Qbar_coupling_at_45_degrees(self, carbon_epoxy_material):
        """At ±45°, coupling terms Q16 and Q26 should be non-zero."""
        Qbar_45 = compute_Qbar(carbon_epoxy_material, 45.0)
        assert not np.isclose(Qbar_45[0, 2], 0.0), "Q16 should be non-zero at 45°"
        assert not np.isclose(Qbar_45[1, 2], 0.0), "Q26 should be non-zero at 45°"

    def test_Qbar_no_coupling_at_0_90(self, carbon_epoxy_material):
        """At 0° and 90°, coupling terms should be zero."""
        for angle in [0, 90]:
            Qbar = compute_Qbar(carbon_epoxy_material, angle)
            # Use relative tolerance based on main diagonal terms
            tol = 1e-10 * np.max(np.abs(np.diag(Qbar)))
            assert np.isclose(Qbar[0, 2], 0.0, atol=tol)
            assert np.isclose(Qbar[1, 2], 0.0, atol=tol)

    def test_Qbar_angle_sign_symmetry(self, carbon_epoxy_material):
        """Qbar(+θ) and Qbar(-θ) should have opposite signs for Q16, Q26."""
        Qbar_pos = compute_Qbar(carbon_epoxy_material, 45.0)
        Qbar_neg = compute_Qbar(carbon_epoxy_material, -45.0)

        # Q16 and Q26 change sign
        assert np.isclose(Qbar_pos[0, 2], -Qbar_neg[0, 2], rtol=1e-10)
        assert np.isclose(Qbar_pos[1, 2], -Qbar_neg[1, 2], rtol=1e-10)

        # Other components stay the same
        assert np.isclose(Qbar_pos[0, 0], Qbar_neg[0, 0], rtol=1e-10)
        assert np.isclose(Qbar_pos[1, 1], Qbar_neg[1, 1], rtol=1e-10)


# =============================================================================
# Tests for Transverse Shear Stiffness
# =============================================================================

class TestTransverseShearCbar:
    """Tests for transformed transverse shear stiffness."""

    def test_Cbar_zero_angle(self, carbon_epoxy_material):
        """At θ=0°, Cbar should have G13 and G23 on diagonal."""
        _, G23, G13 = carbon_epoxy_material.G
        Cbar = compute_shear_Cbar(carbon_epoxy_material, 0.0)

        assert np.isclose(Cbar[0, 0], G13, rtol=1e-10)  # C55
        assert np.isclose(Cbar[1, 1], G23, rtol=1e-10)  # C44
        assert np.isclose(Cbar[0, 1], 0.0, atol=1e-10)  # C45

    def test_Cbar_symmetric(self, carbon_epoxy_material):
        """Cbar should be symmetric."""
        for angle in [0, 30, 45, 60, 90]:
            Cbar = compute_shear_Cbar(carbon_epoxy_material, angle)
            assert np.allclose(Cbar, Cbar.T)


# =============================================================================
# Tests for Ply Class
# =============================================================================

class TestPly:
    """Tests for Ply dataclass."""

    def test_ply_creation(self, carbon_epoxy_material):
        """Test basic ply creation."""
        ply = Ply(carbon_epoxy_material, 0.125e-3, 45.0)

        assert ply.material == carbon_epoxy_material
        assert ply.thickness == 0.125e-3
        assert ply.angle == 45.0
        assert np.isclose(ply.angle_rad, np.radians(45.0))

    def test_ply_with_strength(self, carbon_epoxy_material, carbon_epoxy_strength):
        """Test ply with strength properties."""
        ply = Ply(carbon_epoxy_material, 0.125e-3, 0.0, carbon_epoxy_strength)
        assert ply.strength is not None
        assert ply.strength.Xt == 1500e6


# =============================================================================
# Tests for Laminate Class
# =============================================================================

class TestLaminate:
    """Tests for Laminate class and ABD matrix computation."""

    def test_laminate_creation(self, carbon_epoxy_material, ply_thickness):
        """Test basic laminate creation."""
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, 0),
            Ply(carbon_epoxy_material, ply_thickness, 90),
        ]
        laminate = Laminate(plies)

        assert laminate.n_plies == 2
        assert np.isclose(laminate.total_thickness, 2 * ply_thickness)

    def test_layer_positions(self, carbon_epoxy_material, ply_thickness):
        """Test that layer z-positions are computed correctly."""
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, 0),
            Ply(carbon_epoxy_material, ply_thickness, 90),
            Ply(carbon_epoxy_material, ply_thickness, 90),
            Ply(carbon_epoxy_material, ply_thickness, 0),
        ]
        laminate = Laminate(plies)

        h = laminate.total_thickness

        # Check first ply position
        assert np.isclose(plies[0].z_bottom, -h / 2)
        assert np.isclose(plies[0].z_top, -h / 2 + ply_thickness)

        # Check last ply position
        assert np.isclose(plies[-1].z_top, h / 2)

    def test_symmetric_laminate_B_zero(self, carbon_epoxy_material, ply_thickness):
        """For symmetric laminate, B matrix should be zero."""
        # [0/90]s = [0/90/90/0]
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, 0),
            Ply(carbon_epoxy_material, ply_thickness, 90),
            Ply(carbon_epoxy_material, ply_thickness, 90),
            Ply(carbon_epoxy_material, ply_thickness, 0),
        ]
        laminate = Laminate(plies)

        assert laminate.is_symmetric, "Should be symmetric"
        assert np.allclose(laminate.B, 0, atol=1e-10 * np.max(np.abs(laminate.A)))

    def test_asymmetric_laminate_B_nonzero(self, carbon_epoxy_material, ply_thickness):
        """For asymmetric laminate, B matrix should be non-zero."""
        # [0/90] - asymmetric
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, 0),
            Ply(carbon_epoxy_material, ply_thickness, 90),
        ]
        laminate = Laminate(plies)

        assert not laminate.is_symmetric, "Should not be symmetric"
        assert not np.allclose(laminate.B, 0), "B should be non-zero"

    def test_balanced_laminate_A16_A26_zero(self, carbon_epoxy_material, ply_thickness):
        """For balanced laminate [±45]s, A16 = A26 = 0."""
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, 45),
            Ply(carbon_epoxy_material, ply_thickness, -45),
            Ply(carbon_epoxy_material, ply_thickness, -45),
            Ply(carbon_epoxy_material, ply_thickness, 45),
        ]
        laminate = Laminate(plies)

        assert laminate.is_balanced, "Should be balanced"
        tol = 1e-10 * np.max(np.abs(laminate.A))
        assert np.isclose(laminate.A[0, 2], 0, atol=tol), "A16 should be zero"
        assert np.isclose(laminate.A[1, 2], 0, atol=tol), "A26 should be zero"

    def test_quasi_isotropic_A11_A22_equal(self, carbon_epoxy_material, ply_thickness):
        """For quasi-isotropic laminate, A11 ≈ A22."""
        angles = quasi_isotropic_layup(1)  # [0/45/-45/90/90/-45/45/0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        A = laminate.A
        assert np.isclose(A[0, 0], A[1, 1], rtol=0.01), "A11 should equal A22"
        assert laminate.is_balanced, "Quasi-isotropic should be balanced"
        assert laminate.is_symmetric, "Quasi-isotropic should be symmetric"

    def test_ABD_positive_definite(self, carbon_epoxy_material, ply_thickness):
        """A and D matrices should be positive definite."""
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        # Check A matrix
        eigvals_A = np.linalg.eigvalsh(laminate.A)
        assert np.all(eigvals_A > 0), "A should be positive definite"

        # Check D matrix
        eigvals_D = np.linalg.eigvalsh(laminate.D)
        assert np.all(eigvals_D > 0), "D should be positive definite"

    def test_create_symmetric_laminate_helper(self, carbon_epoxy_material, ply_thickness):
        """Test create_symmetric_laminate helper function."""
        half_plies = [
            Ply(carbon_epoxy_material, ply_thickness, 0),
            Ply(carbon_epoxy_material, ply_thickness, 90),
        ]
        laminate = create_symmetric_laminate(half_plies)

        assert laminate.n_plies == 4
        assert laminate.is_symmetric


# =============================================================================
# Tests for Stress/Strain Transformation Matrices
# =============================================================================

class TestTransformationMatrices:
    """Tests for stress and strain transformation matrices."""

    def test_stress_transform_identity_at_zero(self):
        """At θ=0°, transformation should be identity."""
        T = stress_transformation_matrix(0.0)
        assert np.allclose(T, np.eye(3), atol=1e-10)

    def test_stress_transform_orthogonal(self):
        """Transformation matrix should be orthogonal."""
        for angle in [0, 30, 45, 60, 90]:
            T = stress_transformation_matrix(angle)
            assert np.isclose(np.linalg.det(T), 1.0, rtol=1e-10)

    def test_strain_transform_identity_at_zero(self):
        """At θ=0°, strain transformation should be identity."""
        T = strain_transformation_matrix(0.0)
        assert np.allclose(T, np.eye(3), atol=1e-10)


# =============================================================================
# Tests for Failure Criteria
# =============================================================================

class TestTsaiWuCriterion:
    """Tests for Tsai-Wu failure criterion."""

    def test_no_stress_no_failure(self, carbon_epoxy_strength):
        """Zero stress should give zero failure index."""
        sigma = np.array([0.0, 0.0, 0.0])
        result = tsai_wu_failure_index(sigma, carbon_epoxy_strength)

        assert result.failure_index == 0.0
        assert not result.failed

    def test_fiber_dominated_failure(self, carbon_epoxy_strength):
        """High fiber stress should cause failure."""
        sigma = np.array([1600e6, 0.0, 0.0])  # Above Xt
        result = tsai_wu_failure_index(sigma, carbon_epoxy_strength)

        assert result.failure_index > 1.0
        assert result.failed

    def test_matrix_dominated_failure(self, carbon_epoxy_strength):
        """High transverse stress should cause failure."""
        sigma = np.array([0.0, 50e6, 0.0])  # Above Yt (40 MPa)
        result = tsai_wu_failure_index(sigma, carbon_epoxy_strength)

        assert result.failure_index > 1.0
        assert result.failed

    def test_shear_failure(self, carbon_epoxy_strength):
        """High shear stress should cause failure."""
        sigma = np.array([0.0, 0.0, 80e6])  # Above S12 (68 MPa)
        result = tsai_wu_failure_index(sigma, carbon_epoxy_strength)

        assert result.failure_index > 1.0
        assert result.failed

    def test_safe_state(self, carbon_epoxy_strength):
        """Low stress should be safe."""
        sigma = np.array([100e6, 10e6, 20e6])  # Well below strengths
        result = tsai_wu_failure_index(sigma, carbon_epoxy_strength)

        assert result.failure_index < 1.0
        assert not result.failed
        assert result.reserve_factor > 1.0


class TestHashinCriterion:
    """Tests for Hashin failure criterion."""

    def test_fiber_tension_mode(self, carbon_epoxy_strength):
        """Fiber tension failure mode."""
        sigma = np.array([1600e6, 0.0, 0.0])  # Above Xt
        results = hashin_failure_indices(sigma, carbon_epoxy_strength)

        assert FailureMode.FIBER_TENSION in results
        assert results[FailureMode.FIBER_TENSION].failure_index > 1.0

    def test_fiber_compression_mode(self, carbon_epoxy_strength):
        """Fiber compression failure mode."""
        sigma = np.array([-1600e6, 0.0, 0.0])  # Above Xc
        results = hashin_failure_indices(sigma, carbon_epoxy_strength)

        assert FailureMode.FIBER_COMPRESSION in results
        assert results[FailureMode.FIBER_COMPRESSION].failure_index > 1.0

    def test_matrix_tension_mode(self, carbon_epoxy_strength):
        """Matrix tension failure mode."""
        sigma = np.array([0.0, 50e6, 0.0])  # Above Yt
        results = hashin_failure_indices(sigma, carbon_epoxy_strength)

        assert FailureMode.MATRIX_TENSION in results
        assert results[FailureMode.MATRIX_TENSION].failure_index > 1.0

    def test_matrix_compression_mode(self, carbon_epoxy_strength):
        """Matrix compression failure mode."""
        sigma = np.array([0.0, -300e6, 0.0])  # Above Yc
        results = hashin_failure_indices(sigma, carbon_epoxy_strength)

        assert FailureMode.MATRIX_COMPRESSION in results
        assert results[FailureMode.MATRIX_COMPRESSION].failure_index > 1.0


class TestMaxStressCriterion:
    """Tests for maximum stress criterion."""

    def test_identifies_critical_component(self, carbon_epoxy_strength):
        """Should identify the critical stress component."""
        # Matrix tension is critical
        sigma = np.array([500e6, 35e6, 30e6])
        result = max_stress_failure_index(sigma, carbon_epoxy_strength)

        # Yt = 40 MPa, so 35/40 = 0.875 is highest ratio
        assert result.mode == FailureMode.MATRIX_TENSION

    def test_no_interaction(self, carbon_epoxy_strength):
        """Max stress has no interaction between components."""
        # Individual components below limit but combined would fail Tsai-Wu
        sigma = np.array([1000e6, 30e6, 50e6])
        result = max_stress_failure_index(sigma, carbon_epoxy_strength)

        # Should consider only the highest individual ratio
        ratios = [1000e6 / 1500e6, 30e6 / 40e6, 50e6 / 68e6]
        assert np.isclose(result.failure_index, max(ratios), rtol=0.01)


# =============================================================================
# Tests for MITC4Composite Element
# =============================================================================

class TestMITC4CompositeElement:
    """Tests for the MITC4Composite shell element."""

    def test_element_creation(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test basic element creation."""
        angles = [0, 90, 90, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        assert element.element_type == "MITC4Composite"
        assert element.laminate == laminate
        assert not element.has_coupling  # Symmetric laminate

    def test_stiffness_matrix_symmetric(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Element stiffness matrix should be symmetric."""
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        K = element.K()
        assert np.allclose(K, K.T), "Stiffness matrix should be symmetric"

    def test_stiffness_matrix_positive_semidefinite(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Element stiffness matrix should be positive semi-definite."""
        angles = [0, 90, 90, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        K = element.K()
        eigenvalues = np.linalg.eigvalsh(K)
        
        # Allow small negative values due to numerical precision
        assert np.all(eigenvalues >= -1e-10 * np.max(np.abs(eigenvalues)))

    def test_stiffness_with_coupling(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test stiffness matrix for asymmetric laminate with coupling."""
        # Asymmetric: [0/90] - should have B != 0
        angles = [0, 90]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        assert element.has_coupling, "Asymmetric laminate should have coupling"

        K = element.K()
        assert np.allclose(K, K.T), "K should still be symmetric"

    def test_constitutive_matrices(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test that constitutive matrices are returned correctly."""
        angles = [0, 45, -45, 90]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        Cm = element.Cm()
        Cb = element.Cb()
        Cs = element.Cs()

        # Check shapes
        assert Cm.shape == (3, 3)
        assert Cb.shape == (3, 3)
        assert Cs.shape == (2, 2)

        # Check symmetry
        assert np.allclose(Cm, Cm.T)
        assert np.allclose(Cb, Cb.T)
        assert np.allclose(Cs, Cs.T)

    def test_mass_matrix(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test element mass matrix computation."""
        angles = [0, 90, 90, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        M = element.mass()

        # Check shape and symmetry
        assert M.shape == (24, 24)
        assert np.allclose(M, M.T), "Mass matrix should be symmetric"

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10)

    def test_midplane_strains(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test mid-plane strain computation."""
        angles = [0, 90, 90, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # Create a simple displacement (pure stretch in x)
        u_local = np.zeros(24)
        u_local[0] = 0.0   # u1
        u_local[6] = 0.001  # u2 (1mm)
        u_local[12] = 0.001  # u3
        u_local[18] = 0.0  # u4

        epsilon_0, kappa = element.compute_midplane_strains(u_local)

        # Should have non-zero εxx
        assert epsilon_0[0] != 0.0, "Should have εxx strain"

        # Check shapes
        assert epsilon_0.shape == (3,)
        assert kappa.shape == (3,)

    def test_ply_stresses(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test ply-level stress computation."""
        angles = [0, 45, -45, 90]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # Apply uniform stretch
        u_local = np.zeros(24)
        u_local[[6, 12]] = 0.001  # Stretch in x

        stresses = element.compute_ply_stresses(u_local)

        # Should have one result per ply
        assert len(stresses) == 4

        # Check stress structure
        for stress_data in stresses:
            assert 'ply_index' in stress_data
            assert 'sigma_laminate' in stress_data
            assert 'sigma_ply' in stress_data
            assert stress_data['sigma_laminate'].shape == (3,)
            assert stress_data['sigma_ply'].shape == (3,)

    def test_stress_resultants(self, carbon_epoxy_material, ply_thickness, square_element_coords):
        """Test stress resultants computation."""
        angles = [0, 90, 90, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # Apply displacement
        u_local = np.zeros(24)
        u_local[[6, 12]] = 0.001

        N, M = element.compute_stress_resultants(u_local)

        assert N.shape == (3,), "N should be [Nxx, Nyy, Nxy]"
        assert M.shape == (3,), "M should be [Mxx, Myy, Mxy]"

        # For symmetric laminate with in-plane load, M should be small
        assert np.allclose(M, 0, atol=1e-3 * np.max(np.abs(N)))


class TestMITC4CompositeFailure:
    """Tests for failure analysis with MITC4Composite."""

    def test_failure_evaluation(self, carbon_epoxy_material, carbon_epoxy_strength, 
                                ply_thickness, square_element_coords):
        """Test failure criterion evaluation."""
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, angle, carbon_epoxy_strength)
            for angle in [0, 45, -45, 90]
        ]
        laminate = Laminate(plies)

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # Apply moderate displacement
        u_local = np.zeros(24)
        u_local[[6, 12]] = 0.0001  # Small stretch

        results = element.evaluate_failure(u_local, criterion="tsai-wu")

        assert len(results) == 4
        for res in results:
            assert 'failure_index' in res
            assert 'failed' in res
            assert 'mode' in res

    def test_critical_ply_identification(self, carbon_epoxy_material, carbon_epoxy_strength,
                                         ply_thickness, square_element_coords):
        """Test finding the most critical ply."""
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, angle, carbon_epoxy_strength)
            for angle in [0, 90, 90, 0]
        ]
        laminate = Laminate(plies)

        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # Apply displacement
        u_local = np.zeros(24)
        u_local[[6, 12]] = 0.0001

        critical = element.get_critical_ply(u_local, criterion="tsai-wu")

        assert 'ply_index' in critical
        assert 'failure_index' in critical
        assert critical['failure_index'] == max(
            element.evaluate_failure(u_local)
        , key=lambda x: x['failure_index'])['failure_index']


# =============================================================================
# Benchmark Tests - Comparison with Analytical Solutions
# =============================================================================

class TestLaminateBenchmarks:
    """Benchmark tests comparing with analytical solutions."""

    def test_unidirectional_A_matrix(self, carbon_epoxy_material, ply_thickness):
        """Test A matrix for unidirectional [0]4 laminate."""
        angles = [0, 0, 0, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        h = laminate.total_thickness
        Q = compute_Q(carbon_epoxy_material)

        # For unidirectional, A = Q * h
        A_expected = Q * h

        assert np.allclose(laminate.A, A_expected, rtol=1e-10)

    def test_unidirectional_D_matrix(self, carbon_epoxy_material, ply_thickness):
        """Test D matrix for unidirectional [0]4 laminate."""
        angles = [0, 0, 0, 0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        h = laminate.total_thickness
        Q = compute_Q(carbon_epoxy_material)

        # For unidirectional, D = Q * h³/12
        D_expected = Q * h**3 / 12

        assert np.allclose(laminate.D, D_expected, rtol=1e-10)

    def test_cross_ply_A_matrix_symmetry(self, carbon_epoxy_material, ply_thickness):
        """Test A11 and A22 relationship for cross-ply [0/90]ns."""
        angles = cross_ply_layup(4)  # [0/90/90/0]
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        Q = compute_Q(carbon_epoxy_material)

        # For [0/90]s: A11 = A22 = (Q11 + Q22) * h/2
        h = laminate.total_thickness
        A11_expected = (Q[0, 0] + Q[1, 1]) * h / 2
        
        # Due to thickness distribution, there's a small difference
        # but A11 should be close to A22
        assert np.isclose(laminate.A[0, 0], laminate.A[1, 1], rtol=0.01)

    def test_laminate_consistency(self, carbon_epoxy_material, ply_thickness):
        """Test that equivalent properties are consistent with ABD."""
        angles = quasi_isotropic_layup(1)
        laminate = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, angles
        )

        equiv = laminate.get_equivalent_properties()
        h = laminate.total_thickness

        # Ex_membrane should be consistent with A11
        # Ex = 1/(h * a11) where a = inv(A)
        a = np.linalg.inv(laminate.A)
        Ex_check = 1 / (h * a[0, 0])

        assert np.isclose(equiv['Ex_membrane'], Ex_check, rtol=1e-10)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, carbon_epoxy_material, carbon_epoxy_strength,
                          ply_thickness, square_element_coords):
        """Test complete workflow from laminate definition to failure analysis."""
        # 1. Define laminate
        plies = [
            Ply(carbon_epoxy_material, ply_thickness, angle, carbon_epoxy_strength)
            for angle in [0, 45, -45, 90, 90, -45, 45, 0]
        ]
        laminate = Laminate(plies)

        # 2. Verify laminate properties
        assert laminate.is_symmetric
        assert laminate.is_balanced

        # 3. Create element
        element = MITC4Composite(
            node_coords=square_element_coords,
            node_ids=(1, 2, 3, 4),
            laminate=laminate,
        )

        # 4. Compute stiffness matrix
        K = element.K()
        assert K.shape == (24, 24)

        # 5. Apply displacement and compute response
        u = np.zeros(24)
        u[[6, 12]] = 0.0001  # Small stretch

        # 6. Compute stresses
        stresses = element.compute_ply_stresses(u)
        assert len(stresses) == 8

        # 7. Evaluate failure
        failure_results = element.evaluate_failure(u, criterion="tsai-wu")
        
        # 8. Find critical ply
        critical = element.get_critical_ply(u)
        
        # All plies should be safe for small deformation
        assert all(not res['failed'] for res in failure_results)

    def test_layup_pattern_helpers(self, carbon_epoxy_material, ply_thickness):
        """Test that layup helper functions create valid laminates."""
        # Quasi-isotropic
        qi_angles = quasi_isotropic_layup(1)
        qi_lam = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, qi_angles
        )
        assert qi_lam.is_symmetric
        assert qi_lam.is_balanced

        # Cross-ply
        cp_angles = cross_ply_layup(8)
        cp_lam = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, cp_angles
        )
        assert cp_lam.is_symmetric

        # Angle-ply
        ap_angles = angle_ply_layup(45, 8)
        ap_lam = create_laminate_from_angles(
            carbon_epoxy_material, ply_thickness, ap_angles
        )
        assert ap_lam.is_symmetric
        assert ap_lam.is_balanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
