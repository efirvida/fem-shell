"""
Tests for MITC4 element implementation.

Validates that MITC4:
1. Shares the same API as MITC4 (K, M, body_load)
2. Implements membrane MITC interpolation correctly
3. Produces different results than MITC4 (due to membrane interpolation)
4. Maintains numerical stability and positive semi-definiteness
"""

import numpy as np
import pytest

from fem_shell.core.material import Material
from fem_shell.elements.MITC4 import MITC4


class TestMITC4API:
    """Test that MITC4 shares the same API as MITC4."""

    @pytest.fixture
    def material(self):
        """Create a test material."""
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def node_coords(self):
        """Create flat rectangular element (reference domain)."""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)

    @pytest.fixture
    def node_ids(self):
        """Create node IDs."""
        return (1, 2, 3, 4)

    def test_mitc4plus_initialization(self, node_coords, node_ids, material):
        """Test that MITC4 initializes correctly."""
        elem = MITC4(node_coords, node_ids, material, thickness=0.01)
        
        assert elem.element_type == "MITC4"
        assert elem.thickness == 0.01
        assert elem.dofs_count == 24
        assert elem.dofs_per_node == 6
        assert len(elem._tying_points_eps_xx) == 4
        assert len(elem._tying_points_eps_yy) == 4
        assert len(elem._tying_points_gamma_xy) == 5

    def test_shared_api_K(self, node_coords, node_ids, material):
        """Test that K matrix is computed with same API."""
        elem = MITC4(node_coords, node_ids, material, thickness=0.01)
        K = elem.K
        
        # Check shape
        assert K.shape == (24, 24)
        
        # Check symmetry
        assert np.allclose(K, K.T, atol=1e-10)
        
        # Check positive semi-definite (all eigenvalues >= -1e-5)
        eigs = np.linalg.eigvalsh(K)
        assert np.all(eigs >= -1e-5), f"K has negative eigenvalues: {eigs[:5]}"

    def test_shared_api_M(self, node_coords, node_ids, material):
        """Test that M matrix is computed with same API."""
        elem = MITC4(node_coords, node_ids, material, thickness=0.01)
        M = elem.M
        
        # Check shape
        assert M.shape == (24, 24)
        
        # Check symmetry
        assert np.allclose(M, M.T, atol=1e-10)
        
        # Check positive semi-definite
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs >= -1e-5), f"M has negative eigenvalues: {eigs[:5]}"

    def test_shared_api_body_load(self, node_coords, node_ids, material):
        """Test that body_load method works with same API."""
        elem = MITC4(node_coords, node_ids, material, thickness=0.01)
        body_force = np.array([0, 0, -9.81])  # Gravity in z
        
        f = elem.body_load(body_force)
        
        # Check shape
        assert f.shape == (24,)
        
        # Check that only z-component has nonzero loads
        # (body load in z should give forces on z DOFs)
        assert np.any(f != 0), "body_load should produce nonzero forces for gravity"


class TestMITC4MembraneInterpolation:
    """Test membrane MITC interpolation in MITC4."""

    @pytest.fixture
    def material(self):
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def flat_element(self, material):
        """Flat rectangular element."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        return MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01)

    def test_tying_points_exist(self, flat_element):
        """Test that all tying points are defined."""
        elem = flat_element
        
        # Check tying points are defined
        assert hasattr(elem, '_tying_points_eps_xx')
        assert hasattr(elem, '_tying_points_eps_yy')
        assert hasattr(elem, '_tying_points_gamma_xy')
        
        # Check tying points have correct number
        assert len(elem._tying_points_eps_xx) == 4
        assert len(elem._tying_points_eps_yy) == 4
        assert len(elem._tying_points_gamma_xy) == 5

    def test_evaluate_B_m_at_point(self, flat_element):
        """Test evaluation of B_m at a single point."""
        elem = flat_element
        
        # Evaluate at center
        B_m = elem._evaluate_B_m_at_point(0.0, 0.0)
        
        # Check shape: 3x8 (3 membrane strains, 8 DOFs)
        assert B_m.shape == (3, 8)
        
        # Should have finite values
        assert np.all(np.isfinite(B_m))

    def test_get_membrane_strains_at_tying_points(self, flat_element):
        """Test evaluation of membrane strains at tying points."""
        elem = flat_element
        
        # Evaluate ε_xx at tying points
        eps_xx_list = elem._get_eps_xx_at_tying_points()
        assert len(eps_xx_list) == 4
        assert all(isinstance(e, np.ndarray) for e in eps_xx_list)
        assert all(e.shape == (8,) for e in eps_xx_list)
        
        # Evaluate ε_yy at tying points
        eps_yy_list = elem._get_eps_yy_at_tying_points()
        assert len(eps_yy_list) == 4
        assert all(e.shape == (8,) for e in eps_yy_list)
        
        # Evaluate γ_xy at tying points
        gamma_xy_list = elem._get_gamma_xy_at_tying_points()
        assert len(gamma_xy_list) == 5
        assert all(e.shape == (8,) for e in gamma_xy_list)

    def test_interpolation_functions(self, flat_element):
        """Test that interpolation functions return correct shapes."""
        elem = flat_element
        
        # Get tying point values
        eps_xx_tied = elem._get_eps_xx_at_tying_points()
        eps_yy_tied = elem._get_eps_yy_at_tying_points()
        gamma_xy_tied = elem._get_gamma_xy_at_tying_points()
        
        # Test at Gauss points
        r, s = 1 / np.sqrt(3), 1 / np.sqrt(3)
        
        eps_xx_interp = elem._interpolate_eps_xx(r, s, eps_xx_tied)
        assert eps_xx_interp.shape == (8,)
        assert np.all(np.isfinite(eps_xx_interp))
        
        eps_yy_interp = elem._interpolate_eps_yy(r, s, eps_yy_tied)
        assert eps_yy_interp.shape == (8,)
        assert np.all(np.isfinite(eps_yy_interp))
        
        gamma_xy_interp = elem._interpolate_gamma_xy(r, s, gamma_xy_tied)
        assert gamma_xy_interp.shape == (8,)
        assert np.all(np.isfinite(gamma_xy_interp))

    def test_B_m_mitc4plus(self, flat_element):
        """Test that B_m returns interpolated matrix."""
        elem = flat_element
        
        # Evaluate at Gauss point
        r, s = 1 / np.sqrt(3), 1 / np.sqrt(3)
        B_m = elem.B_m(r, s)
        
        # Check shape: 3x8
        assert B_m.shape == (3, 8)
        
        # Check finite values
        assert np.all(np.isfinite(B_m))


class TestMITC4EdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def material(self):
        return Material(E=210e9, nu=0.3, rho=7850)

    def test_very_thin_element(self, material):
        """Test with very thin element (thin shell limit)."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        # Very thin shell: h/L = 0.001
        elem = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.001)
        
        K = elem.K
        M = elem.M
        
        # Should still be positive semi-definite
        eigs_K = np.linalg.eigvalsh(K)
        eigs_M = np.linalg.eigvalsh(M)
        
        assert np.all(eigs_K >= -1e-5), "Very thin K not positive semi-definite"
        assert np.all(eigs_M >= -1e-5), "Very thin M not positive semi-definite"

    def test_thick_element(self, material):
        """Test with thick element (thick shell limit)."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        # Thick shell: h/L = 0.1
        elem = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.1)
        
        K = elem.K
        M = elem.M
        
        eigs_K = np.linalg.eigvalsh(K)
        eigs_M = np.linalg.eigvalsh(M)
        
        assert np.all(eigs_K >= -1e-5), "Thick K not positive semi-definite"
        assert np.all(eigs_M >= -1e-5), "Thick M not positive semi-definite"

    def test_stiffness_modifiers(self, material):
        """Test with stiffness modifiers."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        # Create with different stiffness modifiers
        elem1 = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01, kx_mod=1.0, ky_mod=1.0)
        elem2 = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01, kx_mod=0.8, ky_mod=1.2)
        
        K1 = elem1.K
        K2 = elem2.K
        
        # Should be different
        assert not np.allclose(K1, K2, atol=1e-10), "Different modifiers should produce different K"
        
        # Both should be positive semi-definite
        eigs_K1 = np.linalg.eigvalsh(K1)
        eigs_K2 = np.linalg.eigvalsh(K2)
        
        assert np.all(eigs_K1 >= -1e-5), "K1 not positive semi-definite"
        assert np.all(eigs_K2 >= -1e-5), "K2 not positive semi-definite"


class TestMITC4NonlinearAnalysis:
    """Test geometric nonlinear analysis capabilities of MITC4+.
    
    Based on Ko, Lee & Bathe (2017) 'The MITC4+ shell element in 
    geometric nonlinear analysis'.
    """

    @pytest.fixture
    def material(self):
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def flat_element_nonlinear(self, material):
        """Flat rectangular element with nonlinear=True."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        return MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01, nonlinear=True)

    def test_nonlinear_flag(self, material):
        """Test that nonlinear flag is set correctly."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        elem_lin = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01)
        elem_nl = MITC4(node_coords, (1, 2, 3, 4), material, thickness=0.01, nonlinear=True)
        
        assert elem_lin.nonlinear is False
        assert elem_nl.nonlinear is True

    def test_update_configuration(self, flat_element_nonlinear):
        """Test that configuration update works correctly."""
        elem = flat_element_nonlinear
        
        # Store initial coords
        initial = elem._initial_coords.copy()
        
        # Apply displacement
        u = np.zeros(24)
        u[2] = 0.01   # w at node 1
        u[8] = 0.02   # w at node 2
        u[14] = 0.03  # w at node 3
        u[20] = 0.01  # w at node 4
        
        elem.update_configuration(u)
        
        # Check that current coords updated
        assert np.allclose(elem._current_coords[0, 2], initial[0, 2] + 0.01)
        assert np.allclose(elem._current_coords[1, 2], initial[1, 2] + 0.02)
        assert np.allclose(elem._current_coords[2, 2], initial[2, 2] + 0.03)
        assert np.allclose(elem._current_coords[3, 2], initial[3, 2] + 0.01)
        
        # Initial should be unchanged
        assert np.allclose(elem._initial_coords, initial)

    def test_reset_configuration(self, flat_element_nonlinear):
        """Test that reset configuration restores initial state."""
        elem = flat_element_nonlinear
        initial = elem._initial_coords.copy()
        
        # Apply and then reset
        u = np.zeros(24)
        u[2] = 0.1
        elem.update_configuration(u)
        elem.reset_configuration()
        
        assert np.allclose(elem._current_coords, initial)
        assert np.allclose(elem._current_displacements, 0)

    def test_green_lagrange_strain_zero_displacement(self, flat_element_nonlinear):
        """Test that GL strain is zero for zero displacement."""
        elem = flat_element_nonlinear
        
        E = elem.compute_green_lagrange_strain(0, 0)
        
        assert np.allclose(E, 0, atol=1e-15)

    def test_green_lagrange_strain_small_displacement(self, flat_element_nonlinear):
        """Test GL strain for small displacements is consistent with kinematics."""
        elem = flat_element_nonlinear
        
        # Apply small out-of-plane displacement (bending)
        # This gives more predictable GL strain behavior
        u = np.zeros(24)
        eps = 1e-4  # small displacement
        u[2] = eps   # w at node 1
        u[8] = eps   # w at node 2
        u[14] = eps  # w at node 3
        u[20] = eps  # w at node 4
        
        elem.update_configuration(u)
        E = elem.compute_green_lagrange_strain(0, 0)
        
        # For uniform translation, GL strain should be very small
        # (only quadratic terms from w derivatives)
        assert np.all(np.abs(E) < 1e-4), f"GL strain should be small for uniform translation, got {E}"
        
        # Test that strain is symmetric
        assert np.allclose(E, E.T), "GL strain tensor should be symmetric"

    def test_tangent_stiffness_equals_K_for_zero_displacement(self, flat_element_nonlinear):
        """Test that K_T equals K_0 for zero displacement."""
        elem = flat_element_nonlinear
        
        K_0 = elem.K
        K_T = elem.compute_tangent_stiffness()
        
        # For zero displacement, K_T should equal K_0
        # (K_L and K_sigma should be zero or negligible)
        assert np.allclose(K_T, K_0, rtol=1e-6), "K_T should equal K_0 for zero displacement"

    def test_tangent_stiffness_symmetric(self, flat_element_nonlinear):
        """Test that tangent stiffness is symmetric."""
        elem = flat_element_nonlinear
        
        # Apply some displacement
        u = np.zeros(24)
        u[2] = 0.001
        u[8] = 0.002
        elem.update_configuration(u)
        
        K_T = elem.compute_tangent_stiffness()
        
        assert np.allclose(K_T, K_T.T, atol=1e-10), "K_T should be symmetric"

    def test_internal_forces_zero_for_zero_displacement(self, flat_element_nonlinear):
        """Test that internal forces are zero for zero displacement."""
        elem = flat_element_nonlinear
        
        f_int = elem.compute_internal_forces()
        
        # Internal forces should be zero for undeformed configuration
        assert np.allclose(f_int, 0, atol=1e-10), "f_int should be zero for zero displacement"

    def test_internal_forces_finite(self, flat_element_nonlinear):
        """Test that internal forces are finite for finite displacement."""
        elem = flat_element_nonlinear
        
        # Apply displacement
        u = np.zeros(24)
        u[2] = 0.001  # w at node 1
        elem.update_configuration(u)
        
        f_int = elem.compute_internal_forces()
        
        assert np.all(np.isfinite(f_int)), "f_int should be finite"
        assert np.linalg.norm(f_int) > 0, "f_int should be non-zero for deformed configuration"

    def test_strain_energy_non_negative(self, flat_element_nonlinear):
        """Test that strain energy is non-negative."""
        elem = flat_element_nonlinear
        
        # Zero displacement
        U_0 = elem.compute_strain_energy()
        assert U_0 >= 0, "Strain energy should be non-negative"
        assert U_0 < 1e-10, "Strain energy should be ~0 for zero displacement"
        
        # With displacement
        u = np.zeros(24)
        u[2] = 0.001
        elem.update_configuration(u)
        
        U = elem.compute_strain_energy()
        assert U > 0, "Strain energy should be positive for deformed configuration"

    def test_residual_computation(self, flat_element_nonlinear):
        """Test residual force computation."""
        elem = flat_element_nonlinear
        
        # Apply displacement
        u = np.zeros(24)
        u[2] = 0.001
        elem.update_configuration(u)
        
        f_ext = np.zeros(24)
        f_ext[2] = 1000  # External force in z at node 1
        
        R = elem.compute_residual(f_ext)
        
        assert R.shape == (24,)
        assert np.all(np.isfinite(R))
        
        # Residual should be f_ext - f_int
        f_int = elem.compute_internal_forces()
        assert np.allclose(R, f_ext - f_int)

    def test_displacement_gradient(self, flat_element_nonlinear):
        """Test displacement gradient computation."""
        elem = flat_element_nonlinear
        
        # Zero displacement -> zero gradient
        H = elem.get_displacement_gradient(0, 0)
        assert np.allclose(H, 0, atol=1e-15)
        
        # Apply uniform stretch
        u = np.zeros(24)
        u[6] = 0.01   # u at node 2
        u[12] = 0.01  # u at node 3
        elem.update_configuration(u)
        
        H = elem.get_displacement_gradient(0, 0)
        
        # Should have non-zero du/dx
        assert abs(H[0, 0]) > 0, "du/dx should be non-zero for x-stretch"

    def test_B_NL_matrix(self, flat_element_nonlinear):
        """Test nonlinear B matrix computation."""
        elem = flat_element_nonlinear
        
        # Zero displacement -> zero B_NL
        B_NL = elem._compute_B_NL(0, 0)
        assert np.allclose(B_NL, 0, atol=1e-15)
        
        # With displacement
        u = np.zeros(24)
        u[2] = 0.01
        elem.update_configuration(u)
        
        B_NL = elem._compute_B_NL(0, 0)
        assert B_NL.shape == (6, 24)
        assert np.all(np.isfinite(B_NL))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
