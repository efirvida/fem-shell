"""
Tests for MITC4Plus element implementation.

Validates that MITC4Plus:
1. Shares the same API as MITC4 (K, M, body_load)
2. Implements membrane MITC interpolation correctly
3. Produces different results than MITC4 (due to membrane interpolation)
4. Maintains numerical stability and positive semi-definiteness
"""

import numpy as np
import pytest

from fem_shell.core.material import Material
from fem_shell.elements.MITC4 import MITC4, MITC4Plus


class TestMITC4PlusAPI:
    """Test that MITC4Plus shares the same API as MITC4."""

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
        """Test that MITC4Plus initializes correctly."""
        elem = MITC4Plus(node_coords, node_ids, material, thickness=0.01)
        
        assert elem.element_type == "MITC4Plus"
        assert elem.thickness == 0.01
        assert elem.dofs_count == 24
        assert elem.dofs_per_node == 6
        assert len(elem._tying_points_eps_xx) == 4
        assert len(elem._tying_points_eps_yy) == 4
        assert len(elem._tying_points_gamma_xy) == 5

    def test_shared_api_K(self, node_coords, node_ids, material):
        """Test that K matrix is computed with same API."""
        elem = MITC4Plus(node_coords, node_ids, material, thickness=0.01)
        K = elem.K
        
        # Check shape
        assert K.shape == (24, 24)
        
        # Check symmetry
        assert np.allclose(K, K.T, atol=1e-10)
        
        # Check positive semi-definite (all eigenvalues >= -1e-10)
        eigs = np.linalg.eigvalsh(K)
        assert np.all(eigs >= -1e-10), f"K has negative eigenvalues: {eigs[:5]}"

    def test_shared_api_M(self, node_coords, node_ids, material):
        """Test that M matrix is computed with same API."""
        elem = MITC4Plus(node_coords, node_ids, material, thickness=0.01)
        M = elem.M
        
        # Check shape
        assert M.shape == (24, 24)
        
        # Check symmetry
        assert np.allclose(M, M.T, atol=1e-10)
        
        # Check positive semi-definite
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs >= -1e-10), f"M has negative eigenvalues: {eigs[:5]}"

    def test_shared_api_body_load(self, node_coords, node_ids, material):
        """Test that body_load method works with same API."""
        elem = MITC4Plus(node_coords, node_ids, material, thickness=0.01)
        body_force = np.array([0, 0, -9.81])  # Gravity in z
        
        f = elem.body_load(body_force)
        
        # Check shape
        assert f.shape == (24,)
        
        # Check that only z-component has nonzero loads
        # (body load in z should give forces on z DOFs)
        assert np.any(f != 0), "body_load should produce nonzero forces for gravity"


class TestMITC4PlusMembraneInterpolation:
    """Test membrane MITC interpolation in MITC4Plus."""

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
        return MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.01)

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


class TestMITC4PlusVsMITC4:
    """Compare MITC4Plus vs MITC4 to ensure differences."""

    @pytest.fixture
    def material(self):
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def curved_element_coords(self):
        """Create a slightly curved element (curved shell test case)."""
        # Cylindrical shell segment
        R = 1.0  # Radius
        theta = 0.1  # Small angular segment
        L = 1.0  # Length
        
        return np.array([
            [R * np.cos(-theta/2), -L/2, R * np.sin(-theta/2)],
            [R * np.cos(+theta/2), -L/2, R * np.sin(+theta/2)],
            [R * np.cos(+theta/2), +L/2, R * np.sin(+theta/2)],
            [R * np.cos(-theta/2), +L/2, R * np.sin(-theta/2)],
        ], dtype=float)

    def test_mitc4plus_different_from_mitc4_curved(self, curved_element_coords, material):
        """Test that MITC4Plus gives different results than MITC4 for curved elements."""
        node_ids = (1, 2, 3, 4)
        thickness = 0.05
        
        # Create both elements
        mitc4 = MITC4(curved_element_coords, node_ids, material, thickness)
        mitc4plus = MITC4Plus(curved_element_coords, node_ids, material, thickness)
        
        # Compare K matrices
        K_mitc4 = mitc4.K
        K_mitc4plus = mitc4plus.K
        
        # They should be different (membrane interpolation makes a difference)
        diff = np.linalg.norm(K_mitc4 - K_mitc4plus)
        assert diff > 1e-6, f"K matrices should differ, difference norm: {diff}"
        
        # Compare M matrices
        M_mitc4 = mitc4.M
        M_mitc4plus = mitc4plus.M
        
        # M should be the same (both use same mass formulation)
        assert np.allclose(M_mitc4, M_mitc4plus, atol=1e-10), \
            "M matrices should be identical (both inherit same formulation)"

    def test_both_matrices_positive_semidefinite(self, curved_element_coords, material):
        """Test that both elements produce positive semi-definite matrices."""
        node_ids = (1, 2, 3, 4)
        thickness = 0.05
        
        mitc4 = MITC4(curved_element_coords, node_ids, material, thickness)
        mitc4plus = MITC4Plus(curved_element_coords, node_ids, material, thickness)
        
        # MITC4
        eigs_K_mitc4 = np.linalg.eigvalsh(mitc4.K)
        assert np.all(eigs_K_mitc4 >= -1e-10), f"MITC4 K not positive semi-definite: {eigs_K_mitc4[:5]}"
        
        eigs_M_mitc4 = np.linalg.eigvalsh(mitc4.M)
        assert np.all(eigs_M_mitc4 >= -1e-10), f"MITC4 M not positive semi-definite: {eigs_M_mitc4[:5]}"
        
        # MITC4Plus
        eigs_K_mitc4plus = np.linalg.eigvalsh(mitc4plus.K)
        assert np.all(eigs_K_mitc4plus >= -1e-10), f"MITC4Plus K not positive semi-definite: {eigs_K_mitc4plus[:5]}"
        
        eigs_M_mitc4plus = np.linalg.eigvalsh(mitc4plus.M)
        assert np.all(eigs_M_mitc4plus >= -1e-10), f"MITC4Plus M not positive semi-definite: {eigs_M_mitc4plus[:5]}"


class TestMITC4PlusEdgeCases:
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
        elem = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.001)
        
        K = elem.K
        M = elem.M
        
        # Should still be positive semi-definite
        eigs_K = np.linalg.eigvalsh(K)
        eigs_M = np.linalg.eigvalsh(M)
        
        assert np.all(eigs_K >= -1e-10), "Very thin K not positive semi-definite"
        assert np.all(eigs_M >= -1e-10), "Very thin M not positive semi-definite"

    def test_thick_element(self, material):
        """Test with thick element (thick shell limit)."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        # Thick shell: h/L = 0.1
        elem = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.1)
        
        K = elem.K
        M = elem.M
        
        eigs_K = np.linalg.eigvalsh(K)
        eigs_M = np.linalg.eigvalsh(M)
        
        assert np.all(eigs_K >= -1e-10), "Thick K not positive semi-definite"
        assert np.all(eigs_M >= -1e-10), "Thick M not positive semi-definite"

    def test_stiffness_modifiers(self, material):
        """Test with stiffness modifiers."""
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        
        # Create with different stiffness modifiers
        elem1 = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.01, kx_mod=1.0, ky_mod=1.0)
        elem2 = MITC4Plus(node_coords, (1, 2, 3, 4), material, thickness=0.01, kx_mod=0.8, ky_mod=1.2)
        
        K1 = elem1.K
        K2 = elem2.K
        
        # Should be different
        assert not np.allclose(K1, K2, atol=1e-10), "Different modifiers should produce different K"
        
        # Both should be positive semi-definite
        eigs_K1 = np.linalg.eigvalsh(K1)
        eigs_K2 = np.linalg.eigvalsh(K2)
        
        assert np.all(eigs_K1 >= -1e-10), "K1 not positive semi-definite"
        assert np.all(eigs_K2 >= -1e-10), "K2 not positive semi-definite"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
