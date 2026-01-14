"""
Tests for MITC3 element implementation.

Validates that MITC3 (Triangular):
1. Initializes correctly with 3 nodes.
2. Computes Stiffness (K) and Mass (M) matrices of correct shape (18x18).
3. Passes Rigid Body Motion tests.
4. Passes Patch Test (constant strain/stress).
"""

import numpy as np
import pytest

from fem_shell.core.material import Material
from fem_shell.elements.MITC3 import MITC3


class TestMITC3:
    """Test MITC3 triangular shell element."""

    @pytest.fixture
    def material(self):
        """Create a test material."""
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def node_coords(self):
        """Create flat right-angled triangular element."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

    @pytest.fixture
    def node_ids(self):
        """Create node IDs."""
        return (1, 2, 3)

    def test_mitc3_initialization(self, node_coords, node_ids, material):
        """Test that MITC3 initializes correctly."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01)

        assert elem.element_type == "MITC3"
        assert elem.thickness == 0.01
        assert elem.dofs_count == 18
        assert elem.dofs_per_node == 6
        assert len(elem._tying_points_constant) == 3

    def test_stiffness_matrix_structure(self, node_coords, node_ids, material):
        """Test K matrix structure."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01)
        K = elem.K

        # Check shape (3 nodes * 6 DOFs = 18)
        assert K.shape == (18, 18)

        # Check symmetry
        assert np.allclose(K, K.T, atol=1e-10)

        # Check positive semi-definite (eigenvalues >= 0)
        # Note: 6 rigid body modes means 6 zero eigenvalues.
        eigs = np.linalg.eigvalsh(K)
        # We expect 6 small eigenvalues (rigid body modes) + some positive ones
        # Use a slightly negative epsilon for numerical noise
        assert np.all(eigs >= -1e-5), f"K has negative eigenvalues: {eigs[:5]}"

        # Count rigid body modes
        zero_eigs = np.sum(eigs < 1e-4)  # Should be around 6
        # Due to drilling rotation stabilization, standard MITC usually keeps 3+3=6 free modes.
        assert zero_eigs >= 3, (
            f"Expected at least 3 zero eigenvalues for membrane modes, got {zero_eigs}"
        )

    def test_mass_matrix_structure(self, node_coords, node_ids, material):
        """Test M matrix structure."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01)
        M = elem.M

        assert M.shape == (18, 18)
        assert np.allclose(M, M.T, atol=1e-10)
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs >= -1e-5), "Mass matrix not positive semi-definite"

    def test_rigid_body_translation(self, node_coords, node_ids, material):
        """Test that rigid body translations produce zero strain energy."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01)
        K = elem.K

        # Rigid body modes: [u, v, w, 0, 0, 0] constant
        # Mode 1: Translation X
        d_x = np.tile([1.0, 0, 0, 0, 0, 0], 3)
        f_x = K @ d_x
        assert np.linalg.norm(f_x) < 1e-5 * np.linalg.norm(K), "Rigid body translation X failed"

        # Mode 3: Translation Z
        d_z = np.tile([0, 0, 1.0, 0, 0, 0], 3)
        f_z = K @ d_z
        assert np.linalg.norm(f_z) < 1e-5 * np.linalg.norm(K), "Rigid body translation Z failed"

    def test_rigid_body_rotation(self, node_ids, material):
        """Test rigid body rotation about Z axis."""
        # Element in XY plane
        nodes = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        elem = MITC3(nodes, node_ids, material, thickness=0.01)
        K = elem.K

        # Rotation about Z-axis (theta_z = constant, u = -y*theta, v = x*theta)
        alpha = 0.01  # rotation angle rad
        d_rot = np.zeros(18)

        for i in range(3):
            x, y, z = nodes[i]
            # u = -alpha * y
            d_rot[6 * i] = -alpha * y
            # v = alpha * x
            d_rot[6 * i + 1] = alpha * x
            # w = 0
            # theta_z = alpha
            d_rot[6 * i + 5] = alpha

        f_rot = K @ d_rot

        # Verify forces are close to zero relative to stiffness scale
        stiffness = np.max(np.abs(K))
        load = np.max(np.abs(f_rot))
        # Relative tolerance
        assert load < 1e-4 * stiffness * alpha, f"Rigid body rotation Z failed. Load: {load}"

    def test_constant_strain_patch_test(self, material, node_ids):
        """
        Patch test for constant membrane strain.
        Specifies displacements corresponding to constant sx.
        """
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        elem = MITC3(nodes, node_ids, material, thickness=0.1)

        # Constant strain field: eps_xx = 0.001
        # u = 0.001 * x
        # v = 0
        eps0 = 0.001

        d = np.zeros(18)
        for i in range(3):
            d[6 * i] = eps0 * nodes[i, 0]  # u = eps * x

        # Compute internal forces / stresses
        # In linear element, stress should be exactly E/(1-nu^2) * eps0
        # Check computed stress at integration point?
        # MITC3.py doesn't expose stress method directly yet, but valid element should pass
        # K @ d = f_ext
        # For constant strain, interior forces cancel out?
        # Check Energy: U = 0.5 * d'Kd = Volume * 0.5 * sigma * eps

        U_fem = 0.5 * d @ elem.K @ d

        # Analytical Energy
        # Volume = Area * thickness = 0.5 * 0.1 = 0.05
        # Energy density = 0.5 * E_effective * eps^2
        # E_eff = E / (1 - nu^2) for plane stress
        E_eff = material.E / (1 - material.nu**2)
        U_analy = 0.05 * 0.5 * E_eff * eps0**2

        assert np.isclose(U_fem, U_analy, rtol=1e-4)


class TestMITC3NonlinearAnalysis:
    """Test MITC3 nonlinear analysis methods."""

    @pytest.fixture
    def material(self):
        """Create a test material."""
        return Material(E=210e9, nu=0.3, rho=7850)

    @pytest.fixture
    def node_coords(self):
        """Create flat triangular element."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

    @pytest.fixture
    def node_ids(self):
        """Create node IDs."""
        return (1, 2, 3)

    def test_nonlinear_flag(self, node_coords, node_ids, material):
        """Test that nonlinear flag is correctly set."""
        elem_linear = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=False)
        elem_nonlinear = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        assert elem_linear.nonlinear == False
        assert elem_nonlinear.nonlinear == True

    def test_update_configuration(self, node_coords, node_ids, material):
        """Test configuration update."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Apply displacement
        displacements = np.zeros(18)
        displacements[0] = 0.1  # Node 1, u
        displacements[6] = 0.1  # Node 2, u
        displacements[12] = 0.1  # Node 3, u

        elem.update_configuration(displacements)

        # Check current coordinates updated
        expected_coords = node_coords.copy()
        expected_coords[:, 0] += 0.1
        assert np.allclose(elem._current_coords, expected_coords)
        assert np.allclose(elem._current_displacements, displacements)

    def test_reset_configuration(self, node_coords, node_ids, material):
        """Test configuration reset."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Apply and then reset
        displacements = np.zeros(18)
        displacements[0] = 0.1
        elem.update_configuration(displacements)
        elem.reset_configuration()

        assert np.allclose(elem._current_coords, elem._initial_coords)
        assert np.allclose(elem._current_displacements, np.zeros(18))

    def test_green_lagrange_strain_zero_displacement(self, node_coords, node_ids, material):
        """Test Green-Lagrange strain is zero for zero displacement."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Zero displacement
        elem.update_configuration(np.zeros(18))
        E = elem.compute_green_lagrange_strain(1 / 3, 1 / 3)

        assert np.allclose(E, np.zeros((3, 3)), atol=1e-14)

    def test_green_lagrange_strain_small_displacement(self, node_coords, node_ids, material):
        """Test Green-Lagrange strain for small displacement approaches linear strain."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Small uniform extension
        eps0 = 1e-6
        displacements = np.zeros(18)
        for i in range(3):
            displacements[6 * i] = eps0 * node_coords[i, 0]

        elem.update_configuration(displacements)
        E = elem.compute_green_lagrange_strain(1 / 3, 1 / 3)

        # For small deformations, E_xx ≈ ε_xx
        assert np.isclose(E[0, 0], eps0, rtol=1e-4)

    def test_green_lagrange_strain_voigt(self, node_coords, node_ids, material):
        """Test Voigt notation of Green-Lagrange strain."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        eps0 = 1e-5
        displacements = np.zeros(18)
        for i in range(3):
            displacements[6 * i] = eps0 * node_coords[i, 0]

        elem.update_configuration(displacements)
        E_voigt = elem.compute_green_lagrange_strain_voigt(1 / 3, 1 / 3)

        assert E_voigt.shape == (6,)
        # E_xx component
        assert np.isclose(E_voigt[0], eps0, rtol=1e-3)

    def test_tangent_stiffness_equals_K_for_zero_displacement(
        self, node_coords, node_ids, material
    ):
        """Test tangent stiffness equals linear stiffness for zero displacement."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        elem.update_configuration(np.zeros(18))
        K_t = elem.compute_tangent_stiffness(transform_to_global=True)
        K = elem.K

        # For zero displacement, tangent ≈ linear
        # Note: K has drilling DOF stabilization, so compare only structural DOFs
        # or use relative tolerance accounting for drilling stiffness
        structural_dofs = [i for i in range(18) if i % 6 != 5]  # Exclude θz DOFs
        K_t_struct = K_t[np.ix_(structural_dofs, structural_dofs)]
        K_struct = K[np.ix_(structural_dofs, structural_dofs)]

        assert np.allclose(K_t_struct, K_struct, rtol=1e-6)

    def test_tangent_stiffness_symmetric(self, node_coords, node_ids, material):
        """Test tangent stiffness is symmetric."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Apply some displacement
        displacements = np.zeros(18)
        displacements[2] = 0.01  # w1
        displacements[8] = 0.005  # w2
        elem.update_configuration(displacements)

        K_t = elem.compute_tangent_stiffness(transform_to_global=True)

        assert np.allclose(K_t, K_t.T, atol=1e-10)

    def test_internal_forces_zero_for_zero_displacement(self, node_coords, node_ids, material):
        """Test internal forces are zero for zero displacement."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        elem.update_configuration(np.zeros(18))
        f_int = elem.compute_internal_forces(transform_to_global=True)

        assert np.allclose(f_int, np.zeros(18), atol=1e-14)

    def test_internal_forces_finite(self, node_coords, node_ids, material):
        """Test internal forces are finite for finite displacement."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        displacements = np.zeros(18)
        displacements[2] = 0.01
        elem.update_configuration(displacements)

        f_int = elem.compute_internal_forces(transform_to_global=True)

        assert np.all(np.isfinite(f_int))
        assert np.linalg.norm(f_int) > 0

    def test_strain_energy_non_negative(self, node_coords, node_ids, material):
        """Test strain energy is non-negative."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Zero displacement
        elem.update_configuration(np.zeros(18))
        U_zero = elem.compute_strain_energy()
        assert U_zero >= 0
        assert np.isclose(U_zero, 0.0, atol=1e-14)

        # Non-zero displacement
        displacements = np.zeros(18)
        displacements[0] = 0.001
        elem.update_configuration(displacements)
        U_nonzero = elem.compute_strain_energy()
        assert U_nonzero > 0

    def test_residual_computation(self, node_coords, node_ids, material):
        """Test residual computation."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # For zero external force and zero displacement, residual = 0
        f_ext = np.zeros(18)
        elem.update_configuration(np.zeros(18))

        residual = elem.compute_residual(f_ext, transform_to_global=True)
        assert np.allclose(residual, np.zeros(18), atol=1e-14)

    def test_displacement_gradient(self, node_coords, node_ids, material):
        """Test displacement gradient computation."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Uniform extension in x
        eps = 0.01
        displacements = np.zeros(18)
        for i in range(3):
            displacements[6 * i] = eps * node_coords[i, 0]

        elem.update_configuration(displacements)
        H = elem.get_displacement_gradient(1 / 3, 1 / 3)

        # ∂u/∂x should be eps
        assert np.isclose(H[0, 0], eps, rtol=1e-10)

    def test_B_NL_matrix(self, node_coords, node_ids, material):
        """Test nonlinear B matrix."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Zero displacement => B_NL = 0
        elem.update_configuration(np.zeros(18))
        B_NL = elem._compute_B_NL(1 / 3, 1 / 3)
        assert np.allclose(B_NL, np.zeros((6, 18)), atol=1e-14)

        # Non-zero displacement => B_NL != 0
        displacements = np.zeros(18)
        displacements[0] = 0.01
        elem.update_configuration(displacements)
        B_NL = elem._compute_B_NL(1 / 3, 1 / 3)
        assert np.linalg.norm(B_NL) > 0

    def test_geometric_stiffness_matrix(self, node_coords, node_ids, material):
        """Test geometric stiffness matrix properties."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Uniform tension
        sigma = np.array([1e6, 1e6, 0.0])  # [σxx, σyy, σxy]

        K_g = elem.compute_geometric_stiffness(sigma, transform_to_global=True)

        # Should be 18x18
        assert K_g.shape == (18, 18)

        # Should be symmetric
        assert np.allclose(K_g, K_g.T, atol=1e-10)

        # For uniform tension, should be positive semi-definite
        eigs = np.linalg.eigvalsh(K_g)
        # Allow small numerical noise
        assert np.all(eigs >= -1e-5 * abs(eigs).max())

    def test_membrane_stress_from_displacement(self, node_coords, node_ids, material):
        """Test membrane stress computation from displacement."""
        elem = MITC3(node_coords, node_ids, material, thickness=0.01, nonlinear=True)

        # Uniform extension in x
        eps = 0.001
        u_local = np.zeros(18)
        for i in range(3):
            u_local[6 * i] = eps * node_coords[i, 0]

        sigma = elem.compute_membrane_stress_from_displacement(u_local)

        # Expected stress: σxx = E/(1-ν²) * ε
        E_eff = material.E / (1 - material.nu**2)
        sigma_xx_expected = E_eff * eps

        assert np.isclose(sigma[0], sigma_xx_expected, rtol=1e-6)

    def test_centrifugal_prestress(self, node_coords, node_ids, material):
        """Test centrifugal prestress computation."""
        # Move element away from rotation axis
        shifted_coords = node_coords.copy()
        shifted_coords[:, 0] += 10.0  # 10m from axis

        elem = MITC3(shifted_coords, (1, 2, 3), material, thickness=0.01, nonlinear=True)

        omega = 10.0  # rad/s
        axis = np.array([0, 0, 1])  # Z-axis rotation
        center = np.array([0, 0, 0])

        sigma_cf = elem.compute_centrifugal_prestress(omega, axis, center)

        # Should be finite and positive (tension)
        assert np.all(np.isfinite(sigma_cf))
        # For radial direction along X, σxx should be dominant
        assert sigma_cf[0] > 0  # Tension in x direction
