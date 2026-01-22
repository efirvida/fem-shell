"""
Unit tests for co-rotational rotor FSI utilities.

Tests coordinate transformations, inertial force calculations,
and the OmegaProvider classes without requiring a full FSI setup.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

# Import the module directly to avoid triggering __init__.py which has PETSc deps
_module_path = (
    Path(__file__).parent.parent / "src" / "fem_shell" / "solvers" / "fsi_rotor_corotational.py"
)
_spec = importlib.util.spec_from_file_location("fsi_rotor_corotational", _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["fsi_rotor_corotational"] = _module
_spec.loader.exec_module(_module)

ConstantOmega = _module.ConstantOmega
CoordinateTransforms = _module.CoordinateTransforms
FunctionOmega = _module.FunctionOmega
InertialForcesCalculator = _module.InertialForcesCalculator
OmegaProvider = _module.OmegaProvider
TableOmega = _module.TableOmega


class TestCoordinateTransforms:
    """Tests for the CoordinateTransforms class."""

    @pytest.fixture
    def z_axis_transform(self):
        """Transform for rotation about Z-axis."""
        return CoordinateTransforms(rotation_axis=[0, 0, 1], rotation_center=[0, 0, 0])

    @pytest.fixture
    def y_axis_transform(self):
        """Transform for rotation about Y-axis."""
        return CoordinateTransforms(rotation_axis=[0, 1, 0], rotation_center=[0, 0, 0])

    @pytest.fixture
    def arbitrary_axis_transform(self):
        """Transform for rotation about arbitrary axis."""
        return CoordinateTransforms(
            rotation_axis=[1, 1, 1],
            rotation_center=[1, 0, 0],  # Will be normalized
        )

    def test_rotation_matrix_identity_at_zero(self, z_axis_transform):
        """Rotation matrix should be identity at theta=0."""
        R = z_axis_transform.rotation_matrix(0.0)
        assert_array_almost_equal(R, np.eye(3), decimal=12)

    def test_rotation_matrix_orthogonality(self, z_axis_transform):
        """Rotation matrix should be orthogonal: R @ R.T = I."""
        for theta in [0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
            R = z_axis_transform.rotation_matrix(theta)
            assert_array_almost_equal(R @ R.T, np.eye(3), decimal=12)
            assert_array_almost_equal(R.T @ R, np.eye(3), decimal=12)

    def test_rotation_matrix_determinant(self, z_axis_transform):
        """Rotation matrix should have determinant = 1."""
        for theta in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            R = z_axis_transform.rotation_matrix(theta)
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_z_axis_90_degree_rotation(self, z_axis_transform):
        """90° rotation about Z should swap X and Y."""
        R = z_axis_transform.rotation_matrix(np.pi / 2)

        # X-axis -> Y-axis
        v_x = np.array([1, 0, 0])
        v_rotated = R @ v_x
        assert_array_almost_equal(v_rotated, [0, 1, 0], decimal=10)

        # Y-axis -> -X-axis
        v_y = np.array([0, 1, 0])
        v_rotated = R @ v_y
        assert_array_almost_equal(v_rotated, [-1, 0, 0], decimal=10)

        # Z-axis -> Z-axis (unchanged)
        v_z = np.array([0, 0, 1])
        v_rotated = R @ v_z
        assert_array_almost_equal(v_rotated, [0, 0, 1], decimal=10)

    def test_round_trip_transformation(self, z_axis_transform):
        """Transform to rotating and back should give original vector."""
        v_original = np.array([3.5, -2.1, 7.8])

        for theta in [0, np.pi / 6, np.pi / 3, np.pi, 5 * np.pi / 3]:
            v_rotating = z_axis_transform.to_rotating(v_original, theta)
            v_back = z_axis_transform.to_inertial(v_rotating, theta)
            assert_array_almost_equal(v_back, v_original, decimal=10)

    def test_round_trip_batch_transformation(self, z_axis_transform):
        """Batch transform to rotating and back should give original vectors."""
        v_original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 2.5]])

        for theta in [np.pi / 4, np.pi / 2, np.pi]:
            v_rotating = z_axis_transform.to_rotating(v_original, theta)
            v_back = z_axis_transform.to_inertial(v_rotating, theta)
            assert_array_almost_equal(v_back, v_original, decimal=10)

    def test_force_transformation(self, z_axis_transform):
        """Force transformation should preserve magnitude."""
        force_global = np.array([[100, 0, 0], [0, 50, 0], [0, 0, 25]])

        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            force_local = z_axis_transform.transform_force_to_rotating(
                force_global.flatten(), theta
            )
            force_local_2d = force_local.reshape(-1, 3)

            # Magnitudes should be preserved
            for i in range(3):
                mag_original = np.linalg.norm(force_global[i])
                mag_transformed = np.linalg.norm(force_local_2d[i])
                assert_allclose(mag_transformed, mag_original, atol=1e-10)

    def test_displacement_transformation(self, z_axis_transform):
        """Displacement transformation should preserve magnitude."""
        disp_local = np.array([[0.01, 0.02, 0.005], [0.0, 0.015, 0.0]])

        for theta in [0, np.pi / 3, 2 * np.pi / 3]:
            disp_global = z_axis_transform.transform_displacement_to_inertial(
                disp_local.flatten(), theta
            )
            disp_global_2d = disp_global.reshape(-1, 3)

            for i in range(2):
                mag_original = np.linalg.norm(disp_local[i])
                mag_transformed = np.linalg.norm(disp_global_2d[i])
                assert_allclose(mag_transformed, mag_original, atol=1e-10)

    def test_axis_normalization(self):
        """Non-unit rotation axis should be normalized."""
        transform = CoordinateTransforms(rotation_axis=[0, 0, 10])
        assert_allclose(np.linalg.norm(transform.axis), 1.0)
        assert_array_almost_equal(transform.axis, [0, 0, 1])

    def test_zero_axis_raises_error(self):
        """Zero rotation axis should raise ValueError."""
        with pytest.raises(ValueError, match="zero vector"):
            CoordinateTransforms(rotation_axis=[0, 0, 0])

    def test_arbitrary_axis_orthogonality(self, arbitrary_axis_transform):
        """Rotation about arbitrary axis should still be orthogonal."""
        for theta in [np.pi / 4, np.pi / 2, np.pi]:
            R = arbitrary_axis_transform.rotation_matrix(theta)
            assert_array_almost_equal(R @ R.T, np.eye(3), decimal=12)


class TestInertialForcesCalculator:
    """Tests for the InertialForcesCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Default calculator for Z-axis rotation."""
        return InertialForcesCalculator(rotation_axis=[0, 0, 1], rotation_center=[0, 0, 0])

    @pytest.fixture
    def simple_rotor_setup(self):
        """Simple rotor with nodes along X-axis."""
        # Nodes at r = 1, 2, 3 m along X-axis
        nodal_coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        nodal_masses = np.array([10.0, 10.0, 10.0])  # 10 kg each
        return nodal_coords, nodal_masses

    def test_centrifugal_force_direction(self, calculator, simple_rotor_setup):
        """Centrifugal force should point radially outward."""
        coords, masses = simple_rotor_setup
        omega = 10.0  # rad/s

        F_cf = calculator.compute_centrifugal_force(coords, masses, omega)

        # For nodes on X-axis, centrifugal should point in +X direction
        for i in range(3):
            # Force should be in X direction (radially outward)
            assert F_cf[i, 0] > 0, "Centrifugal should be positive X"
            assert_allclose(F_cf[i, 1], 0, atol=1e-10)
            assert_allclose(F_cf[i, 2], 0, atol=1e-10)

    def test_centrifugal_force_magnitude(self, calculator, simple_rotor_setup):
        """Centrifugal force should equal m * omega^2 * r."""
        coords, masses = simple_rotor_setup
        omega = 10.0

        F_cf = calculator.compute_centrifugal_force(coords, masses, omega)

        for i, (r, m) in enumerate(zip([1.0, 2.0, 3.0], masses)):
            expected_magnitude = m * omega**2 * r
            actual_magnitude = np.linalg.norm(F_cf[i])
            assert_allclose(actual_magnitude, expected_magnitude, rtol=1e-10)

    def test_centrifugal_force_zero_omega(self, calculator, simple_rotor_setup):
        """Centrifugal force should be zero when omega=0."""
        coords, masses = simple_rotor_setup

        F_cf = calculator.compute_centrifugal_force(coords, masses, omega=0.0)
        assert_array_almost_equal(F_cf, np.zeros_like(coords))

    def test_coriolis_force_direction(self, calculator):
        """Coriolis force should be perpendicular to both omega and velocity."""
        # Node moving in +X direction
        velocities = np.array([[10.0, 0.0, 0.0]])  # 10 m/s in X
        masses = np.array([1.0])
        omega = 5.0  # rad/s about Z

        F_cor = calculator.compute_coriolis_force(velocities, masses, omega)

        # omega × v = [0,0,5] × [10,0,0] = [0, 50, 0]
        # F_cor = -2 * m * (omega × v) = -2 * 1 * [0, 50, 0] = [0, -100, 0]
        expected = np.array([[0, -100, 0]])
        assert_array_almost_equal(F_cor, expected, decimal=10)

    def test_coriolis_force_perpendicular(self, calculator):
        """Coriolis force should be perpendicular to velocity."""
        velocities = np.array([[5.0, 3.0, 0.0], [0.0, 7.0, 2.0]])
        masses = np.array([1.0, 1.0])
        omega = 2.0

        F_cor = calculator.compute_coriolis_force(velocities, masses, omega)

        # Check perpendicularity: F · v = 0
        for i in range(2):
            dot_product = np.dot(F_cor[i], velocities[i])
            assert_allclose(dot_product, 0, atol=1e-10)

    def test_coriolis_force_zero_omega(self, calculator):
        """Coriolis force should be zero when omega=0."""
        velocities = np.array([[10.0, 5.0, 2.0]])
        masses = np.array([5.0])

        F_cor = calculator.compute_coriolis_force(velocities, masses, omega=0.0)
        assert_array_almost_equal(F_cor, np.zeros_like(velocities))

    def test_euler_force_direction(self, calculator, simple_rotor_setup):
        """Euler force should be tangential (perpendicular to radius)."""
        coords, masses = simple_rotor_setup
        alpha = 2.0  # rad/s^2

        F_euler = calculator.compute_euler_force(coords, masses, alpha)

        # For nodes on X-axis with Z rotation, Euler force should be in -Y
        # alpha × r = [0,0,2] × [r,0,0] = [0, 2r, 0]
        # F_euler = -m * (alpha × r) = -m * [0, 2r, 0] = [0, -2mr, 0]
        for i, (r, m) in enumerate(zip([1.0, 2.0, 3.0], masses)):
            expected_y = -m * alpha * r
            assert_allclose(F_euler[i, 0], 0, atol=1e-10)
            assert_allclose(F_euler[i, 1], expected_y, rtol=1e-10)
            assert_allclose(F_euler[i, 2], 0, atol=1e-10)

    def test_euler_force_zero_alpha(self, calculator, simple_rotor_setup):
        """Euler force should be zero when alpha=0."""
        coords, masses = simple_rotor_setup

        F_euler = calculator.compute_euler_force(coords, masses, alpha=0.0)
        assert_array_almost_equal(F_euler, np.zeros_like(coords))

    def test_compute_all_inertial_forces(self, calculator, simple_rotor_setup):
        """Combined inertial forces should equal sum of components."""
        coords, masses = simple_rotor_setup
        velocities = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
        omega = 5.0
        alpha = 1.0

        F_total, diagnostics = calculator.compute_all_inertial_forces(
            nodal_coords=coords,
            nodal_velocities=velocities,
            nodal_masses=masses,
            omega=omega,
            alpha=alpha,
        )

        # Compute components separately
        F_cf = calculator.compute_centrifugal_force(coords, masses, omega)
        F_cor = calculator.compute_coriolis_force(velocities, masses, omega)
        F_euler = calculator.compute_euler_force(coords, masses, alpha)

        expected_total = F_cf + F_cor + F_euler
        assert_array_almost_equal(F_total, expected_total, decimal=10)

        # Check diagnostics
        assert "centrifugal" in diagnostics
        assert "coriolis" in diagnostics
        assert "euler" in diagnostics
        assert "total_inertial" in diagnostics

    def test_selective_inertial_forces(self, calculator, simple_rotor_setup):
        """Should be able to disable individual force components."""
        coords, masses = simple_rotor_setup
        velocities = np.zeros_like(coords)
        omega = 5.0

        # Only centrifugal
        F_cf_only, _ = calculator.compute_all_inertial_forces(
            nodal_coords=coords,
            nodal_velocities=velocities,
            nodal_masses=masses,
            omega=omega,
            include_centrifugal=True,
            include_coriolis=False,
            include_euler=False,
        )

        F_cf = calculator.compute_centrifugal_force(coords, masses, omega)
        assert_array_almost_equal(F_cf_only, F_cf)

    def test_off_axis_node(self, calculator):
        """Test centrifugal for node not on a principal axis."""
        # Node at (1, 1, 0) - distance from Z-axis is sqrt(2)
        coords = np.array([[1.0, 1.0, 0.0]])
        masses = np.array([10.0])
        omega = 10.0

        F_cf = calculator.compute_centrifugal_force(coords, masses, omega)

        # Centrifugal magnitude = m * omega^2 * r_perp
        r_perp = np.sqrt(2)
        expected_mag = masses[0] * omega**2 * r_perp
        actual_mag = np.linalg.norm(F_cf[0])
        assert_allclose(actual_mag, expected_mag, rtol=1e-10)

        # Direction should be radially outward in XY plane
        expected_direction = np.array([1, 1, 0]) / np.sqrt(2)
        actual_direction = F_cf[0] / actual_mag
        assert_array_almost_equal(actual_direction, expected_direction, decimal=10)


class TestOmegaProviders:
    """Tests for OmegaProvider implementations."""

    def test_constant_omega(self):
        """ConstantOmega should return constant values."""
        provider = ConstantOmega(omega=15.0)

        for t in [0.0, 0.5, 1.0, 10.0, 100.0]:
            omega, alpha = provider.get_omega(t)
            assert omega == 15.0
            assert alpha == 0.0

        assert provider.initial_omega == 15.0

    def test_constant_omega_repr(self):
        """ConstantOmega should have readable repr."""
        provider = ConstantOmega(omega=12.5)
        assert "12.5" in repr(provider)

    def test_table_omega_interpolation(self):
        """TableOmega should interpolate between data points."""
        times = [0.0, 1.0, 2.0, 3.0]
        omegas = [0.0, 10.0, 10.0, 5.0]

        provider = TableOmega(times, omegas)

        # At data points
        omega, _ = provider.get_omega(0.0)
        assert omega == 0.0

        omega, _ = provider.get_omega(1.0)
        assert omega == 10.0

        # Interpolated
        omega, _ = provider.get_omega(0.5)
        assert_allclose(omega, 5.0)

        omega, _ = provider.get_omega(2.5)
        assert_allclose(omega, 7.5)

    def test_table_omega_extrapolation(self):
        """TableOmega should extrapolate beyond data range."""
        times = [1.0, 2.0]
        omegas = [10.0, 20.0]

        provider = TableOmega(times, omegas)

        # Before first point - uses first value
        omega, _ = provider.get_omega(0.0)
        assert omega == 10.0

        # After last point - uses last value
        omega, _ = provider.get_omega(5.0)
        assert omega == 20.0

    def test_table_omega_alpha_computation(self):
        """TableOmega should compute alpha from gradient."""
        # Linear ramp: omega = 10*t, so alpha = 10
        times = [0.0, 1.0, 2.0]
        omegas = [0.0, 10.0, 20.0]

        provider = TableOmega(times, omegas)

        omega, alpha = provider.get_omega(0.5)
        assert_allclose(alpha, 10.0, rtol=0.01)

    def test_table_omega_validation(self):
        """TableOmega should validate inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            TableOmega([0, 1, 2], [0, 10])

        # Too few points
        with pytest.raises(ValueError, match="at least 2"):
            TableOmega([0], [0])

        # Non-monotonic times
        with pytest.raises(ValueError, match="monotonically increasing"):
            TableOmega([0, 2, 1], [0, 10, 20])

    def test_function_omega(self):
        """FunctionOmega should evaluate provided functions."""
        # omega = 5*t, alpha = 5
        provider = FunctionOmega(omega_func=lambda t: 5.0 * t, alpha_func=lambda t: 5.0)

        omega, alpha = provider.get_omega(2.0)
        assert omega == 10.0
        assert alpha == 5.0

    def test_function_omega_numerical_alpha(self):
        """FunctionOmega should compute alpha numerically if not provided."""
        # omega = t^2, alpha = 2*t
        provider = FunctionOmega(omega_func=lambda t: t**2)

        omega, alpha = provider.get_omega(3.0)
        assert_allclose(omega, 9.0)
        # Numerical derivative should be close to 2*3 = 6
        assert_allclose(alpha, 6.0, rtol=0.01)

    def test_function_omega_initial_omega(self):
        """FunctionOmega should correctly report initial omega."""
        # With explicit initial_omega
        provider1 = FunctionOmega(omega_func=lambda t: t + 5, initial_omega=5.0)
        assert provider1.initial_omega == 5.0

        # Computed from function
        provider2 = FunctionOmega(omega_func=lambda t: t + 5)
        assert provider2.initial_omega == 5.0


class TestIntegration:
    """Integration tests with simulated forces (no preCICE needed)."""

    def test_force_transform_and_inertial_combination(self):
        """Test complete force processing pipeline."""
        # Setup
        transforms = CoordinateTransforms([0, 0, 1], [0, 0, 0])
        inertial_calc = InertialForcesCalculator([0, 0, 1], [0, 0, 0])

        # Simulate CFD forces in global frame (thrust in X direction)
        F_global = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])

        # Rotor at 45 degrees
        theta = np.pi / 4
        omega = 10.0

        # Nodes on rotor blade
        coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        masses = np.array([5.0, 5.0, 5.0])
        velocities = np.zeros((3, 3))

        # Transform forces to rotating frame
        F_local = transforms.transform_force_to_rotating(F_global.flatten(), theta)
        F_local = F_local.reshape(-1, 3)

        # Compute inertial forces
        F_inertial, _ = inertial_calc.compute_all_inertial_forces(coords, velocities, masses, omega)

        # Combine
        F_total = F_local + F_inertial

        # Verify total force has both components
        assert F_total.shape == (3, 3)

        # Magnitudes should be reasonable
        total_mag = np.linalg.norm(np.sum(F_total, axis=0))
        assert total_mag > 0

    def test_theta_accumulation_simulation(self):
        """Simulate theta accumulation over time steps."""
        provider = ConstantOmega(omega=10.0)  # 10 rad/s
        dt = 0.001  # 1 ms time step

        theta = 0.0
        n_steps = 1000  # 1 second of simulation

        for i in range(n_steps):
            t = i * dt
            omega, _ = provider.get_omega(t)
            theta += omega * dt

        # After 1 second at 10 rad/s, should be at 10 radians
        assert_allclose(theta, 10.0, rtol=0.001)

    def test_displacement_consistency_over_rotation(self):
        """
        Test that displacement magnitude is preserved through
        the full rotation cycle.
        """
        transforms = CoordinateTransforms([0, 0, 1], [0, 0, 0])

        # Displacement in local frame
        u_local = np.array([0.01, 0.005, 0.0])

        # Check at various rotation angles
        for theta in np.linspace(0, 2 * np.pi, 37):  # Every 10 degrees
            u_global = transforms.to_inertial(u_local, theta)
            u_back = transforms.to_rotating(u_global, theta)

            # Magnitude preserved
            assert_allclose(np.linalg.norm(u_global), np.linalg.norm(u_local), atol=1e-12)

            # Round-trip is exact
            assert_array_almost_equal(u_back, u_local, decimal=12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
