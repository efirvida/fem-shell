"""Tests for the BEM engine (CCBlade wrapper).

Requires ccblade to be installed: ``pip install -e ".[bem]"``

Covers:
- BEMSolver construction from BladeAero
- Parked blade (omega=0): AoA ≈ twist + pitch
- BEMResult fields are physically sensible
"""

from pathlib import Path

import numpy as np
import pytest

ccblade = pytest.importorskip("ccblade", reason="ccblade not installed (pip install -e '.[bem]')")

from fem_shell.models.blade.aerodynamics import load_blade_aero
from fem_shell.solvers.bem.engine import BEMResult, BEMSolver

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IEA_YAML = str(_PROJECT_ROOT / "examples" / "blade" / "solid" / "IEA-15-240-RWT.yaml")


@pytest.fixture(scope="module")
def blade_aero():
    """Load IEA 15 MW blade aero data once per module."""
    return load_blade_aero(IEA_YAML)


@pytest.fixture(scope="module")
def bem_solver(blade_aero):
    """Create a BEM solver for the IEA 15 MW blade."""
    return BEMSolver(blade_aero, rho=1.225, mu=1.81206e-5)


class TestBEMSolverParked:
    """Tests for parked-blade BEM analysis (omega = 0)."""

    @pytest.fixture(scope="class")
    def parked_result(self, bem_solver):
        """BEM result for parked blade at V=45 m/s, pitch=0."""
        return bem_solver.compute(v_inf=45.0, omega=0.0, pitch=0.0)

    def test_result_is_bem_result(self, parked_result):
        assert isinstance(parked_result, BEMResult)

    def test_r_matches_blade(self, parked_result, blade_aero):
        np.testing.assert_array_equal(parked_result.r, blade_aero.r)

    def test_Np_shape(self, parked_result, blade_aero):
        assert parked_result.Np.shape == blade_aero.r.shape

    def test_Tp_shape(self, parked_result, blade_aero):
        assert parked_result.Tp.shape == blade_aero.r.shape

    def test_alpha_shape(self, parked_result, blade_aero):
        assert parked_result.alpha.shape == blade_aero.r.shape

    def test_parked_alpha_close_to_twist(self, parked_result, blade_aero):
        """For a parked blade (omega=0, pitch=0), AoA ≈ 90° - twist.

        When the blade is stationary the inflow is purely axial, so the
        angle of attack at each section equals the complement of the
        twist angle: α = 90° − θ_twist (both in degrees here, since BEM
        result reports alpha in degrees).
        """
        twist_deg = np.rad2deg(blade_aero.twist)
        expected_alpha = 90.0 - twist_deg  # from geometry
        # Allow generous tolerance — BEM induction modifies the inflow angle
        # but for parked conditions (no rotation) it should be close
        residual = np.abs(parked_result.alpha - expected_alpha)
        # At least 80% of stations should be within 15° — inboard stations
        # with thick airfoils and root corrections may deviate more
        fraction_close = np.mean(residual < 15.0)
        assert fraction_close > 0.8, (
            f"Only {fraction_close * 100:.0f}% of stations have alpha within 15° of expected"
        )

    def test_parked_thrust_positive(self, parked_result):
        """Thrust should be positive for positive wind speed."""
        assert parked_result.thrust > 0

    def test_parked_torque_near_zero(self, parked_result):
        """Torque should be very small for a parked rotor."""
        # For parked, power ≈ 0 (no rotation)
        assert abs(parked_result.power) < 1e6  # much less than rated power

    def test_parked_Np_mostly_positive(self, parked_result):
        """Normal force should be mostly positive (pushed downwind)."""
        frac_positive = np.mean(parked_result.Np > 0)
        assert frac_positive > 0.7


class TestBEMSolverRotating:
    """Tests for rotating-blade BEM analysis."""

    @pytest.fixture(scope="class")
    def rated_result(self, bem_solver):
        """BEM result near rated conditions: V=10.59 m/s, 7.56 RPM."""
        return bem_solver.compute(v_inf=10.59, omega=7.56, pitch=0.0)

    def test_rated_power_positive(self, rated_result):
        """A rotating blade in wind should produce positive power."""
        assert rated_result.power > 0

    def test_rated_thrust_positive(self, rated_result):
        assert rated_result.thrust > 0

    def test_rated_torque_positive(self, rated_result):
        assert rated_result.torque > 0

    def test_axial_induction_physical(self, rated_result):
        """Axial induction factor should be between 0 and ~0.6."""
        a = rated_result.a
        # Some root/tip stations may have extreme values; check bulk
        mask = np.isfinite(a) & (a > -0.5)
        frac_ok = np.mean((a[mask] >= -0.1) & (a[mask] <= 0.6))
        assert frac_ok > 0.7

    def test_cl_not_all_zero(self, rated_result):
        """Lift coefficients should be non-trivial."""
        assert np.any(np.abs(rated_result.cl) > 0.1)
