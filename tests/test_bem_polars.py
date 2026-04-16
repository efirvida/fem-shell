"""Tests for the BEM aerodynamic data layer (aerodynamics.py).

Covers:
- PolarData interpolation and periodic wrapping
- AirfoilAero Reynolds-number selection
- BladeAero loading from WindIO YAML (IEA 15 MW)
- BladeAero property accessors (r, chord, twist)
"""

from pathlib import Path

import numpy as np
import pytest

from fem_shell.models.blade.aerodynamics import (
    AirfoilAero,
    BladeAero,
    PolarData,
    load_blade_aero,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IEA_YAML = str(_PROJECT_ROOT / "examples" / "blade" / "solid" / "IEA-15-240-RWT.yaml")


# =====================================================================
# PolarData
# =====================================================================


class TestPolarData:
    """Unit tests for the PolarData dataclass."""

    @pytest.fixture
    def naca0012_polar(self):
        """Simple NACA 0012-like polar (symmetric)."""
        alpha = np.linspace(-np.pi, np.pi, 361)
        cl = 2 * np.pi * np.sin(alpha)  # thin-airfoil Cl(alpha) ≈ 2π sin(α)
        cd = 0.01 + 0.1 * np.sin(alpha) ** 2
        cm = np.zeros_like(alpha)
        return PolarData(alpha=alpha, cl=cl, cd=cd, cm=cm, re=1e6)

    def test_evaluate_at_zero(self, naca0012_polar):
        """Cl should be ~0 at alpha = 0 for a symmetric airfoil."""
        cl, cd, cm = naca0012_polar.evaluate(np.array([0.0]))
        assert abs(cl[0]) < 1e-3
        assert cd[0] > 0

    def test_evaluate_at_known_alpha(self, naca0012_polar):
        """Cl at 5° should be close to 2π * sin(5°)."""
        alpha_5 = np.deg2rad(5.0)
        cl, _, _ = naca0012_polar.evaluate(np.array([alpha_5]))
        expected = 2 * np.pi * np.sin(alpha_5)
        np.testing.assert_allclose(cl[0], expected, atol=0.05)

    def test_evaluate_periodic_wrapping(self, naca0012_polar):
        """Alpha outside [-π, π] should wrap correctly."""
        alpha_pos = np.deg2rad(10.0)
        alpha_wrapped = alpha_pos + 2 * np.pi  # 370° should == 10°
        cl_ref, _, _ = naca0012_polar.evaluate(np.array([alpha_pos]))
        cl_wrap, _, _ = naca0012_polar.evaluate(np.array([alpha_wrapped]))
        np.testing.assert_allclose(cl_wrap, cl_ref, atol=1e-6)

    def test_evaluate_vectorised(self, naca0012_polar):
        """evaluate() should handle arrays."""
        alphas = np.deg2rad(np.array([-10.0, 0.0, 5.0, 10.0]))
        cl, cd, cm = naca0012_polar.evaluate(alphas)
        assert cl.shape == (4,)
        assert cd.shape == (4,)
        # Cl should increase from -10° to +10°
        assert cl[0] < cl[1] < cl[3]


# =====================================================================
# AirfoilAero
# =====================================================================


class TestAirfoilAero:
    """Reynolds number selection tests."""

    @pytest.fixture
    def multi_re_airfoil(self):
        """Airfoil with polars at Re = 3e6, 6e6, 9e6."""
        alpha = np.linspace(-np.pi, np.pi, 37)
        polars = []
        for re in [3e6, 6e6, 9e6]:
            cl = 2 * np.pi * np.sin(alpha) * (1 + 0.1 * re / 9e6)
            polars.append(
                PolarData(
                    alpha=alpha,
                    cl=cl,
                    cd=np.full_like(alpha, 0.01),
                    cm=np.zeros_like(alpha),
                    re=re,
                )
            )
        return AirfoilAero(
            name="test_af",
            coordinates=np.zeros((10, 2)),
            relative_thickness=0.21,
            aerodynamic_center=0.25,
            polars=polars,
        )

    def test_get_polar_exact_match(self, multi_re_airfoil):
        """Exact Re match should return the right polar."""
        polar = multi_re_airfoil.get_polar(6e6)
        assert polar.re == 6e6

    def test_get_polar_closest(self, multi_re_airfoil):
        """Nearest-Re selection should work."""
        polar = multi_re_airfoil.get_polar(7e6)
        assert polar.re == 6e6  # 7e6 is closer to 6e6 than 9e6

    def test_get_polar_single(self):
        """Single polar should be returned regardless of Re."""
        alpha = np.linspace(-np.pi, np.pi, 37)
        polar = PolarData(alpha=alpha, cl=np.zeros(37), cd=np.zeros(37), cm=np.zeros(37), re=1e6)
        af = AirfoilAero(
            name="single",
            coordinates=np.zeros((5, 2)),
            relative_thickness=0.1,
            aerodynamic_center=0.25,
            polars=[polar],
        )
        result = af.get_polar(9e6)
        assert result.re == 1e6


# =====================================================================
# BladeAero — YAML loading (IEA 15 MW)
# =====================================================================


class TestBladeAeroYAML:
    """Integration tests that load polars from the IEA 15 MW WindIO YAML."""

    @pytest.fixture(scope="class")
    def blade_aero(self):
        """Load IEA 15 MW blade aero data once per class."""
        return load_blade_aero(IEA_YAML)

    def test_blade_length_positive(self, blade_aero):
        assert blade_aero.blade_length > 100.0  # IEA 15 MW ~ 117 m

    def test_hub_radius_positive(self, blade_aero):
        assert blade_aero.hub_radius > 0.0

    def test_rotor_radius(self, blade_aero):
        assert blade_aero.rotor_radius > blade_aero.hub_radius

    def test_n_blades(self, blade_aero):
        assert blade_aero.n_blades == 3

    def test_stations_ordered(self, blade_aero):
        """Stations should be root-to-tip ordered."""
        r = blade_aero.r
        assert len(r) > 5
        assert np.all(np.diff(r) >= 0)

    def test_chord_physical(self, blade_aero):
        """Chord should be positive and < 10 m for IEA 15 MW."""
        chord = blade_aero.chord
        assert np.all(chord > 0)
        assert np.all(chord < 10.0)

    def test_twist_range(self, blade_aero):
        """Twist (rad) should be within ±30° for a utility-scale blade."""
        twist = blade_aero.twist
        assert np.all(np.abs(twist) < np.deg2rad(30))

    def test_airfoils_have_polars(self, blade_aero):
        """Every airfoil should have at least one polar table."""
        for af in blade_aero.airfoils:
            assert len(af.polars) >= 1, f"Airfoil {af.name} has no polars"

    def test_polar_alpha_covers_range(self, blade_aero):
        """Polars should cover a wide alpha range (at least ±90°)."""
        af = blade_aero.airfoils[0]
        polar = af.polars[0]
        alpha_range = polar.alpha.max() - polar.alpha.min()
        assert alpha_range >= np.pi  # at least 180°

    def test_polar_cl_not_constant(self, blade_aero):
        """Cl should vary with alpha for a non-cylindrical airfoil."""
        # Skip the root cylinder (airfoils[0]) — pick a mid-span airfoil
        af = blade_aero.airfoils[len(blade_aero.airfoils) // 2]
        polar = af.polars[0]
        assert np.std(polar.cl) > 0.1

    def test_r_property_shape(self, blade_aero):
        """r property should match number of stations."""
        assert blade_aero.r.shape == (len(blade_aero.stations),)

    def test_chord_property_shape(self, blade_aero):
        assert blade_aero.chord.shape == (len(blade_aero.stations),)

    def test_twist_property_shape(self, blade_aero):
        assert blade_aero.twist.shape == (len(blade_aero.stations),)
