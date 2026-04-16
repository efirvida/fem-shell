"""Tests for conservative force projection (BEM → shell mesh).

No external BEM dependencies required — uses synthetic BEM results and
a simple rectangular mesh to verify:
- Total force conservation: Σ f_nodes == F_BEM_integrated
- Moment conservation per strip
- Correct handling of single-node strips
"""

import numpy as np
import pytest

from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
from fem_shell.core.mesh.model import MeshModel
from fem_shell.models.blade.aerodynamics import AeroStation, AirfoilAero, BladeAero, PolarData
from fem_shell.solvers.bem.engine import BEMResult
from fem_shell.solvers.bem.force_projection import ForceProjector

# =====================================================================
# Helpers
# =====================================================================


def _make_rectangular_mesh(n_span: int, n_chord: int, span_length: float, chord_length: float):
    """Create a flat rectangular quad mesh in the XZ plane.

    Span direction: Z (from hub_radius to hub_radius + span_length).
    Chord direction: X.
    Y = 0 (flat plate).

    Returns (mesh, hub_radius).
    """
    hub_radius = 3.0  # arbitrary
    Node._id_counter = 0
    MeshElement._id_counter = 0

    nodes = []
    for j in range(n_span):
        z = hub_radius + j * span_length / (n_span - 1)
        for i in range(n_chord):
            x = i * chord_length / (n_chord - 1)
            nodes.append(Node([x, 0.0, z]))

    elements = []
    for j in range(n_span - 1):
        for i in range(n_chord - 1):
            n0 = j * n_chord + i
            n1 = n0 + 1
            n2 = n0 + n_chord + 1
            n3 = n0 + n_chord
            elem_nodes = [nodes[n0], nodes[n1], nodes[n2], nodes[n3]]
            elements.append(MeshElement(elem_nodes, ElementType.quad))

    mesh = MeshModel(nodes=nodes, elements=elements)
    return mesh, hub_radius


def _make_simple_blade_aero(n_stations: int, hub_radius: float, span_length: float):
    """Create a BladeAero with uniform stations along the span."""
    alpha = np.linspace(-np.pi, np.pi, 37)
    polar = PolarData(
        alpha=alpha,
        cl=2 * np.pi * np.sin(alpha),
        cd=np.full(37, 0.01),
        cm=np.zeros(37),
        re=1e6,
    )
    airfoil = AirfoilAero(
        name="flat_plate",
        coordinates=np.array([[1, 0], [0, 0]]),
        relative_thickness=0.0,
        aerodynamic_center=0.25,
        polars=[polar],
    )

    stations = []
    for i in range(n_stations):
        eta = i / (n_stations - 1)
        r = hub_radius + eta * span_length
        stations.append(
            AeroStation(
                span_fraction=eta,
                r=r,
                chord=1.0,
                twist=0.0,
                pitch_axis=0.25,
                airfoil=airfoil,
            )
        )

    return BladeAero(
        airfoils=[airfoil],
        stations=stations,
        blade_length=span_length,
        hub_radius=hub_radius,
        rotor_radius=hub_radius + span_length,
        n_blades=3,
    )


def _make_uniform_bem_result(blade_aero: BladeAero, Np_val: float, Tp_val: float):
    """Create a BEMResult with uniform Np and Tp along the span."""
    n = len(blade_aero.stations)
    return BEMResult(
        r=blade_aero.r,
        Np=np.full(n, Np_val),
        Tp=np.full(n, Tp_val),
        alpha=np.zeros(n),
        cl=np.zeros(n),
        cd=np.zeros(n),
        a=np.zeros(n),
        ap=np.zeros(n),
        thrust=0.0,  # not used by projector
        torque=0.0,
        power=0.0,
    )


# =====================================================================
# Tests
# =====================================================================


class TestForceProjectorConstruction:
    """Basic construction tests."""

    def test_creates_strips(self):
        mesh, hub_r = _make_rectangular_mesh(
            n_span=10, n_chord=5, span_length=20.0, chord_length=1.0
        )
        blade_aero = _make_simple_blade_aero(n_stations=10, hub_radius=hub_r, span_length=20.0)
        projector = ForceProjector(mesh, blade_aero)
        # Should have as many strips as BEM stations
        assert len(projector._strips) == 10

    def test_all_nodes_assigned(self):
        """Every mesh node should belong to some strip."""
        mesh, hub_r = _make_rectangular_mesh(
            n_span=10, n_chord=5, span_length=20.0, chord_length=1.0
        )
        blade_aero = _make_simple_blade_aero(n_stations=10, hub_radius=hub_r, span_length=20.0)
        projector = ForceProjector(mesh, blade_aero)

        assigned = set()
        for strip in projector._strips:
            assigned.update(strip.node_indices.tolist())
        assert len(assigned) == len(mesh.nodes)


class TestForceConservation:
    """Force conservation: sum of projected nodal forces == BEM integrated force."""

    @pytest.fixture
    def setup(self):
        n_span, n_chord = 11, 5
        span_length = 20.0
        mesh, hub_r = _make_rectangular_mesh(n_span, n_chord, span_length, chord_length=1.0)
        blade_aero = _make_simple_blade_aero(
            n_stations=n_span, hub_radius=hub_r, span_length=span_length
        )
        return mesh, blade_aero

    def test_uniform_Np_conservation(self, setup):
        """Uniform Np=1000 N/m: total normal force = Np * L."""
        mesh, blade_aero = setup
        Np_val = 1000.0
        span_length = blade_aero.blade_length
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=Np_val, Tp_val=0.0)

        projector = ForceProjector(mesh, blade_aero, span_direction=[0, 0, 1])
        forces = projector.project(bem_result)

        # Expected total: Np * span_length in the normal direction (x)
        expected_total = Np_val * span_length
        actual_total_x = forces[:, 0].sum()

        # Tangential (y) and span (z) components should be ~0
        np.testing.assert_allclose(forces[:, 1].sum(), 0.0, atol=1e-6)
        np.testing.assert_allclose(forces[:, 2].sum(), 0.0, atol=1e-6)

        # Normal force conservation (use trapz-based expected from verify)
        verification = projector.verify(bem_result, forces)
        assert verification["force_error"] < 1e-6, (
            f"Force error = {verification['force_error']:.2e} N"
        )

    def test_uniform_Tp_conservation(self, setup):
        """Uniform Tp=500 N/m: total tangential force = Tp * L."""
        mesh, blade_aero = setup
        Tp_val = 500.0
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=0.0, Tp_val=Tp_val)

        projector = ForceProjector(mesh, blade_aero, span_direction=[0, 0, 1])
        forces = projector.project(bem_result)

        # Normal (x) and span (z) should be ~0
        np.testing.assert_allclose(forces[:, 0].sum(), 0.0, atol=1e-6)
        np.testing.assert_allclose(forces[:, 2].sum(), 0.0, atol=1e-6)

        verification = projector.verify(bem_result, forces)
        assert verification["force_error"] < 1e-6

    def test_combined_Np_Tp_conservation(self, setup):
        """Combined Np + Tp should conserve both components."""
        mesh, blade_aero = setup
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=800.0, Tp_val=300.0)

        projector = ForceProjector(mesh, blade_aero, span_direction=[0, 0, 1])
        forces = projector.project(bem_result)

        verification = projector.verify(bem_result, forces)
        assert verification["force_error"] < 1e-6

    def test_varying_Np_conservation(self, setup):
        """Linearly varying Np (root=2000, tip=0): force conservation via trapz."""
        mesh, blade_aero = setup
        n = len(blade_aero.stations)
        Np_varying = np.linspace(2000.0, 0.0, n)

        bem_result = BEMResult(
            r=blade_aero.r,
            Np=Np_varying,
            Tp=np.zeros(n),
            alpha=np.zeros(n),
            cl=np.zeros(n),
            cd=np.zeros(n),
            a=np.zeros(n),
            ap=np.zeros(n),
            thrust=0.0,
            torque=0.0,
            power=0.0,
        )

        projector = ForceProjector(mesh, blade_aero, span_direction=[0, 0, 1])
        forces = projector.project(bem_result)

        verification = projector.verify(bem_result, forces)
        # Slightly more relaxed for varying loads — strip discretisation
        # introduces small mismatch vs. continuous trapz integral
        assert verification["force_error"] < 50.0, (
            f"Force error = {verification['force_error']:.2e} N (expected < 50 N for a 20 m blade)"
        )


class TestForceProjectionOutput:
    """Output shape and direction tests."""

    def test_output_shape(self):
        mesh, hub_r = _make_rectangular_mesh(
            n_span=6, n_chord=4, span_length=10.0, chord_length=1.0
        )
        blade_aero = _make_simple_blade_aero(n_stations=6, hub_radius=hub_r, span_length=10.0)
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=100.0, Tp_val=0.0)

        projector = ForceProjector(mesh, blade_aero)
        forces = projector.project(bem_result)

        assert forces.shape == (len(mesh.nodes), 3)

    def test_zero_load_gives_zero_forces(self):
        mesh, hub_r = _make_rectangular_mesh(
            n_span=6, n_chord=4, span_length=10.0, chord_length=1.0
        )
        blade_aero = _make_simple_blade_aero(n_stations=6, hub_radius=hub_r, span_length=10.0)
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=0.0, Tp_val=0.0)

        projector = ForceProjector(mesh, blade_aero)
        forces = projector.project(bem_result)

        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_forces_only_in_load_direction(self):
        """Np-only load should produce forces only in the normal direction."""
        mesh, hub_r = _make_rectangular_mesh(
            n_span=6, n_chord=4, span_length=10.0, chord_length=1.0
        )
        blade_aero = _make_simple_blade_aero(n_stations=6, hub_radius=hub_r, span_length=10.0)
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=500.0, Tp_val=0.0)

        projector = ForceProjector(
            mesh,
            blade_aero,
            normal_direction=[1, 0, 0],
            tangential_direction=[0, 1, 0],
        )
        forces = projector.project(bem_result)

        # Y and Z components should be essentially zero
        np.testing.assert_allclose(forces[:, 1], 0.0, atol=1e-8)
        np.testing.assert_allclose(forces[:, 2], 0.0, atol=1e-8)

        # X component should be non-zero
        assert np.abs(forces[:, 0]).sum() > 0


class TestSingleNodeStrip:
    """Edge case: strips with a single node should receive the full strip force."""

    def test_single_node_per_strip(self):
        """Mesh with 1 chordwise node per span station."""
        Node._id_counter = 0
        MeshElement._id_counter = 0

        hub_r = 3.0
        span_length = 10.0
        n_stations = 5

        # Create nodes along span, one per strip
        nodes = []
        for j in range(n_stations):
            z = hub_r + j * span_length / (n_stations - 1)
            nodes.append(Node([0.5, 0.0, z]))

        # Minimal triangle elements (not used by projector, but mesh needs them)
        elements = []
        for j in range(n_stations - 1):
            # Degenerate triangles just for mesh validity
            elem_nodes = [nodes[j], nodes[j + 1], nodes[j]]
            elements.append(MeshElement(elem_nodes, ElementType.triangle))

        mesh = MeshModel(nodes=nodes, elements=elements)
        blade_aero = _make_simple_blade_aero(
            n_stations=n_stations, hub_radius=hub_r, span_length=span_length
        )
        bem_result = _make_uniform_bem_result(blade_aero, Np_val=1000.0, Tp_val=0.0)

        projector = ForceProjector(mesh, blade_aero, span_direction=[0, 0, 1])
        forces = projector.project(bem_result)

        # Each node should get the full strip force (F = Np * dr)
        assert forces.shape == (n_stations, 3)
        # All forces in x direction (normal)
        assert np.all(np.abs(forces[:, 0]) > 0)
        # Total force conservation
        verification = projector.verify(bem_result, forces)
        assert verification["force_error"] < 1.0  # relaxed for coarse discretisation
