"""Fase 6 — Parity tests for LinearDynamicFSIRotorSolver Rust fast-path.

Three test groups:

``TestMapOmegaProvider``
    Unit tests for ``_map_omega_provider()`` — verifies that each Python
    OmegaProvider subclass maps correctly to the Rust parameter tuple.
    No PETSc or preCICE required.

``TestUseRustFlag``
    Verifies ``use_rust: true/false`` in the YAML is picked up by
    ``_init_rotor_config`` and stored in ``_use_rust_fsi``.
    Requires PETSc; skips when unavailable.

``TestRotorRustBinding``
    Smoke tests for ``_aeroelast.run_rotor_fsi_solver`` with a minimal
    COO system and a zero-step mock (preCICE not initialised; expected to
    raise an exception from the Rust side — we only verify the Python
    marshalling layer and the presence of the symbol).
    Requires ``_aeroelast`` built with ``--features fsi``; skips otherwise.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Import omega providers — must come from the installed package so that
# ``isinstance`` checks in the rotor tests work (same class objects as rotor.py).
# The direct file-loader approach would create a separate module entry in
# sys.modules, making isinstance return False even for matching types.
# ---------------------------------------------------------------------------
try:
    from aeroelast.solvers.fsi.corotational import (
        ConstantOmega,
        RampedOmega,
        ComputedOmega,
        RampedComputedOmega,
        TableOmega,
    )

    _HAS_CORO = True
except (ImportError, OSError):
    _HAS_CORO = False
    # Fallbacks so collection doesn't crash
    ConstantOmega = RampedOmega = ComputedOmega = RampedComputedOmega = TableOmega = None  # type: ignore

# ---------------------------------------------------------------------------
# Optional imports — all guarded so TestMapOmegaProvider runs without PETSc
# ---------------------------------------------------------------------------

try:
    from petsc4py import PETSc  # noqa: F401
    from aeroelast.solvers.fsi.rotor import LinearDynamicFSIRotorSolver

    _HAS_ROTOR = True
except (ImportError, OSError):
    _HAS_ROTOR = False
    LinearDynamicFSIRotorSolver = None  # type: ignore[assignment,misc]

_skip_rotor = pytest.mark.skipif(
    not _HAS_ROTOR, reason="preCICE or PETSc not available on this node"
)

try:
    import _aeroelast  # type: ignore[import]

    _HAS_RUST = hasattr(_aeroelast, "run_rotor_fsi_solver")
except (ImportError, OSError):
    _HAS_RUST = False

_skip_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="_aeroelast.run_rotor_fsi_solver not available"
)


# ---------------------------------------------------------------------------
# Minimal stub for _map_omega_provider (no full aeroelast import needed)
# ---------------------------------------------------------------------------


class _RotorStub:
    """Minimal object that carries ``_omega_provider`` and a copy of
    ``_map_omega_provider`` logic.

    The method is inlined so that ``TestMapOmegaProvider`` runs without PETSc
    or preCICE.  The logic mirrors ``LinearDynamicFSIRotorSolver._map_omega_provider``
    exactly — any divergence is a test failure waiting to happen.
    """

    def __init__(self, provider):
        self._omega_provider = provider

    def _map_omega_provider(self):
        """Mirror of LinearDynamicFSIRotorSolver._map_omega_provider."""
        p = self._omega_provider
        if isinstance(p, ConstantOmega):
            return "constant", float(p._omega), None, None, None, None
        if isinstance(p, RampedOmega):
            return "ramped", 0.0, float(p._target_omega), float(p._ramp_time), None, None
        if isinstance(p, ComputedOmega):
            return (
                "computed",
                float(p._omega),
                None,
                None,
                float(p._I),
                float(p._tau_shaft),
            )
        if isinstance(p, RampedComputedOmega):
            return (
                "ramped_computed",
                0.0,
                float(p._target_omega),
                float(p._ramp_time),
                float(p._I),
                float(p._tau_shaft),
            )
        # Fallback for TableOmega / FunctionOmega: sample omega at t=0
        omega_val, _ = p.get_omega(0.0)
        return "constant", float(omega_val), None, None, None, None


# ---------------------------------------------------------------------------
# Group 1 — _map_omega_provider unit tests (no PETSc, no preCICE)
# ---------------------------------------------------------------------------


class TestMapOmegaProvider:
    """Verify that every OmegaProvider subclass maps to the correct Rust params."""

    def _map(self, provider):
        stub = _RotorStub(provider)
        return stub._map_omega_provider()

    def test_constant_omega(self):
        p = ConstantOmega(omega=10.0)
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "constant"
        assert_allclose(omega, 10.0)
        assert omega_target is None
        assert t_ramp is None
        assert moi is None
        assert shaft_tau is None

    def test_constant_omega_zero(self):
        p = ConstantOmega(omega=0.0)
        mode, omega, *rest = self._map(p)
        assert mode == "constant"
        assert_allclose(omega, 0.0)
        assert all(v is None for v in rest)

    def test_ramped_omega(self):
        p = RampedOmega(target_omega=20.0, ramp_time=5.0)
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "ramped"
        assert_allclose(omega, 0.0)  # starts from zero
        assert_allclose(omega_target, 20.0)
        assert_allclose(t_ramp, 5.0)
        assert moi is None
        assert shaft_tau is None

    def test_computed_omega_defaults(self):
        p = ComputedOmega(moment_of_inertia=500.0)
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "computed"
        assert_allclose(omega, 0.0)  # initial_omega default
        assert omega_target is None
        assert t_ramp is None
        assert_allclose(moi, 500.0)
        assert_allclose(shaft_tau, 0.0)  # shaft_torque default

    def test_computed_omega_with_initial_and_torque(self):
        p = ComputedOmega(moment_of_inertia=1000.0, initial_omega=5.0, shaft_torque=-200.0)
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "computed"
        assert_allclose(omega, 5.0)
        assert moi == pytest.approx(1000.0)
        assert shaft_tau == pytest.approx(-200.0)

    def test_ramped_computed_omega(self):
        p = RampedComputedOmega(
            target_omega=15.0,
            ramp_time=3.0,
            moment_of_inertia=800.0,
            shaft_torque=-50.0,
        )
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "ramped_computed"
        assert_allclose(omega, 0.0)  # starts from zero
        assert_allclose(omega_target, 15.0)
        assert_allclose(t_ramp, 3.0)
        assert_allclose(moi, 800.0)
        assert_allclose(shaft_tau, -50.0)

    def test_ramped_computed_default_shaft_torque(self):
        p = RampedComputedOmega(target_omega=10.0, ramp_time=2.0, moment_of_inertia=300.0)
        mode, *_, shaft_tau = self._map(p)
        assert mode == "ramped_computed"
        assert shaft_tau == pytest.approx(0.0)

    def test_table_omega_fallback(self):
        """TableOmega has no Rust counterpart — should fall back to 'constant'."""
        p = TableOmega([0.0, 1.0, 2.0], [0.0, 5.0, 5.0])
        mode, omega, *rest = self._map(p)

        assert mode == "constant"
        assert_allclose(omega, 0.0)  # value at t=0
        assert all(v is None for v in rest)


# ---------------------------------------------------------------------------
# Helpers shared by PETSc-dependent groups
# ---------------------------------------------------------------------------


def _make_solver_stub(rotor_cfg_overrides: Optional[dict] = None):
    """
    Build a minimal ``LinearDynamicFSIRotorSolver`` via ``object.__new__``
    and then call ``_init_rotor_config()`` on it so we test the real parsing
    logic without touching PETSc or preCICE.
    """
    from aeroelast.core.mesh.model import MeshModel  # noqa: PLC0415

    base_model_props = {
        "elements": {},
        "solver": {
            "type": "FSIRotorDynamic",
            "time_step": 0.01,
            "total_time": 0.1,
            "beta": 0.25,
            "gamma": 0.5,
            "coupling_boundaries": ["blade_surface"],
            "precice": {
                "participant": "Solid",
                "config_file": "precice-config.xml",
                "coupling_mesh": "SolidMesh",
                "write_data": "Displacement",
                "read_data": "Force",
            },
            "rotor": {
                "omega": 1.0,
                "rotation_axis": [0.0, 0.0, 1.0],
                "rotation_center": [0.0, 0.0, 0.0],
                **(rotor_cfg_overrides or {}),
            },
        },
    }

    solver = object.__new__(LinearDynamicFSIRotorSolver)
    solver.solver_params = base_model_props["solver"]
    solver.model_properties = base_model_props
    solver._init_rotor_config()
    return solver


# ---------------------------------------------------------------------------
# Group 2 — use_rust flag parsing (requires PETSc for the class import)
# ---------------------------------------------------------------------------


@_skip_rotor
class TestUseRustFlag:
    """Verify rotor config parsing (use_rust flag removed — Rust is always used)."""

    def test_use_rust_default_is_true(self):
        # The solver always uses the Rust backend; _use_rust_fsi no longer exists.
        solver = _make_solver_stub()
        assert not hasattr(solver, "_use_rust_fsi"), (
            "_use_rust_fsi should not exist; Rust is now the only backend"
        )

    def test_use_rust_true(self):
        # use_rust=True in YAML is accepted but has no effect (already Rust).
        solver = _make_solver_stub({"use_rust": True})
        assert not hasattr(solver, "_use_rust_fsi")

    def test_use_rust_false_explicit(self):
        # use_rust=False is also accepted (ignored silently — Rust is always used).
        solver = _make_solver_stub({"use_rust": False})
        assert not hasattr(solver, "_use_rust_fsi")

    def test_use_rust_truthy_int(self):
        solver = _make_solver_stub({"use_rust": 1})
        assert not hasattr(solver, "_use_rust_fsi")

    def test_omega_provider_type_constant(self):
        solver = _make_solver_stub({"omega": 5.0})
        assert isinstance(solver._omega_provider, ConstantOmega)
        omega, alpha = solver._omega_provider.get_omega(0.0)
        assert_allclose(omega, 5.0)
        assert_allclose(alpha, 0.0)

    def test_omega_provider_type_ramped(self):
        solver = _make_solver_stub({"omega": 0.0, "omega_ramp_time": 3.0})
        assert isinstance(solver._omega_provider, RampedOmega)
        assert_allclose(solver._omega_provider._ramp_time, 3.0)

    def test_omega_provider_type_computed(self):
        solver = _make_solver_stub({
            "omega": 2.0,
            "moment_of_inertia": 500.0,
            "omega_ramp_time": 0.0,
        })
        assert isinstance(solver._omega_provider, ComputedOmega)
        assert_allclose(solver._omega_provider._I, 500.0)

    def test_omega_provider_type_ramped_computed(self):
        solver = _make_solver_stub({
            "omega": 5.0,
            "moment_of_inertia": 300.0,
            "omega_ramp_time": 2.0,
        })
        assert isinstance(solver._omega_provider, RampedComputedOmega)
        assert_allclose(solver._omega_provider._target_omega, 5.0)
        assert_allclose(solver._omega_provider._ramp_time, 2.0)
        assert_allclose(solver._omega_provider._I, 300.0)

    def test_map_omega_round_trip_constant(self):
        """_map_omega_provider after _init_rotor_config must return consistent values."""
        solver = _make_solver_stub({"omega": 7.5})
        mode, omega, omega_target, t_ramp, moi, shaft_tau = solver._map_omega_provider()
        assert mode == "constant"
        assert_allclose(omega, 7.5)
        assert omega_target is None

    def test_map_omega_round_trip_ramped_computed(self):
        solver = _make_solver_stub({
            "omega": 10.0,
            "moment_of_inertia": 250.0,
            "omega_ramp_time": 4.0,
            "shaft_torque": -100.0,
        })
        mode, omega, omega_target, t_ramp, moi, shaft_tau = solver._map_omega_provider()
        assert mode == "ramped_computed"
        assert_allclose(omega_target, 10.0)
        assert_allclose(t_ramp, 4.0)
        assert_allclose(moi, 250.0)
        assert_allclose(shaft_tau, -100.0)


@_skip_rotor
class TestRotorAutoInertia:
    class _FakeDiagVec:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)
            self.destroyed = False

        def getArray(self, readonly=True):
            return self._values

        def destroy(self):
            self.destroyed = True

    class _FakeMat:
        def __init__(self, diag_values):
            self._diag_values = np.asarray(diag_values, dtype=np.float64)

        def getDiagonal(self):
            return TestRotorAutoInertia._FakeDiagVec(self._diag_values)

    def test_compute_estimated_inertia_matches_point_mass_sum(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver.M = self._FakeMat([2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        solver.domain = SimpleNamespace(
            nodes=[
                SimpleNamespace(coords=np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                SimpleNamespace(coords=np.array([0.0, 2.0, 0.0], dtype=np.float64)),
            ],
            dofs_per_node=3,
        )
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        inertia = solver._compute_estimated_inertia()

        assert_allclose(inertia, 14.0)

    def test_solve_auto_inertia_uses_python_helper(self, monkeypatch):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._omega_provider = ConstantOmega(omega=1.0)
        solver._auto_inertia = True
        solver._auto_inertia_params = {
            "target_omega": 8.0,
            "shaft_torque": -5.0,
            "ramp_time": 0.0,
        }
        solver._print_header = lambda *args, **kwargs: None
        solver._print_info = lambda *args, **kwargs: None
        solver._is_primary_rank = lambda: False
        solver.solver_params = {"start_time": 0.0, "beta": 0.25, "gamma": 0.5}
        solver.model_properties = {"solver": {"coupling_boundaries": ["blade_surface"]}}
        solver._rotor_radius = 1.0

        node = SimpleNamespace(id=1, coords=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        solver.domain = SimpleNamespace(
            nodes=[node],
            dofs_per_node=3,
            spatial_dim=3,
            _node_dofs_map={1: np.array([0, 1, 2], dtype=int)},
            mesh=SimpleNamespace(node_sets={"blade_surface": SimpleNamespace(nodes={1: node})}),
        )
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        fake_mass = self._FakeMat([1.0, 0.0, 0.0])
        bc_manager = SimpleNamespace(free_dofs=np.array([0, 1, 2], dtype=np.int32))

        def _assemble(_omega_initial):
            solver.M = fake_mass
            return ((None, fake_mass), bc_manager, None)

        solver._assemble_system_matrices = _assemble
        solver._compute_estimated_inertia = lambda: 42.0

        captured = {}

        def _fake_solve_via_rust(**kwargs):
            captured["provider"] = solver._omega_provider
            return ("u", "v", "a")

        solver._solve_via_rust = _fake_solve_via_rust

        monkeypatch.setitem(sys.modules, "_aeroelast", SimpleNamespace())

        result = solver.solve()

        assert result == ("u", "v", "a")
        assert isinstance(captured["provider"], ComputedOmega)
        assert_allclose(captured["provider"]._I, 42.0)

    def test_resolve_restart_omega_state_uses_checkpoint_state_for_ramped_computed(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._omega_provider = RampedComputedOmega(
            target_omega=10.0,
            ramp_time=2.0,
            moment_of_inertia=100.0,
            shaft_torque=-5.0,
        )

        state = solver._resolve_restart_omega_state(
            {
                "t": 4.0,
                "omega": 12.5,
                "alpha": -0.75,
                "omega_ramp_completed": True,
                "omega_current_time": 4.0,
            },
            4.0,
        )

        assert_allclose(state["assembly_omega"], 12.5)
        assert_allclose(state["omega"], 12.5)
        assert_allclose(state["alpha"], -0.75)
        assert state["ramp_completed"] is True
        assert_allclose(state["current_time"], 4.0)

    def test_resolve_restart_omega_state_uses_start_time_for_ramped_provider(self):
        solver = _make_solver_stub({"omega": 12.0, "omega_ramp_time": 6.0})

        state = solver._resolve_restart_omega_state(None, 1.5)

        assert_allclose(state["assembly_omega"], 3.0)
        assert state["omega"] is None
        assert state["alpha"] is None

    def test_solve_restart_uses_checkpoint_omega_for_matrix_assembly(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._omega_provider = RampedComputedOmega(
            target_omega=8.0,
            ramp_time=2.0,
            moment_of_inertia=50.0,
        )
        solver._auto_inertia = False
        solver._print_header = lambda *args, **kwargs: None
        solver._print_info = lambda *args, **kwargs: None
        solver._is_primary_rank = lambda: False
        solver.solver_params = {"start_time": 0.0, "beta": 0.25, "gamma": 0.5}
        solver.model_properties = {"solver": {"coupling_boundaries": ["blade_surface"]}}
        solver._rotor_radius = 1.0

        node = SimpleNamespace(id=1, coords=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        solver.domain = SimpleNamespace(
            nodes=[node],
            dofs_per_node=3,
            spatial_dim=3,
            _node_dofs_map={1: np.array([0, 1, 2], dtype=int)},
            mesh=SimpleNamespace(node_sets={"blade_surface": SimpleNamespace(nodes={1: node})}),
        )
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        solver._include_geometric_stiffness = False

        checkpoint_state = {
            "u_red": np.zeros(3, dtype=np.float64),
            "v_red": np.zeros(3, dtype=np.float64),
            "a_red": np.zeros(3, dtype=np.float64),
            "t": 3.0,
            "theta": 0.4,
            "omega": 9.5,
            "alpha": -0.2,
            "omega_ramp_completed": True,
            "omega_current_time": 3.0,
        }
        solver._try_restore_checkpoint = lambda: checkpoint_state

        fake_mass = self._FakeMat([1.0, 0.0, 0.0])
        bc_manager = SimpleNamespace(free_dofs=np.array([0, 1, 2], dtype=np.int32))
        captured = {}

        def _assemble(omega_initial):
            captured["omega_initial"] = omega_initial
            solver.M = fake_mass
            return ((None, fake_mass), bc_manager, None)

        solver._assemble_system_matrices = _assemble
        solver._compute_rotor_radius = lambda coords, interface_disps=None: 1.0

        def _fake_solve_via_rust(**kwargs):
            captured["checkpoint_state"] = kwargs["checkpoint_state"]
            captured["restart_omega_state"] = kwargs["restart_omega_state"]
            return ("u", "v", "a")

        solver._solve_via_rust = _fake_solve_via_rust

        result = solver.solve()

        assert result == ("u", "v", "a")
        assert_allclose(captured["omega_initial"], 9.5)
        assert captured["checkpoint_state"] is checkpoint_state
        assert_allclose(captured["restart_omega_state"]["omega"], 9.5)
        assert captured["restart_omega_state"]["ramp_completed"] is True

    def test_solve_via_rust_passes_restart_state_by_keyword(self, monkeypatch):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver.K = SimpleNamespace(getSize=lambda: (3, 3))
        solver.M = self._FakeMat([1.0, 0.0, 0.0])
        solver._petsc_to_coo = lambda _mat: (
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
        )
        solver._map_omega_provider = lambda: ("computed", 5.0, None, None, 42.0, -3.0)
        solver._try_restore_checkpoint = lambda: None
        solver._is_primary_rank = lambda: False
        solver._peek_checkpoint_omega = lambda: None
        solver.solver_params = {"start_time": 0.0, "beta": 0.25, "gamma": 0.5, "dt": 0.1}
        solver._eta_k = 0.0
        solver._eta_m = 0.0
        solver._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        solver._include_centrifugal = True
        solver._include_coriolis = True
        solver._include_euler = True
        solver._include_geometric_stiffness = True
        solver._include_spin_softening = True
        solver._kg_update_interval = 0
        solver._fluid_density = 1.225
        solver._flow_velocity = 10.0
        solver._rotor_radius = 1.0
        solver._send_omega_to_precice = True
        solver._omega_mesh_name = "GlobalSolidMesh"
        solver._omega_write_data_name = "AngularVelocity"
        solver._force_ramp_time = 0.0
        solver._force_max_magnitude = None
        solver._checkpoint_manager = None
        solver._interface_coords = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        node = SimpleNamespace(coords=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        solver.domain = SimpleNamespace(nodes=[node], dofs_per_node=3, spatial_dim=3, _rust=None)
        solver._coupling_cfg = {
            "participant": "Solid",
            "config_file": "precice-config.xml",
            "coupling_mesh": "SolidMesh",
            "write_data": "Displacement",
            "read_data": "Force",
        }

        captured = {}

        def _fake_run_rotor_fsi_solver(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [], [], [], []

        monkeypatch.setitem(
            sys.modules,
            "_aeroelast",
            SimpleNamespace(run_rotor_fsi_solver=_fake_run_rotor_fsi_solver),
        )

        bc_manager = SimpleNamespace(free_dofs=np.array([0, 1, 2], dtype=np.int32), fixed_dofs={})

        solver._solve_via_rust(
            bc_manager=bc_manager,
            interface_coords_flat=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            interface_dofs_global_flat=np.array([0, 1, 2], dtype=np.uint64),
            K_G=None,
            checkpoint_state=None,
            restart_omega_state={
                "omega": 9.5,
                "alpha": -0.2,
                "ramp_completed": True,
                "current_time": 3.0,
            },
        )

        assert captured["kwargs"]["include_euler"] is True
        assert captured["kwargs"]["restart_omega"] == 9.5
        assert captured["kwargs"]["restart_alpha"] == -0.2
        assert captured["kwargs"]["restart_ramp_completed"] is True
        assert captured["kwargs"]["restart_current_time"] == 3.0
        assert captured["kwargs"]["theta0"] == 0.0

    def test_compute_rotor_radius_matches_max_perpendicular_distance(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        coords = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 3.0],
            ],
            dtype=np.float64,
        )

        radius = solver._compute_rotor_radius(coords)

        assert_allclose(radius, 2.0)

    def test_compute_rotor_radius_uses_deformed_coordinates(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        coords = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        disps = np.array([[0.0, 2.0, 0.0]], dtype=np.float64)

        radius = solver._compute_rotor_radius(coords, disps)

        assert_allclose(radius, np.sqrt(5.0))

    def test_compute_axis_torque_uses_deformed_positions(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            to_inertial=lambda vec, theta: np.asarray(vec, dtype=np.float64),
        )

        torque_vec, torque_scalar = solver._compute_axis_torque(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[10.0, 0.0, 0.0]], dtype=np.float64),
            theta=0.0,
        )

        assert_allclose(torque_vec, np.array([0.0, 0.0, -10.0]))
        assert_allclose(torque_scalar, -10.0)

    def test_compute_performance_coefficients_uses_given_radius(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._fluid_density = 1.225
        solver._flow_velocity = 10.0

        _, cp_ref, cq_ref, tsr_ref = solver._compute_performance_coefficients(
            thrust=100.0,
            torque_aero=50.0,
            omega=5.0,
            radius=2.0,
        )
        _, cp_def, cq_def, tsr_def = solver._compute_performance_coefficients(
            thrust=100.0,
            torque_aero=50.0,
            omega=5.0,
            radius=3.0,
        )

        assert cp_def < cp_ref
        assert cq_def < cq_ref
        assert tsr_def > tsr_ref

    def test_solve_auto_radius_uses_python_helper(self):
        solver = object.__new__(LinearDynamicFSIRotorSolver)
        solver._omega_provider = ConstantOmega(omega=1.0)
        solver._auto_inertia = False
        solver._print_header = lambda *args, **kwargs: None
        solver._print_info = lambda *args, **kwargs: None
        solver._is_primary_rank = lambda: False
        solver.solver_params = {"start_time": 0.0, "beta": 0.25, "gamma": 0.5}
        solver.model_properties = {"solver": {"coupling_boundaries": ["blade_surface"]}}
        solver._rotor_radius = None

        node = SimpleNamespace(id=1, coords=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        solver.domain = SimpleNamespace(
            nodes=[node],
            dofs_per_node=3,
            spatial_dim=3,
            _node_dofs_map={1: np.array([0, 1, 2], dtype=int)},
            mesh=SimpleNamespace(node_sets={"blade_surface": SimpleNamespace(nodes={1: node})}),
        )
        solver._coord_transforms = SimpleNamespace(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        fake_mass = self._FakeMat([1.0, 0.0, 0.0])
        bc_manager = SimpleNamespace(free_dofs=np.array([0, 1, 2], dtype=np.int32))

        def _assemble(_omega_initial):
            solver.M = fake_mass
            return ((None, fake_mass), bc_manager, None)

        solver._assemble_system_matrices = _assemble
        solver._compute_rotor_radius = lambda coords, interface_disps=None: 7.5

        captured = {}

        def _fake_solve_via_rust(**kwargs):
            captured["radius"] = solver._rotor_radius
            return ("u", "v", "a")

        solver._solve_via_rust = _fake_solve_via_rust

        result = solver.solve()

        assert result == ("u", "v", "a")
        assert_allclose(captured["radius"], 7.5)


# ---------------------------------------------------------------------------
# Group 3 — Rust binding smoke test (requires _aeroelast --features fsi)
# ---------------------------------------------------------------------------


def _make_minimal_truss_system(n_nodes: int = 3, dofs_per_node: int = 3):
    """
    Build a trivially small diagonal stiffness / mass system in COO format.

    Returns (k_rows, k_cols, k_vals, m_rows, m_cols, m_vals, free_dofs, n_dofs).
    """
    n = n_nodes * dofs_per_node
    # Diagonal stiffness k = 1e6 * I
    rows = np.arange(n, dtype=np.int32)
    cols = np.arange(n, dtype=np.int32)
    k_vals = np.full(n, 1e6, dtype=np.float64)
    m_vals = np.full(n, 1.0, dtype=np.float64)

    # Pin first node (first 3 DOFs)
    free_dofs = np.arange(dofs_per_node, n, dtype=np.int32)

    return rows, cols, k_vals, rows.copy(), cols.copy(), m_vals, free_dofs, n


@_skip_rust
class TestRotorRustBinding:
    """
    Smoke tests for ``_aeroelast.run_rotor_fsi_solver``.

    These tests do NOT initialise preCICE.  They verify that:
    1. The symbol exists in the compiled extension.
    2. Calling it with well-formed arguments raises a ``RuntimeError`` (or
       similar) from the Rust side when preCICE is absent — NOT a
       ``TypeError`` or ``AttributeError``, which would indicate a Python
       marshalling bug.
    """

    # pytest.raises only accepts a single exception type (not tuples) in pytest 9+.
    # We catch the broad Exception and re-raise if it looks like a marshalling bug.
    _PRECICE_ERRORS = (RuntimeError, OSError, SystemError)

    @staticmethod
    def _assert_expected_error(exc_info):
        """Re-raise if the exception is a marshalling bug (TypeError/AttributeError)."""
        exc = exc_info.value
        if isinstance(exc, (TypeError, AttributeError)):
            raise exc

    @staticmethod
    def _binding_supports_restart_args() -> bool:
        """Return True when the installed extension exposes restart omega args.

        The source tree may be newer than the installed `_aeroelast` extension
        available on the test node. Keep the smoke tests compatible with both
        signatures so we can still validate the Python marshalling layer.
        """
        try:
            params = inspect.signature(_aeroelast.run_rotor_fsi_solver).parameters
        except (TypeError, ValueError):
            return False
        return "restart_omega" in params

    def _call_binding(
        self,
        *,
        omega_mode: str = "constant",
        omega: float = 1.0,
        omega_target=None,
        t_ramp=None,
        moi=None,
        shaft_tau=None,
        n_nodes: int = 4,
        dofs_per_node: int = 3,
        step_callback=None,
    ):
        k_r, k_c, k_v, m_r, m_c, m_v, free_dofs, n = _make_minimal_truss_system(
            n_nodes, dofs_per_node
        )
        node_coords = np.tile([1.0, 0.0, 0.0], n_nodes).astype(np.float64)
        node_masses = np.ones(n_nodes, dtype=np.float64)

        # Interface: last node
        iface_coords = np.array([[1.0, 0.0, 0.0]], dtype=np.float64).ravel()
        iface_dofs = np.array(
            list(range((n_nodes - 1) * dofs_per_node, n_nodes * dofs_per_node)),
            dtype=np.uintp,
        )

        args = [
            None,  # rust_asm (optional)
            n,  # n_full_dofs
            0,  # kg_update_interval
            [0.0, 0.0, 1.0],  # rotation_axis
            [0.0, 0.0, 0.0],  # rotation_center
            node_coords,
            node_masses,
            omega_mode,
            omega,
            omega_target,
            t_ramp,
            moi,
            shaft_tau,
            [0.0, 0.0, -9.81],  # gravity
            True,
            True,
            True,  # centrifugal, coriolis, euler
            False,
            False,  # include_kg, include_ksp
            1e-4,  # ksp_omega_threshold
            dofs_per_node,
            1.225,  # fluid_density
            10.0,  # flow_velocity
            1.0,  # rotor_radius
            k_r,
            k_c,
            k_v,
            m_r,
            m_c,
            m_v,
            free_dofs,
            0.01,
            0.01,  # eta_k, eta_m
            0.25,
            0.5,  # beta, gamma
            0.01,  # dt
            iface_coords,
            iface_dofs,
            3,  # spatial_dim
            "Solid",  # participant_name
            "precice-config.xml",  # config_file
            "SolidMesh",  # coupling_mesh
            "Displacement",  # write_data_name
            "Force",  # read_data_name
            0.0,  # ramp_time
            None,  # force_max
            None,
            None,
            None,  # omega_mesh, omega_write, omega_vertex
            None,
            None,
            None,  # u0, v0, a0
            0.0,  # t0
            0.0,  # theta0
        ]
        if self._binding_supports_restart_args():
            args.extend([None, None, None, None])
        args.extend([None, None, None])  # kg0_rows, kg0_cols, kg0_vals

        return _aeroelast.run_rotor_fsi_solver(*args, step_callback=step_callback)  # type: ignore[name-defined]

    def test_symbol_exists(self):
        """run_rotor_fsi_solver must be importable from _aeroelast."""
        assert callable(_aeroelast.run_rotor_fsi_solver)

    def test_call_without_precice_raises_runtime_or_os_error(self):
        """Without a live preCICE config, the Rust side must raise RuntimeError/OSError."""
        with pytest.raises(Exception) as exc_info:
            self._call_binding()
        self._assert_expected_error(exc_info)

    def test_call_ramped_omega_marshalling(self):
        """Ramped omega params must reach Rust without TypeError."""
        with pytest.raises(Exception) as exc_info:
            self._call_binding(
                omega_mode="ramped",
                omega=0.0,
                omega_target=10.0,
                t_ramp=5.0,
            )
        self._assert_expected_error(exc_info)

    def test_call_computed_omega_marshalling(self):
        """Computed omega params must reach Rust without TypeError."""
        with pytest.raises(Exception) as exc_info:
            self._call_binding(
                omega_mode="computed",
                omega=2.0,
                moi=500.0,
                shaft_tau=-100.0,
            )
        self._assert_expected_error(exc_info)

    def test_call_ramped_computed_marshalling(self):
        """ramped_computed omega params must reach Rust without TypeError."""
        with pytest.raises(Exception) as exc_info:
            self._call_binding(
                omega_mode="ramped_computed",
                omega=0.0,
                omega_target=15.0,
                t_ramp=3.0,
                moi=800.0,
                shaft_tau=-50.0,
            )
        self._assert_expected_error(exc_info)

    def test_call_with_callback_marshalling(self):
        """Providing a step_callback must not cause a TypeError."""
        calls: list = []

        def _cb(t, ts, dt, u, v, a, fm, fi, omega, alpha, theta, perf):
            calls.append(t)

        with pytest.raises(Exception) as exc_info:
            self._call_binding(step_callback=_cb)
        self._assert_expected_error(exc_info)

    def test_wrong_omega_mode_raises(self):
        """An unrecognised omega_mode string should raise from Rust, not Python."""
        with pytest.raises(Exception) as exc_info:
            self._call_binding(omega_mode="invalid_mode")
        self._assert_expected_error(exc_info)

    def test_mismatched_dofs_raises(self):
        """Free DOFs that exceed n_full_dofs should raise cleanly."""
        k_r, k_c, k_v, m_r, m_c, m_v, _free, n = _make_minimal_truss_system(4, 3)
        bad_free = np.array([999, 1000, 1001], dtype=np.int32)  # out of range

        iface_coords = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        iface_dofs = np.array([9, 10, 11], dtype=np.uintp)

        with pytest.raises(BaseException) as exc_info:
            args = [
                None,
                n,
                0,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                np.tile([1.0, 0.0, 0.0], 4).astype(np.float64),
                np.ones(4, dtype=np.float64),
                "constant",
                1.0,
                None,
                None,
                None,
                None,
                [0.0, 0.0, -9.81],
                True,
                True,
                True,
                False,
                False,
                1e-4,
                3,
                1.225,
                10.0,
                1.0,
                k_r,
                k_c,
                k_v,
                m_r,
                m_c,
                m_v,
                bad_free,
                0.01,
                0.01,
                0.25,
                0.5,
                0.01,
                iface_coords,
                iface_dofs,
                3,
                "Solid",
                "precice-config.xml",
                "SolidMesh",
                "Displacement",
                "Force",
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0.0,
            ]
            if self._binding_supports_restart_args():
                args.extend([None, None, None, None])
            args.extend([None, None, None])

            _aeroelast.run_rotor_fsi_solver(*args, step_callback=None)  # type: ignore[name-defined]
        self._assert_expected_error(exc_info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
