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

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Import corotational module directly (avoids PETSc import at collection time)
# ---------------------------------------------------------------------------
_CORO_PATH = (
    Path(__file__).parent.parent
    / "src" / "aeroelast" / "solvers" / "fsi" / "corotational.py"
)
_spec = importlib.util.spec_from_file_location("_coro_test", _CORO_PATH)
_coro = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_coro_test", _coro)
_spec.loader.exec_module(_coro)

ConstantOmega = _coro.ConstantOmega
RampedOmega = _coro.RampedOmega
ComputedOmega = _coro.ComputedOmega
RampedComputedOmega = _coro.RampedComputedOmega
TableOmega = _coro.TableOmega

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
        assert_allclose(omega, 0.0)          # starts from zero
        assert_allclose(omega_target, 20.0)
        assert_allclose(t_ramp, 5.0)
        assert moi is None
        assert shaft_tau is None

    def test_computed_omega_defaults(self):
        p = ComputedOmega(moment_of_inertia=500.0)
        mode, omega, omega_target, t_ramp, moi, shaft_tau = self._map(p)

        assert mode == "computed"
        assert_allclose(omega, 0.0)           # initial_omega default
        assert omega_target is None
        assert t_ramp is None
        assert_allclose(moi, 500.0)
        assert_allclose(shaft_tau, 0.0)       # shaft_torque default

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
        assert_allclose(omega, 0.0)           # starts from zero
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
        assert_allclose(omega, 0.0)           # value at t=0
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
    """Verify that ``rotor.use_rust`` is correctly parsed into ``_use_rust_fsi``."""

    def test_use_rust_default_is_false(self):
        solver = _make_solver_stub()
        assert solver._use_rust_fsi is False

    def test_use_rust_true(self):
        solver = _make_solver_stub({"use_rust": True})
        assert solver._use_rust_fsi is True

    def test_use_rust_false_explicit(self):
        solver = _make_solver_stub({"use_rust": False})
        assert solver._use_rust_fsi is False

    def test_use_rust_truthy_int(self):
        solver = _make_solver_stub({"use_rust": 1})
        assert solver._use_rust_fsi is True

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

    _PRECICE_ERRORS = (RuntimeError, OSError, SystemError)

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
            dtype=np.intp,
        )

        return _aeroelast.run_rotor_fsi_solver(  # type: ignore[name-defined]
            None,           # rust_asm (optional)
            n,              # n_full_dofs
            0,              # kg_update_interval
            [0.0, 0.0, 1.0],       # rotation_axis
            [0.0, 0.0, 0.0],       # rotation_center
            node_coords,
            node_masses,
            omega_mode,
            omega,
            omega_target,
            t_ramp,
            moi,
            shaft_tau,
            [0.0, 0.0, -9.81],    # gravity
            True, True, True,      # centrifugal, coriolis, euler
            False, False,          # include_kg, include_ksp
            1e-4,                  # ksp_omega_threshold
            dofs_per_node,
            1.225,                 # fluid_density
            10.0,                  # flow_velocity
            1.0,                   # rotor_radius
            k_r, k_c, k_v,
            m_r, m_c, m_v,
            free_dofs,
            0.01, 0.01,            # eta_k, eta_m
            0.25, 0.5,             # beta, gamma
            0.01,                  # dt
            iface_coords,
            iface_dofs,
            3,                     # spatial_dim
            "Solid",               # participant_name
            "precice-config.xml",  # config_file
            "SolidMesh",           # coupling_mesh
            "Displacement",        # write_data_name
            "Force",               # read_data_name
            0.0,                   # ramp_time
            None,                  # force_max
            None, None, None,      # omega_mesh, omega_write, omega_vertex
            None, None, None,      # u0, v0, a0
            0.0,                   # t0
            0.0,                   # theta0
            step_callback=step_callback,
        )

    def test_symbol_exists(self):
        """run_rotor_fsi_solver must be importable from _aeroelast."""
        assert callable(_aeroelast.run_rotor_fsi_solver)

    def test_call_without_precice_raises_runtime_or_os_error(self):
        """Without a live preCICE config, the Rust side must raise RuntimeError/OSError."""
        with pytest.raises(self._PRECICE_ERRORS):
            self._call_binding()

    def test_call_ramped_omega_marshalling(self):
        """Ramped omega params must reach Rust without TypeError."""
        with pytest.raises(self._PRECICE_ERRORS):
            self._call_binding(
                omega_mode="ramped",
                omega=0.0,
                omega_target=10.0,
                t_ramp=5.0,
            )

    def test_call_computed_omega_marshalling(self):
        """Computed omega params must reach Rust without TypeError."""
        with pytest.raises(self._PRECICE_ERRORS):
            self._call_binding(
                omega_mode="computed",
                omega=2.0,
                moi=500.0,
                shaft_tau=-100.0,
            )

    def test_call_ramped_computed_marshalling(self):
        """ramped_computed omega params must reach Rust without TypeError."""
        with pytest.raises(self._PRECICE_ERRORS):
            self._call_binding(
                omega_mode="ramped_computed",
                omega=0.0,
                omega_target=15.0,
                t_ramp=3.0,
                moi=800.0,
                shaft_tau=-50.0,
            )

    def test_call_with_callback_marshalling(self):
        """Providing a step_callback must not cause a TypeError."""
        calls: list = []

        def _cb(t, ts, dt, u, v, a, fm, fi, omega, alpha, theta, perf):
            calls.append(t)

        with pytest.raises(self._PRECICE_ERRORS):
            self._call_binding(step_callback=_cb)

    def test_wrong_omega_mode_raises(self):
        """An unrecognised omega_mode string should raise from Rust, not Python."""
        with pytest.raises((self._PRECICE_ERRORS, ValueError)):
            self._call_binding(omega_mode="invalid_mode")

    def test_mismatched_dofs_raises(self):
        """Free DOFs that exceed n_full_dofs should raise cleanly."""
        k_r, k_c, k_v, m_r, m_c, m_v, _free, n = _make_minimal_truss_system(4, 3)
        bad_free = np.array([999, 1000, 1001], dtype=np.int32)  # out of range

        iface_coords = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        iface_dofs = np.array([9, 10, 11], dtype=np.intp)

        with pytest.raises((self._PRECICE_ERRORS, ValueError, OverflowError)):
            _aeroelast.run_rotor_fsi_solver(  # type: ignore[name-defined]
                None, n, 0,
                [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                np.tile([1.0, 0.0, 0.0], 4).astype(np.float64),
                np.ones(4, dtype=np.float64),
                "constant", 1.0, None, None, None, None,
                [0.0, 0.0, -9.81],
                True, True, True, False, False, 1e-4,
                3, 1.225, 10.0, 1.0,
                k_r, k_c, k_v, m_r, m_c, m_v, bad_free,
                0.01, 0.01, 0.25, 0.5, 0.01,
                iface_coords, iface_dofs, 3,
                "Solid", "precice-config.xml", "SolidMesh",
                "Displacement", "Force",
                0.0, None, None, None, None,
                None, None, None, 0.0, 0.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
