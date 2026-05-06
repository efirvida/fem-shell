"""Tests for rotor_performance.csv logging in the rotor FSI solver."""

from __future__ import annotations

import csv
import sys
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("petsc4py", reason="PETSc not available")

try:
    import precice  # noqa: F401
except (ImportError, OSError):

    class _DummyParticipant:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["precice"] = SimpleNamespace(Participant=_DummyParticipant)

try:
    import _aeroelast  # noqa: F401
except (ImportError, OSError):

    class _DummyElementFamily(Enum):
        PLANE = "plane"
        SHELL = "shell"
        SOLID = "solid"

    sys.modules["_aeroelast"] = SimpleNamespace(ElementFamily=_DummyElementFamily)

try:
    from aeroelast.solvers.fsi.rotor import LinearDynamicFSIRotorSolver

    _HAS_FSI = True
except (ImportError, OSError):
    _HAS_FSI = False
    LinearDynamicFSIRotorSolver = None  # type: ignore[assignment,misc]

_skip_fsi = pytest.mark.skipif(not _HAS_FSI, reason="preCICE shared library not available")


@_skip_fsi
def test_rotor_performance_report_creates_output_folder_and_csv(tmp_path):
    solver = object.__new__(LinearDynamicFSIRotorSolver)
    solver.solver_params = {"output_folder": str(tmp_path / "nested" / "results")}
    solver._is_primary_rank = lambda: True

    solver._log_rotor_performance(
        t=0.25,
        omega_rpm=12.5,
        omega_rad=1.308996939,
        alpha=0.0,
        angle_deg=15.0,
        thrust=100.0,
        torque_aero=20.0,
        torque_non_aero=-2.0,
        torque_inertial=1.0,
        torque_gravity=1.0,
        torque_total=18.0,
        power_aero=26.17993878,
        power_total=23.561944902,
        structural_efficiency=0.1,
        cp=0.45,
        cq=0.12,
        ct=0.8,
        tsr=6.5,
        torque_aero_global=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        torque_total_global=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        max_displacement=0.02,
        deformed_radius=7.5,
    )

    csv_path = tmp_path / "nested" / "results" / "rotor_performance.csv"
    assert csv_path.exists()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    assert reader.fieldnames is not None
    assert "Aero Thrust [N]" in reader.fieldnames
    assert "Max Displacement [m]" in reader.fieldnames
    assert float(row["Time [s]"]) == pytest.approx(0.25)
    assert float(row["Aero Torque [Nm]"]) == pytest.approx(20.0)
    assert float(row["Total Torque Z [Nm]"]) == pytest.approx(6.0)
    assert float(row["Max Displacement [m]"]) == pytest.approx(0.02)
    assert float(row["Deformed Radius [m]"]) == pytest.approx(7.5)
