"""Tests for structural_report.csv logging in the FSI solver."""

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
    from aeroelast.solvers.fsi.linear_dynamic import LinearDynamicFSISolver

    _HAS_FSI = True
except (ImportError, OSError):
    _HAS_FSI = False
    LinearDynamicFSISolver = None  # type: ignore[assignment,misc]

_skip_fsi = pytest.mark.skipif(not _HAS_FSI, reason="preCICE shared library not available")


@_skip_fsi
def test_structural_report_logs_max_displacement_components(tmp_path):
    solver = object.__new__(LinearDynamicFSISolver)
    solver.solver_params = {"output_folder": str(tmp_path)}
    solver.domain = SimpleNamespace(
        nodes=[
            SimpleNamespace(id=10, x=0.0, y=0.0, z=0.0),
            SimpleNamespace(id=20, x=1.0, y=2.0, z=3.0),
        ]
    )
    solver._is_primary_rank = lambda: True

    u_full = np.array(
        [
            1.0e-2,
            2.0e-2,
            3.0e-2,
            0.0,
            0.0,
            0.0,
            4.0e-1,
            -5.0e-1,
            6.0e-1,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    v_full = np.zeros_like(u_full)
    a_full = np.zeros_like(u_full)
    stress_fields = {
        "TOP_von_mises": np.array([1.0, 2.0], dtype=np.float64),
        "MID_von_mises": np.array([3.0, 4.0], dtype=np.float64),
        "BOT_von_mises": np.array([5.0, 6.0], dtype=np.float64),
        "TOP_sigma_1": np.array([7.0, 8.0], dtype=np.float64),
    }

    solver._log_structural_report(
        t=0.25,
        time_step=3,
        u_full=u_full,
        v_full=v_full,
        a_full=a_full,
        stress_fields=stress_fields,
        applied_force_mag=12.5,
    )

    csv_path = tmp_path / "structural_report.csv"
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    assert reader.fieldnames is not None
    assert "Max Disp X [m]" in reader.fieldnames
    assert "Max Disp Y [m]" in reader.fieldnames
    assert "Max Disp Z [m]" in reader.fieldnames
    assert float(row["Max Disp [m]"]) == pytest.approx(np.linalg.norm([0.4, -0.5, 0.6]))
    assert float(row["Max Disp X [m]"]) == pytest.approx(0.4)
    assert float(row["Max Disp Y [m]"]) == pytest.approx(-0.5)
    assert float(row["Max Disp Z [m]"]) == pytest.approx(0.6)
    assert row["Max Disp Node"] == "20"
    assert row["Max Disp Pos (x;y;z)"] == "1.0000;2.0000;3.0000"
