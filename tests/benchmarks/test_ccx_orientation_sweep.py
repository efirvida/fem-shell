"""Orientation and load-scaling sweep for shell plate FY/FZ parity: AeroElast vs CCX.

Comprehensive test matrix:
- Orientation sweep: span_direction at 0°, 30°, 45°, 60°, 90° from mesh X-axis
- Nonlinear load scaling: 0.1x, 0.25x, 0.5x, 1.0x
- FY/FZ load cases (pure transverse, most sensitive to drilling/shear)
- Materials: isotropic + composite laminate
- CCX order: linear + quadratic
- Tolerance: 20% for FY/FZ (liberal while diagnostic)

Consolidates FY/Uy error table by orientation × load_configuration.

Drilling/shear diagnostics:
- Identifies dominant error patterns (membrane-dominant vs bending-dominant)
- Proposes adjustments: drilling_scale, shear_correction
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from aeroelast.core.bc import DirichletCondition, NodalLoad
from aeroelast.core.laminate import create_laminate_from_angles
from aeroelast.core.material import IsotropicMaterial, OrthotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import CompositeShellProperty, ShellProperty
from aeroelast.elements import ElementFamily
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver
from aeroelast.solvers.elasticity.static_nonlinear import StaticNonlinearSolver


pytestmark = [pytest.mark.slow]


# ============================================================================
# Material and geometry constants
# ============================================================================

MAT_ISO = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.3, rho=7800.0)
MAT_ORTHO = OrthotropicMaterial(
    name="CarbonEpoxy",
    E=(181e9, 10.3e9, 10.3e9),
    G=(7.17e9, 3.78e9, 7.17e9),
    nu=(0.28, 0.28, 0.28),
    rho=1600.0,
)


# ============================================================================
# Configuration sweep space
# ============================================================================

# Orientation sweep: (angle_deg, label)
ORIENTATION_SWEEP: tuple[tuple[float, str], ...] = (
    (0.0, "0deg"),
    (30.0, "30deg"),
    (45.0, "45deg"),
    (60.0, "60deg"),
    (90.0, "90deg"),
)

# Load scaling sweep (nonlinear FY/FZ only)
LOAD_SCALE_SWEEP: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

# Nonlinear step controls
NL_STEP_CONTROLS = {
    "nl_initial_increment": 0.05,
    "nl_min_increment": 1.0e-7,
    "nl_max_increment": 0.1,
    "nl_max_increments": 400,
}


# ============================================================================
# Data classes for error table
# ============================================================================

@dataclass
class ErrorEntry:
    orientation_deg: float
    load_scale: float
    material: str
    ccx_order: str
    case: str
    ae_norm: float
    ccx_norm: float
    rel_err: float
    ux_ae: float
    ux_ccx: float
    uy_ae: float
    uy_ccx: float
    uz_ae: float
    uz_ccx: float
    ux_rel: float
    uy_rel: float
    uz_rel: float
    failure: Optional[str] = None


# Global error table — populated by tests for consolidation
_ERROR_TABLE: list[ErrorEntry] = []
_ERROR_TABLE_LOCK = __name__ + ".error_table"


# ============================================================================
# Helpers
# ============================================================================

def _ccx_bin_or_skip() -> str:
    ccx_bin = os.environ.get("CCX_BIN")
    if not ccx_bin:
        pytest.skip("CCX_BIN not set; skipping CalculiX parity tests")
    p = Path(ccx_bin)
    if not p.exists():
        pytest.skip(f"CCX_BIN path does not exist: {ccx_bin}")
    if not os.access(p, os.X_OK):
        pytest.skip(f"CCX_BIN is not executable: {ccx_bin}")
    return ccx_bin


def _span_direction(angle_deg: float) -> tuple[float, float, float]:
    """Return span direction rotated *angle_deg* from the mesh X-axis, in the XY plane.
    
    A cantilever plate with span_direction aligned with the structural axis.
    Angle is measured from global X-axis in degrees.
    """
    theta = np.radians(angle_deg)
    return (float(np.cos(theta)), float(np.sin(theta)), 0.0)


def _build_cantilever_mesh(
    *,
    nx: int = 8,
    ny: int = 4,
    Lx: float = 1.0,
    Ly: float = 0.1,
) -> MeshModel:
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)

    grid: dict[tuple[int, int], Node] = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(node)
            grid[(i, j)] = node

    for j in range(ny):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[
                        grid[(i, j)],
                        grid[(i + 1, j)],
                        grid[(i + 1, j + 1)],
                        grid[(i, j + 1)],
                    ],
                    element_type=ElementType.quad,
                )
            )

    clamped_nodes = {n for n in mesh.nodes if np.isclose(n.x, 0.0, atol=1e-12)}
    free_nodes = {n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12)}

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_edge", free_nodes))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))
    return mesh


def _center_free_edge_node(mesh: MeshModel, *, Lx: float = 1.0, Ly: float = 0.1) -> Node:
    target_y = 0.5 * Ly
    candidates = [n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12)]
    return min(candidates, key=lambda n: abs(float(n.y) - target_y))


def _model_cfg(
    material_kind: str,
    span_dir: tuple[float, float, float],
    *,
    drilling_scale: float = 1.0,
    shear_correction: float = 5.0 / 6.0,
):
    if material_kind == "isotropic":
        prop = ShellProperty(
            material=MAT_ISO,
            thickness=1.0e-3,
            shear_correction=shear_correction,
        )
    elif material_kind == "composite":
        lam = create_laminate_from_angles(
            MAT_ORTHO,
            0.125e-3,
            [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0],
        )
        # Note: shear_correction not supported for CompositeShellProperty yet
        prop = CompositeShellProperty(
            laminate=lam,
        )
    else:
        raise ValueError(f"Unknown material kind: {material_kind}")

    return {
        "solver": {
            "atol": 1e-10,
            "rtol": 1e-8,
            "stol": 1e-8,
            "max_it": 60,
        },
        "elements": {
            "element_family": ElementFamily.SHELL,
            "properties": {"plate": prop},
            "span_direction": span_dir,
            "drilling_scale": drilling_scale,
            "shear_correction": shear_correction,
        },
    }


def _clamped_dofs(mesh: MeshModel, dofs_per_node: int) -> list[int]:
    dofs: list[int] = []
    m = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = m[node.id] * dofs_per_node
        dofs.extend(range(i0, i0 + dofs_per_node))
    return sorted(set(dofs))


def _nodal_loads(
    mesh: MeshModel,
    dofs_per_node: int,
    load6: tuple[float, float, float, float, float, float],
    nodeset: str = "free_edge",
) -> list[NodalLoad]:
    nodes = sorted(mesh.get_node_set(nodeset).nodes.values(), key=lambda n: n.id)
    n = max(len(nodes), 1)
    per_node = np.asarray(load6, dtype=float) / float(n)
    idx = mesh.node_id_to_index
    out: list[NodalLoad] = []
    for node in nodes:
        i0 = idx[node.id] * dofs_per_node
        dofs = list(range(i0, i0 + dofs_per_node))
        out.append(NodalLoad(dofs, per_node.tolist()))
    return out


def _aero_linear_disp(
    mesh: MeshModel,
    cfg: dict,
    case: str,
    load_scale: float = 1.0,
) -> np.ndarray:
    """Return translational displacement at center free edge for FY or FZ load case."""
    solver = StaticLinearSolver(mesh, cfg)
    dpn = solver.domain.dofs_per_node
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)])

    # FY load (pure Y direction)
    if case == "fy":
        loads = _nodal_loads(
            mesh, dpn, (0.0, 600.0 * load_scale, 0.0, 0.0, 0.0, 0.0)
        )
    elif case == "fz":
        loads = _nodal_loads(
            mesh, dpn, (0.0, 0.0, 600.0 * load_scale, 0.0, 0.0, 0.0)
        )
    else:
        raise ValueError(f"Unknown case: {case}")

    solver.add_nodal_loads(loads)
    u = solver.solve()
    arr = np.asarray(u.array, dtype=float)
    c = _center_free_edge_node(mesh)
    i0 = mesh.node_id_to_index[c.id] * dpn
    return arr[i0 : i0 + 3]


def _aero_nonlinear_disp(
    mesh: MeshModel,
    cfg: dict,
    case: str,
    load_scale: float = 1.0,
) -> np.ndarray:
    """Return translational displacement at center free edge (nonlinear, FY or FZ)."""
    solver = StaticNonlinearSolver(mesh, cfg)
    dpn = solver.domain.dofs_per_node
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)])

    if case == "fy":
        loads = _nodal_loads(
            mesh, dpn, (0.0, 600.0 * load_scale, 0.0, 0.0, 0.0, 0.0)
        )
    elif case == "fz":
        loads = _nodal_loads(
            mesh, dpn, (0.0, 0.0, 600.0 * load_scale, 0.0, 0.0, 0.0)
        )
    else:
        raise ValueError(f"Unknown case: {case}")

    solver.add_nodal_loads(loads)
    u = np.asarray(solver.solve(), dtype=float)
    c = _center_free_edge_node(mesh)
    i0 = mesh.node_id_to_index[c.id] * dpn
    return u[i0 : i0 + 3]


def _run_ccx(ccx_bin: str, workdir: Path, stem: str) -> None:
    proc = subprocess.run(
        [ccx_bin, stem],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CCX run failed for {stem} (code={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


_FRD_NODE_ID = re.compile(r"^\s*-1\s+(\d+)\s+(.*)$")
_FRD_FLOAT = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?")


def _parse_frd_node_xyz(line: str) -> tuple[int, float, float, float] | None:
    m = _FRD_NODE_ID.match(line.strip())
    if not m:
        return None
    nid = int(m.group(1))
    tail = m.group(2)
    vals = [float(tok.replace("D", "E")) for tok in _FRD_FLOAT.findall(tail)]
    if len(vals) < 3:
        return None
    return nid, vals[0], vals[1], vals[2]


def _parse_frd_last_disp(frd_file: Path) -> dict[int, tuple[float, float, float]]:
    last: dict[int, tuple[float, float, float]] = {}
    cur: dict[int, tuple[float, float, float]] = {}
    in_disp = False

    for raw in frd_file.read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("-4") and "DISP" in s.upper():
            in_disp = True
            cur = {}
            continue
        if s == "-3":
            if in_disp and cur:
                last = cur
            in_disp = False
            continue
        if in_disp and s.startswith("-1"):
            parsed = _parse_frd_node_xyz(s)
            if parsed is None:
                continue
            nid, ux, uy, uz = parsed
            cur[nid] = (ux, uy, uz)

    if not last:
        raise RuntimeError(f"No DISP block parsed from {frd_file}")
    return last


def _parse_frd_initial_coords(frd_file: Path) -> dict[int, tuple[float, float, float]]:
    coords: dict[int, tuple[float, float, float]] = {}
    in_coords = False
    for raw in frd_file.read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("2C"):
            in_coords = True
            continue
        if in_coords and s == "-3":
            break
        if in_coords and s.startswith("-1"):
            parsed = _parse_frd_node_xyz(s)
            if parsed is None:
                continue
            nid, x, y, z = parsed
            coords[nid] = (x, y, z)
    if not coords:
        raise RuntimeError(f"No coordinate block parsed from {frd_file}")
    return coords


def _ccx_center_free_edge_disp(
    frd_file: Path,
    *,
    Lx: float = 1.0,
    Ly: float = 0.1,
) -> np.ndarray:
    """Extract center free-edge displacement from FRD by geometric matching."""
    disp = _parse_frd_last_disp(frd_file)
    coords = _parse_frd_initial_coords(frd_file)

    target_y = 0.5 * Ly
    available = set(coords).intersection(disp)
    x_candidates = [nid for nid in available if np.isclose(coords[nid][0], Lx, atol=1e-9)]
    if not x_candidates:
        x_candidates = [nid for nid, (x, _y, _z) in coords.items() if np.isclose(x, Lx, atol=1e-9)]
        if not x_candidates:
            raise RuntimeError(f"No FRD nodes found on free edge x={Lx}")

    min_dy = min(abs(coords[nid][1] - target_y) for nid in x_candidates)
    y_candidates = [nid for nid in x_candidates if abs(coords[nid][1] - target_y) <= (min_dy + 1e-9)]
    disp_candidates = [np.asarray(disp[nid], dtype=float) for nid in y_candidates if nid in disp]
    if not disp_candidates:
        raise RuntimeError("No displacement records found for center free-edge FRD nodes")
    return np.mean(np.vstack(disp_candidates), axis=0)


def _ccx_static_disp(
    mesh: MeshModel,
    cfg: dict,
    case: str,
    load_scale: float,
    quadratic: bool,
    nonlinear: bool,
    ccx_bin: str,
    workdir: Path,
) -> np.ndarray:
    """Run CCX and extract center free-edge displacement."""
    kind = "nl" if nonlinear else "lin"
    load_vector: list[float]
    if case == "fy":
        load_vector = [0.0, 600.0 * load_scale, 0.0, 0.0, 0.0, 0.0]
    elif case == "fz":
        load_vector = [0.0, 0.0, 600.0 * load_scale, 0.0, 0.0, 0.0]
    else:
        raise ValueError(f"Unknown case: {case}")

    inp = workdir / f"{kind}_{case}_s{load_scale:.2f}_{'q2' if quadratic else 'q1'}.inp"
    write_ccx_mesh(
        mesh,
        str(inp),
        properties=cfg["elements"]["properties"],
        boundary_nodeset="clamped",
        solver_type="NonlinearStatic" if nonlinear else "LinearStatic",
        load_nodeset="free_edge",
        load_vector=load_vector,
        quadratic=quadratic,
        span_direction=cfg["elements"]["span_direction"],
        **(NL_STEP_CONTROLS if nonlinear else {}),
    )
    _run_ccx(ccx_bin, workdir, inp.stem)
    return _ccx_center_free_edge_disp(workdir / f"{inp.stem}.frd")


def _rel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    den = np.maximum(np.abs(y), 1e-14)
    return np.abs(x - y) / den


# ============================================================================
# Test class
# ============================================================================

class TestOrientationLoadSweepFYFZ:
    """Orientation sweep × load scaling sweep for FY/FZ plate parity."""

    # ---- Linear static: orientation sweep (all load scales at 1.0) ----

    @pytest.mark.parametrize("orientation_deg", [p[0] for p in ORIENTATION_SWEEP], ids=[p[1] for p in ORIENTATION_SWEEP])
    @pytest.mark.parametrize("case", ["fy", "fz"])
    @pytest.mark.parametrize("material_kind", ["isotropic", "composite"])
    def test_linear_orientation_sweep_center_free_edge(self, orientation_deg, case, material_kind, tmp_path: Path):
        """Linear static: orientation sweep for FY/FZ at full load scale."""
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()
        span_dir = _span_direction(orientation_deg)
        cfg = _model_cfg(material_kind, span_dir)

        ae = _aero_linear_disp(mesh, cfg, case, load_scale=1.0)
        ae_norm = float(np.linalg.norm(ae))
        ae_ux, ae_uy, ae_uz = float(ae[0]), float(ae[1]), float(ae[2])

        for quadratic in (False, True):
            try:
                ccx = _ccx_static_disp(
                    mesh, cfg, case, load_scale=1.0,
                    quadratic=quadratic, nonlinear=False,
                    ccx_bin=ccx_bin, workdir=tmp_path,
                )
                ccx_norm = float(np.linalg.norm(ccx))
                ccx_ux, ccx_uy, ccx_uz = float(ccx[0]), float(ccx[1]), float(ccx[2])
                rel_tot = abs(ccx_norm - ae_norm) / max(abs(ccx_norm), 1e-14)
                rel_ux = float(_rel(np.array([ae_ux]), np.array([ccx_ux]))[0])
                rel_uy = float(_rel(np.array([ae_uy]), np.array([ccx_uy]))[0])
                rel_uz = float(_rel(np.array([ae_uz]), np.array([ccx_uz]))[0])

                _ERROR_TABLE.append(ErrorEntry(
                    orientation_deg=orientation_deg,
                    load_scale=1.0,
                    material=material_kind,
                    ccx_order="q2" if quadratic else "q1",
                    case=case,
                    ae_norm=ae_norm,
                    ccx_norm=ccx_norm,
                    rel_err=rel_tot,
                    ux_ae=ae_ux,
                    ux_ccx=ccx_ux,
                    uy_ae=ae_uy,
                    uy_ccx=ccx_uy,
                    uz_ae=ae_uz,
                    uz_ccx=ccx_uz,
                    ux_rel=rel_ux,
                    uy_rel=rel_uy,
                    uz_rel=rel_uz,
                ))
            except RuntimeError as exc:
                _ERROR_TABLE.append(ErrorEntry(
                    orientation_deg=orientation_deg,
                    load_scale=1.0,
                    material=material_kind,
                    ccx_order="q2" if quadratic else "q1",
                    case=case,
                    ae_norm=ae_norm,
                    ccx_norm=np.nan,
                    rel_err=np.nan,
                    ux_ae=ae_ux,
                    ux_ccx=np.nan,
                    uy_ae=ae_uy,
                    uy_ccx=np.nan,
                    uz_ae=ae_uz,
                    uz_ccx=np.nan,
                    ux_rel=np.nan,
                    uy_rel=np.nan,
                    uz_rel=np.nan,
                    failure=str(exc).splitlines()[0],
                ))

    # ---- Nonlinear static: load scaling sweep (FY/FZ, 0deg only = aligned with mesh axes) ----

    @pytest.mark.parametrize("load_scale", LOAD_SCALE_SWEEP, ids=[f"{s:.2f}x" for s in LOAD_SCALE_SWEEP])
    @pytest.mark.parametrize("case", ["fy", "fz"])
    @pytest.mark.parametrize("material_kind", ["isotropic", "composite"])
    def test_nonlinear_load_scale_sweep_center_free_edge(self, load_scale, case, material_kind, tmp_path: Path):
        """Nonlinear static: load scaling sweep for FY/FZ at 0° orientation."""
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()
        # Fixed at 0° orientation (aligned with mesh X-axis)
        span_dir = _span_direction(0.0)
        cfg = _model_cfg(material_kind, span_dir)

        ae = _aero_nonlinear_disp(mesh, cfg, case, load_scale=load_scale)
        ae_norm = float(np.linalg.norm(ae))
        ae_ux, ae_uy, ae_uz = float(ae[0]), float(ae[1]), float(ae[2])

        for quadratic in (False, True):
            try:
                ccx = _ccx_static_disp(
                    mesh, cfg, case, load_scale=load_scale,
                    quadratic=quadratic, nonlinear=True,
                    ccx_bin=ccx_bin, workdir=tmp_path,
                )
                ccx_norm = float(np.linalg.norm(ccx))
                ccx_ux, ccx_uy, ccx_uz = float(ccx[0]), float(ccx[1]), float(ccx[2])
                rel_tot = abs(ccx_norm - ae_norm) / max(abs(ccx_norm), 1e-14)
                rel_ux = float(_rel(np.array([ae_ux]), np.array([ccx_ux]))[0])
                rel_uy = float(_rel(np.array([ae_uy]), np.array([ccx_uy]))[0])
                rel_uz = float(_rel(np.array([ae_uz]), np.array([ccx_uz]))[0])

                _ERROR_TABLE.append(ErrorEntry(
                    orientation_deg=0.0,
                    load_scale=load_scale,
                    material=material_kind,
                    ccx_order="q2" if quadratic else "q1",
                    case=case,
                    ae_norm=ae_norm,
                    ccx_norm=ccx_norm,
                    rel_err=rel_tot,
                    ux_ae=ae_ux,
                    ux_ccx=ccx_ux,
                    uy_ae=ae_uy,
                    uy_ccx=ccx_uy,
                    uz_ae=ae_uz,
                    uz_ccx=ccx_uz,
                    ux_rel=rel_ux,
                    uy_rel=rel_uy,
                    uz_rel=rel_uz,
                ))
            except RuntimeError as exc:
                _ERROR_TABLE.append(ErrorEntry(
                    orientation_deg=0.0,
                    load_scale=load_scale,
                    material=material_kind,
                    ccx_order="q2" if quadratic else "q1",
                    case=case,
                    ae_norm=ae_norm,
                    ccx_norm=np.nan,
                    rel_err=np.nan,
                    ux_ae=ae_ux,
                    ux_ccx=np.nan,
                    uy_ae=ae_uy,
                    uy_ccx=np.nan,
                    uz_ae=ae_uz,
                    uz_ccx=np.nan,
                    ux_rel=np.nan,
                    uy_rel=np.nan,
                    uz_rel=np.nan,
                    failure=str(exc).splitlines()[0],
                ))

    def test_report_error_table(self, tmp_path: Path):
        """Generate FY/Uy error table by orientation × load configuration.
        
        This test always passes; it generates a detailed report for analysis.
        """
        global _ERROR_TABLE
        if not _ERROR_TABLE:
            pytest.skip("No error entries collected (no tests were executed); cannot generate table")

        print("\n" + "=" * 100)
        print("CONSOLIDATED FY/Uy ERROR TABLE")
        print("  Orientation  LoadScale  Material   Order  Case  |u|_AE         |u|_CCX        rel_err    ux_rel    uy_rel    uz_rel  Failure")
        print("-" * 100)

        failures: list[str] = []
        TOL = 0.20  # Liberal tolerance while diagnostic

        for e in _ERROR_TABLE:
            flag = ""
            if e.failure:
                flag = f" [FAIL: {e.failure}]"
            elif e.rel_err > TOL:
                flag = f" [TOL exceeded: {e.rel_err:.1%}]"
                failures.append(
                    f"orient={e.orientation_deg:.0f}deg load={e.load_scale:.2f} "
                    f"{e.material}/{e.ccx_order}/{e.case}: rel={e.rel_err:.3%}"
                )

            print(
                f"  {e.orientation_deg:8.1f}°  "
                f"{e.load_scale:8.2f}x  "
                f"{e.material:10s}  "
                f"{e.ccx_order:4s}  "
                f"{e.case:4s}  "
                f"{e.ae_norm:12.4e}  "
                f"{e.ccx_norm:12.4e}  "
                f"{e.rel_err:7.1%}  "
                f"{e.ux_rel:7.1%}  "
                f"{e.uy_rel:7.1%}  "
                f"{e.uz_rel:7.1%}  "
                f"{flag}"
            )

        print("-" * 100)

        # ---- Diagnostic summary ----
        print("\nDIAGNOSTIC SUMMARY:")
        print("-" * 60)

        # Group by case
        fy_entries = [e for e in _ERROR_TABLE if e.case == "fy"]
        fz_entries = [e for e in _ERROR_TABLE if e.case == "fz"]

        for case_name, entries in [("FY", fy_entries), ("FZ", fz_entries)]:
            valid = [e for e in entries if not np.isnan(e.rel_err)]
            if not valid:
                print(f"  {case_name}: no valid data")
                continue

            mean_rel = float(np.nanmean([e.rel_err for e in valid]))
            max_rel = float(np.nanmax([e.rel_err for e in valid]))
            worst = max(valid, key=lambda e: e.rel_err)

            # Component-level
            ux_mean = float(np.nanmean([e.ux_rel for e in valid]))
            uy_mean = float(np.nanmean([e.uy_rel for e in valid]))
            uz_mean = float(np.nanmean([e.uz_rel for e in valid]))

            print(f"  {case_name}: {len(valid)} valid entries")
            print(f"    mean rel_err = {mean_rel:.2%}, max = {max_rel:.2%}")
            print(f"    mean component rel: ux={ux_mean:.2%}  uy={uy_mean:.2%}  uz={uz_mean:.2%}")
            print(f"    worst: orient={worst.orientation_deg:.0f}° load={worst.load_scale:.2f} {worst.material}/{worst.ccx_order}")
            print(f"           ux_ae={worst.ux_ae:.4e} ux_ccx={worst.ux_ccx:.4e}")
            print(f"           uy_ae={worst.uy_ae:.4e} uy_ccx={worst.uy_ccx:.4e}")
            print(f"           uz_ae={worst.uz_ae:.4e} uz_ccx={worst.uz_ccx:.4e}")

        # ---- Drilling/shear pattern analysis ----
        print("\nDRILLING/SHEAR PATTERN ANALYSIS:")
        print("-" * 60)

        # For FY load: the main transverse direction is Y. The UY component
        # is the dominant displacement. For FZ load: UZ is dominant.
        for case_name, dom_comp, alt_comp in [
            ("FY", "uy_rel", "ux_rel"),
            ("FZ", "uz_rel", "ux_rel"),
        ]:
            entries = [e for e in _ERROR_TABLE if e.case == case_name and not np.isnan(e.rel_err)]
            if not entries:
                continue

            dom_mean = float(np.nanmean([getattr(e, dom_comp) for e in entries]))
            alt_mean = float(np.nanmean([getattr(e, alt_comp) for e in entries]))
            dom_max = float(np.nanmax([getattr(e, dom_comp) for e in entries]))
            alt_max = float(np.nanmax([getattr(e, alt_comp) for e in entries]))

            print(f"  {case_name}: dominant={dom_comp}, alternative={alt_comp}")
            print(f"    dominant: mean={dom_mean:.2%}, max={dom_max:.2%}")
            print(f"    alternative: mean={alt_mean:.2%}, max={alt_max:.2%}")

            # Pattern: if dominant error >> alternative, it's a membrane/bending coupling issue
            # If both are large and similar, it's a general stiffness issue
            if dom_max > TOL and alt_max < TOL * 0.5:
                print(f"  → DIAGNOSIS: dominant-{dom_comp} error suggests drilling or shear coupling")
                print(f"    Recommendation: adjust drilling_scale or shear_correction_factor")
            elif dom_max > TOL and alt_max > TOL * 0.5:
                print(f"  → DIAGNOSIS: multi-component error suggests general membrane/bending stiffness mismatch")
                print(f"    Recommendation: check material constants (E, nu) and section thickness parity")

        print("=" * 100)

        if failures:
            print("\nFAILURES (rel_err > 20%):")
            for f in failures:
                print(f"  - {f}")

        # Always pass — this is a report-only test
        # Store table for downstream analysis
        return


# ============================================================================
# Diagnostic helper: propose drilling/shear adjustments
# ============================================================================

def _propose_drilling_adjustment(error_table: list[ErrorEntry]) -> dict:
    """Analyze error table and propose drilling/shear parameter adjustments.
    
    Returns a dict with:
    - recommended_drilling_scale: suggested modification to drilling_scale
    - recommended_shear_correction: suggested modification to shear_correction_factor
    - reasoning: text explaining the recommendation
    """
    valid = [e for e in error_table if not np.isnan(e.rel_err) and not e.failure]
    if not valid:
        return {
            "drilling_scale": None,
            "shear_correction": None,
            "reasoning": "No valid data in error table",
        }

    # Analyze dominant error patterns
    fy_entries = [e for e in valid if e.case == "fy"]
    fz_entries = [e for e in valid if e.case == "fz"]

    recommendations: dict = {
        "drilling_scale": None,
        "shear_correction": None,
        "reasoning": "",
    }

    if fy_entries:
        fy_worst = max(fy_entries, key=lambda e: e.uy_rel)
        fy_dominant = float(np.nanmean([e.uy_rel for e in fy_entries]))
        fy_alternative = float(np.nanmean([e.ux_rel for e in fy_entries]))

        if fy_dominant > 0.10 and fy_dominant > fy_alternative * 1.5:
            # FY dominant error is in UY — suggests in-plane shear/diaphragm coupling
            # Reduce drilling stiffness to lower UY resistance (CCX uses less drilling)
            recommendations["drilling_scale"] = 0.5
            recommendations["reasoning"] += (
                f" FY case shows dominant UY error ({fy_dominant:.1%}) vs UX ({fy_alternative:.1%}). "
                f"Worst case: orient={fy_worst.orientation_deg}° {fy_worst.material}/{fy_worst.ccx_order}. "
                f"Drilling_scale=0.5 suggested to reduce in-plane rotational coupling. "
            )
        elif fy_dominant > 0.15 and fy_alternative > 0.10:
            # Both large — general shear coupling issue
            recommendations["shear_correction"] = 0.6
            recommendations["reasoning"] += (
                f" FY case shows large multi-component error (uy={fy_dominant:.1%}, ux={fy_alternative:.1%}). "
                f"Shear_correction=0.6 suggested to improve shear-rotation coupling. "
            )

    if fz_entries:
        fz_worst = max(fz_entries, key=lambda e: e.uz_rel)
        fz_dominant = float(np.nanmean([e.uz_rel for e in fz_entries]))
        fz_alternative = float(np.nanmean([e.uy_rel for e in fz_entries]))

        if fz_dominant > 0.10:
            # FZ is transverse bending — sensitive to shear locking
            recommendations["reasoning"] += (
                f" FZ case dominant UZ error ({fz_dominant:.1%}). "
                f"Worst: orient={fz_worst.orientation_deg}° {fz_worst.material}/{fz_worst.ccx_order}. "
                f"Check shear_correction: current 5/6 may lock; consider 0.6-0.75 range. "
            )

    return recommendations


class TestDrillingShearAdjustments:
    """Apply and validate drilling/shear parameter adjustments."""

    @pytest.mark.parametrize("case", ["fy", "fz"])
    @pytest.mark.parametrize("material_kind", ["isotropic", "composite"])
    def test_linear_adjusted_drilling_scale(self, case, material_kind, tmp_path: Path):
        """Compare baseline vs adjusted drilling_scale=0.5 for FY/FZ linear static.
        
        Run with both drilling_scale=1.0 (baseline) and drilling_scale=0.5
        (adjusted). Report whether the adjustment improves or degrades parity.
        """
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()
        span_dir = (1.0, 0.0, 0.0)

        cfg_baseline = _model_cfg(material_kind, span_dir, drilling_scale=1.0)
        cfg_adjusted = _model_cfg(material_kind, span_dir, drilling_scale=0.5)

        ae_b = _aero_linear_disp(mesh, cfg_baseline, case, load_scale=1.0)
        ae_a = _aero_linear_disp(mesh, cfg_adjusted, case, load_scale=1.0)
        ae_b_norm = float(np.linalg.norm(ae_b))
        ae_a_norm = float(np.linalg.norm(ae_a))

        results: dict[str, float] = {"baseline": ae_b_norm, "adjusted": ae_a_norm}

        for label, cfg in [("baseline", cfg_baseline), ("adjusted", cfg_adjusted)]:
            try:
                ccx = _ccx_static_disp(
                    mesh, cfg, case, load_scale=1.0,
                    quadratic=False, nonlinear=False,
                    ccx_bin=ccx_bin, workdir=tmp_path,
                )
                results[f"ccx_{label}"] = float(np.linalg.norm(ccx))
            except RuntimeError:
                results[f"ccx_{label}"] = np.nan

        print(
            f"\n[{case}/{material_kind}] drilling_scale comparison:\n"
            f"  AE (baseline drilling) = {ae_b_norm:.6e}\n"
            f"  AE (adjusted drilling) = {ae_a_norm:.6e}\n"
            f"  CCX (baseline drilling) = {results.get('ccx_baseline', np.nan):.6e}\n"
            f"  CCX (adjusted drilling) = {results.get('ccx_adjusted', np.nan):.6e}\n"
            f"  rel baseline = {abs(results['ccx_baseline'] - ae_b_norm) / max(abs(results['ccx_baseline']), 1e-14):.3%}\n"
            f"  rel adjusted = {abs(results['ccx_adjusted'] - ae_a_norm) / max(abs(results['ccx_adjusted']), 1e-14):.3%}"
        )

        # Suggest whether the adjustment improves parity
        rel_b = abs(results["ccx_baseline"] - ae_b_norm) / max(abs(results["ccx_baseline"]), 1e-14)
        rel_a = abs(results["ccx_adjusted"] - ae_a_norm) / max(abs(results["ccx_adjusted"]), 1e-14)
        if not np.isnan(rel_b) and not np.isnan(rel_a):
            delta = rel_b - rel_a
            if delta > 0.01:
                print(f"  → Drilling adjustment IMPROVES parity by {delta:.3%}")
            elif delta < -0.01:
                print(f"  → Drilling adjustment DEGRADES parity by {-delta:.3%}")
            else:
                print(f"  → Drilling adjustment has negligible effect ({delta:.3%})")