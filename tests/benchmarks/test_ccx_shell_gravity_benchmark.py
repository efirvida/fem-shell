"""Shell gravity benchmark parity: AeroElast vs CalculiX (CCX).

Inspired by CalculiX Elements/Shell example (shell.inp + test.py):
- Cantilever-like shell plate
- Fully clamped root edge
- Gravity loading via CCX ``*DLOAD, GRAV``
- Mesh-density sweep with manageable sizes

Coverage:
- Center and tip displacement parity (first-order + quadratic CCX exports)
- Optional bending/rotation stress-proxy parity when FRD exposes rotational DOFs
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from aeroelast.core.bc import BodyForce, DirichletCondition
from aeroelast.core.material import IsotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import ShellProperty
from aeroelast.elements import ElementFamily
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver


pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

_DEFAULT_CCX_BIN = "/scratch/leahk/eduardo.donestevez/venv/bin/ccx"
_FRD_NODE_ID = re.compile(r"^\s*-1\s+(\d+)\s+(.*)$")
_FRD_FLOAT = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?")

MAT_ISO = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.333333333, rho=7850.0)
THICKNESS = 5.0e-3
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


def _ccx_bin_or_skip() -> str:
    ccx_bin = os.environ.get("CCX_BIN", _DEFAULT_CCX_BIN)
    p = Path(ccx_bin)
    if not p.exists():
        pytest.skip(f"CCX binary not found: {ccx_bin}")
    if not os.access(p, os.X_OK):
        pytest.skip(f"CCX binary is not executable: {ccx_bin}")
    return ccx_bin


def _build_shell_mesh(*, nx: int, ny: int, Lx: float = 1.0, Ly: float = 0.2) -> MeshModel:
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

    clamped = {n for n in mesh.nodes if np.isclose(n.x, 0.0, atol=1e-12)}
    free_edge = {n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12)}
    free_center = min(free_edge, key=lambda n: abs(float(n.y) - 0.5 * Ly))
    free_tip = max(free_edge, key=lambda n: float(n.y))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_edge", free_edge))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_node_set(NodeSet("free_tip", {free_tip}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))
    return mesh


def _model_cfg() -> dict:
    return {
        "solver": {"atol": 1e-10, "rtol": 1e-8},
        "elements": {
            "element_family": ElementFamily.SHELL,
            "properties": {"plate": ShellProperty(material=MAT_ISO, thickness=THICKNESS)},
            "span_direction": (1.0, 0.0, 0.0),
        },
    }


def _clamped_dofs(mesh: MeshModel, dofs_per_node: int) -> list[int]:
    out: list[int] = []
    idx = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = idx[node.id] * dofs_per_node
        out.extend(range(i0, i0 + dofs_per_node))
    return sorted(set(out))


def _aero_center_tip(mesh: MeshModel) -> tuple[np.ndarray, np.ndarray]:
    cfg = _model_cfg()
    solver = StaticLinearSolver(mesh, cfg)
    dpn = solver.domain.dofs_per_node
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)])
    solver.add_body_forces([BodyForce(GRAVITY)])

    u = np.asarray(solver.solve().array, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    tip = next(iter(mesh.get_node_set("free_tip").nodes.values()))
    ic = mesh.node_id_to_index[center.id] * dpn
    it = mesh.node_id_to_index[tip.id] * dpn
    return u[ic : ic + dpn], u[it : it + dpn]


def _patch_inp_with_gravity(inp_file: Path, *, gravity: np.ndarray) -> None:
    text = inp_file.read_text()
    grav = f"*DLOAD\nEPLATE,GRAV,{abs(float(gravity[2])):.6E},{float(gravity[0]):.6E},{float(gravity[1]):.6E},{float(gravity[2]):.6E}\n"
    marker = "*NODE FILE\n"
    if marker not in text:
        raise RuntimeError(f"Cannot inject *DLOAD, marker '{marker.strip()}' not found in {inp_file}")
    text = text.replace(marker, grav + marker, 1)
    inp_file.write_text(text)


def _run_ccx(ccx_bin: str, workdir: Path, stem: str) -> None:
    proc = subprocess.run([ccx_bin, stem], cwd=workdir, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CCX run failed for {stem} (code={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _parse_frd_components(line: str) -> tuple[int, np.ndarray] | None:
    m = _FRD_NODE_ID.match(line.strip())
    if not m:
        return None
    nid = int(m.group(1))
    vals = [float(tok.replace("D", "E")) for tok in _FRD_FLOAT.findall(m.group(2))]
    if len(vals) < 3:
        return None
    return nid, np.asarray(vals, dtype=float)


def _parse_frd_initial_coords(frd_file: Path) -> dict[int, np.ndarray]:
    coords: dict[int, np.ndarray] = {}
    in_coords = False
    for raw in frd_file.read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("2C"):
            in_coords = True
            continue
        if in_coords and s == "-3":
            break
        if in_coords and s.startswith("-1"):
            parsed = _parse_frd_components(s)
            if parsed is None:
                continue
            nid, vals = parsed
            coords[nid] = vals[:3]
    if not coords:
        raise RuntimeError(f"No coordinate block parsed from {frd_file}")
    return coords


def _parse_frd_last_disp(frd_file: Path) -> dict[int, np.ndarray]:
    last: dict[int, np.ndarray] = {}
    cur: dict[int, np.ndarray] = {}
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
            parsed = _parse_frd_components(s)
            if parsed is None:
                continue
            nid, vals = parsed
            cur[nid] = vals

    if not last:
        raise RuntimeError(f"No DISP block parsed from {frd_file}")
    return last


def _frd_result_at_xy(frd_file: Path, *, x: float, y: float, atol: float = 1e-9) -> np.ndarray:
    disp = _parse_frd_last_disp(frd_file)
    coords = _parse_frd_initial_coords(frd_file)
    available = set(disp).intersection(coords)
    if not available:
        raise RuntimeError("No common nodes in FRD coord/disp blocks")

    candidates = [nid for nid in available if np.isclose(coords[nid][0], x, atol=atol)]
    if not candidates:
        raise RuntimeError(f"No FRD nodes found at x={x}")

    min_dy = min(abs(float(coords[nid][1]) - y) for nid in candidates)
    selected = [nid for nid in candidates if abs(float(coords[nid][1]) - y) <= (min_dy + atol)]
    vals = [disp[nid] for nid in selected if nid in disp]
    if not vals:
        raise RuntimeError("No FRD displacement records for selected geometric location")
    return np.mean(np.vstack(vals), axis=0)


def _ccx_center_tip(mesh: MeshModel, *, quadratic: bool, ccx_bin: str, workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = _model_cfg()
    stem = f"shell_gravity_{'q2' if quadratic else 'q1'}"
    inp = workdir / f"{stem}.inp"
    write_ccx_mesh(
        mesh,
        str(inp),
        properties=cfg["elements"]["properties"],
        boundary_nodeset="clamped",
        solver_type="LinearStatic",
        quadratic=quadratic,
        span_direction=(1.0, 0.0, 0.0),
    )
    _patch_inp_with_gravity(inp, gravity=GRAVITY)
    _run_ccx(ccx_bin, workdir, stem)

    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    tip = next(iter(mesh.get_node_set("free_tip").nodes.values()))
    frd = workdir / f"{stem}.frd"
    ccx_center = _frd_result_at_xy(frd, x=float(center.x), y=float(center.y))
    ccx_tip = _frd_result_at_xy(frd, x=float(tip.x), y=float(tip.y))
    return ccx_center, ccx_tip


MESH_SWEEP = [
    pytest.param((6, 2), id="mesh_6x2"),
    pytest.param((12, 4), id="mesh_12x4"),
]


class TestCCXShellGravityBenchmark:
    @pytest.mark.parametrize("nx_ny", MESH_SWEEP)
    def test_gravity_displacement_parity_mesh_sweep(self, nx_ny: tuple[int, int], tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        nx, ny = nx_ny
        mesh = _build_shell_mesh(nx=nx, ny=ny)

        ae_center, ae_tip = _aero_center_tip(mesh)
        center_tol = 0.12
        tip_tol = 0.12
        proxy_tol = 0.20
        failures: list[str] = []

        for quadratic in (False, True):
            ccx_center, ccx_tip = _ccx_center_tip(mesh, quadratic=quadratic, ccx_bin=ccx_bin, workdir=tmp_path)

            for label, ae, ccx, tol in (
                ("center", ae_center[:3], ccx_center[:3], center_tol),
                ("tip", ae_tip[:3], ccx_tip[:3], tip_tol),
            ):
                ae_m = float(np.linalg.norm(ae))
                ccx_m = float(np.linalg.norm(ccx))
                rel = abs(ae_m - ccx_m) / max(abs(ccx_m), 1e-14)
                if rel > tol:
                    failures.append(
                        f"{label}/mesh={nx}x{ny}/{'q2' if quadratic else 'q1'} "
                        f"|u|_ae={ae_m:.6e} |u|_ccx={ccx_m:.6e} rel={rel:.3%} tol={tol:.1%}"
                    )

            # Rotation-based stress proxy (only when FRD exposes rotational shell DOFs).
            if ae_center.size >= 6 and ccx_center.size >= 6 and ae_tip.size >= 6 and ccx_tip.size >= 6:
                ae_proxy = float(np.linalg.norm(ae_tip[3:6] - ae_center[3:6]))
                ccx_proxy = float(np.linalg.norm(ccx_tip[3:6] - ccx_center[3:6]))
                rel_proxy = abs(ae_proxy - ccx_proxy) / max(abs(ccx_proxy), 1e-14)
                if rel_proxy > proxy_tol:
                    failures.append(
                        f"rot_proxy/mesh={nx}x{ny}/{'q2' if quadratic else 'q1'} "
                        f"ae={ae_proxy:.6e} ccx={ccx_proxy:.6e} rel={rel_proxy:.3%} tol={proxy_tol:.1%}"
                    )

            print(
                "[shell-gravity] "
                f"mesh={nx}x{ny} {'q2' if quadratic else 'q1'} "
                f"center_ae={np.linalg.norm(ae_center[:3]):.6e} center_ccx={np.linalg.norm(ccx_center[:3]):.6e} "
                f"tip_ae={np.linalg.norm(ae_tip[:3]):.6e} tip_ccx={np.linalg.norm(ccx_tip[:3]):.6e}"
            )

        if failures:
            pytest.fail("Shell gravity parity exceeded tolerance:\n" + "\n".join(failures))
