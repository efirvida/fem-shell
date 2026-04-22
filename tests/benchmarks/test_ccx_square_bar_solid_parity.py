"""3D solid cantilever square-bar parity tests: AeroElast core vs CalculiX.

Purpose:
- Distinguish solver/core assembly issues from shell formulation issues.
- Use only solid C3D8/Hexa8 elements in a simple cantilever bar.

Coverage:
- Linear static center/tip displacement under axial and transverse loads.
- Modal first modes frequency parity (if CCX available).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler, modal_solve_coo

from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel


pytestmark = [pytest.mark.slow]


MAT_STEEL = {"type": "solid_3d", "e": 2.1e11, "nu": 0.3, "rho": 7800.0}


@dataclass(frozen=True)
class SolidLoadCase:
    name: str
    nodeset: str
    load3: tuple[float, float, float]


LOAD_CASES: tuple[SolidLoadCase, ...] = (
    SolidLoadCase("free_face_fx", "free_face", (1.0e5, 0.0, 0.0)),
    SolidLoadCase("free_face_fz", "free_face", (0.0, 0.0, 1.0e4)),
)


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


def _build_square_bar_mesh(
    *,
    Lx: float = 1.0,
    Ly: float = 0.05,
    Lz: float = 0.05,
    nx: int = 10,
    ny: int = 2,
    nz: int = 2,
) -> MeshModel:
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(0.0, Lz, nz + 1)

    grid: dict[tuple[int, int, int], Node] = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node = Node([float(x), float(y), float(z)], geometric_node=False)
                mesh.add_node(node)
                grid[(i, j, k)] = node

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = grid[(i, j, k)]
                n100 = grid[(i + 1, j, k)]
                n110 = grid[(i + 1, j + 1, k)]
                n010 = grid[(i, j + 1, k)]
                n001 = grid[(i, j, k + 1)]
                n101 = grid[(i + 1, j, k + 1)]
                n111 = grid[(i + 1, j + 1, k + 1)]
                n011 = grid[(i, j + 1, k + 1)]
                mesh.add_element(
                    MeshElement(
                        nodes=[n000, n100, n110, n010, n001, n101, n111, n011],
                        element_type=ElementType.hexahedron,
                    )
                )

    clamped_nodes = {n for n in mesh.nodes if np.isclose(n.x, 0.0, atol=1e-12)}
    free_face_nodes = {n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12)}
    center = next(
        n
        for n in mesh.nodes
        if np.isclose(n.x, Lx, atol=1e-12)
        and np.isclose(n.y, 0.5 * Ly, atol=1e-12)
        and np.isclose(n.z, 0.5 * Lz, atol=1e-12)
    )
    tip_corner = next(
        n
        for n in mesh.nodes
        if np.isclose(n.x, Lx, atol=1e-12) and np.isclose(n.y, Ly, atol=1e-12) and np.isclose(n.z, Lz, atol=1e-12)
    )

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_face", free_face_nodes))
    mesh.add_node_set(NodeSet("free_center", {center}))
    mesh.add_node_set(NodeSet("free_tip", {tip_corner}))
    mesh.add_element_set(ElementSet("bar", set(mesh.elements)))
    return mesh


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


_FREQ_ROW = re.compile(
    r"^\s*(\d+)\s+([+\-]?\d+[\d.]*(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+[\d.]*(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+[\d.]*(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+[\d.]*(?:[EeDd][+\-]?\d+)?)\s*$"
)


def _parse_ccx_frequencies(dat_file: Path, n_modes: int = 5) -> np.ndarray:
    lines = dat_file.read_text(errors="replace").splitlines()
    out: dict[int, float] = {}
    in_eig = False
    for line in lines:
        up = line.upper()
        if "E I G E N V A L U E" in up and "O U T P U T" in up:
            in_eig = True
            continue
        if not in_eig:
            continue
        m = _FREQ_ROW.match(line)
        if m:
            mode = int(m.group(1))
            freq_hz = float(m.group(4).replace("D", "E"))
            out[mode] = freq_hz
    if not out:
        raise RuntimeError(f"No frequencies parsed from {dat_file}")
    return np.array([out[k] for k in sorted(out)][:n_modes], dtype=float)


_FRD_NODE_ID = re.compile(r"^\s*-1\s+(\d+)\s+(.*)$")
_FRD_FLOAT = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?")


def _parse_frd_node_xyz(line: str) -> tuple[int, float, float, float] | None:
    m = _FRD_NODE_ID.match(line.strip())
    if not m:
        return None
    nid = int(m.group(1))
    vals = [float(tok.replace("D", "E")) for tok in _FRD_FLOAT.findall(m.group(2))]
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


def _frd_disp_at_coord(frd_file: Path, x: float, y: float, z: float) -> np.ndarray:
    disp = _parse_frd_last_disp(frd_file)
    coords = _parse_frd_initial_coords(frd_file)
    available = set(coords).intersection(disp)
    if not available:
        raise RuntimeError("No common FRD nodes between coordinates and displacements")

    min_dist = min(
        (coords[nid][0] - x) ** 2 + (coords[nid][1] - y) ** 2 + (coords[nid][2] - z) ** 2
        for nid in available
    )
    selected = [
        nid
        for nid in available
        if (coords[nid][0] - x) ** 2 + (coords[nid][1] - y) ** 2 + (coords[nid][2] - z) ** 2 <= min_dist + 1e-18
    ]
    vals = [np.asarray(disp[nid], dtype=float) for nid in selected]
    return np.mean(np.vstack(vals), axis=0)


def _build_py_assembler(mesh: MeshModel) -> PyMeshAssembler:
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [208] * len(mesh.elements)  # Hexa8
    mats = [MAT_STEEL] * len(mesh.elements)
    return PyMeshAssembler(node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats)


def _clamped_dofs(mesh: MeshModel, dofs_per_node: int = 3) -> np.ndarray:
    dofs: list[int] = []
    idx = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = idx[node.id] * dofs_per_node
        dofs.extend(range(i0, i0 + dofs_per_node))
    return np.array(sorted(set(dofs)), dtype=np.int64)


def _nodal_force_vector(mesh: MeshModel, case: SolidLoadCase, dofs_per_node: int = 3) -> np.ndarray:
    f = np.zeros(len(mesh.nodes) * dofs_per_node, dtype=float)
    nodes = sorted(mesh.get_node_set(case.nodeset).nodes.values(), key=lambda n: n.id)
    n = max(len(nodes), 1)
    per_node = np.asarray(case.load3, dtype=float) / float(n)
    idx = mesh.node_id_to_index
    for node in nodes:
        i0 = idx[node.id] * dofs_per_node
        f[i0 : i0 + dofs_per_node] += per_node
    return f


def _aero_static_solution(mesh: MeshModel, case: SolidLoadCase) -> np.ndarray:
    asm = _build_py_assembler(mesh)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    f = _nodal_force_vector(mesh, case)

    fixed = _clamped_dofs(mesh, dofs_per_node=3)
    free_mask = np.ones(n, dtype=bool)
    free_mask[fixed] = False
    free = np.where(free_mask)[0]

    u = np.zeros(n, dtype=float)
    u[free] = spsolve(K[free][:, free], f[free])
    return u


def _aero_modal_freqs(mesh: MeshModel, n_modes: int = 5) -> np.ndarray:
    asm = _build_py_assembler(mesh)
    k_rows, k_cols, k_vals = asm.assemble_k()
    m_rows, m_cols, m_vals = asm.assemble_m()
    n = asm.dofs_count
    fixed = _clamped_dofs(mesh, dofs_per_node=3)
    free = np.array([i for i in range(n) if i not in set(fixed)], dtype=np.int64)
    freqs, _modes = modal_solve_coo(
        k_rows.astype(np.int64),
        k_cols.astype(np.int64),
        k_vals.astype(np.float64),
        m_rows.astype(np.int64),
        m_cols.astype(np.int64),
        m_vals.astype(np.float64),
        n,
        free,
        n_modes,
    )
    return np.asarray(freqs, dtype=float)


def _node_disp_from_u(mesh: MeshModel, u: np.ndarray, nodeset: str) -> np.ndarray:
    node = next(iter(mesh.get_node_set(nodeset).nodes.values()))
    i0 = mesh.node_id_to_index[node.id] * 3
    return np.asarray(u[i0 : i0 + 3], dtype=float)


def _write_solid_ccx_inp(
    inp: Path,
    *,
    mesh: MeshModel,
    stem: str,
    solver_type: str,
    load_case: SolidLoadCase | None,
) -> None:
    with inp.open("wt") as f:
        f.write("**\n")
        f.write("** Solid cantilever bar parity case\n")
        f.write("*INCLUDE, INPUT=%s.msh\n" % stem)
        f.write("*INCLUDE, INPUT=%s.nam\n" % stem)
        f.write("**\n")
        f.write("*MATERIAL, NAME=STEEL\n")
        f.write("*ELASTIC\n")
        f.write("2.100000E+11, 0.3\n")
        f.write("*DENSITY\n")
        f.write("7.800000E+03\n")
        f.write("*SOLID SECTION, ELSET=EBAR, MATERIAL=STEEL\n")
        f.write("**\n")
        f.write("*BOUNDARY\n")
        f.write("NCLAMPED, 1, 3, 0.0\n")
        f.write("**\n")
        f.write("*STEP\n")
        if solver_type == "modal":
            f.write("*FREQUENCY\n")
            f.write("5\n")
        else:
            f.write("*STATIC\n")
            if load_case is not None:
                f.write("*CLOAD\n")
                n_nodes = len(mesh.get_node_set(load_case.nodeset).nodes)
                fx, fy, fz = (c / max(n_nodes, 1) for c in load_case.load3)
                if fx != 0.0:
                    f.write(f"N{load_case.nodeset.upper()}, 1, {fx:.6E}\n")
                if fy != 0.0:
                    f.write(f"N{load_case.nodeset.upper()}, 2, {fy:.6E}\n")
                if fz != 0.0:
                    f.write(f"N{load_case.nodeset.upper()}, 3, {fz:.6E}\n")
        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*END STEP\n")


def _ccx_static_displacements(mesh: MeshModel, case: SolidLoadCase, ccx_bin: str, workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    stem = f"solid_lin_{case.name}"
    inp = workdir / f"{stem}.inp"
    write_ccx_mesh(mesh, str(inp), properties=None)
    _write_solid_ccx_inp(inp, mesh=mesh, stem=stem, solver_type="linear", load_case=case)
    _run_ccx(ccx_bin, workdir, stem)
    frd = workdir / f"{stem}.frd"
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    tip = next(iter(mesh.get_node_set("free_tip").nodes.values()))
    return (
        _frd_disp_at_coord(frd, center.x, center.y, center.z),
        _frd_disp_at_coord(frd, tip.x, tip.y, tip.z),
    )


def _ccx_modal(mesh: MeshModel, ccx_bin: str, workdir: Path) -> np.ndarray:
    stem = "solid_modal"
    inp = workdir / f"{stem}.inp"
    write_ccx_mesh(mesh, str(inp), properties=None)
    _write_solid_ccx_inp(inp, mesh=mesh, stem=stem, solver_type="modal", load_case=None)
    _run_ccx(ccx_bin, workdir, stem)
    return _parse_ccx_frequencies(workdir / f"{stem}.dat", n_modes=5)


class TestSquareBarSolidCCXParity:
    def test_linear_static_center_tip_displacement(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_square_bar_mesh()
        tol = 0.08
        failures: list[str] = []

        for case in LOAD_CASES:
            u = _aero_static_solution(mesh, case)
            ae_center = _node_disp_from_u(mesh, u, "free_center")
            ae_tip = _node_disp_from_u(mesh, u, "free_tip")

            ccx_center, ccx_tip = _ccx_static_displacements(mesh, case, ccx_bin, tmp_path)

            for loc, ae, ccx in (("center", ae_center, ccx_center), ("tip", ae_tip, ccx_tip)):
                ae_m = float(np.linalg.norm(ae))
                ccx_m = float(np.linalg.norm(ccx))
                rel = abs(ae_m - ccx_m) / max(abs(ccx_m), 1e-14)
                if rel > tol:
                    failures.append(
                        f"linear/{case.name}/{loc} ae={ae_m:.6e} ccx={ccx_m:.6e} rel={rel:.3%} tol={tol:.1%}"
                    )

        if failures:
            pytest.fail("3D solid linear static parity exceeded tolerance:\n" + "\n".join(failures))

    def test_modal_first_modes(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_square_bar_mesh()
        tol = 0.08

        ae = _aero_modal_freqs(mesh, n_modes=5)
        ccx = _ccx_modal(mesh, ccx_bin, tmp_path)
        rel = np.abs(ccx - ae) / np.maximum(np.abs(ccx), 1e-14)
        if np.any(rel > tol):
            pytest.fail(
                "3D solid modal parity exceeded tolerance: "
                f"ae={ae.tolist()} ccx={ccx.tolist()} rel={rel.tolist()} tol={tol}"
            )
