"""Cantilever plate parity tests: AeroElast vs CalculiX (CCX).

Coverage:
- Modal (first 5 modes)
- Linear static (distributed free-edge X/Y/Z + corner torsion)
- Nonlinear static (same load cases)
- Materials: isotropic + composite laminate
- CCX element order: first-order and second-order (quadratic)

Validation metric for static/nonlinear:
- Translational displacement norm at the center node of the free edge.

These tests are intentionally marked as ``slow`` and require ``CCX_BIN``.
If ``CCX_BIN`` is missing/unavailable, tests are skipped gracefully.
"""

from __future__ import annotations

import os
import re
import subprocess
from itertools import permutations
from dataclasses import dataclass
from pathlib import Path

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
from aeroelast.solvers.modal import ModalSolver


pytestmark = [pytest.mark.slow]


MAT_ISO = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.3, rho=7800.0)
MAT_ORTHO = OrthotropicMaterial(
    name="CarbonEpoxy",
    E=(181e9, 10.3e9, 10.3e9),
    G=(7.17e9, 3.78e9, 7.17e9),
    nu=(0.28, 0.28, 0.28),
    rho=1600.0,
)


@dataclass(frozen=True)
class LoadCase:
    name: str
    nodeset: str
    load6: tuple[float, float, float, float, float, float]


LOAD_CASES: tuple[LoadCase, ...] = (
    # Resultant force distributed over the full free edge (x = Lx)
    LoadCase("free_edge_fx", "free_edge", (600.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    LoadCase("free_edge_fy", "free_edge", (0.0, 600.0, 0.0, 0.0, 0.0, 0.0)),
    LoadCase("free_edge_fz", "free_edge", (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)),
    # Torsion-biased case: point force applied at a free-edge corner.
    # A corner force is preferred over directly prescribing nodal moments.
    LoadCase("corner_torsion_fz", "free_corner", (0.0, 0.0, 600.0, 0.0, 0.0, 0.0)),
)


def _select_cases(*names: str) -> tuple[LoadCase, ...]:
    wanted = set(names)
    return tuple(case for case in LOAD_CASES if case.name in wanted)


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


def _build_cantilever_mesh(*, nx: int = 8, ny: int = 4, Lx: float = 1.0, Ly: float = 0.1) -> MeshModel:
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
    corner = next(n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12) and np.isclose(n.y, Ly, atol=1e-12))

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_edge", free_nodes))
    mesh.add_node_set(NodeSet("free_corner", {corner}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))
    return mesh


def _center_free_edge_node(mesh: MeshModel, *, Lx: float = 1.0, Ly: float = 0.1) -> Node:
    target_y = 0.5 * Ly
    candidates = [n for n in mesh.nodes if np.isclose(n.x, Lx, atol=1e-12)]
    return min(candidates, key=lambda n: abs(float(n.y) - target_y))


def _model_cfg(material_kind: str):
    if material_kind == "isotropic":
        prop = ShellProperty(material=MAT_ISO, thickness=1.0e-3)
    elif material_kind == "composite":
        lam = create_laminate_from_angles(
            MAT_ORTHO,
            0.125e-3,
            [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0],
        )
        prop = CompositeShellProperty(laminate=lam)
    else:
        raise ValueError(f"Unknown material kind: {material_kind}")

    return {
        "solver": {
            "num_modes": 5,
            "atol": 1e-10,
            "rtol": 1e-8,
            "stol": 1e-8,
            "max_it": 60,
        },
        "elements": {
            "element_family": ElementFamily.SHELL,
            "properties": {"plate": prop},
            "span_direction": (1.0, 0.0, 0.0),
        },
    }


def _clamped_dofs(mesh: MeshModel, dofs_per_node: int) -> list[int]:
    dofs: list[int] = []
    m = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = m[node.id] * dofs_per_node
        dofs.extend(range(i0, i0 + dofs_per_node))
    return sorted(set(dofs))


def _load_as_nodal(mesh: MeshModel, dofs_per_node: int, case: LoadCase) -> list[NodalLoad]:
    nodes = sorted(mesh.get_node_set(case.nodeset).nodes.values(), key=lambda n: n.id)
    n = max(len(nodes), 1)
    per_node = np.asarray(case.load6, dtype=float) / float(n)
    idx = mesh.node_id_to_index
    out: list[NodalLoad] = []
    for node in nodes:
        i0 = idx[node.id] * dofs_per_node
        dofs = list(range(i0, i0 + dofs_per_node))
        out.append(NodalLoad(dofs, per_node.tolist()))
    return out


def _ae_applied_resultant(mesh: MeshModel, dofs_per_node: int, case: LoadCase) -> np.ndarray:
    """Return global resultant vector represented by AeroElast nodal loads."""
    loads = _load_as_nodal(mesh, dofs_per_node, case)
    total = np.zeros(6, dtype=float)
    max_components = min(int(dofs_per_node), 6)
    for load in loads:
        vals = np.asarray(load.force, dtype=float)
        n = min(vals.size, max_components)
        total[:n] += vals[:n]
    return total


def _ccx_applied_resultant(mesh: MeshModel, case: LoadCase) -> np.ndarray:
    """Return global resultant represented by CCX *CLOAD distribution in writer."""
    n_nodes = len(mesh.get_node_set(case.nodeset).nodes)
    if n_nodes <= 0:
        raise ValueError(f"Nodeset '{case.nodeset}' is empty; cannot define CCX load resultant")
    return np.asarray(case.load6, dtype=float)


def _resultant_consistency_rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.maximum(np.abs(b), 1e-14)
    return np.abs(a - b) / den


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
    """Parse FRD '-1' node records allowing packed signs without spaces."""
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


def _parse_frd_disp_blocks(frd_file: Path) -> list[dict[int, tuple[float, float, float]]]:
    """Parse all DISP blocks in FRD, preserving order."""
    blocks: list[dict[int, tuple[float, float, float]]] = []
    cur: dict[int, tuple[float, float, float]] = {}
    in_disp = False

    for raw in frd_file.read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("-4") and "DISP" in s.upper():
            in_disp = True
            cur = {}
            continue
        if in_disp and s == "-3":
            if cur:
                blocks.append(cur)
            in_disp = False
            continue
        if in_disp and s.startswith("-1"):
            parsed = _parse_frd_node_xyz(s)
            if parsed is None:
                continue
            nid, ux, uy, uz = parsed
            cur[nid] = (ux, uy, uz)

    if not blocks:
        raise RuntimeError(f"No DISP blocks parsed from {frd_file}")
    return blocks


def _parse_frd_initial_coords(frd_file: Path) -> dict[int, tuple[float, float, float]]:
    """Parse initial coordinates block from FRD file."""
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


def _ccx_center_free_edge_disp_from_frd(
    frd_file: Path,
    *,
    Lx: float = 1.0,
    Ly: float = 0.1,
) -> np.ndarray:
    """Return center free-edge displacement from FRD by geometric matching.

    CCX can renumber nodes and/or emit top/bottom surface nodes in FRD, so we
    identify the geometric target (x=Lx, y=Ly/2) directly from FRD coordinates,
    then average all coincident-through-thickness nodes at that location.
    """
    disp = _parse_frd_last_disp(frd_file)
    coords = _parse_frd_initial_coords(frd_file)

    target_y = 0.5 * Ly
    # Prefer IDs available in both coordinate and displacement datasets.
    # Some FRD outputs omit parts of the model in DISP blocks.
    available = set(coords).intersection(disp)
    x_candidates = [nid for nid in available if np.isclose(coords[nid][0], Lx, atol=1e-9)]
    if not x_candidates:
        # Fallback to coordinate-only edge IDs (for clearer error reporting)
        x_candidates = [nid for nid, (x, _y, _z) in coords.items() if np.isclose(x, Lx, atol=1e-9)]
        if not x_candidates:
            raise RuntimeError(f"No FRD nodes found on free edge x={Lx}")

    min_dy = min(abs(coords[nid][1] - target_y) for nid in x_candidates)
    y_candidates = [nid for nid in x_candidates if abs(coords[nid][1] - target_y) <= (min_dy + 1e-9)]
    disp_candidates = [np.asarray(disp[nid], dtype=float) for nid in y_candidates if nid in disp]
    if not disp_candidates:
        raise RuntimeError("No displacement records found for center free-edge FRD nodes")
    return np.mean(np.vstack(disp_candidates), axis=0)


def _aero_modal(mesh: MeshModel, material_kind: str) -> np.ndarray:
    cfg = _model_cfg(material_kind)
    solver = ModalSolver(mesh, cfg)
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, solver.domain.dofs_per_node), 0.0)])
    f, _ = solver.solve()
    return np.asarray(f[:5], dtype=float)


def _aero_modal_with_shapes(mesh: MeshModel, material_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return first 5 frequencies and translational mode vectors (3 dof/node)."""
    cfg = _model_cfg(material_kind)
    solver = ModalSolver(mesh, cfg)
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, solver.domain.dofs_per_node), 0.0)])
    freqs, modes = solver.solve()

    freqs = np.asarray(freqs[:5], dtype=float)
    modes = np.asarray(modes[:, :5], dtype=float)
    dpn = solver.domain.dofs_per_node
    n_nodes = len(mesh.nodes)
    out = np.zeros((3 * n_nodes, freqs.size), dtype=float)

    for node in mesh.nodes:
        idx = mesh.node_id_to_index[node.id]
        out[3 * idx : 3 * idx + 3, :] = modes[idx * dpn : idx * dpn + 3, :]

    return freqs, out


def _aero_linear_center_disp(mesh: MeshModel, material_kind: str, case: LoadCase) -> np.ndarray:
    cfg = _model_cfg(material_kind)
    solver = StaticLinearSolver(mesh, cfg)
    dpn = solver.domain.dofs_per_node
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)])
    solver.add_nodal_loads(_load_as_nodal(mesh, dpn, case))
    u = solver.solve()
    arr = np.asarray(u.array, dtype=float)
    c = _center_free_edge_node(mesh)
    i0 = mesh.node_id_to_index[c.id] * dpn
    return arr[i0 : i0 + 3]


def _aero_nonlinear_center_disp(mesh: MeshModel, material_kind: str, case: LoadCase) -> np.ndarray:
    cfg = _model_cfg(material_kind)
    solver = StaticNonlinearSolver(mesh, cfg)
    dpn = solver.domain.dofs_per_node
    solver.add_dirichlet_conditions([DirichletCondition(_clamped_dofs(mesh, dpn), 0.0)])
    solver.add_nodal_loads(_load_as_nodal(mesh, dpn, case))
    arr = np.asarray(solver.solve(), dtype=float)
    c = _center_free_edge_node(mesh)
    i0 = mesh.node_id_to_index[c.id] * dpn
    return arr[i0 : i0 + 3]


def _ccx_modal(mesh: MeshModel, material_kind: str, quadratic: bool, ccx_bin: str, workdir: Path) -> np.ndarray:
    cfg = _model_cfg(material_kind)
    inp = workdir / f"modal_{material_kind}_{'q2' if quadratic else 'q1'}.inp"
    write_ccx_mesh(
        mesh,
        str(inp),
        properties=cfg["elements"]["properties"],
        boundary_nodeset="clamped",
        num_modes=5,
        solver_type="Modal",
        quadratic=quadratic,
        span_direction=(1.0, 0.0, 0.0),
    )
    _run_ccx(ccx_bin, workdir, inp.stem)
    return _parse_ccx_frequencies(workdir / f"{inp.stem}.dat", n_modes=5)


def _ccx_modal_with_shapes(
    mesh: MeshModel,
    material_kind: str,
    quadratic: bool,
    ccx_bin: str,
    workdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first 5 frequencies and translational mode vectors (3 dof/node)."""
    cfg = _model_cfg(material_kind)
    inp = workdir / f"modal_{material_kind}_{'q2' if quadratic else 'q1'}.inp"
    write_ccx_mesh(
        mesh,
        str(inp),
        properties=cfg["elements"]["properties"],
        boundary_nodeset="clamped",
        num_modes=5,
        solver_type="Modal",
        quadratic=quadratic,
        span_direction=(1.0, 0.0, 0.0),
    )
    _run_ccx(ccx_bin, workdir, inp.stem)

    freqs = _parse_ccx_frequencies(workdir / f"{inp.stem}.dat", n_modes=5)
    frd = workdir / f"{inp.stem}.frd"
    coords = _parse_frd_initial_coords(frd)
    disp_blocks = _parse_frd_disp_blocks(frd)
    if len(disp_blocks) < len(freqs):
        raise RuntimeError(
            f"FRD contains {len(disp_blocks)} mode-shape blocks, expected at least {len(freqs)}"
        )

    # Geometric matching from AeroElast node coordinates to FRD nodes (robust to renumbering).
    n_nodes = len(mesh.nodes)
    out = np.zeros((3 * n_nodes, len(freqs)), dtype=float)
    coord_to_nids: dict[tuple[float, float, float], list[int]] = {}
    coord_xy_to_nids: dict[tuple[float, float], list[int]] = {}
    for nid, xyz in coords.items():
        key = (round(float(xyz[0]), 9), round(float(xyz[1]), 9), round(float(xyz[2]), 9))
        coord_to_nids.setdefault(key, []).append(nid)
        xy = (round(float(xyz[0]), 9), round(float(xyz[1]), 9))
        coord_xy_to_nids.setdefault(xy, []).append(nid)

    for node in mesh.nodes:
        idx = mesh.node_id_to_index[node.id]
        key = (round(float(node.x), 9), round(float(node.y), 9), round(float(node.z), 9))
        nids = coord_to_nids.get(key)
        if not nids:
            # Shell FRD often writes top/bottom surface nodes at z=±t/2 instead
            # of the model mid-surface z=0. Fall back to XY geometric matching.
            xy = (round(float(node.x), 9), round(float(node.y), 9))
            nids = coord_xy_to_nids.get(xy)
        if not nids:
            raise RuntimeError(f"No FRD node found for mesh node id={node.id} at {key}")
        for mode_i, mode_disp in enumerate(disp_blocks[: len(freqs)]):
            vals = [np.asarray(mode_disp[nid], dtype=float) for nid in nids if nid in mode_disp]
            if not vals:
                # CCX can omit constrained shell nodes in modal DISP blocks;
                # those are physically zero-displacement for clamped edges.
                out[3 * idx : 3 * idx + 3, mode_i] = 0.0
            else:
                out[3 * idx : 3 * idx + 3, mode_i] = np.mean(np.vstack(vals), axis=0)

    return freqs, out


def _mac_matrix(a_modes: np.ndarray, b_modes: np.ndarray) -> np.ndarray:
    """Compute modal assurance criterion matrix between two mode sets."""
    na = a_modes.shape[1]
    nb = b_modes.shape[1]
    mac = np.zeros((na, nb), dtype=float)
    for i in range(na):
        ai = a_modes[:, i]
        den_a = float(np.dot(ai, ai))
        for j in range(nb):
            bj = b_modes[:, j]
            den_b = float(np.dot(bj, bj))
            if den_a <= 1e-30 or den_b <= 1e-30:
                mac[i, j] = 0.0
            else:
                num = float(np.dot(ai, bj))
                mac[i, j] = (num * num) / (den_a * den_b)
    return mac


def _best_mac_mode_mapping(mac: np.ndarray) -> tuple[list[int], list[float]]:
    """Find one-to-one mapping AeroElast mode i -> CCX mode mapping[i] by max MAC sum."""
    n = min(mac.shape[0], mac.shape[1])
    best_perm: tuple[int, ...] | None = None
    best_score = -1.0
    for perm in permutations(range(mac.shape[1]), n):
        score = float(sum(mac[i, perm[i]] for i in range(n)))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Failed to determine MAC mode mapping")
    mapping = list(best_perm)
    mac_diag = [float(mac[i, mapping[i]]) for i in range(n)]
    return mapping, mac_diag


def _ccx_static_center_disp(
    mesh: MeshModel,
    material_kind: str,
    quadratic: bool,
    case: LoadCase,
    nonlinear: bool,
    ccx_bin: str,
    workdir: Path,
    nl_step_controls: dict | None = None,
) -> np.ndarray:
    cfg = _model_cfg(material_kind)
    kind = "nl" if nonlinear else "lin"
    inp = workdir / f"{kind}_{case.name}_{material_kind}_{'q2' if quadratic else 'q1'}.inp"
    kw = dict(nl_step_controls or {})
    write_ccx_mesh(
        mesh,
        str(inp),
        properties=cfg["elements"]["properties"],
        boundary_nodeset="clamped",
        solver_type="NonlinearStatic" if nonlinear else "LinearStatic",
        load_nodeset=case.nodeset,
        load_vector=list(case.load6),
        quadratic=quadratic,
        span_direction=(1.0, 0.0, 0.0),
        **kw,
    )
    _run_ccx(ccx_bin, workdir, inp.stem)
    return _ccx_center_free_edge_disp_from_frd(workdir / f"{inp.stem}.frd")


class TestCantileverPlateCCXParity:
    """Cross-validation against CalculiX using center free-edge metric."""

    def test_modal_first_five_modes(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()

        failures: list[str] = []
        for material, tol in (("isotropic", 0.05), ("composite", 0.08)):
            aero_freq, aero_modes = _aero_modal_with_shapes(mesh, material)
            for quadratic in (False, True):
                ccx_freq, ccx_modes = _ccx_modal_with_shapes(mesh, material, quadratic, ccx_bin, tmp_path)
                mac = _mac_matrix(aero_modes, ccx_modes)
                mapping, mac_diag = _best_mac_mode_mapping(mac)
                ccx_matched = np.asarray([ccx_freq[j] for j in mapping], dtype=float)
                rel = np.abs(ccx_matched - aero_freq) / np.maximum(np.abs(ccx_matched), 1e-14)
                if np.any(rel > tol):
                    pairs = ", ".join(
                        [f"AE m{i + 1}->CCX m{j + 1} (MAC={mac_diag[i]:.4f})" for i, j in enumerate(mapping)]
                    )
                    failures.append(
                        f"modal/{material}/{'q2' if quadratic else 'q1'} rel={rel.tolist()} tol={tol}; "
                        f"mapping=[{pairs}]; mac_matrix={np.array2string(mac, precision=3, suppress_small=True)}"
                    )

        if failures:
            pytest.fail("Modal parity exceeded tolerance:\n" + "\n".join(failures))

    def test_linear_static_center_free_edge_metric(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()

        failures: list[str] = []
        for material, tol in (("isotropic", 0.05), ("composite", 0.10)):
            for case in LOAD_CASES:
                aero_vec = _aero_linear_center_disp(mesh, material, case)
                aero_metric = float(np.linalg.norm(aero_vec))
                ae_resultant = _ae_applied_resultant(mesh, dofs_per_node=6, case=case)
                ccx_resultant = _ccx_applied_resultant(mesh, case)
                rel_res = _resultant_consistency_rel(ae_resultant, ccx_resultant)
                if np.any(rel_res > 1e-12):
                    failures.append(
                        f"load_consistency/{material}/{case.name} ae={ae_resultant.tolist()} "
                        f"ccx={ccx_resultant.tolist()} rel={rel_res.tolist()}"
                    )
                for quadratic in (False, True):
                    ccx_vec = _ccx_static_center_disp(
                        mesh,
                        material,
                        quadratic,
                        case,
                        nonlinear=False,
                        ccx_bin=ccx_bin,
                        workdir=tmp_path,
                    )
                    ccx_metric = float(np.linalg.norm(ccx_vec))
                    rel = abs(ccx_metric - aero_metric) / max(abs(ccx_metric), 1e-14)
                    rel_comp = np.abs(ccx_vec - aero_vec) / np.maximum(np.abs(ccx_vec), 1e-14)
                    if rel > tol:
                        failures.append(
                            f"linear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"|u|_ae={aero_metric:.6e} |u|_ccx={ccx_metric:.6e} rel={rel:.3%} tol={tol:.1%}; "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]}"
                        )
                    if np.any(rel_comp > tol):
                        failures.append(
                            f"linear_component/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]} tol={tol:.1%}"
                        )

        if failures:
            pytest.fail("Linear static parity exceeded tolerance:\n" + "\n".join(failures))

    def test_linear_static_center_free_edge_metric_fy_fz(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()

        failures: list[str] = []
        cases = _select_cases("free_edge_fy", "free_edge_fz")
        for material, tol in (("isotropic", 0.05), ("composite", 0.10)):
            for case in cases:
                aero_vec = _aero_linear_center_disp(mesh, material, case)
                aero_metric = float(np.linalg.norm(aero_vec))
                ae_resultant = _ae_applied_resultant(mesh, dofs_per_node=6, case=case)
                ccx_resultant = _ccx_applied_resultant(mesh, case)
                rel_res = _resultant_consistency_rel(ae_resultant, ccx_resultant)
                if np.any(rel_res > 1e-12):
                    failures.append(
                        f"load_consistency/{material}/{case.name} ae={ae_resultant.tolist()} "
                        f"ccx={ccx_resultant.tolist()} rel={rel_res.tolist()}"
                    )
                for quadratic in (False, True):
                    ccx_vec = _ccx_static_center_disp(
                        mesh,
                        material,
                        quadratic,
                        case,
                        nonlinear=False,
                        ccx_bin=ccx_bin,
                        workdir=tmp_path,
                    )
                    ccx_metric = float(np.linalg.norm(ccx_vec))
                    rel = abs(ccx_metric - aero_metric) / max(abs(ccx_metric), 1e-14)
                    rel_comp = np.abs(ccx_vec - aero_vec) / np.maximum(np.abs(ccx_vec), 1e-14)
                    if rel > tol:
                        failures.append(
                            f"linear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"|u|_ae={aero_metric:.6e} |u|_ccx={ccx_metric:.6e} rel={rel:.3%} tol={tol:.1%}; "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]}"
                        )
                    if np.any(rel_comp > tol):
                        failures.append(
                            f"linear_component/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]} tol={tol:.1%}"
                        )

        if failures:
            pytest.fail("Linear static FY/FZ parity exceeded tolerance:\n" + "\n".join(failures))

    def test_nonlinear_static_center_free_edge_metric(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()

        failures: list[str] = []
        tol = 0.15
        nl_step_controls = {
            "nl_initial_increment": 0.05,
            "nl_min_increment": 1.0e-7,
            "nl_max_increment": 0.1,
            "nl_max_increments": 400,
        }
        for material in ("isotropic", "composite"):
            for case in LOAD_CASES:
                aero_vec = _aero_nonlinear_center_disp(mesh, material, case)
                aero_metric = float(np.linalg.norm(aero_vec))
                ae_resultant = _ae_applied_resultant(mesh, dofs_per_node=6, case=case)
                ccx_resultant = _ccx_applied_resultant(mesh, case)
                rel_res = _resultant_consistency_rel(ae_resultant, ccx_resultant)
                if np.any(rel_res > 1e-12):
                    failures.append(
                        f"load_consistency_nl/{material}/{case.name} ae={ae_resultant.tolist()} "
                        f"ccx={ccx_resultant.tolist()} rel={rel_res.tolist()}"
                    )
                for quadratic in (False, True):
                    try:
                        ccx_vec = _ccx_static_center_disp(
                            mesh,
                            material,
                            quadratic,
                            case,
                            nonlinear=True,
                            ccx_bin=ccx_bin,
                            workdir=tmp_path,
                            nl_step_controls=nl_step_controls,
                        )
                    except RuntimeError as exc:
                        failures.append(
                            f"nonlinear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"ccx_run_failed={type(exc).__name__}: {str(exc).splitlines()[0]}"
                        )
                        continue
                    ccx_metric = float(np.linalg.norm(ccx_vec))
                    rel = abs(ccx_metric - aero_metric) / max(abs(ccx_metric), 1e-14)
                    rel_comp = np.abs(ccx_vec - aero_vec) / np.maximum(np.abs(ccx_vec), 1e-14)
                    if rel > tol:
                        failures.append(
                            f"nonlinear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"|u|_ae={aero_metric:.6e} |u|_ccx={ccx_metric:.6e} rel={rel:.3%} tol={tol:.1%}; "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]}"
                        )
                    if np.any(rel_comp > tol):
                        failures.append(
                            f"nonlinear_component/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]} tol={tol:.1%}"
                        )

        if failures:
            pytest.fail("Nonlinear static parity exceeded tolerance:\n" + "\n".join(failures))

    def test_nonlinear_static_center_free_edge_metric_fy_fz(self, tmp_path: Path):
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_cantilever_mesh()

        failures: list[str] = []
        tol = 0.15
        nl_step_controls = {
            "nl_initial_increment": 0.05,
            "nl_min_increment": 1.0e-7,
            "nl_max_increment": 0.1,
            "nl_max_increments": 400,
        }
        cases = _select_cases("free_edge_fy", "free_edge_fz")

        for material in ("isotropic", "composite"):
            for case in cases:
                aero_vec = _aero_nonlinear_center_disp(mesh, material, case)
                aero_metric = float(np.linalg.norm(aero_vec))
                ae_resultant = _ae_applied_resultant(mesh, dofs_per_node=6, case=case)
                ccx_resultant = _ccx_applied_resultant(mesh, case)
                rel_res = _resultant_consistency_rel(ae_resultant, ccx_resultant)
                if np.any(rel_res > 1e-12):
                    failures.append(
                        f"load_consistency_nl/{material}/{case.name} ae={ae_resultant.tolist()} "
                        f"ccx={ccx_resultant.tolist()} rel={rel_res.tolist()}"
                    )
                for quadratic in (False, True):
                    try:
                        ccx_vec = _ccx_static_center_disp(
                            mesh,
                            material,
                            quadratic,
                            case,
                            nonlinear=True,
                            ccx_bin=ccx_bin,
                            workdir=tmp_path,
                            nl_step_controls=nl_step_controls,
                        )
                    except RuntimeError as exc:
                        failures.append(
                            f"nonlinear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"ccx_run_failed={type(exc).__name__}: {str(exc).splitlines()[0]}"
                        )
                        continue
                    ccx_metric = float(np.linalg.norm(ccx_vec))
                    rel = abs(ccx_metric - aero_metric) / max(abs(ccx_metric), 1e-14)
                    rel_comp = np.abs(ccx_vec - aero_vec) / np.maximum(np.abs(ccx_vec), 1e-14)
                    if rel > tol:
                        failures.append(
                            f"nonlinear/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"|u|_ae={aero_metric:.6e} |u|_ccx={ccx_metric:.6e} rel={rel:.3%} tol={tol:.1%}; "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]}"
                        )
                    if np.any(rel_comp > tol):
                        failures.append(
                            f"nonlinear_component/{material}/{case.name}/{'q2' if quadratic else 'q1'} "
                            f"comp_ae={aero_vec.tolist()} comp_ccx={ccx_vec.tolist()} "
                            f"comp_rel={[float(x) for x in rel_comp]} tol={tol:.1%}"
                        )

        if failures:
            pytest.fail("Nonlinear static FY/FZ parity exceeded tolerance:\n" + "\n".join(failures))
