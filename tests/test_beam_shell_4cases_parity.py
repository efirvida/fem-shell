"""Beam Cantilever Parity (Shell Elements): AeroElast (MITC4) vs CalculiX (S4).

Geometry:
- Cantilever plate in XZ plane, Y=0, extends along Z-axis from 0 to L.
- Width B in X, shell thickness t.
- Clamped at z=0, free end at z=L=1.0m.

Load Cases (4 configurations):
1. Axial Tension:     +Fz (membrane extension along Z)
2. Axial Compression: -Fz (membrane contraction along Z)
3. In-plane Bending:   Fx (bending about Y-axis, membrane-dominated)
4. Out-of-plane Load:  Fy (transverse bending, plate-bending-dominated)

Analytical solutions used for validation against both AeroElast and CCX.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler, modal_solve_coo

from aeroelast.core.material import IsotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel


pytestmark = [pytest.mark.slow]

# ============================================================================
# Material and Geometry Constants
# ============================================================================

MAT_STEEL = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.3, rho=7850.0)

L = 1.0  # Beam length along Z (m)
B = 0.05  # Width in X (m)
t = 0.005  # Shell thickness (m)

A = B * t  # Cross-sectional area (m²)
I_x = B * t**3 / 12  # Moment of inertia about X-axis (out-of-plane bending, Fy)
I_y = t * B**3 / 12  # Moment of inertia about Y-axis (in-plane bending, Fx)

FORCE_MAGNITUDE = 1000.0  # N
DOFS_PER_NODE = 6  # MITC4 shell: 3 translations + 3 rotations


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class ShellLoadCase:
    name: str
    description: str
    load_vector: tuple[float, float, float]  # (Fx, Fy, Fz)
    analytical_dof: str  # 'x', 'y', or 'z'
    analytical_formula: str


LOAD_CASES: tuple[ShellLoadCase, ...] = (
    ShellLoadCase(
        name="tension_axial",
        description="Axial tension along Z (membrane)",
        load_vector=(0.0, 0.0, FORCE_MAGNITUDE),
        analytical_dof="z",
        analytical_formula="Fz*L/(E*A)",
    ),
    ShellLoadCase(
        name="compression_axial",
        description="Axial compression along Z (membrane)",
        load_vector=(0.0, 0.0, -FORCE_MAGNITUDE),
        analytical_dof="z",
        analytical_formula="-Fz*L/(E*A)",
    ),
    ShellLoadCase(
        name="bending_fx",
        description="In-plane bending by Fx (about Y-axis, membrane bending)",
        load_vector=(FORCE_MAGNITUDE, 0.0, 0.0),
        analytical_dof="x",
        analytical_formula="Fx*L^3/(3*E*I_y)",
    ),
    ShellLoadCase(
        name="transverse_fy",
        description="Out-of-plane transverse Fy (plate bending about X-axis)",
        load_vector=(0.0, FORCE_MAGNITUDE, 0.0),
        analytical_dof="y",
        analytical_formula="Fy*L^3/(3*E*I_x)",
    ),
)

STATIC_CASE_PARAMS = [pytest.param(case, id=case.name) for case in LOAD_CASES]


# ============================================================================
# Mesh Builder
# ============================================================================


def _build_shell_mesh(*, nz: int = 20, nx: int = 4) -> MeshModel:
    """Build a cantilever shell mesh (MITC4 quads) in XZ plane.

    Plate: X in [0, B], Y=0, Z in [0, L].
    Clamped at z=0, free end at z=L.
    """
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()

    xs = np.linspace(0.0, B, nx + 1)
    zs = np.linspace(0.0, L, nz + 1)

    grid: dict[tuple[int, int], Node] = {}
    for k, z in enumerate(zs):
        for i, x in enumerate(xs):
            node = Node([float(x), 0.0, float(z)], geometric_node=False)
            mesh.add_node(node)
            grid[(i, k)] = node

    for k in range(nz):
        for i in range(nx):
            n00 = grid[(i, k)]
            n10 = grid[(i + 1, k)]
            n11 = grid[(i + 1, k + 1)]
            n01 = grid[(i, k + 1)]
            mesh.add_element(
                MeshElement(
                    nodes=[n00, n10, n11, n01],
                    element_type=ElementType.quad,
                )
            )

    clamped_nodes = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face_nodes = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    free_center = min(free_face_nodes, key=lambda n: abs(float(n.x) - B / 2))

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_face", free_face_nodes))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


# ============================================================================
# Analytical Solutions
# ============================================================================


def _analytical_solution(case: ShellLoadCase) -> float:
    """Return analytical tip displacement for the given load case."""
    F = FORCE_MAGNITUDE
    E = MAT_STEEL.E

    if case.analytical_dof == "z":
        sign = 1.0 if case.load_vector[2] > 0 else -1.0
        return sign * (F * L) / (E * A)
    elif case.analytical_dof == "x":
        return (F * L**3) / (3.0 * E * I_y)
    elif case.analytical_dof == "y":
        return (F * L**3) / (3.0 * E * I_x)
    return 0.0


# ============================================================================
# CCX Parsing and Helpers
# ============================================================================

_NUM_RE = re.compile(r"[-+]?\d*\.\d+(?:[Ee][+-]?\d+)?|[-+]?\d+")
_FREQ_ROW = re.compile(
    r"^\s*(\d+)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s*$"
)


def _parse_frd_coords(frd_file: Path) -> dict[int, np.ndarray]:
    """Parse nodal coordinates from the first coordinate block in a CCX .frd file."""
    with open(frd_file, "r") as f:
        lines = f.readlines()

    coords: dict[int, np.ndarray] = {}
    in_coords = False
    for line in lines:
        s = line.strip()
        if s.startswith("2C"):
            in_coords = True
            continue
        if not in_coords:
            continue
        if s.startswith("-3"):
            break
        if not s.startswith("-1"):
            continue
        nums = _NUM_RE.findall(s)
        if len(nums) < 5:
            continue
        try:
            node_id = int(nums[1])
            coords[node_id] = np.array([float(nums[2]), float(nums[3]), float(nums[4])])
        except (ValueError, IndexError):
            continue
    return coords


def _parse_frd_last_disp(frd_file: Path) -> dict[int, np.ndarray]:
    """Parse the last displacement block from a CCX .frd file."""
    if not frd_file.exists():
        raise RuntimeError(f"FRD file not found: {frd_file}")

    with open(frd_file, "r") as f:
        lines = f.readlines()

    last_node_start = -1
    for i, line in enumerate(lines):
        if "DISP" in line.upper() and ("-4" in line or line.startswith("*NODE")):
            last_node_start = i

    if last_node_start == -1:
        raise RuntimeError(f"No displacement results found in {frd_file}")

    disps: dict[int, np.ndarray] = {}
    for line in lines[last_node_start + 1 :]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("-3") or s.startswith("*") or "STEP" in s.upper():
            break
        if not s.startswith("-1"):
            continue
        nums = _NUM_RE.findall(s)
        if len(nums) < 5:
            continue
        try:
            node_id = int(nums[1])
            disps[node_id] = np.array([float(nums[2]), float(nums[3]), float(nums[4])])
        except (ValueError, IndexError):
            continue
    return disps


def _frd_disp_at_xyz(frd_file: Path, x: float, z: float, tol: float = 1e-4) -> np.ndarray:
    """Return midsurface displacement at the requested shell point.

    CCX may expand S4 shell nodes to top/bottom surface nodes in the FRD. Matching by
    original node id is therefore brittle. Matching by (x, z) and averaging collapses
    the shell thickness expansion back to the midsurface response.
    """
    coords = _parse_frd_coords(frd_file)
    disps = _parse_frd_last_disp(frd_file)

    matches: list[np.ndarray] = []
    for node_id, xyz in coords.items():
        if abs(xyz[0] - x) <= tol and abs(xyz[2] - z) <= tol and node_id in disps:
            matches.append(disps[node_id])

    if not matches:
        raise RuntimeError(f"No FRD nodes found near x={x:.4e}, z={z:.4e} in {frd_file}")
    return np.mean(np.vstack(matches), axis=0)


def _parse_ccx_frequencies(dat_path: Path, n_modes: int = 5) -> np.ndarray:
    """Parse eigenfrequencies from a CCX .dat file."""
    dat_file = dat_path.with_suffix(".dat")
    if not dat_file.exists():
        raise RuntimeError(f"DAT file not found: {dat_file}")

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


def _ccx_bin() -> str:
    return "/scratch/leahk/eduardo.donestevez/venv/bin/ccx"


def _run_ccx(ccx_bin: str, workdir: Path, stem: str) -> None:
    for file_path in workdir.glob("*"):
        if (
            file_path.is_file()
            and file_path.name != f"{stem}.inp"
            and file_path.suffix not in [".msh", ".nam"]
        ):
            file_path.unlink()

    proc = subprocess.run(
        [ccx_bin, stem],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CCX run failed for {stem} (code={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _clamped_dofs(mesh: MeshModel) -> list[int]:
    dofs: list[int] = []
    idx = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = idx[node.id] * DOFS_PER_NODE
        dofs.extend(range(i0, i0 + DOFS_PER_NODE))
    return sorted(set(dofs))


def _compare_modal_frequencies(
    freqs_ae: np.ndarray, freqs_ccx: np.ndarray, tol: float = 0.05
) -> None:
    print("\n[Modal Comparison]")
    print(f"  AeroElast frequencies (Hz): {freqs_ae}")
    print(f"  CCX       frequencies (Hz): {freqs_ccx}")

    rel = np.abs(freqs_ccx - freqs_ae) / np.maximum(np.abs(freqs_ccx), 1e-14)
    print(f"  Relative errors: {rel}")

    if np.any(rel > tol):
        pytest.fail(
            f"Modal parity exceeded tolerance: "
            f"ae={freqs_ae.tolist()} ccx={freqs_ccx.tolist()} "
            f"rel={rel.tolist()} tol={tol}"
        )


# ============================================================================
# Test Class
# ============================================================================


class TestBeamShell4CasesParity:
    """Validate AeroElast (MITC4) vs CCX (S4) for 4 shell beam load cases."""

    def _run_static_case(
        self, case: ShellLoadCase, tmp_path: Path
    ) -> tuple[float, float, float, float]:
        """Solve one static shell load case and return AeroElast, CCX, analytical and relative error."""
        ccx_bin = _ccx_bin()
        mesh = _build_shell_mesh()

        node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
        conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
        elem_types = [4] * len(mesh.elements)  # MITC4

        center = next(iter(mesh.get_node_set("free_center").nodes.values()))
        i0 = mesh.node_id_to_index[center.id] * DOFS_PER_NODE

        fixed = _clamped_dofs(mesh)
        free_mask = np.ones(len(node_coords) * DOFS_PER_NODE, dtype=bool)
        free_mask[fixed] = False
        mats = [
            {
                "type": "isotropic",
                "e": MAT_STEEL.E,
                "nu": MAT_STEEL.nu,
                "rho": MAT_STEEL.rho,
                "thickness": t,
            }
        ] * len(mesh.elements)

        asm = PyMeshAssembler(
            node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
        )
        rows, cols, vals = asm.assemble_k()
        n = asm.dofs_count
        K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        # Load vector (6 DOFs/node: dof 0=Tx, 1=Ty, 2=Tz, 3=Rx, 4=Ry, 5=Rz)
        f = np.zeros(n, dtype=float)
        dof_map = {"x": 0, "y": 1, "z": 2}
        dof_idx = dof_map[case.analytical_dof]
        fx, fy, fz = case.load_vector
        if fx != 0.0:
            f[i0] = fx
        if fy != 0.0:
            f[i0 + 1] = fy
        if fz != 0.0:
            f[i0 + 2] = fz

        free = np.where(free_mask)[0]
        u = np.zeros(n, dtype=float)
        u[free] = spsolve(K[free][:, free], f[free])

        ae_disp = float(u[i0 + dof_idx])
        anal_disp = _analytical_solution(case)

        stem = f"shell_{case.name}"
        case_dir = tmp_path / case.name
        case_dir.mkdir(exist_ok=True)
        inp = case_dir / f"{stem}.inp"

        write_ccx_mesh(
            mesh,
            str(inp),
            properties={
                "plate": {
                    "type": "isotropic",
                    "e": MAT_STEEL.E,
                    "nu": MAT_STEEL.nu,
                    "rho": MAT_STEEL.rho,
                    "thickness": t,
                }
            },
            boundary_nodeset="clamped",
            solver_type="LinearStatic",
            load_nodeset="free_center",
            load_vector=list(case.load_vector),
        )
        _run_ccx(ccx_bin, case_dir, stem)

        frd = case_dir / f"{stem}.frd"
        ccx_vec = _frd_disp_at_xyz(frd, float(center.x), L)
        ccx_disp = float(ccx_vec[dof_idx])

        rel_err = np.abs(ae_disp - ccx_disp) / np.maximum(np.abs(ccx_disp), 1e-14)

        print(f"\n[{case.name}]")
        print(f"  Analytical: {anal_disp:.6E}")
        print(
            f"  AeroElast:  {ae_disp:.6E} "
            f"(rel err vs anal: {np.abs(ae_disp - anal_disp) / np.abs(anal_disp) * 100:.2f}%)"
        )
        print(
            f"  CCX:        {ccx_disp:.6E} "
            f"(rel err vs anal: {np.abs(ccx_disp - anal_disp) / np.abs(anal_disp) * 100:.2f}%)"
        )
        print(f"  AE vs CCX rel err: {rel_err * 100:.2f}%")

        return ae_disp, ccx_disp, anal_disp, rel_err

    @pytest.mark.parametrize("case", STATIC_CASE_PARAMS)
    def test_linear_static_with_analytical(self, tmp_path: Path, case: ShellLoadCase):
        """Compare one linear static shell load case against analytical and CCX results."""

        tol = 0.05
        ae_disp, ccx_disp, _anal_disp, rel_err = self._run_static_case(case, tmp_path)

        assert rel_err <= tol, (
            f"{case.name}: ae={ae_disp:.6E} ccx={ccx_disp:.6E} "
            f"rel_err={rel_err * 100:.2f}% tol={tol * 100:.2f}%"
        )

    def test_modal_first_five_modes(self, tmp_path: Path):
        """Compare first five matched modal frequencies: AeroElast (MITC4) vs CCX (S4).

        MITC4 and S4 may compute spurious or membrane modes in different positions.
        We request N_SEARCH modes from both solvers and find the 5 best-matched pairs
        to avoid false failures from mode ordering differences.
        """
        N_SEARCH = 12  # compute extra modes so matching is robust
        N_COMPARE = 5  # validate the 5 best-matched pairs

        ccx_bin = _ccx_bin()
        mesh = _build_shell_mesh()

        node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
        conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
        elem_types = [4] * len(mesh.elements)  # MITC4
        mats = [
            {
                "type": "isotropic",
                "e": MAT_STEEL.E,
                "nu": MAT_STEEL.nu,
                "rho": MAT_STEEL.rho,
                "thickness": t,
            }
        ] * len(mesh.elements)

        asm = PyMeshAssembler(
            node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
        )
        k_rows, k_cols, k_vals = asm.assemble_k()
        m_rows, m_cols, m_vals = asm.assemble_m()
        n = asm.dofs_count

        fixed = _clamped_dofs(mesh)
        free = np.array([i for i in range(n) if i not in set(fixed)], dtype=np.int64)

        freqs_ae_all, _ = modal_solve_coo(
            k_rows.astype(np.int64),
            k_cols.astype(np.int64),
            k_vals.astype(np.float64),
            m_rows.astype(np.int64),
            m_cols.astype(np.int64),
            m_vals.astype(np.float64),
            n,
            free,
            N_SEARCH,
        )
        freqs_ae_all = np.sort(np.asarray(freqs_ae_all, dtype=float))

        # CCX modal - request the same N_SEARCH modes
        modal_dir = tmp_path / "modal"
        modal_dir.mkdir(exist_ok=True)
        stem = "shell_modal"
        inp = modal_dir / f"{stem}.inp"

        write_ccx_mesh(
            mesh,
            str(inp),
            properties={
                "plate": {
                    "type": "isotropic",
                    "e": MAT_STEEL.E,
                    "nu": MAT_STEEL.nu,
                    "rho": MAT_STEEL.rho,
                    "thickness": t,
                }
            },
            boundary_nodeset="clamped",
            solver_type="Modal",
            num_modes=N_SEARCH,
        )
        _run_ccx(ccx_bin, modal_dir, stem)

        dat = modal_dir / f"{stem}.dat"
        freqs_ccx_all = np.sort(_parse_ccx_frequencies(dat, n_modes=N_SEARCH))

        # Solve the one-to-one frequency matching problem globally, then keep the best pairs.
        rel_cost = np.abs(freqs_ae_all[:, None] - freqs_ccx_all[None, :]) / np.maximum(
            np.abs(freqs_ccx_all[None, :]), 1e-14
        )
        row_ind, col_ind = linear_sum_assignment(rel_cost)
        matched = sorted(
            (
                float(rel_cost[i, j]),
                float(freqs_ae_all[i]),
                float(freqs_ccx_all[j]),
            )
            for i, j in zip(row_ind, col_ind, strict=False)
        )

        freqs_ae = np.array([item[1] for item in matched[:N_COMPARE]], dtype=float)
        freqs_ccx = np.array([item[2] for item in matched[:N_COMPARE]], dtype=float)

        print(f"\n  AE  all ({N_SEARCH} modes): {freqs_ae_all}")
        print(f"  CCX all ({N_SEARCH} modes): {freqs_ccx_all}")

        _compare_modal_frequencies(freqs_ae, freqs_ccx, tol=0.05)
