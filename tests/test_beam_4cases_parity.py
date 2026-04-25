"""Beam Cantilever Parity: AeroElast vs CalculiX (CCX).

Geometry:
- Beam centered at (0,0,0) in XY plane, extends along Z-axis from 0 to L.
- Cross-section: B x H in XY plane (B=0.05m, H=0.05m).
- Clamped at z=0, free end at z=L=1.0m.

Load Cases (4 configurations):
1. Axial Tension: +Fz (extension along Z)
2. Axial Compression: -Fz (contraction along Z)
3. Bending (Flexion): Fx applied at free end (causes bending about Y-axis)
4. Shear (Cortante): Fy applied at free end (causes bending about X-axis)

Analytical solutions used for validation.
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

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler, modal_solve_coo

from aeroelast.core.bc import DirichletCondition, NodalLoad
from aeroelast.core.material import IsotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import ShellProperty
from aeroelast.elements import ElementFamily
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver
from aeroelast.solvers.elasticity.static_nonlinear import StaticNonlinearSolver
from aeroelast.solvers.modal import ModalSolver


pytestmark = [pytest.mark.slow]

# ============================================================================
# Material and Geometry Constants
# ============================================================================

MAT_STEEL = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.3, rho=7850.0)

L = 1.0  # Beam length along Z (m)
B = 0.05  # Width in X (m)
H = 0.05  # Height in Y (m)
A = B * H  # Cross-sectional area (m²)
I_x = (H * B**3) / 12  # Moment of inertia about X-axis (m⁴)
I_y = (B * H**3) / 12  # Moment of inertia about Y-axis (m⁴)

FORCE_MAGNITUDE = 1000.0  # N


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class BeamLoadCase:
    name: str
    description: str
    load_vector: tuple[float, float, float]  # (Fx, Fy, Fz)
    analytical_dof: str  # Which DOF to check ('x', 'y', or 'z')
    analytical_formula: str  # Description of formula


# 4 Load Cases as specified
LOAD_CASES: tuple[BeamLoadCase, ...] = (
    BeamLoadCase(
        name="tension_axial",
        description="Axial tension along Z",
        load_vector=(0.0, 0.0, FORCE_MAGNITUDE),
        analytical_dof="z",
        analytical_formula="Fz*L/(E*A)",
    ),
    BeamLoadCase(
        name="compression_axial",
        description="Axial compression along Z",
        load_vector=(0.0, 0.0, -FORCE_MAGNITUDE),
        analytical_dof="z",
        analytical_formula="-Fz*L/(E*A)",
    ),
    BeamLoadCase(
        name="bending_fx",
        description="Bending by transverse Fx (about Y-axis)",
        load_vector=(FORCE_MAGNITUDE, 0.0, 0.0),
        analytical_dof="x",
        analytical_formula="Fx*L^3/(3*E*I_y)",
    ),
    BeamLoadCase(
        name="shear_fy",
        description="Shear/Transverse Fy (about X-axis)",
        load_vector=(0.0, FORCE_MAGNITUDE, 0.0),
        analytical_dof="y",
        analytical_formula="Fy*L^3/(3*E*I_x)",
    ),
)


# ============================================================================
# CCX Result Parsers
# ============================================================================


def _parse_frd_last_disp(frd_file: Path) -> dict[int, tuple[float, float, float]]:
    # Path to .frd file
    frd_file = frd_file.with_suffix(".frd")
    if not frd_file.exists():
        raise RuntimeError(f"FRD file not found: {frd_file}")

    with open(frd_file, "r") as f:
        lines = f.readlines()

    # Find last displacement results
    last_node_start = -1
    for i, line in enumerate(lines):
        if "DISP" in line.upper() and ("-4" in line or line.startswith("*NODE")):
            last_node_start = i

    if last_node_start == -1:
        raise RuntimeError(f"No displacement results found in {frd_file}")

    # Parse displacements
    disps: dict[int, np.ndarray] = {}
    for line in lines[last_node_start + 1 :]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("-3") or s.startswith("*") or "STEP" in s.upper():
            break
        if s.startswith("-1"):
            # Use regex to find all numbers (including scientific notation)
            import re
            # Match the ID and the 3 floats
            # The ID is usually after the -1, then 3 floats
            # Example: -1        46 0.00000E+00 0.00000E+00-5.00000E-04
            # We need to extract: 46, 0.0, 0.0, -5e-4

            # Find all numbers including signs and exponents
            nums = re.findall(r"[-+]?\d*\.\d+(?:[Ee][+-]?\d+)?|[-+]?\d+", s)
            if len(nums) >= 4:
                try:
                    # nums[0] is -1, nums[1] is node_id, nums[2:5] are U, V, W
                    node_id = int(nums[1])
                    u = np.array([float(nums[2]), float(nums[3]), float(nums[4])])
                    disps[node_id] = u
                except (ValueError, IndexError):
                    continue
    return disps


def _frd_disp_at_node_id(frd_file: Path, node_id: int) -> np.ndarray:
    """Get displacement for a specific node ID."""
    disp = _parse_frd_last_disp(frd_file)
    if node_id not in disp:
        # Try to find it in the dictionary (some versions of CCX might have shifted IDs)
        # but usually it's exact.
        raise RuntimeError(f"Node ID {node_id} not found in FRD displacement block")
    return disp[node_id]


_FREQ_ROW = re.compile(
    r"^\s*(\d+)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s+"
    r"([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)\s*$"
)


def _parse_ccx_frequencies(dat_path: Path, n_modes: int = 5) -> np.ndarray:
    # Path to .dat file
    dat_file = dat_path.with_suffix(".dat")
    if not dat_file.exists():
        raise RuntimeError(f"No frequencies parsed from {dat_file}")

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


# ============================================================================
# Helper Functions
# ============================================================================


def _compare_modal_frequencies(
    freqs_ae: np.ndarray, freqs_ccx: np.ndarray, tol: float = 0.08
) -> None:
    """Compare modal frequencies between AeroElast and CCX.

    Prints both frequency arrays and relative errors, then fails via pytest if any
    error exceeds the tolerance.
    """
    print(f"\n[Modal Comparison]")
    print(f"  AeroElast frequencies: {freqs_ae}")
    print(f"  CCX frequencies:        {freqs_ccx}")

    rel = np.abs(freqs_ccx - freqs_ae) / np.maximum(np.abs(freqs_ccx), 1e-14)
    print(f"  Relative errors: {rel}")

    if np.any(rel > tol):
        pytest.fail(
            f"Modal parity exceeded tolerance: "
            f"ae={freqs_ae.tolist()} ccx={freqs_ccx.tolist()} "
            f"rel={rel.tolist()} tol={tol}"
        )


def _ccx_bin_or_skip() -> str:
    ccx_bin = "/scratch/leahk/eduardo.donestevez/venv/bin/ccx"
    return ccx_bin


def _build_beam_mesh(*, nz: int = 10, nx: int = 2, ny: int = 2) -> MeshModel:
    """Build a cantilever beam mesh extending along Z-axis.

    Beam is centered at (0,0,0) in XY, extends from z=0 (clamped) to z=L (free).
    """
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()

    # Grid in XYZ: X from -B/2 to B/2, Y from -H/2 to H/2, Z from 0 to L
    xs = np.linspace(-B / 2, B / 2, nx + 1)
    ys = np.linspace(-H / 2, H / 2, ny + 1)
    zs = np.linspace(0.0, L, nz + 1)

    grid: dict[tuple[int, int, int], Node] = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node = Node([float(x), float(y), float(z)], geometric_node=False)
                mesh.add_node(node)
                grid[(i, j, k)] = node

    # Create hexahedral elements
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

    # Node sets
    clamped_nodes = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face_nodes = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}

    # Center node at free face (x=0, y=0, z=L)
    free_center = min(free_face_nodes, key=lambda n: abs(float(n.x)) + abs(float(n.y)))

    mesh.add_node_set(NodeSet("clamped", clamped_nodes))
    mesh.add_node_set(NodeSet("free_face", free_face_nodes))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("beam", set(mesh.elements)))

    return mesh


def _analytical_solution(case: BeamLoadCase) -> float:
    """Return analytical displacement at free end."""
    F = FORCE_MAGNITUDE
    E = MAT_STEEL.E

    if case.analytical_dof == "z":  # Axial
        sign = 1.0 if case.load_vector[2] > 0 else -1.0
        return sign * (F * L) / (E * A)
    elif case.analytical_dof == "x":  # Bending about Y (Fx)
        return (F * L**3) / (3 * E * I_y)
    elif case.analytical_dof == "y":  # Bending about X (Fy)
        return (F * L**3) / (3 * E * I_x)
    return 0.0


def _clamped_dofs(mesh: MeshModel, dofs_per_node: int = 3) -> list:
    dofs: list = []
    idx = mesh.node_id_to_index
    for node in mesh.get_node_set("clamped").nodes.values():
        i0 = idx[node.id] * dofs_per_node
        dofs.extend(range(i0, i0 + dofs_per_node))
    return sorted(set(dofs))


def _write_chunks(file, ids, per_line: int) -> None:
    """Write list of ids in chunks of per_line per line."""
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        file.write(", ".join(str(i) for i in chunk) + "\n")


def _run_ccx(ccx_bin: str, workdir: Path, stem: str) -> None:
    # Ensure only necessary input files exist to avoid duplicate loading of residual mesh files
    for f in workdir.glob("*"):
        if f.is_file() and f.name != f"{stem}.inp" and f.suffix not in [".msh", ".nam"]:
            f.unlink()

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


def _write_beam_ccx_inp(
    inp: Path,
    *,
    mesh: MeshModel,
    stem: str,
    solver_type: str,
    load_case: BeamLoadCase | None,
) -> None:
    """Write a minimal, strictly formatted CCX input file."""
    lines = []
    lines.append("*NODE, NSET=NALL")
    for i, node in enumerate(mesh.nodes):
        lines.append(f"{i + 1},{node.x:.6f},{node.y:.6f},{node.z:.6f}")
    lines.append("*NSET, NSET=NCLAMPED")
    c_ids = sorted(
        mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("clamped").nodes.values()
    )
    for i in range(0, len(c_ids), 16):
        lines.append(",".join(str(x) for x in c_ids[i : i + 16]))
    lines.append("*NSET, NSET=NFREE_FACE")
    f_ids = sorted(
        mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_face").nodes.values()
    )
    for i in range(0, len(f_ids), 16):
        lines.append(",".join(str(x) for x in f_ids[i : i + 16]))
    lines.append("*NSET, NSET=NCENTER")
    center_node = next(iter(mesh.get_node_set("free_center").nodes.values()))
    lines.append(f"{mesh.node_id_to_index[center_node.id] + 1}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EBEAM")
    for i, el in enumerate(mesh.elements):
        conn = ",".join(str(int(nid) + 1) for nid in el.node_ids)
        lines.append(f"{i + 1},{conn}")
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append(f"{MAT_STEEL.E:.6E},{MAT_STEEL.nu:.6f}")
    lines.append("*DENSITY")
    lines.append(f"{MAT_STEEL.rho:.6E}")
    lines.append("*SOLID SECTION, ELSET=EBEAM, MATERIAL=STEEL")
    lines.append("*BOUNDARY")
    lines.append("NCLAMPED,1,3,0.0")
    lines.append("*STEP, INC=10000")
    if solver_type == "modal":
        lines.append("*FREQUENCY")
        lines.append("5")
    elif solver_type == "linear":
        lines.append("*STATIC")
        if load_case:
            lines.append("*CLOAD")
            center_idx = mesh.node_id_to_index[center_node.id] + 1
            fx, fy, fz = load_case.load_vector
            if fx != 0.0:
                lines.append(f"{center_idx},1,{fx:.6E}")
            if fy != 0.0:
                lines.append(f"{center_idx},2,{fy:.6E}")
            if fz != 0.0:
                lines.append(f"{center_idx},3,{fz:.6E}")
    lines.append("*NODE FILE")
    lines.append("U")
    if solver_type == "modal":
        lines.append("*EL FILE")
        lines.append("S")
    lines.append("*END STEP")
    inp.write_text("\n".join(lines) + "\n")


# ============================================================================
# Test Class
# ============================================================================


class TestBeam4CasesParity:
    """Validate AeroElast vs CCX for 4 beam load cases with analytical solutions."""

    def test_linear_static_with_analytical(self, tmp_path: Path):
        """Compare linear static results against analytical solutions."""
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_beam_mesh()

        failures: list[str] = []
        tol = 0.05  # 5% tolerance

        for case in LOAD_CASES:
            # AeroElast solution using Rust backend
            node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
            conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
            elem_types = [208] * len(mesh.elements)  # Hexa8
            mats = [
                {"type": "solid_3d", "e": MAT_STEEL.E, "nu": MAT_STEEL.nu, "rho": MAT_STEEL.rho}
            ] * len(mesh.elements)

            asm = PyMeshAssembler(
                node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
            )
            rows, cols, vals = asm.assemble_k()
            n = asm.dofs_count
            K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

            # Load vector
            f = np.zeros(n, dtype=float)
            center = next(iter(mesh.get_node_set("free_center").nodes.values()))
            i0 = mesh.node_id_to_index[center.id] * 3
            fx, fy, fz = case.load_vector
            if fx != 0:
                f[i0] = fx
            if fy != 0:
                f[i0 + 1] = fy
            if fz != 0:
                f[i0 + 2] = fz

            # Apply boundary conditions
            fixed = _clamped_dofs(mesh, dofs_per_node=3)
            free_mask = np.ones(n, dtype=bool)
            free_mask[fixed] = False
            free = np.where(free_mask)[0]

            u = np.zeros(n, dtype=float)
            u[free] = spsolve(K[free][:, free], f[free])

            # Get displacement at free center
            dof_map = {"x": 0, "y": 1, "z": 2}
            dof_idx = dof_map[case.analytical_dof]
            ae_disp = float(u[i0 + dof_idx])

            # Analytical solution
            anal_disp = _analytical_solution(case)

            # CCX solution
            stem = f"beam_{case.name}"
            case_dir = tmp_path / case.name
            case_dir.mkdir(exist_ok=True)
            inp = case_dir / f"{stem}.inp"

            from aeroelast.core.mesh.io.writers import write_ccx_mesh

            write_ccx_mesh(
                mesh,
                str(inp),
                properties={
                    "beam": {
                        "type": "isotropic",
                        "e": MAT_STEEL.E,
                        "nu": MAT_STEEL.nu,
                        "rho": MAT_STEEL.rho,
                        "thickness": 0.05,
                    }
                },
                boundary_nodeset="clamped",
                solver_type="LinearStatic",
                load_nodeset="free_center",
                load_vector=case.load_vector,
            )
            _run_ccx(ccx_bin, case_dir, stem)

            # Parse CCX displacement
            frd = case_dir / f"{stem}.frd"
            center_node = next(iter(mesh.get_node_set("free_center").nodes.values()))
            ccx_disp_vec = _frd_disp_at_node_id(frd, center_node.id + 1)  # CCX is 1-based
            ccx_disp = float(ccx_disp_vec[dof_idx])

            # Parity check
            rel_err = np.abs(ae_disp - ccx_disp) / np.maximum(np.abs(ccx_disp), 1e-14)

            print(f"\n[{case.name}]")
            print(f"  Analytical: {anal_disp:.6E}")
            print(
                f"  AeroElast:  {ae_disp:.6E} (rel err vs anal: {np.abs(ae_disp - anal_disp) / np.abs(anal_disp) * 100:.2f}%)"
            )
            print(
                f"  CCX:        {ccx_disp:.6E} (rel err vs anal: {np.abs(ccx_disp - anal_disp) / np.abs(anal_disp) * 100:.2f}%)"
            )
            print(f"  AE vs CCX rel err: {rel_err * 100:.2f}%")

            if rel_err > tol:
                failures.append(
                    f"{case.name}: ae={ae_disp:.6E} ccx={ccx_disp:.6E} rel_err={rel_err * 100:.2f}% tol={tol * 100:.2f}%"
                )

        if failures:
            pytest.fail("Linear static parity exceeded tolerance:\n" + "\n".join(failures))

    def test_modal_first_five_modes(self, tmp_path: Path):
        """Compare first five modal frequencies against AeroElast."""
        ccx_bin = _ccx_bin_or_skip()
        mesh = _build_beam_mesh()

        # AeroElast modal
        node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
        conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
        elem_types = [208] * len(mesh.elements)
        mats = [
            {"type": "solid_3d", "e": MAT_STEEL.E, "nu": MAT_STEEL.nu, "rho": MAT_STEEL.rho}
        ] * len(mesh.elements)

        asm = PyMeshAssembler(
            node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
        )
        k_rows, k_cols, k_vals = asm.assemble_k()
        m_rows, m_cols, m_vals = asm.assemble_m()
        n = asm.dofs_count

        fixed = _clamped_dofs(mesh, dofs_per_node=3)
        free = np.array([i for i in range(n) if i not in set(fixed)], dtype=np.int64)

        freqs_ae, _, _ = modal_solve_coo(
            k_rows.astype(np.int64),
            k_cols.astype(np.int64),
            k_vals.astype(np.float64),
            m_rows.astype(np.int64),
            m_cols.astype(np.int64),
            m_vals.astype(np.float64),
            n,
            free,
            5,
        )
        freqs_ae = np.asarray(freqs_ae, dtype=float)

        # CCX solution
        modal_dir = tmp_path / "modal"
        modal_dir.mkdir(exist_ok=True)
        stem = "beam_modal"
        inp = modal_dir / f"{stem}.inp"

        from aeroelast.core.mesh.io.writers import write_ccx_mesh

        write_ccx_mesh(
            mesh,
            str(inp),
            properties={
                "beam": {
                    "type": "isotropic",
                    "e": MAT_STEEL.E,
                    "nu": MAT_STEEL.nu,
                    "rho": MAT_STEEL.rho,
                    "thickness": 0.05,
                }
            },
            boundary_nodeset="clamped",
            solver_type="Modal",
            num_modes=5,
        )
        _run_ccx(ccx_bin, modal_dir, stem)

        # Parse CCX frequencies
        dat = modal_dir / f"{stem}.dat"
        freqs_ccx = _parse_ccx_frequencies(dat, n_modes=5)

        # Compare using helper (tolerance = 8%)
        _compare_modal_frequencies(freqs_ae, freqs_ccx, tol=0.08)
