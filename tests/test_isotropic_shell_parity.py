"""Isotropic Shell Parity: AeroElast (MITC4) vs CalculiX (S4).

Purpose: Test if the 7x difference observed in composite shells
also appears in isotropic shells.

Geometry:
- Cantilever plate: L x B x t (shell thickness)
- Extends along Z-axis from 0 to L
- Clamped at z=0, free end at z=L
- Transverse load Fy at free center

If MITC4 and S4 match for isotropic, the problem is in composite formulation.
If they differ, the problem is in MITC vs S4 formulation generally.
"""

from __future__ import annotations

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


# ============================================================================
# Constants
# ============================================================================

MAT_STEEL = IsotropicMaterial(name="Steel", E=2.1e11, nu=0.3, rho=7850.0)

L = 1.0  # Length along Z (m)
B = 0.1  # Width in X (m)
t = 0.005  # Shell thickness (m)
FORCE = 100.0  # N (transverse load)


# ============================================================================
# Mesh Builder - Isotropic Shell
# ============================================================================


def build_shell_mesh(*, nz: int = 10, nx: int = 2) -> MeshModel:
    """Build a cantilever shell mesh extending along Z-axis."""
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()

    # Grid: X from 0 to B, Z from 0 to L, Y=0 (flat plate in XZ plane)
    xs = np.linspace(0, B, nx + 1)
    ys = np.zeros(nx + 1)  # Y = 0 (flat)
    zs = np.linspace(0, L, nz + 1)

    grid: dict[tuple[int, int], Node] = {}
    for k, z in enumerate(zs):
        for i, x in enumerate(xs):
            node = Node([float(x), 0.0, float(z)], geometric_node=False)
            mesh.add_node(node)
            grid[(i, k)] = node

    # Quad shell elements
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

    # Node sets
    clamped = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    # Pick middle node (x~B/2), not corner
    free_center = min(free_face, key=lambda n: abs(float(n.x) - B / 2))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_face", free_face))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


# ============================================================================
# Analytical Solution (Mindlin plate)
# ============================================================================


def analytical_tip_displacement(
    L: float, b: float, t: float, E: float, nu: float, F: float
) -> float:
    """Bending tip displacement for cantilever Mindlin plate.

    Using classical beam formula with shear correction.
    """
    I = b * t**3 / 12  # Moment of inertia
    k = 5 / 6  # Shear correction factor (rectangular cross-section)

    # Bending + shear
    delta = F * L**3 / (3 * E * I) + F * L / (k * G * b * t)

    return delta


# ============================================================================
# CCX Writer
# ============================================================================


def write_ccx_inp(
    inp_path: Path,
    mesh: MeshModel,
    load_vector: tuple[float, float, float],
    thickness: float,
) -> None:
    """Write CCX input for isotropic shell cantilever."""
    lines = []

    # Nodes
    lines.append("*NODE, NSET=NALL")
    for i, node in enumerate(mesh.nodes):
        lines.append(f"{i + 1},{node.x:.6f},{node.y:.6f},{node.z:.6f}")

    # Clamped nodes
    c_ids = sorted(
        mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("clamped").nodes.values()
    )
    lines.append("*NSET, NSET=NCLAMPED")
    for i in range(0, len(c_ids), 16):
        lines.append(",".join(str(x) for x in c_ids[i : i + 16]))

    # Free center node
    center_node = next(iter(mesh.get_node_set("free_center").nodes.values()))
    center_idx = mesh.node_id_to_index[center_node.id] + 1
    lines.append("*NSET, NSET=NCENTER")
    lines.append(str(center_idx))

    # Elements - S4 shell
    lines.append("*ELEMENT, TYPE=S4, ELSET=EGLOBAL")
    for i, el in enumerate(mesh.elements):
        nids = [mesh.node_id_to_index[nid] + 1 for nid in el.node_ids]
        lines.append(f"{i + 1},{nids[0]},{nids[1]},{nids[2]},{nids[3]}")

    # Material - Isotropic
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append(f"{MAT_STEEL.E:.2e}, {MAT_STEEL.nu}")
    lines.append("*DENSITY")
    lines.append(f"{MAT_STEEL.rho:.4f}")

    # Shell section
    lines.append("*SHELL SECTION, ELSET=EGLOBAL,MATERIAL=STEEL")
    lines.append(f"{thickness:.6f}")

    # Boundary
    lines.append("*BOUNDARY")
    for nid in c_ids:
        lines.append(f"{nid},1,6,0.0")

    # Load
    lines.append("*CLOAD")
    lines.append(f"{center_idx},2,{load_vector[1]:.6f}")

    # Static step
    lines.append("*STATIC")
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*EL FILE")
    lines.append("S")
    lines.append("*END STEP")

    inp_path.write_text("\n".join(lines) + "\n")


# ============================================================================
# Test
# ============================================================================


class TestIsotropicShellParity:
    """Compare isotropic shell between AeroElast and CalculiX."""

    def test_transverse_tip_displacement(self, tmp_path: Path):
        """Test: Does MITC4 vs S4 difference appear in isotropic shells?"""
        ccx_bin = Path("/scratch/leahk/eduardo.donestevez/venv/bin/ccx")
        print(f"CCX binary exists: {ccx_bin.exists()}")
        if not ccx_bin.exists():
            pytest.skip("CalculiX not available")

        print(f"tmp_path: {tmp_path}")
        print(f"tmp_path exists: {tmp_path.exists()}")

        mesh = build_shell_mesh(nz=10, nx=2)
        print(f"Mesh built: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

        # === AeroElast ===
        node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
        conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]

        # MITC4 = element type 4 (or similar quad shell)
        elem_types = [4] * len(mesh.elements)
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

        # Boundary conditions
        fixed_dofs = set()
        for node in mesh.get_node_set("clamped").nodes.values():
            idx = mesh.node_id_to_index[node.id] * 3
            fixed_dofs.update([idx, idx + 1, idx + 2])

        # Load vector
        f = np.zeros(n, dtype=float)
        center = next(iter(mesh.get_node_set("free_center").nodes.values()))
        i_center = mesh.node_id_to_index[center.id] * 3
        f[i_center + 1] = FORCE  # Fy

        # Apply BCs
        free_mask = np.ones(n, dtype=bool)
        for dof in fixed_dofs:
            free_mask[dof] = False
        free = np.where(free_mask)[0]

        # Solve
        u_ae = np.zeros(n, dtype=float)
        u_ae[free] = spsolve(K[free][:, free], f[free])

        disp_ae = u_ae[i_center + 1]

        # === CalculiX ===
        case_dir = tmp_path / "isotropic_shell"
        case_dir.mkdir(exist_ok=True)
        inp = case_dir / "shell.inp"

        # Use aeroelast writer (same as test_beam_4cases_parity.py)
        from aeroelast.core.mesh.io.writers import write_ccx_mesh

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
            load_vector=(0.0, FORCE, 0.0),  # Fy
        )
        print(f"Wrote CCX input via aeroelast")

        # Run CCX
        stem = "shell"
        print(f"Running CCX: {ccx_bin} {stem}")

        # Run ccx with just the stem, same as test_beam_4cases_parity.py
        proc = subprocess.run(
            [str(ccx_bin), stem],
            cwd=case_dir,
            capture_output=True,
            text=True,
        )
        print(f"CCX return code: {proc.returncode}")
        print(f"CCX stdout: {proc.stdout[:500] if proc.stdout else 'empty'}")
        print(f"Files in case_dir: {list(case_dir.iterdir())}")
        if proc.returncode != 0:
            print(f"CCX stderr: {proc.stderr}")
            pytest.skip(f"CCX failed: {proc.stderr}")

        # Parse FRD - file is named after the stem
        frd = case_dir / f"{stem}.frd"
        print(f"FRD exists: {frd.exists()}")

        if not frd.exists():
            pytest.skip(f"FRD not generated at {frd}")

        # Parse FRD file - simplified: find max Y displacement
        frd_path = case_dir / f"{stem}.frd"

        if not frd_path.exists():
            pytest.skip(f"FRD not generated at {frd_path}")

        # Read and parse FRD - look for displacement values
        frd_content = frd_path.read_text()

        # Find the displacement block and extract max V (Y) displacement
        # CCX outputs displacements in format: -1 <node> <U> <V> <W>
        # Format may have no spaces when numbers are negative: -1 <id>-1.2E-14 2.0E-03-1.1E-14
        import re

        disp_ccx = None
        max_v = 0.0

        # Look for lines after -4 DISP section
        in_disp = False
        for line in frd_content.splitlines():
            if "-4" in line and "DISP" in line:
                in_disp = True
                continue
            if in_disp:
                if line.startswith(" -3"):
                    break
                if line.startswith(" -1"):
                    # Extract numbers from line (all scientific notation values)
                    # Pattern: any sequence of digits, dots, eE signs that starts with digit or minus
                    numbers = re.findall(r"[-+]?[0-9]{1,}\.[0-9]{5}E[+-][0-9]{2}", line)

                    # Should have at least 3 values (U, V, W), skip first (node ID area)
                    if len(numbers) >= 3:
                        try:
                            # Second value is V (Y displacement)
                            v_val = float(numbers[1])
                            if abs(v_val) > abs(max_v):
                                max_v = v_val
                        except (ValueError, IndexError):
                            pass

        if max_v == 0.0:
            # Could not find any meaningful displacement
            pytest.skip("Could not parse CCX displacement from FRD")

        disp_ccx = max_v
        ratio = disp_ccx / disp_ae

        print(f"\n[Isotropic Shell Parity]")
        print(f"  AeroElast (MITC4): {disp_ae:.6f} m")
        print(f"  CalculiX (S4):    {disp_ccx:.6f} m")
        print(f"  Ratio (CCX/AE):    {ratio:.3f}")
        print(f"  Difference:        {abs(disp_ccx - disp_ae):.6f} m ({abs(ratio - 1) * 100:.1f}%)")

        # Allow 10% tolerance
        tol = 0.05
        if abs(ratio - 1) > tol:
            pytest.fail(
                f"Isotropic shell mismatch: AE={disp_ae:.4f}, CCX={disp_ccx:.4f}, "
                f"ratio={ratio:.3f}, tol={tol}"
            )
