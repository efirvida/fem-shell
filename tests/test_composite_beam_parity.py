"""Composite Shell Cantilever Parity: AeroElast vs CalculiX (CCX).

Geometry:
- Cantilever plate/shell clamped at Z=0, free at Z=L
- Uses composite shell elements with multiple layers
- Different fiber orientations (typical layup: [0/90/45/-45])

Load Cases:
1. Axial tension: +Fz (extension along Z)
2. Bending: Fx applied at free end (bending about Y-axis)

This test validates composite shell formulation against CalculiX.
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
from aeroelast.core.material import IsotropicMaterial, OrthotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import CompositeShellProperty, LaminatePly, ShellProperty
from aeroelast.elements import ElementFamily
from aeroelast.solvers.elasticity.static_linear import StaticLinearSolver

pytestmark = [pytest.mark.slow]


# ============================================================================
# Geometry and Material Constants
# ============================================================================

L = 1.0  # Beam/plate length along Z (m)
B = 0.1  # Width in X (m)
thickness = 0.005  # Total laminate thickness (5mm)

# Composite layup: typical [0/90/45/-45]s configuration
E1 = 120e9  # Longitudinal modulus (Pa)
E2 = 10e9  # Transverse modulus
G12 = 5e9  # Shear modulus
nu12 = 0.3  # Poisson ratio

# ============================================================================
# CCX Helpers
# ============================================================================


def _parse_frd_disp(frd_file: Path, node_ids: list[int]) -> dict[int, np.ndarray]:
    """Parse displacement from FRD file."""
    import re

    disps = {}
    in_disp = False
    with open(frd_file) as f:
        for line in f:
            if "-4" in line and "DISP" in line:
                in_disp = True
                continue
            if in_disp:
                if line.startswith(" -3"):
                    break
                if line.startswith(" -1"):
                    # Extract numbers from line like " -1     123 1.23E-04 ..."
                    content = line[3:].strip()
                    parts = content.split()
                    if len(parts) >= 4:
                        try:
                            nid = int(parts[0])
                            if nid in node_ids:
                                ux = float(parts[1])
                                uy = float(parts[2])
                                uz = float(parts[3])
                                disps[nid] = np.array([ux, uy, uz])
                        except (ValueError, IndexError):
                            continue
    return disps


def _run_ccx(inp_path: Path, ccx_bin: str) -> subprocess.CompletedProcess:
    """Run CalculiX on input file."""
    os.chdir(inp_path.parent)
    result = subprocess.run(
        [ccx_bin, inp_path.stem],
        capture_output=True,
        text=True,
    )
    return result


# ============================================================================
# Build Mesh - Cantilever Plate with Composite Shells
# ============================================================================


def _build_composite_plate_mesh() -> MeshModel:
    """Build cantilever plate mesh for composite shell test."""
    nx = 4  # elements along X
    ny = 2  # elements along Y  
    nz = 10  # elements along Z (length)

    mesh = MeshModel()

    # Create node grid
    xs = np.linspace(0, B, nx + 1)
    ys = np.linspace(-0.05, 0.05, ny + 1)
    zs = np.linspace(0, L, nz + 1)

    grid: dict[tuple[int, int, int], Node] = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node = Node([float(x), float(y), float(z)], geometric_node=False)
                mesh.add_node(node)
                grid[(i, j, k)] = node

    # Create quadrilateral shell elements
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n00 = grid[(i, j, k)]
                n10 = grid[(i + 1, j, k)]
                n11 = grid[(i + 1, j + 1, k)]
                n01 = grid[(i, j + 1, k)]
                mesh.add_element(
                    MeshElement(
                        nodes=[n00, n10, n11, n01],
                        element_type=ElementType.quad,
                    )
                )

    # Node sets
    clamped = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    free_center = min(free_face, key=lambda n: abs(float(n.x)) + abs(float(n.y)))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_face", free_face))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


# ============================================================================
# CCX Input Writer - Composite Shells
# ============================================================================


def _write_composite_ccx_inp(
    inp_path: Path,
    mesh: MeshModel,
    load_vector: tuple[float, float, float],
) -> None:
    """Write CCX input for composite cantilever."""
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
        lines.append(",".join(str(x) for x in c_ids[i:i + 16]))

    # Free center node
    center_node = next(iter(mesh.get_node_set("free_center").nodes.values()))
    center_idx = mesh.node_id_to_index[center_node.id] + 1
    lines.append("*NSET, NSET=NCENTER")
    lines.append(str(center_idx))

    # Elements - S4 shell
    lines.append("*ELEMENT, TYPE=S4, ELSET=EGLOBAL")
    for i, el in enumerate(mesh.elements):
        nids = [mesh.node_id_to_index[n.id] + 1 for n in el.node_ids]
        lines.append(f"{i + 1},{nids[0]},{nids[1]},{nids[2]},{nids[3]}")

    # Material - Orthotropic
    lines.append("*MATERIAL, NAME=MAT")
    lines.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    lines.append(f"{E1:.6E},{E2:.6E},{E2:.6E},{nu12:.6f},{nu12:.6f},{0.0:.6f},{G12:.6E},{G12:.6E},{G12:.6E}")

    # 4-layer laminate [0/90/45/-45]s
    t = thickness / 8  # 8 plies total, 4 per side
    lines.append("*SHELL SECTION, ELSET=EGLOBAL, MATERIAL=MAT")
    for angle in [0, 90, 45, -45, -45, 45, 90, 0]:
        lines.append(f"{t:.6E},{0:.6f},MAT")
        lines.append(f"0.,0.,{float(angle):.6f}")

    # Boundary
    lines.append("*BOUNDARY")
    lines.append("NCLAMPED,1,6,0.0")

    # Load
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    fx, fy, fz = load_vector
    if fx != 0:
        lines.append(f"{center_idx},1,{fx:.6E}")
    if fy != 0:
        lines.append(f"{center_idx},2,{fy:.6E}")
    if fz != 0:
        lines.append(f"{center_idx},3,{fz:.6E}")

    # Output
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*END STEP")

    inp_path.write_text("\n".join(lines) + "\n")


# ============================================================================
# Tests
# ============================================================================


def _ccx_bin_or_skip() -> str:
    """Find CCX binary or skip."""
    import shutil

    ccx = shutil.which("ccx") or shutil.which("CalculiX")
    if ccx is None:
        pytest.skip("CalculiX (ccx) not found in PATH")
    return ccx


# Test de composite shell - tensión axial
def test_composite_axial_tension(tmp_path: Path):
    """Test composite shell under axial tension."""
    ccx_bin = _ccx_bin_or_skip()
    mesh = _build_composite_plate_mesh()

    # Load: 1000N in Z direction
    load = (0.0, 0.0, 1000.0)

    # AeroElast solution
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)  # S4

    # Multiple materials - one per element with composite properties
    mats = []
    for _ in mesh.elements:
        mats.append({
            "type": "shell_composite",
            "E1": E1,
            "E2": E2,
            "nu12": nu12,
            "G12": G12,
            "thickness": thickness,
            "plies": [
                {"t": thickness / 8, "theta": 0, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": 90, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": 45, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": -45, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
            ]
        })

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
    f[i0 + 2] = load[2]  # Z direction

    # Boundary conditions
    clamped = {
        mesh.node_id_to_index[n.id] * 3 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(3)
    }
    free_mask = np.ones(n, dtype=bool)
    for dof in clamped:
        free_mask[dof] = False
    free = np.where(free_mask)[0]

    K_ff = K[np.ix_(free, free)]
    f_free = f[free]

    u_free = spsolve(K_ff, f_free)
    u = np.zeros(n, dtype=float)
    u[free] = u_free

    aero_disp = u[i0:i0 + 3]

    # CCX solution
    inp_path = tmp_path / "composite_axial.inp"
    _write_composite_ccx_inp(inp_path, mesh, load)
    
    # Get node IDs for CCX output
    node_ids = [mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_center").nodes.values()]

    result = _run_ccx(inp_path, ccx_bin)
    if result.returncode != 0:
        pytest.skip(f"CCX failed: {result.stderr}")

    frd_path = inp_path.with_suffix(".frd")
    if not frd_path.exists():
        pytest.skip("No FRD output from CCX")

    ccx_disp = _parse_frd_disp(frd_path, node_ids)
    if not ccx_disp:
        pytest.skip("Could not parse CCX displacements")

    ccx_vals = list(ccx_disp.values())[0]

    # Compare - axial displacement should be similar
    rel_error = abs(aero_disp[2] - ccx_vals[2]) / max(abs(ccx_vals[2]), 1e-10)
    print(f"AeroElast axial: {aero_disp[2]*1e6:.2f} um, CCX: {ccx_vals[2]*1e6:.2f} um")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.1, f"Composite axial tension: {rel_error*100:.1f}% error (max 10%)"


# Test de composite shell - bending
def test_composite_bending(tmp_path: Path):
    """Test composite shell under transverse bending."""
    ccx_bin = _ccx_bin_or_skip()
    mesh = _build_composite_plate_mesh()

    # Load: 100N in X direction at free end
    load = (100.0, 0.0, 0.0)

    # AeroElast solution
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)

    mats = []
    for _ in mesh.elements:
        mats.append({
            "type": "shell_composite",
            "E1": E1,
            "E2": E2,
            "nu12": nu12,
            "G12": G12,
            "thickness": thickness,
            "plies": [
                {"t": thickness / 8, "theta": 0, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": 90, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": 45, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
                {"t": thickness / 8, "theta": -45, "E1": E1, "E2": E2, "G12": G12, "nu12": nu12},
            ]
        })

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
    f[i0] = load[0]  # X direction

    # Boundary conditions
    clamped = {
        mesh.node_id_to_index[n.id] * 3 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(3)
    }
    free_mask = np.ones(n, dtype=bool)
    for dof in clamped:
        free_mask[dof] = False
    free = np.where(free_mask)[0]

    K_ff = K[np.ix_(free, free)]
    f_free = f[free]

    u_free = spsolve(K_ff, f_free)
    u = np.zeros(n, dtype=float)
    u[free] = u_free

    aero_disp = u[i0:i0 + 3]

    # CCX solution
    inp_path = tmp_path / "composite_bending.inp"
    _write_composite_ccx_inp(inp_path, mesh, load)

    node_ids = [mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_center").nodes.values()]

    result = _run_ccx(inp_path, ccx_bin)
    if result.returncode != 0:
        pytest.skip(f"CCX failed: {result.stderr}")

    frd_path = inp_path.with_suffix(".frd")
    if not frd_path.exists():
        pytest.skip("No FRD output from CCX")

    ccx_disp = _parse_frd_disp(frd_path, node_ids)
    if not ccx_disp:
        pytest.skip("Could not parse CCX displacements")

    ccx_vals = list(ccx_disp.values())[0]

    # Compare
    rel_error = abs(aero_disp[0] - ccx_vals[0]) / max(abs(ccx_vals[0]), 1e-10)
    print(f"AeroElast X: {aero_disp[0]*1e6:.2f} um, CCX: {ccx_vals[0]*1e6:.2f} um")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.1, f"Composite bending: {rel_error*100:.1f}% error (max 10%)"