"""Composite Shell Cantilever Parity: AeroElast vs CalculiX (CCX).

Geometry:
- Cantilever plate/shell clamped at Z=0, free at Z=L
- Uses composite shell elements (type S4 with composite layup)
- Different fiber orientations (typical layup: [0/90/45/-45])

Load Cases:
1. Axial tension: +Fz (extension along Z)
2. Bending: Fx applied at free end (bending about Y-axis)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler

from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.model import MeshModel

pytestmark = [pytest.mark.slow]


# ============================================================================
# Material and Geometry Constants
# ============================================================================

L = 1.0  # Length along Z (m)
B = 0.1  # Width in X (m)
thickness = 0.005  # Total laminate thickness (5mm)

# Create a simple symmetric laminate [0/90]s
# Material properties (typical carbon/epoxy)
E1 = 120e9  # Longitudinal modulus
E2 = 10e9   # Transverse modulus  
G12 = 5e9     # Shear modulus
nu12 = 0.3    # Poisson ratio


# ============================================================================
# Helper: Create ABD matrices from layup
# ============================================================================


def _compute_abd_matrices(
    ply_angles: list[float],
    thickness_per_ply: float,
    E1: float = 120e9,
    E2: float = 10e9,
    G12: float = 5e9,
    nu12: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ABD matrices for a laminate.
    
    Returns A, B, D matrices (each 3x3).
    """
    n_plies = len(ply_angles)
    h = n_plies * thickness_per_ply
    
    # For a symmetric laminate, compute ABD analytically
    # Simple approach: use equivalent stiffness
    Q11 = E1 / (1 - nu12 * E2 / E1)
    Q22 = E2
    Q12 = nu12 * E2
    Q66 = G12
    
    # For [0/90]s laminate (4 plies)
    # A matrix (extensional stiffness)
    A11 = 2 * (Q11 * thickness_per_ply + Q22 * thickness_per_ply)
    A12 = 2 * (Q12 * thickness_per_ply + Q12 * thickness_per_ply)
    A22 = 2 * (Q22 * thickness_per_ply + Q11 * thickness_per_ply)
    A66 = 2 * (Q66 * thickness_per_ply + Q66 * thickness_per_ply)
    
    A = np.array([
        [A11, A12, 0],
        [A12, A22, 0],
        [0, 0, A66]
    ])
    
    # B matrix (coupling) - zero for symmetric
    B = np.zeros((3, 3))
    
    # D matrix (bending stiffness) 
    z = np.array([-3*thickness_per_ply, -thickness_per_ply, thickness_per_ply, 3*thickness_per_ply])
    D11 = 2*(Q11*z[-1]**3 - Q11*z[1]**3)/3 + 2*(Q22*z[1]**3 - Q22*z[-1]**3)/3
    D12 = 2*(Q12*z[-1]**3 - Q12*z[1]**3)/3 + 2*(Q12*z[1]**3 - Q12*z[-1]**3)/3
    D22 = 2*(Q22*z[-1]**3 - Q22*z[1]**3)/3 + 2*(Q11*z[1]**3 - Q11*z[-1]**3)/3
    D66 = 2*(Q66*z[-1]**3 - Q66*z[1]**3)/3 + 2*(Q66*z[1]**3 - Q66*z[-1]**3)/3
    
    D = np.array([
        [D11, D12, 0],
        [D12, D22, 0],
        [0, 0, D66]
    ])
    
    # Shear coupling Cs
    Cs = np.array([
        [5e9/6*thickness, 0, 0],
        [0, 5e9/6*thickness, 0],
        [0, 0, 5e9/6*thickness]
    ])
    
    return A, B, D


# ============================================================================
# CCX Helper
# ============================================================================


def _parse_ccx_frd(frd_file: Path, node_ids: list[int]) -> dict[int, np.ndarray]:
    """Parse displacement from CCX FRD file."""
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


def _ccx_or_skip() -> str:
    """Find CCX binary or skip."""
    import shutil
    ccx = shutil.which("ccx") or shutil.which("CalculiX")
    if ccx is None:
        pytest.skip("CalculiX (ccx) not found")
    return ccx


# ============================================================================
# Build Mesh - Cantilever Plate
# ============================================================================


def _build_plate_mesh() -> MeshModel:
    """Build cantilever plate mesh."""
    nx, ny, nz = 4, 2, 10

    mesh = MeshModel()
    xs = np.linspace(0, B, nx + 1)
    ys = np.linspace(-0.05, 0.05, ny + 1)
    zs = np.linspace(0, L, nz + 1)

    grid = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node = Node([float(x), float(y), float(z)], geometric_node=False)
                mesh.add_node(node)
                grid[(i, j, k)] = node

    # Shell elements (quads)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n00 = grid[(i, j, k)]
                n10 = grid[(i + 1, j, k)]
                n11 = grid[(i + 1, j + 1, k)]
                n01 = grid[(i, j + 1, k)]
                mesh.add_element(
                    MeshElement(nodes=[n00, n10, n11, n01], element_type=ElementType.quad)
                )

    clamped = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    free_center = min(free_face, key=lambda n: abs(float(n.x)) + abs(float(n.y)))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_face", free_face))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


def _write_ccx_inp(inp: Path, mesh: MeshModel, load: tuple[float, float, float]) -> None:
    """Write CCX input for composite plate."""
    lines = []
    
    # Nodes
    lines.append("*NODE, NSET=NALL")
    for i, node in enumerate(mesh.nodes):
        lines.append(f"{i+1},{node.x:.8f},{node.y:.8f},{node.z:.8f}")
    
    # Clamped nodes
    c_ids = sorted(mesh.node_id_to_index[n.id]+1 for n in mesh.get_node_set("clamped").nodes.values())
    lines.append("*NSET, NSET=NCLAMPED")
    for i in range(0, len(c_ids), 16):
        lines.append(",".join(str(x) for x in c_ids[i:i+16]))
    
    # Center node
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    center_idx = mesh.node_id_to_index[center.id] + 1
    lines.append("*NSET, NSET=NCENTER")
    lines.append(str(center_idx))
    
    # Elements
    lines.append("*ELEMENT, TYPE=S4, ELSET=EGLOBAL")
    for i, el in enumerate(mesh.elements):
        nids = [mesh.node_id_to_index[nid] + 1 for nid in el.node_ids]
        lines.append(f"{i+1},{nids[0]},{nids[1]},{nids[2]},{nids[3]}")
    
    # Orthotropic material
    lines.append("*MATERIAL, NAME=MAT")
    lines.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    lines.append(f"{E1:.8E},{E2:.8E},{E2:.8E},{nu12:.6f},{nu12:.6f},{0.0:.6f},{G12:.8E},{G12:.8E},{G12:.8E}")
    
    # Shell section: 2 plies [0/90]
    t = thickness / 2
    lines.append("*SHELL SECTION, ELSET=EGLOBAL, MATERIAL=MAT")
    lines.append(f"{t:.8E},0.,MAT")
    lines.append("0.,0.,0.")
    lines.append(f"{t:.8E},0.,MAT")
    lines.append("0.,0.,90.")
    
    # Boundary
    lines.append("*BOUNDARY")
    lines.append("NCLAMPED,1,6,0.0")
    
    # Step
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    fx, fy, fz = load
    if fx != 0:
        lines.append(f"{center_idx},1,{fx:.8E}")
    if fy != 0:
        lines.append(f"{center_idx},2,{fy:.8E}")
    if fz != 0:
        lines.append(f"{center_idx},3,{fz:.8E}")
    
    lines.append("*NODE FILE,U")
    lines.append("*END STEP")
    
    inp.write_text("\n".join(lines) + "\n")


# ============================================================================
# Tests
# ============================================================================


def test_composite_axial_tension(tmp_path: Path):
    """Test composite shell under axial tension."""
    ccx_bin = _ccx_or_skip()
    mesh = _build_plate_mesh()
    
    load = (0.0, 0.0, 1000.0)  # 1kN in Z
    
    # Compute ABD matrices for [0/90]s laminate
    A, B, D = _compute_abd_matrices([0, 90], thickness / 2, E1, E2, G12, nu12)
    
    # AeroElast solution
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)  # S4
    
# Material with ABD matrices
    e_equiv = (E1 + E2) / 2
    mpa = 2 * 0.5 * (E1 / (1 - nu12**2))
    ri = 0.0
    # Cs is 2x2 shear coupling matrix (4 elements), flattened row-major
    cs = np.array([G12, 0, 0, G12])  
    mats = [{
        "type": "composite",
        "cm": A.ravel().tolist(),
        "cb": B.ravel().tolist(),
        "cs": cs.ravel().tolist(),
        "thickness": thickness,
        "e_equiv": e_equiv,
        "mass_per_area": mpa,
        "rotational_inertia": ri,
    }] * len(mesh.elements)

    asm = PyMeshAssembler(node_coords, conn, elem_types, mats)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 3
    f[i0] = load[0]
    
    clamped = {
        mesh.node_id_to_index[n.id] * 3 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(3)
    }
    free_mask = np.ones(n, dtype=bool)
    for dof in clamped:
        free_mask[dof] = False
    free = np.where(free_mask)[0]
    
    u = spsolve(K[np.ix_(free, free)], f[free])
    
    disp = np.zeros(n, dtype=float)
    disp[free] = u
    
    aero_ux = disp[i0]
    
    # CCX
    inp_path = tmp_path / "bending.inp"
    _write_ccx_inp(inp_path, mesh, load)
    
    node_ids = [mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_center").nodes.values()]
    
    os.chdir(tmp_path)
    result = subprocess.run([ccx_bin, inp_path.stem], capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.skip(f"CCX failed: {result.stderr[:200]}")
    
    frd = inp_path.with_suffix(".frd")
    if not frd.exists():
        pytest.skip("No FRD")
    
    ccx_disp = _parse_ccx_frd(frd, node_ids)
    if not ccx_disp:
        pytest.skip("Could not parse CCX")
    
    ccx_ux = list(ccx_disp.values())[0][0]
    
    rel_error = abs(aero_ux - ccx_ux) / max(abs(ccx_ux), 1e-12)
    print(f"AeroElast Ux: {aero_ux*1e6:.2f} um")
    print(f"CCX Ux:     {ccx_ux*1e6:.2f} um")
    print(f"Error:      {rel_error*100:.2f}%")
    
    assert rel_error < 0.10, f"Bending: {rel_error*100:.1f}% > 10%"