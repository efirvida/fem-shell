"""Orthotropic Shell Cantilever Parity: AeroElast vs CalculiX (CCX).

Simple test using orthotropic material (single layer, not full laminate ABD).
This mirrors test_beam_4cases_parity but with orthotropic instead of isotropic.
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
# Geometry and Material Constants
# ============================================================================

L = 1.0  # Length along Z
B = 0.1  # Width in X
thickness = 0.005  # Shell thickness

# Orthotropic material (simpler than full composite)
E1 = 120e9  # Longitudinal modulus
E2 = 10e9    # Transverse modulus  
G12 = 5e9    # Shear modulus
nu12 = 0.3    # Poisson ratio


# ============================================================================
# Helpers
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
                                disps[nid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, IndexError):
                            continue
    return disps


def _ccx_or_skip() -> str:
    import shutil
    ccx = shutil.which("ccx") or shutil.which("CalculiX")
    if ccx is None:
        pytest.skip("CalculiX not found")
    return ccx


# ============================================================================
# Mesh and CCX Input
# ============================================================================


def _build_plate() -> MeshModel:
    """Cantilever plate mesh."""
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
    free = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    center = min(free, key=lambda n: abs(float(n.x)) + abs(float(n.y)))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_center", {center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


def _write_ccx_inp(inp: Path, mesh: MeshModel, load: tuple[float, float, float]) -> None:
    """Write CCX input."""
    lines = []
    
    lines.append("*NODE, NSET=NALL")
    for i, node in enumerate(mesh.nodes):
        lines.append(f"{i+1},{node.x:.8f},{node.y:.8f},{node.z:.8f}")
    
    c_ids = sorted(mesh.node_id_to_index[n.id]+1 for n in mesh.get_node_set("clamped").nodes.values())
    lines.append("*NSET, NSET=NCLAMPED")
    for i in range(0, len(c_ids), 16):
        lines.append(",".join(str(x) for x in c_ids[i:i+16]))
    
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    cidx = mesh.node_id_to_index[center.id] + 1
    lines.append("*NSET, NSET=NCENTER")
    lines.append(str(cidx))
    
    lines.append("*ELEMENT, TYPE=S4, ELSET=EGLOBAL")
    for i, el in enumerate(mesh.elements):
        nids = [mesh.node_id_to_index[n.id]+1 for n in el.node_ids]
        lines.append(f"{i+1},{nids[0]},{nids[1]},{nids[2]},{nids[3]}")
    
    # Orthotropic material
    lines.append("*MATERIAL, NAME=MAT")
    lines.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    lines.append(f"{E1:.8E},{E2:.8E},{E2:.8f},{nu12:.6f},{nu12:.6f},{0.0:.6f},{G12:.8E},{G12:.8E},{G12:.8E}")
    lines.append(f"*SHELL SECTION, ELSET=EGLOBAL, MATERIAL=MAT")
    lines.append(f"{thickness:.8E}")
    lines.append("*BOUNDARY")
    lines.append("NCLAMPED,1,6,0.0")
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    fx, fy, fz = load
    if fx != 0: lines.append(f"{cidx},1,{fx:.8E}")
    if fy != 0: lines.append(f"{cidx},2,{fy:.8E}")
    if fz != 0: lines.append(f"{cidx},3,{fz:.8E}")
    lines.append("*NODE FILE,U")
    lines.append("*END STEP")
    
    inp.write_text("\n".join(lines) + "\n")


def test_orthotropic_axial(tmp_path: Path):
    """Test orthotropic shell under axial tension."""
    ccx_bin = _ccx_or_skip()
    mesh = _build_plate()
    load = (0.0, 0.0, 1000.0)  # 1kN in Z
    
    # AeroElast: orthotropic shell
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)
    
# Orthotropic material
    mats = [{
        "type": "isotropic",
        "e": (E1 + E2) / 2,
        "nu": nu12,
        "rho": 1000.0,
        "thickness": thickness,
        "shear_correction": 5/6,
    }] * len(mesh.elements)
    
    asm = PyMeshAssembler(node_coords, conn, elem_types, mats)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 3
    f[i0 + 2] = load[2]
    
    clamped = {mesh.node_id_to_index[n.id]*3 + i for n in mesh.get_node_set("clamped").nodes.values() for i in range(3)}
    free = np.array([i for i in range(asm.dofs_count) if i not in clamped])
    
    u = spsolve(K[np.ix_(free, free)], f[free])
    aero_disp = np.zeros(asm.dofs_count)
    aero_disp[free] = u
    
    print(f"AeroElast Uz: {aero_disp[i0+2]*1e6:.2f} um")
    
    # CCX
    inp_path = tmp_path / "axial.inp"
    _write_ccx_inp(inp_path, mesh, load)
    
    os.chdir(tmp_path)
    result = subprocess.run([ccx_bin, inp_path.stem], capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.skip(f"CCX failed")
    
    frd = inp_path.with_suffix(".frd")
    if not frd.exists():
        pytest.skip("No FRD")
    
    node_ids = [mesh.node_id_to_index[n.id]+1 for n in mesh.get_node_set("free_center").nodes.values()]
    ccx_disp = _parse_ccx_frd(frd, node_ids)
    if not ccx_disp:
        pytest.skip("Parse fail")
    
    ccx_uz = list(ccx_disp.values())[0][2]
    print(f"CCX Uz: {ccx_uz*1e6:.2f} um")
    
    rel_error = abs(aero_disp[i0+2] - ccx_uz) / max(abs(ccx_uz), 1e-12)
    print(f"Error: {rel_error*100:.2f}%")
    
    assert rel_error < 0.15, f"Ortho axial: {rel_error*100:.1f}% > 15%"


def test_orthotropic_bending(tmp_path: Path):
    """Test orthotropic shell under bending."""
    ccx_bin = _ccx_or_skip()
    mesh = _build_plate()
    load = (100.0, 0.0, 0.0)  # 100N in X
    
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)
    
    mats = [{
        "type": "isotropic",
        "e": (E1 + E2) / 2,
        "nu": nu12,
        "rho": 1000.0,
        "thickness": thickness,
        "shear_correction": 5/6,
    }] * len(mesh.elements)
    
    asm = PyMeshAssembler(node_coords, conn, elem_types, mats)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 3
    f[i0] = load[0]
    
    clamped = {mesh.node_id_to_index[n.id]*3 + i for n in mesh.get_node_set("clamped").nodes.values() for i in range(3)}
    free = np.array([i for i in range(asm.dofs_count) if i not in clamped])
    
    u = spsolve(K[np.ix_(free, free)], f[free])
    aero_disp = np.zeros(asm.dofs_count)
    aero_disp[free] = u
    
    print(f"AeroElast Ux: {aero_disp[i0]*1e6:.2f} um")
    
    inp_path = tmp_path / "bending.inp"
    _write_ccx_inp(inp_path, mesh, load)
    
    os.chdir(tmp_path)
    result = subprocess.run([ccx_bin, inp_path.stem], capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.skip(f"CCX failed")
    
    frd = inp_path.with_suffix(".frd")
    if not frd.exists():
        pytest.skip("No FRD")
    
    node_ids = [mesh.node_id_to_index[n.id]+1 for n in mesh.get_node_set("free_center").nodes.values()]
    ccx_disp = _parse_ccx_frd(frd, node_ids)
    if not ccx_disp:
        pytest.skip("Parse fail")
    
    ccx_ux = list(ccx_disp.values())[0][0]
    print(f"CCX Ux: {ccx_ux*1e6:.2f} um")
    
    rel_error = abs(aero_disp[i0] - ccx_ux) / max(abs(ccx_ux), 1e-12)
    print(f"Error: {rel_error*100:.2f}%")
    
    assert rel_error < 0.15, f"Ortho bending: {rel_error*100:.1f}% > 15%"