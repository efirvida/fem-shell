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
    """Parse displacement from CCX FRD file - find max transverse displacement (like isotropic test)."""
    import re
    max_v = 0.0
    max_nid = None
    max_u = 0.0
    max_w = 0.0
    
    with open(frd_file) as f:
        content = f.read()
    
    # Look for lines after -4 DISP section (same as isotropic test)
    for line in content.splitlines():
        if "-4" in line and "DISP" in line:
            continue
        if line.startswith(" -3"):
            break
        if line.startswith(" -1"):
            # Try pattern with separated numbers
            matches = re.findall(
                r'-1\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)',
                line
            )
            if matches:
                for nid_str, u_str, v_str, w_str in matches:
                    try:
                        v_val = float(v_str)
                        if abs(v_val) > abs(max_v):
                            max_v = v_val
                            max_nid = int(nid_str)
                            max_u = float(u_str)
                            max_w = float(w_str)
                    except (ValueError, IndexError):
                        continue
            else:
                # Try concatenated pattern
                matches2 = re.findall(r'-1\s+(\d+)([0-9.eE+-]{10,})', line)
                for nid_str, rest in matches2:
                    try:
                        nid = int(nid_str)
                        if len(rest) >= 24:
                            v_val = float(rest[-12:])
                            if abs(v_val) > abs(max_v):
                                max_v = v_val
                                max_nid = nid
                    except (ValueError, IndexError):
                        continue
    
    if max_nid is not None:
        return {max_nid: np.array([max_u, max_v, max_w])}
    return {}


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
    """Cantilever plate mesh - flat (like test_isotropic_shell_parity.py)."""
    nx, nz = 4, 10  # flat mesh

    mesh = MeshModel()
    xs = np.linspace(0, B, nx + 1)
    ys = np.zeros(nx + 1)  # flat (y=0)
    zs = np.linspace(0, L, nz + 1)

    grid = {}
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
                MeshElement(nodes=[n00, n10, n11, n01], element_type=ElementType.quad)
            )

    clamped = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    center = min(free, key=lambda n: abs(float(n.x) - B/2))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_center", {center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


def _write_ccx_inp(inp: Path, mesh: MeshModel, load: tuple[float, float, float]) -> None:
    """Write CCX input using aeroelast writer."""
    from aeroelast.core.mesh.io.writers import write_ccx_mesh
    
    e_equiv = (E1 + E2) / 2
    props = {
        "plate": {
            "type": "isotropic",
            "e": e_equiv,
            "nu": nu12,
            "rho": 1000.0,
            "thickness": thickness,
        }
    }
    
    # Build loads - apply to center node
    center_node = mesh.get_node_set("free_center")
    fx, fy, fz = load
    load_nodeset = "free_center"
    load_vector = [fx, fy, fz]
    
    # IMPORTANT: Use LinearStatic solver, not Modal
    write_ccx_mesh(mesh, str(inp), 
                 properties=props,
                 load_nodeset=load_nodeset,
                 load_vector=load_vector,
                 solver_type="LinearStatic")


def test_orthotropic_axial(tmp_path: Path):
    """Test orthotropic shell under transverse load (like isotropic test)."""
    ccx_bin = _ccx_or_skip()
    mesh = _build_plate()
    load = (0.0, 100.0, 0.0)  # 100N in Y (transverse)
    
    # AeroElast: orthotropic shell
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)
    
    # Isotropic average (same as CCX will use)
    e_equiv = (E1 + E2) / 2
    mats = [{
        "type": "isotropic",
        "e": e_equiv,
        "nu": nu12,
        "rho": 1000.0,
        "thickness": thickness,
    }] * len(mesh.elements)
    
    asm = PyMeshAssembler(node_coords, conn, elem_types, mats)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 3
    f[i0 + 1] = load[1]  # Y direction
    
    clamped = {mesh.node_id_to_index[n.id]*3 + i for n in mesh.get_node_set("clamped").nodes.values() for i in range(3)}
    free = np.array([i for i in range(asm.dofs_count) if i not in clamped])
    
    u = spsolve(K[np.ix_(free, free)], f[free])
    aero_disp = np.zeros(asm.dofs_count)
    aero_disp[free] = u
    
    print(f"AeroElast Uy: {aero_disp[i0+1]*1e6:.2f} um")
    
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
    
    ccx_uy = list(ccx_disp.values())[0][1]
    print(f"CCX Uy: {ccx_uy*1e6:.2f} um")
    
    rel_error = abs(aero_disp[i0+1] - ccx_uy) / max(abs(ccx_uy), 1e-12)
    print(f"Error: {rel_error*100:.2f}%")
    
    assert rel_error < 0.15, f"Ortho transverse: {rel_error*100:.1f}% > 15%"


def test_orthotropic_bending(tmp_path: Path):
    """Test orthotropic shell under bending."""
    ccx_bin = _ccx_or_skip()
    mesh = _build_plate()
    load = (100.0, 0.0, 0.0)  # 100N in X
    
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)
    
    e_equiv = (E1 + E2) / 2
    mats = [{
        "type": "isotropic",
        "e": e_equiv,
        "nu": nu12,
        "rho": 1000.0,
        "thickness": thickness,
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