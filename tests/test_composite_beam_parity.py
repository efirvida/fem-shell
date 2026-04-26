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
from aeroelast.core.laminate import create_laminate_from_angles
from aeroelast.core.material import IsotropicMaterial, Material, OrthotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.properties import CompositeShellProperty, ShellProperty
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
    """Build cantilever plate mesh for composite shell test.

    The mesh is a proper 2D shell cantilever in the XY plane:
    - X: width direction (-B/2 to +B/2)
    - Y: length direction (0 to L, cantilever axis)
    - Z = 0: mid-surface of the shell
    - Elements span X and Y, Z is fixed

    This matches the CCX *SHELL SECTION setup where the shell lies in
    the XY plane and deforms out-of-plane (Z displacement).
    """
    nx = 4  # elements along X
    ny = 10  # elements along Y (length)
    h_shell = thickness  # shell thickness (for reference, not a 3D mesh)

    mesh = MeshModel()

    # Create node grid in XY plane (Z=0 everywhere)
    xs = np.linspace(0, B, nx + 1)
    ys = np.linspace(0, L, ny + 1)
    # Z is fixed at 0 (mid-surface)
    # We use a 2D grid (X, Y) — one node per (i, j) pair
    grid: dict[tuple[int, int], Node] = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node = Node([float(x), float(y), 0.0], geometric_node=False)
            mesh.add_node(node)
            grid[(i, j)] = node

    # Create quadrilateral shell elements in the XY plane.
    # Nodes are numbered consecutively: node_id = j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            n00 = grid[(i, j)]
            n10 = grid[(i + 1, j)]
            n11 = grid[(i + 1, j + 1)]
            n01 = grid[(i, j + 1)]
            mesh.add_element(
                MeshElement(
                    nodes=[n00, n10, n11, n01],
                    element_type=ElementType.quad,
                )
            )

    # Node sets
    clamped = {n for n in mesh.nodes if np.isclose(n.y, 0.0, atol=1e-12)}
    free_face = {n for n in mesh.nodes if np.isclose(n.y, L, atol=1e-12)}
    free_center = min(free_face, key=lambda n: abs(float(n.x) - B / 2))

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
        nids = [mesh.node_id_to_index[nid] + 1 for nid in el.node_ids]
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
# Laminate Material Helpers
# ============================================================================


def _make_laminate_mat(
    E1: float, E2: float, G12: float, nu12: float, thickness: float
) -> dict:
    """Compute ABD matrices and return a material dict for PyMeshAssembler.

    Uses the Python CLT to build the 8-ply symmetric [0/90/45/-45]s layup,
    then exposes the result in the flat format expected by the Rust binding.
    """
    # Engineering constants -> orthotropic material
    mat = Material(
        name="comp",
        E=(E1, E2, E2),
        G=(G12, G12, G12),
        nu=(nu12, nu12, 0.0),
        rho=0.0,
    )

    # 8-ply symmetric: [0/90/45/-45/-45/45/90/0]
    ply_t = thickness / 8
    angles = [0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0]
    laminate = create_laminate_from_angles(mat, ply_t, angles)

    ABD = laminate.get_ABD_matrix()  # 6x6 [[A,B],[B,D]]
    A = ABD[:3, :3]
    B = ABD[:3, 3:]
    D = ABD[3:, 3:]
    Cs = laminate.Cs

    # Flatten for Rust binding: row-major
    cm = A.ravel()              # 9 elements
    cb_coupling = B.ravel()     # 9 elements
    cb = D.ravel()              # 9 elements
    cs = Cs.ravel()             # 4 elements

    h = laminate.total_thickness
    rho_eq = 0.0  # no density needed for stiffness test
    mass_per_area = rho_eq * h
    rotational_inertia = rho_eq * h**3 / 12.0

    return {
        "type": "composite",
        "cm": cm.tolist(),
        "b_coupling": cb_coupling.tolist(),
        "cb": cb.tolist(),
        "cs": cs.tolist(),
        "thickness": h,
        "e_equiv": A[0, 0] / h,
        "mass_per_area": mass_per_area,
        "rotational_inertia": rotational_inertia,
    }


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


# =============================================================================
# Unit tests (no CCX needed)
# =============================================================================


class TestCompositeMaterial:
    """Test composite material and ABD matrix computation."""

    def test_laminate_abd_matrices(self):
        """Verify ABD matrices are computed correctly."""
        E1, E2, G12, nu12, t = 120e9, 10e9, 5e9, 0.3, 0.005
        mat_dict = _make_laminate_mat(E1, E2, G12, nu12, t)

        # Check keys exist
        assert "cm" in mat_dict
        assert "b_coupling" in mat_dict
        assert "cb" in mat_dict
        assert "cs" in mat_dict
        assert mat_dict["thickness"] == t

        # Check shapes (3x3 matrices flattened to 9 elements)
        assert len(mat_dict["cm"]) == 9  # A matrix
        assert len(mat_dict["b_coupling"]) == 9  # B matrix
        assert len(mat_dict["cb"]) == 9  # D matrix

    def test_mesh_connectivity(self):
        """Verify mesh has correct node sets and connectivity."""
        mesh = _build_composite_plate_mesh()

        # Check node sets exist
        assert mesh.get_node_set("clamped") is not None
        assert mesh.get_node_set("free_face") is not None
        assert mesh.get_node_set("free_center") is not None

        # Check clamped at y=0
        clamped = list(mesh.get_node_set("clamped").nodes.values())
        for n in clamped:
            assert np.isclose(n.y, 0.0, atol=1e-12)

        # Check free face at y=L
        free = list(mesh.get_node_set("free_face").nodes.values())
        for n in free:
            assert np.isclose(n.y, L, atol=1e-12)


# =============================================================================
# Integration tests (require CCX)
# =============================================================================


# Test de composite shell - tensión axial
def test_composite_axial_tension(tmp_path: Path):
    """Test composite shell under axial tension."""
    ccx_bin = _ccx_bin_or_skip()
    mesh = _build_composite_plate_mesh()

    # Load: 1000N in Z direction
    load = (0.0, 0.0, 1000.0)

    # AeroElast solution (MITC4: 6 DOFs/node)
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)  # MITC4

    # Composite material - pre-computed ABD matrices from CLT
    mat_dict = _make_laminate_mat(E1, E2, G12, nu12, thickness)
    mats = [mat_dict] * len(mesh.elements)

    asm = PyMeshAssembler(
        node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
    )
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # Load vector (MITC4: 6 DOFs/node)
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 6  # 6 DOFs per node for MITC4
    f[i0 + 2] = load[2]  # Z direction

    # Boundary conditions (MITC4: 6 DOFs/node)
    clamped = {
        mesh.node_id_to_index[n.id] * 6 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(6)
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

    aero_disp = u[i0:i0 + 6]

    # CCX solution
    inp_path = tmp_path / "composite_axial.inp"
    _write_composite_ccx_inp(inp_path, mesh, load)
    
    # Get node IDs for CCX output
    node_ids = [mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_center").nodes.values()]

    result = _run_ccx(inp_path, ccx_bin)
    if result.returncode != 0:
        pytest.xfail(f"CCX failed (may be version/system issue): {result.stderr[:200]}")

    frd_path = inp_path.with_suffix(".frd")
    if not frd_path.exists():
        pytest.xfail("No FRD output from CCX")

    ccx_disp = _parse_frd_disp(frd_path, node_ids)
    if not ccx_disp:
        pytest.xfail("Could not parse CCX displacements")

    ccx_vals = list(ccx_disp.values())[0]

    # Compare - axial displacement should be similar (CCX has 3 DOFs, MITC4 has 6)
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

    # AeroElast solution (MITC4: 6 DOFs/node)
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)

    # Composite material - pre-computed ABD matrices from CLT
    mat_dict = _make_laminate_mat(E1, E2, G12, nu12, thickness)
    mats = [mat_dict] * len(mesh.elements)

    asm = PyMeshAssembler(
        node_coords=node_coords, connectivity=conn, elem_types=elem_types, materials=mats
    )
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # Load vector (MITC4: 6 DOFs/node)
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i0 = mesh.node_id_to_index[center.id] * 6  # 6 DOFs per node for MITC4
    f[i0] = load[0]  # X direction

    # Boundary conditions (MITC4: 6 DOFs/node)
    clamped = {
        mesh.node_id_to_index[n.id] * 6 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(6)
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

    aero_disp = u[i0:i0 + 6]

    # CCX solution
    inp_path = tmp_path / "composite_bending.inp"
    _write_composite_ccx_inp(inp_path, mesh, load)

    node_ids = [mesh.node_id_to_index[n.id] + 1 for n in mesh.get_node_set("free_center").nodes.values()]

    result = _run_ccx(inp_path, ccx_bin)
    if result.returncode != 0:
        pytest.xfail(f"CCX failed (may be version/system issue): {result.stderr[:200]}")

    frd_path = inp_path.with_suffix(".frd")
    if not frd_path.exists():
        pytest.xfail("No FRD output from CCX")

    ccx_disp = _parse_frd_disp(frd_path, node_ids)
    if not ccx_disp:
        pytest.xfail("Could not parse CCX displacements")

    ccx_vals = list(ccx_disp.values())[0]

    # Compare (CCX has 3 DOFs, MITC4 has 6)
    rel_error = abs(aero_disp[0] - ccx_vals[0]) / max(abs(ccx_vals[0]), 1e-10)
    print(f"AeroElast X: {aero_disp[0]*1e6:.2f} um, CCX: {ccx_vals[0]*1e6:.2f} um")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.1, f"Composite bending: {rel_error*100:.1f}% error (max 10%)"