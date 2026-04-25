"""Quadratic Composite Shell: AeroElast (MITC4) vs CalculiX (S8R).

Tests if using quadratic mesh with *SHELL SECTION, COMPOSITE
resolves the 7x difference seen with linear S4 mesh.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

pytest.importorskip("petsc4py", reason="PETSc not available")
pytest.importorskip("_aeroelast", reason="Rust backend not available")

from _aeroelast import PyMeshAssembler

from aeroelast.core.material import OrthotropicMaterial
from aeroelast.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet
from aeroelast.core.mesh.io.writers import write_ccx_mesh
from aeroelast.core.mesh.model import MeshModel
from aeroelast.core.laminate import Ply, Laminate


# ============================================================================
# Constants
# ============================================================================

# Carbon/epoxy typical
E1 = 28.7e9  # Pa (fiber direction)
E2 = 10.0e9  # Pa (transverse)
G12 = 8.4e9   # Pa (shear)
nu12 = 0.28
rho = 1600.0   # kg/m³

thickness = 0.002  # 2mm total (1mm per ply)
L = 1.0        # Length
B = 0.1        # Width


# ============================================================================
# Mesh Builder
# ============================================================================


def build_composite_mesh(*, nz: int = 10, nx: int = 4) -> MeshModel:
    """Build cantilever plate with quad elements."""
    Node._id_counter = 0
    MeshElement._id_counter = 0

    mesh = MeshModel()

    xs = np.linspace(0, B, nx + 1)
    zs = np.linspace(0, L, nz + 1)

    grid = {}
    for k, z in enumerate(zs):
        for i, x in enumerate(xs):
            node = Node([float(x), 0.0, float(z)], geometric_node=False)
            mesh.add_node(node)
            grid[(i, k)] = node

    for k in range(nz):
        for i in range(nx):
            mesh.add_element(
                MeshElement(
                    nodes=[
                        grid[(i, k)],
                        grid[(i + 1, k)],
                        grid[(i + 1, k + 1)],
                        grid[(i, k + 1)],
                    ],
                    element_type=ElementType.quad,
                )
            )

    clamped = {n for n in mesh.nodes if np.isclose(n.z, 0.0, atol=1e-12)}
    free_face = {n for n in mesh.nodes if np.isclose(n.z, L, atol=1e-12)}
    free_center = min(free_face, key=lambda n: abs(float(n.x)))

    mesh.add_node_set(NodeSet("clamped", clamped))
    mesh.add_node_set(NodeSet("free_face", free_face))
    mesh.add_node_set(NodeSet("free_center", {free_center}))
    mesh.add_element_set(ElementSet("plate", set(mesh.elements)))

    return mesh


# ============================================================================
# Parse FRD
# ============================================================================


def parse_frd_disp(frd: Path, node_id: int) -> np.ndarray:
    """Parse displacement for node_id from FRD file."""
    disps = {}
    in_disp = False
    for line in frd.read_text().splitlines():
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
                        disps[nid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    except (ValueError, IndexError):
                        continue
    if node_id in disps:
        return disps[node_id]
    return None


# ============================================================================
# Test
# ============================================================================


def test_quadratic_composite_shell(tmp_path: Path):
    """Test if quadratic mesh with SHELL SECTION COMPOSITE fixes the 7x diff."""
    ccx_bin = Path("/scratch/leahk/eduardo.donestevez/venv/bin/ccx")
    if not ccx_bin.exists():
        pytest.skip("CalculiX not available")

    mesh = build_composite_mesh(nz=20, nx=4)

    # Material
    mat = OrthotropicMaterial(
        name="CarbonEpoxy",
        E=[E1, E2, E2],
        nu=[nu12, nu12, 0.0],
        G=[G12, G12, G12],
        rho=rho,
    )

    # Laminate [0/90]
    ply1 = Ply(material=mat, thickness=thickness / 2, angle=0)
    ply2 = Ply(material=mat, thickness=thickness / 2, angle=90)
    lam = Laminate(plies=[ply1, ply2])

    # === AeroElast with full ABD ===
    node_coords = np.asarray([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)
    conn = [[mesh.node_id_to_index[nid] for nid in el.node_ids] for el in mesh.elements]
    elem_types = [4] * len(mesh.elements)  # MITC4

    # Pass ABD directly
    cm = lam.A.ravel().tolist()
    cb = lam.D.ravel().tolist()
    cs = lam.Cs.ravel().tolist()
    h = lam.total_thickness
    e_equiv = float(np.trace(lam.A) / (3 * h))
    mpa = sum(p.material.rho * p.thickness for p in lam.plies)
    ri = sum(p.material.rho * (p.z_top**3 - p.z_bottom**3) / 3 for p in lam.plies)

    mats = [
        {
            "type": "composite",
            "cm": cm,
            "cb": cb,
            "cs": cs,
            "thickness": h,
            "e_equiv": e_equiv,
            "mass_per_area": mpa,
            "rotational_inertia": ri,
        }
    ] * len(mesh.elements)

    asm = PyMeshAssembler(node_coords, conn, elem_types, mats)
    rows, cols, vals = asm.assemble_k()
    n = asm.dofs_count
    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # Load: Fy at free center
    f = np.zeros(n, dtype=float)
    center = next(iter(mesh.get_node_set("free_center").nodes.values()))
    i_center = mesh.node_id_to_index[center.id] * 3
    f[i_center + 1] = 100.0  # Fy = 100N

    # BCs
    clamped = {
        mesh.node_id_to_index[n.id] * 3 + i
        for n in mesh.get_node_set("clamped").nodes.values()
        for i in range(3)
    }
    free_mask = np.ones(n, dtype=bool)
    for dof in clamped:
        free_mask[dof] = False
    free = np.where(free_mask)[0]

    u_ae = np.zeros(n, dtype=float)
    u_ae[free] = spsolve(K[free][:, free], f[free])
    disp_ae = u_ae[i_center + 1]

    # === CalculiX with QUADRATIC mesh + SHELL SECTION COMPOSITE ===
    case_dir = tmp_path / "quad_composite"
    case_dir.mkdir(exist_ok=True)
    inp = case_dir / "quad.inp"

    # Use quadratic=True to get S8R + COMPOSITE
    from aeroelast.core.properties import CompositeShellProperty

    comp_prop = CompositeShellProperty(laminate=lam)

    write_ccx_mesh(
        mesh,
        str(inp),
        properties={"plate": comp_prop},
        boundary_nodeset="clamped",
        solver_type="LinearStatic",
        load_nodeset="free_center",
        load_vector=(0.0, 100.0, 0.0),
        quadratic=True,  # <-- KEY: quadratic mesh
    )

    # Run CCX
    stem = "quad"
    result = subprocess.run(
        [str(ccx_bin), stem],
        cwd=case_dir,
        capture_output=True,
        text=True,
    )

    print(f"CCX return code: {result.returncode}")
    print(f"CCX stdout: {result.stdout[:500] if result.stdout else 'empty'}")
    print(f"Files in case_dir: {list(case_dir.iterdir())}")

    if result.returncode != 0:
        print(f"CCX stderr: {result.stderr[:500]}")
        pytest.skip(f"CCX failed: {result.stderr[:500]}")

    frd = case_dir / f"{stem}.frd"
    if not frd.exists():
        pytest.skip("FRD not generated")

    center_ccx_id = mesh.node_id_to_index[center.id] + 1
    disp_ccx = parse_frd_disp(frd, center_ccx_id)

    print(f"Parsed CCX displacement for original node: {disp_ccx}")

    disp_ccx_y = None
    if disp_ccx is not None:
        disp_ccx_y = disp_ccx[1]
    else:
        # Find max displacement (midside nodes have higher values)
        max_disp = 0.0
        max_node = None
        in_disp = False
        for line in frd.read_text().splitlines():
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
                            uy = float(parts[2])
                            if abs(uy) > abs(max_disp):
                                max_disp = uy
                                max_node = nid
                        except:
                            continue
        print(f"Max displacement: node={max_node}, uy={max_disp:.6f}")
        disp_ccx_y = max_disp

    if disp_ccx_y is None:
        pytest.skip("Could not parse CCX displacement")

    # === Compare ===
    ratio = disp_ccx_y / disp_ae

    print(f"\n[Quadratic Composite Shell]")
    print(f"  AeroElast (MITC4): {disp_ae * 1000:.3f} mm")
    print(f"  CalculiX (S8R):  {disp_ccx_y * 1000:.3f} mm")
    print(f"  Ratio (CCX/AE):    {ratio:.3f}")

    # Allow 20% tolerance
    tol = 0.20
    if abs(ratio - 1) > tol:
        pytest.fail(
            f"Quadratic composite mismatch: AE={disp_ae*1000:.2f}mm, "
            f"CCX={disp_ccx_y*1000:.2f}mm, ratio={ratio:.2f}"
        )