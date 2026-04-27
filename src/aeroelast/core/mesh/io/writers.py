"""
Mesh I/O writers module.

This module contains functions for exporting meshes to various file formats:
- meshio formats (VTK, VTU, OBJ, STL, etc.)
- Gmsh MSH format
- CalculiX format
- Plot3D format
- HDF5 format
- Pickle format
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import meshio
import numpy as np

if TYPE_CHECKING:
    from aeroelast.core.mesh.model import MeshModel

from aeroelast.core.mesh.entities import ElementType

# ============================================================================
# Property duck-typing helpers — support both Rust-native and Python types
# ============================================================================


def _prop_is_composite(prop) -> bool:
    """Return True if *prop* represents a composite (multi-ply) shell property.

    Accepts both ``_aeroelast.Laminate`` (Rust) and legacy Python
    ``CompositeShellProperty``.
    """
    try:
        from _aeroelast import Laminate as _RL  # noqa: PLC0415

        if isinstance(prop, _RL):
            return True
    except ImportError:
        pass
    try:
        from aeroelast.core.properties import CompositeShellProperty as _CSP  # noqa: PLC0415

        if isinstance(prop, _CSP):
            return True
    except ImportError:
        pass
    return False


def _prop_is_isotropic(prop) -> bool:
    """Return True if *prop* represents a single-layer isotropic shell property."""
    if isinstance(prop, dict) and prop.get("type") == "isotropic":
        return True
    try:
        from aeroelast.core.properties import ShellProperty as _SP  # noqa: PLC0415

        if isinstance(prop, _SP):
            return True
    except ImportError:
        pass
    return False


def _bucket_elset_name(set_name: str, bucket_tenths: int) -> str:
    """Build deterministic per-angle-bucket ELSET name.

    Example: ``05_00_HP_LE`` + ``-123`` (=-12.3°) ->
    ``05_00_HP_LE__BKT_N12_3``.
    """
    sign = "N" if bucket_tenths < 0 else "P"
    a = abs(int(bucket_tenths))
    return f"{set_name}__BKT_{sign}{a // 10}_{a % 10}"


def _build_angle_bucket_sets(
    mesh: "MeshModel",
    properties: Dict[str, "ShellPropertyType"],
    span_direction: tuple,
) -> Dict[str, Dict[int, list[int]]]:
    """Group elements by per-element orientation angle bucket (tenths of degree).

    This mirrors Rust assembler logic for composite laminates:
    - Compute `element_angle_offset` from span-direction projection in element plane.
    - Bucket by `round(angle_deg * 10)`.
    - Build per-(set, bucket) element lists.
    """
    if span_direction is None:
        return {}

    sd = np.array(span_direction, dtype=float)
    sd_norm = np.linalg.norm(sd)
    if sd_norm < 1e-12:
        return {}
    sd = sd / sd_norm

    # Preload node coordinates (0-based node id -> xyz)
    coords = np.array([[n.x, n.y, n.z] for n in mesh.nodes], dtype=float)

    # Only composite sets with known properties
    out: Dict[str, Dict[int, list[int]]] = {}
    for set_name, element_set in mesh.element_sets.items():
        prop = properties.get(set_name)
        if prop is None or not _prop_is_composite(prop):
            continue

        buckets: Dict[int, list[int]] = {}
        for elem_id in element_set.element_ids:
            el = mesh.elements[elem_id]
            if len(el.node_ids) < 3:
                continue
            # Use first 4 nodes for quads, first 3 for tris (matching Rust logic intent)
            node_ids = el.node_ids[:4] if len(el.node_ids) >= 4 else el.node_ids[:3]
            pts = coords[np.array(node_ids, dtype=int)]

            p0, p1, p2 = pts[0], pts[1], pts[2]
            v1 = p1 - p0
            v2 = p2 - p0

            n1 = np.cross(v1, v2)
            n1_norm = np.linalg.norm(n1)

            if len(node_ids) >= 4:
                p3 = pts[3]
                n2 = np.cross(p2 - p0, p3 - p0)
                n2_norm = np.linalg.norm(n2)
                if n1_norm > 1e-12 and n2_norm > 1e-12:
                    avg = n1 / n1_norm + n2 / n2_norm
                    avg_norm = np.linalg.norm(avg)
                    e3 = avg / avg_norm if avg_norm > 1e-30 else np.array([0.0, 0.0, 1.0])
                elif n1_norm > 1e-12:
                    e3 = n1 / n1_norm
                elif n2_norm > 1e-12:
                    e3 = n2 / n2_norm
                else:
                    e3 = np.array([0.0, 0.0, 1.0])
            else:
                e3 = n1 / n1_norm if n1_norm > 1e-12 else np.array([0.0, 0.0, 1.0])

            # e1: projected edge 01, fallback edge 02
            e1 = v1 - np.dot(v1, e3) * e3
            e1_norm = np.linalg.norm(e1)
            if e1_norm < 1e-12:
                e1 = v2 - np.dot(v2, e3) * e3
                e1_norm = np.linalg.norm(e1)
            if e1_norm < 1e-30:
                angle_deg = 0.0
            else:
                e1 = e1 / e1_norm
                e2 = np.cross(e3, e1)
                e2_norm = np.linalg.norm(e2)
                if e2_norm < 1e-30:
                    angle_deg = 0.0
                else:
                    e2 = e2 / e2_norm
                    sd_proj = sd - np.dot(sd, e3) * e3
                    sd_proj_norm = np.linalg.norm(sd_proj)
                    if sd_proj_norm < 1e-10:
                        angle_deg = 0.0
                    else:
                        cos_a = float(np.dot(sd_proj, e1) / sd_proj_norm)
                        sin_a = float(np.dot(sd_proj, e2) / sd_proj_norm)
                        angle_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))

            bucket = int(np.round(angle_deg * 10.0))
            buckets.setdefault(bucket, []).append(elem_id)

        if buckets:
            out[set_name] = buckets

    return out


def _prop_plies(prop):
    """Yield (material_name, E1, E2, E3, G12, G23, G31, nu12, nu23, nu31, rho, thickness, angle)
    tuples for each ply in a composite property.

    Works for both Rust ``Laminate`` and Python ``CompositeShellProperty``.
    """
    try:
        from _aeroelast import Laminate as _RL  # noqa: PLC0415

        if isinstance(prop, _RL):
            # Prefer direct property when available.
            plies = getattr(prop, "plies", None)
            # Compatibility path for older/newer bindings exposing method form.
            if plies is None and hasattr(prop, "plies_data"):
                plies = prop.plies_data()
            if plies is not None:
                for i, ply in enumerate(plies):
                    m = ply["material"]
                    yield (
                        f"PLY_{i}",
                        m["e1"],
                        m["e2"],
                        m["e3"],
                        m["g12"],
                        m["g23"],
                        m["g13"],
                        m["nu12"],
                        m["nu23"],
                        m["nu31"],
                        m["rho"],
                        ply["thickness"],
                        ply["angle"],
                    )
                return
    except ImportError:
        pass
    try:
        from aeroelast.core.properties import CompositeShellProperty as _CSP  # noqa: PLC0415

        if isinstance(prop, _CSP):
            for ply in prop.laminate.plies:
                m = ply.material
                yield (
                    m.name,
                    m.E[0],
                    m.E[1],
                    m.E[2],
                    m.G[0],
                    m.G[1],
                    m.G[2],
                    m.nu[0],
                    m.nu[1],
                    m.nu[2],
                    m.rho,
                    ply.thickness,
                    ply.angle,
                )
    except ImportError:
        pass


def _prop_isotropic_data(prop) -> dict:
    """Return isotropic property data as a plain dict with keys:
    ``name``, ``e``, ``nu``, ``rho``, ``thickness``, ``shear_correction``.
    """
    if isinstance(prop, dict):
        return prop
    try:
        from aeroelast.core.properties import ShellProperty as _SP  # noqa: PLC0415

        if isinstance(prop, _SP):
            return {
                "name": getattr(prop.material, "name", "MAT"),
                "e": float(prop.material.E),
                "nu": float(prop.material.nu),
                "rho": float(prop.material.rho),
                "thickness": float(prop.thickness),
                "shear_correction": float(getattr(prop, "shear_correction", 5.0 / 6.0)),
            }
    except ImportError:
        pass
    return {}


# ============================================================================
# Boundary loop utilities for STL tip closing
# ============================================================================


def _find_boundary_loops(mesh: "MeshModel") -> list[list[int]]:
    """
    Find closed loops of boundary edges in a shell mesh.

    Boundary edges are those shared by exactly one element.

    Parameters
    ----------
    mesh : MeshModel
        The mesh model (expected to contain shell elements).

    Returns
    -------
    list of list of int
        Each inner list is an ordered sequence of node IDs forming a closed loop.
    """
    from collections import defaultdict

    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for el in mesh.elements:
        nids = el.node_ids
        n = len(nids)
        for i in range(n):
            edge = tuple(sorted([nids[i], nids[(i + 1) % n]]))
            edge_count[edge] += 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return []

    # Build adjacency from boundary edges
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Trace connected loops
    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adj:
        if start in visited:
            continue
        loop: list[int] = []
        current = start
        prev = None
        while current not in visited:
            visited.add(current)
            loop.append(current)
            next_node = None
            for nb in adj[current]:
                if nb != prev:
                    next_node = nb
                    break
            if next_node is None:
                break
            prev = current
            current = next_node
        if len(loop) > 2:
            loops.append(loop)
    return loops


def _cap_tip_loop(
    loop_node_ids: list[int],
    points: np.ndarray,
    node_id_to_index: dict[int, int],
) -> list[list[int]]:
    """
    Triangulate a boundary loop using a fan from its centroid.

    A new point is appended to *points* (via resize) and fan triangles
    referencing point indices are returned.

    Parameters
    ----------
    loop_node_ids : list of int
        Ordered node IDs forming the boundary loop.
    points : np.ndarray
        (N, 3) coordinate array — will be extended in-place with the centroid.
    node_id_to_index : dict
        Maps node IDs to indices into *points*.

    Returns
    -------
    list of list[int]
        Triangle connectivity (point indices) that caps the loop.
    """
    indices = [node_id_to_index[nid] for nid in loop_node_ids]
    centroid = points[indices].mean(axis=0)
    centroid_idx = len(points)  # will be appended later
    triangles = []
    n = len(indices)
    for i in range(n):
        triangles.append([centroid_idx, indices[i], indices[(i + 1) % n]])
    return triangles, centroid


# ============================================================================
# Element type mappings for different formats
# ============================================================================

ELEMENTS_TO_CALCULIX = {
    # 2D Shell elements
    "triangle": "S3",
    "triangle6": "S6",
    "quad": "S4",
    "quad8": "S8R",
    "quad9": "S9",
    # 3D Solid elements
    "tetra": "C3D4",
    "tetra10": "C3D10",
    "hexahedron": "C3D8",
    "hexahedron20": "C3D20",
    "hexahedron27": "C3D27",
    "wedge": "C3D6",
    "wedge15": "C3D15",
    "pyramid": "C3D5",
    "pyramid13": "C3D13",
}

ELEMENT_TYPE_TO_GMSH = {
    # 2D Surface elements
    ElementType.triangle: 2,  # 3-node triangle
    ElementType.triangle6: 9,  # 6-node second order triangle
    ElementType.quad: 3,  # 4-node quadrangle
    ElementType.quad8: 16,  # 8-node second order quadrangle
    ElementType.quad9: 10,  # 9-node second order quadrangle
    # 3D Volumetric elements
    ElementType.tetra: 4,  # 4-node tetrahedron
    ElementType.tetra10: 11,  # 10-node second order tetrahedron
    ElementType.hexahedron: 5,  # 8-node hexahedron
    ElementType.hexahedron20: 17,  # 20-node second order hexahedron
    ElementType.hexahedron27: 12,  # 27-node second order hexahedron
    ElementType.wedge: 6,  # 6-node prism/wedge
    ElementType.wedge15: 18,  # 15-node second order prism
    ElementType.pyramid: 7,  # 5-node pyramid
    ElementType.pyramid13: 19,  # 13-node second order pyramid
}


# ============================================================================
# Generic writer functions
# ============================================================================


def write_mesh(mesh: "MeshModel", filename: str, **kwargs) -> None:
    """
    Write the mesh to a file.

    Dispatches to the appropriate writer based on file extension.

    Parameters
    ----------
    mesh : MeshModel
        The mesh to write.
    filename : str
        The path to the output file. Format is inferred from extension.
    **kwargs
        Additional arguments passed to the underlying writer.
    """
    if not mesh.nodes:
        raise ValueError("Mesh has no nodes.")
    if not mesh.elements:
        raise ValueError("Mesh has no elements.")

    ext = Path(filename).suffix.lower()

    # Dispatch by extension
    if ext == ".inp":
        write_ccx_mesh(mesh, filename, **kwargs)
    elif ext == ".msh":
        write_gmsh_mesh(mesh, filename)
    elif ext in (".h5", ".hdf5"):
        write_hdf5(mesh, filename)
    elif ext in (".pkl", ".pickle"):
        write_pickle(mesh, filename)
    elif ext == ".fmt":
        write_plot3d(mesh, filename)
    else:
        # Default: use meshio for standard formats
        write_meshio(mesh, filename, **kwargs)


# ============================================================================
# Meshio writer (VTK, VTU, OBJ, STL, etc.)
# ============================================================================


def write_meshio(mesh: "MeshModel", filename: str, close_tip: bool = None, **kwargs) -> None:
    """Write mesh using meshio library (geometry only, no metadata).

    Parameters
    ----------
    mesh : MeshModel
        The mesh to export.
    filename : str
        Output file path.  Format is inferred from extension.
    close_tip : bool, optional
        When *True* and the output format is STL, boundary loops at the
        blade tip (maximum spanwise coordinate) are capped with fan
        triangles so the resulting STL is watertight.
        *None* (default) auto-enables for STL format.
    **kwargs
        Extra arguments forwarded to :func:`meshio.write`.
    """
    from collections import defaultdict

    points = mesh.coords_array
    cells = []
    ext = Path(filename).suffix.lower()

    # Auto-enable tip closing for STL format
    if close_tip is None:
        close_tip = ext == ".stl"

    if ext == ".stl":
        # Check if mesh has volume elements
        has_volume = any(
            el.element_type.name
            in (
                "tetra",
                "tetra10",
                "hexahedron",
                "hexahedron20",
                "hexahedron27",
                "wedge",
                "wedge15",
                "pyramid",
                "pyramid13",
            )
            for el in mesh.elements
        )

        if has_volume:
            face_count = defaultdict(int)
            for element in mesh.elements:
                faces = mesh._get_element_faces(element)
                if not faces:
                    continue
                for face in faces:
                    face_key = tuple(sorted(face))
                    face_count[face_key] += 1

            boundary_faces = []
            for element in mesh.elements:
                faces = mesh._get_element_faces(element)
                if not faces:
                    continue
                for face in faces:
                    face_key = tuple(sorted(face))
                    if face_count[face_key] == 1:
                        # Convert node IDs to indices
                        indices = [mesh.node_id_to_index[nid] for nid in face]
                        if len(indices) == 3:
                            boundary_faces.append(indices)
                        elif len(indices) == 4:
                            # Triangulate quad
                            boundary_faces.append([indices[0], indices[1], indices[2]])
                            boundary_faces.append([indices[0], indices[2], indices[3]])

            if boundary_faces:
                cells = [("triangle", np.array(boundary_faces))]
        else:
            # Extract surface triangles or triangulate quads
            triangles = []
            for el in mesh.elements:
                indices = tuple(mesh.node_id_to_index[nid] for nid in el.node_ids)
                if len(indices) == 3:
                    triangles.append(indices)
                elif len(indices) == 4:
                    triangles.append([indices[0], indices[1], indices[2]])
                    triangles.append([indices[0], indices[2], indices[3]])

            # Close blade tip boundary loops
            if close_tip and triangles:
                loops = _find_boundary_loops(mesh)
                if loops:
                    # Identify the tip loop: highest average spanwise coord (Z)
                    coords = mesh.coords_array
                    id_to_idx = mesh.node_id_to_index
                    loop_avg_z = []
                    for loop in loops:
                        avg_z = np.mean([coords[id_to_idx[nid], 2] for nid in loop])
                        loop_avg_z.append(avg_z)
                    tip_loop_idx = int(np.argmax(loop_avg_z))
                    tip_loop = loops[tip_loop_idx]

                    cap_tris, centroid = _cap_tip_loop(tip_loop, points, id_to_idx)
                    # Append centroid to points array
                    points = np.vstack([points, centroid.reshape(1, 3)])
                    triangles.extend(cap_tris)

            if triangles:
                cells = [("triangle", np.array(triangles))]

    # Default grouping for non-STL or if STL didn't yield bound cells
    if not cells:
        cells_dict = defaultdict(list)
        for el in mesh.elements:
            indices = tuple(mesh.node_id_to_index[nid] for nid in el.node_ids)
            cells_dict[el.element_type.name].append(indices)
        cells = [(el_type, np.array(el_list)) for el_type, el_list in cells_dict.items()]

    mesh_io = meshio.Mesh(points=points, cells=cells)
    meshio.write(filename, mesh_io, **kwargs)


# ============================================================================
# Plot3D writer
# ============================================================================


def write_plot3d(mesh: "MeshModel", filename: str) -> None:
    """Write mesh in Plot3D format."""
    points = mesh.coords_array
    cells = []
    for el in mesh.elements:
        indices = tuple(mesh.node_id_to_index[nid] for nid in el.node_ids)
        cells.append((el.element_type.name, np.array([indices])))
    mesh_io = meshio.Mesh(points=points, cells=cells)
    _write_plot3d_internal(mesh_io, filename)


def _write_plot3d_internal(mesh: meshio.Mesh, filename: str):
    """Internal Plot3D writer implementation."""
    points = mesh.points
    imax = np.unique(points[:, 0]).size
    jmax = np.unique(points[:, 1]).size
    kmax = np.unique(points[:, 2]).size
    _2d = kmax == 1
    with open(filename, "w") as p3dfile:
        if not _2d:
            print(imax, jmax, kmax, file=p3dfile)
            for value in points.flatten(order="F"):
                print(value, file=p3dfile)
        else:
            print(imax, jmax, file=p3dfile)
            for value in points[:, 0:2].flatten(order="F"):
                print(value, file=p3dfile)


# ============================================================================
# CalculiX writer
# ============================================================================


def write_ccx_mesh(
    mesh: "MeshModel",
    filename: str,
    properties: Optional[Dict[str, "ShellPropertyType"]] = None,
    boundary_nodeset: Optional[str] = None,
    num_modes: int = 10,
    span_direction: Optional[tuple] = None,
    orientation_reference_axis: Optional[tuple] = None,
    solver_type: str = "Modal",
    load_nodeset: Optional[str] = None,
    load_vector: Optional[list] = None,
    dt: float = 0.01,
    t_end: float = 1.0,
    quadratic: bool = False,
    nl_initial_increment: Optional[float] = None,
    nl_min_increment: Optional[float] = None,
    nl_max_increment: Optional[float] = None,
    nl_max_increments: Optional[int] = None,
) -> None:
    """
    Write the mesh to CalculiX format following CGX conventions.

    This method generates multiple files as recommended by CalculiX documentation:
    - {basename}.msh: Mesh file containing nodes and elements
    - {basename}.nam: Node sets and element sets definitions
    - {basename}.sur: Surface definitions (if applicable)
    - {basename}.inp: Main input file with *INCLUDE statements

    Parameters
    ----------
    mesh : MeshModel
        The mesh model to export.
    filename : str
        Path to the main .inp file.
    properties : dict, optional
        Mapping of element-set name to ShellProperty or CompositeShellProperty.
        When provided, material definitions, shell sections, and a modal
        analysis step are written into the .inp file.
    boundary_nodeset : str, optional
        Name of the node set to clamp (all 6 DOFs fixed). Required when
        properties are provided.
    num_modes : int, optional
        Number of eigenvalues for the *FREQUENCY step. Default 10.
    span_direction : tuple of 3 floats, optional
        Global direction of the structural span axis (e.g. ``(0, 0, 1)`` for
        a blade along Z).  When provided, ply orientations in *ORIENTATION
        cards are measured from this axis projected onto each element's
        surface, so that a 0-degree ply is fibre-in-span rather than
        fibre-in-the-first-element-edge.  Pass ``(0., 0., 1.)`` for a blade
        model with span along Z.
    quadratic : bool, optional
        When ``True``, upgrade the linear mesh (MITC4/MITC3) to second-order
        (S8R/S6) by inserting midside nodes at edge midpoints, and use
        ``*SHELL SECTION, COMPOSITE`` with per-ply data.  CalculiX requires
        S8R/S6 for ``*SHELL SECTION, COMPOSITE``.
        When ``False`` (default), export as S4/S3 with an orthotropic
        equivalent material derived from the A-matrix inversion — matching
        the first-order MITC4/MITC3 mesh used internally by AeroElast.
    """

    def split_list(arr, chunk_size: int = 7):
        """Split list into chunks for formatted output."""
        arr = list(arr)
        return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

    # Optional per-element orientation buckets (set_name -> bucket_tenths -> [elem_ids])
    # Used to keep CCX export consistent with Rust assembler orientation bucketing.
    angle_bucket_sets: Optional[Dict[str, Dict[int, list[int]]]] = None
    if properties is not None and span_direction is not None:
        angle_bucket_sets = _build_angle_bucket_sets(mesh, properties, span_direction)

    # Extract base path and name
    base_path = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Define file paths
    msh_file = os.path.join(base_path, f"{base_name}.msh") if base_path else f"{base_name}.msh"
    nam_file = os.path.join(base_path, f"{base_name}.nam") if base_path else f"{base_name}.nam"
    sur_file = os.path.join(base_path, f"{base_name}.sur") if base_path else f"{base_name}.sur"
    inp_file = filename

    # quadratic=True  → S8R/S6 + *SHELL SECTION, COMPOSITE (per-ply detail)
    # quadratic=False → S4/S3  + *SHELL SECTION, MATERIAL=  (equivalent ortho)
    # CalculiX requires S8R/S6 for *SHELL SECTION, COMPOSITE; S4/S3 use the
    # orthotropic equivalent derived from the full A-matrix inversion.
    quadratic_data = None
    if quadratic:
        quadratic_data = _build_quadratic_mesh_data(mesh)
        print(
            f"  Converted to quadratic: {quadratic_data['n_nodes']} nodes "
            f"({quadratic_data['n_midside']} midside nodes added)"
        )

    # Write files
    _write_ccx_msh_file(mesh, msh_file, quadratic_data=quadratic_data)
    _write_ccx_nam_file(
        mesh,
        nam_file,
        split_list,
        quadratic_data=quadratic_data,
        angle_bucket_sets=angle_bucket_sets,
    )
    has_surfaces = _write_ccx_sur_file(mesh, sur_file, split_list)
    _write_ccx_inp_file(
        mesh,
        inp_file,
        base_name,
        has_surfaces,
        properties=properties,
        boundary_nodeset=boundary_nodeset,
        num_modes=num_modes,
        span_direction=span_direction,
        orientation_reference_axis=orientation_reference_axis,
        solver_type=solver_type,
        load_nodeset=load_nodeset,
        load_vector=load_vector,
        dt=dt,
        t_end=t_end,
        quadratic=quadratic,
        nl_initial_increment=nl_initial_increment,
        nl_min_increment=nl_min_increment,
        nl_max_increment=nl_max_increment,
        nl_max_increments=nl_max_increments,
        angle_bucket_sets=angle_bucket_sets,
        quadratic_data=quadratic_data,
    )

    print("CalculiX mesh files written:")
    print(f"  - Mesh file: {msh_file}")
    print(f"  - Names file: {nam_file}")
    if has_surfaces:
        print(f"  - Surface file: {sur_file}")
    print(f"  - Main input file: {inp_file}")


def _build_quadratic_mesh_data(mesh: "MeshModel") -> Dict:
    """Convert linear shell mesh to quadratic by adding midside nodes.

    CalculiX requires S8R (quad8) or S6 (triangle6) for composite shell
    sections.  This function computes midside nodes for every edge and
    returns the data needed by the msh/nam writers.

    Returns
    -------
    dict
        ``all_nodes``  – (N, 3) array of all node coordinates,
        ``elements``   – list of (ccx_type, node_ids_1based) per element,
        ``n_nodes``    – total node count,
        ``n_midside``  – number of midside nodes added,
        ``node_set_extra`` – dict mapping original node-set names to
            additional midside node IDs (1-based) that lie on edges
            between two nodes of the set.
    """
    nodes = np.array([[n.x, n.y, n.z] for n in mesh.nodes])
    n_original = len(nodes)

    edge_to_mid: Dict[tuple, int] = {}
    new_nodes: list = []

    def _get_midside(n1: int, n2: int) -> int:
        edge = (min(n1, n2), max(n1, n2))
        mid = edge_to_mid.get(edge)
        if mid is not None:
            return mid
        mid = n_original + len(new_nodes)
        new_nodes.append((nodes[n1] + nodes[n2]) / 2.0)
        edge_to_mid[edge] = mid
        return mid

    elements: list = []
    for el in mesh.elements:
        nids = el.node_ids  # 0-based
        etype = el.element_type.name

        if etype == "triangle":
            m01 = _get_midside(nids[0], nids[1])
            m12 = _get_midside(nids[1], nids[2])
            m20 = _get_midside(nids[2], nids[0])
            elements.append((
                "S6",
                [nids[0] + 1, nids[1] + 1, nids[2] + 1, m01 + 1, m12 + 1, m20 + 1],
            ))
        elif etype == "quad":
            m01 = _get_midside(nids[0], nids[1])
            m12 = _get_midside(nids[1], nids[2])
            m23 = _get_midside(nids[2], nids[3])
            m30 = _get_midside(nids[3], nids[0])
            elements.append((
                "S8R",
                [
                    nids[0] + 1,
                    nids[1] + 1,
                    nids[2] + 1,
                    nids[3] + 1,
                    m01 + 1,
                    m12 + 1,
                    m23 + 1,
                    m30 + 1,
                ],
            ))
        else:
            ccx_type = ELEMENTS_TO_CALCULIX.get(etype, etype)
            elements.append((ccx_type, [n + 1 for n in nids]))

    # Build extra midside nodes for node sets (edge between two set members)
    node_set_extra: Dict[str, list] = {}
    for ns_name, ns in mesh.node_sets.items():
        ns_ids = set(ns.node_ids)  # 0-based
        extras = []
        for (n1, n2), mid in edge_to_mid.items():
            if n1 in ns_ids and n2 in ns_ids:
                extras.append(mid + 1)  # 1-based
        node_set_extra[ns_name] = extras

    all_nodes = np.vstack([nodes, np.array(new_nodes)]) if new_nodes else nodes

    return {
        "all_nodes": all_nodes,
        "elements": elements,
        "n_nodes": len(all_nodes),
        "n_midside": len(new_nodes),
        "node_set_extra": node_set_extra,
    }


def _write_ccx_msh_file(
    mesh: "MeshModel",
    filename: str,
    quadratic_data: Optional[Dict] = None,
) -> None:
    """Write the .msh file containing nodes and elements."""
    with open(filename, "wt") as f:
        if quadratic_data is not None:
            # --- Quadratic mesh ---
            all_nodes = quadratic_data["all_nodes"]
            f.write("*NODE, NSET=Nall\n")
            for i, nd in enumerate(all_nodes):
                f.write(f"{i + 1:8d}, {nd[0]:14.6E}, {nd[1]:14.6E}, {nd[2]:14.6E}\n")

            # Group elements by CCX type
            elements_by_type: Dict[str, list] = {}
            for i, (ccx_type, nids) in enumerate(quadratic_data["elements"]):
                elements_by_type.setdefault(ccx_type, []).append((i, nids))

            for ccx_type, elems in elements_by_type.items():
                f.write(f"*ELEMENT, TYPE={ccx_type}, ELSET=Eall\n")
                for i, nids in elems:
                    node_ids_str = ", ".join(str(n) for n in nids)
                    f.write(f"{i + 1:8d}, {node_ids_str}\n")
        else:
            # --- Original linear mesh ---
            f.write("*NODE, NSET=Nall\n")
            for i, nd in enumerate(mesh.nodes):
                f.write(f"{i + 1:8d}, {nd.x:14.6E}, {nd.y:14.6E}, {nd.z:14.6E}\n")

            elements_by_type: Dict[str, list] = {}
            for i, el in enumerate(mesh.elements):
                el_type_name = el.element_type.name
                if el_type_name not in elements_by_type:
                    elements_by_type[el_type_name] = []
                elements_by_type[el_type_name].append((i, el))

            for el_type_name, elements in elements_by_type.items():
                if el_type_name in ELEMENTS_TO_CALCULIX:
                    ccx_type = ELEMENTS_TO_CALCULIX[el_type_name]
                    f.write(f"*ELEMENT, TYPE={ccx_type}, ELSET=Eall\n")
                    for i, el in elements:
                        node_ids_str = ", ".join(str(n + 1) for n in el.node_ids)
                        f.write(f"{i + 1:8d}, {node_ids_str}\n")


def _write_ccx_nam_file(
    mesh: "MeshModel",
    filename: str,
    split_func,
    quadratic_data: Optional[Dict] = None,
    angle_bucket_sets: Optional[Dict[str, Dict[int, list[int]]]] = None,
) -> None:
    """Write the .nam file containing node sets and element sets."""
    with open(filename, "wt") as f:
        # Write element sets
        for name, element_set in mesh.element_sets.items():
            f.write(f"*ELSET, ELSET=E{name.upper()}\n")
            labels = [el_id + 1 for el_id in element_set.element_ids]
            for chunk in split_func(labels):
                f.write(", ".join(f"{e:8d}" for e in chunk) + "\n")

        # Additional per-angle-bucket ELSETs for CCX orientation consistency
        if angle_bucket_sets:
            for set_name, buckets in angle_bucket_sets.items():
                for bucket_tenths, elem_ids in sorted(buckets.items()):
                    if not elem_ids:
                        continue
                    bucket_elset = _bucket_elset_name(set_name, bucket_tenths)
                    f.write(f"*ELSET, ELSET=E{bucket_elset.upper()}\n")
                    labels = [el_id + 1 for el_id in elem_ids]
                    for chunk in split_func(labels):
                        f.write(", ".join(f"{e:8d}" for e in chunk) + "\n")

        # Write node sets (with extra midside nodes when quadratic)
        ns_extra = quadratic_data["node_set_extra"] if quadratic_data else {}
        for name, node_set in mesh.node_sets.items():
            f.write(f"*NSET, NSET=N{name.upper()}\n")
            labels = sorted([n_id + 1 for n_id in node_set.node_ids])
            extras = ns_extra.get(name, [])
            if extras:
                labels = sorted(labels + extras)
            for chunk in split_func(labels):
                f.write(", ".join(f"{n:8d}" for n in chunk) + "\n")


def _write_ccx_sur_file(mesh: "MeshModel", filename: str, split_func) -> bool:
    """Write the .sur file containing surface definitions."""
    boundary_sets = []
    for name in mesh.node_sets:
        if name.lower() in ["top", "bottom", "left", "right", "front", "back"]:
            boundary_sets.append(name)

    if not boundary_sets:
        return False

    with open(filename, "wt") as f:
        for name in boundary_sets:
            node_set = mesh.node_sets[name]
            node_ids = node_set.node_ids

            boundary_elements = []
            for i, el in enumerate(mesh.elements):
                if el.element_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                    corner_nodes = el.node_ids[:4]
                elif el.element_type in (ElementType.triangle, ElementType.triangle6):
                    corner_nodes = el.node_ids[:3]
                else:
                    corner_nodes = el.node_ids

                if any(n in node_ids for n in corner_nodes):
                    boundary_elements.append(i + 1)

            if boundary_elements:
                f.write(f"*SURFACE, NAME=S{name.upper()}, TYPE=ELEMENT\n")
                for el_id in boundary_elements:
                    f.write(f"{el_id:8d}, SPOS\n")

    return True


def _write_ccx_inp_file(
    mesh: "MeshModel",
    filename: str,
    base_name: str,
    has_surfaces: bool,
    properties: Optional[Dict[str, "ShellPropertyType"]] = None,
    boundary_nodeset: Optional[str] = None,
    num_modes: int = 10,
    span_direction: Optional[tuple] = None,
    orientation_reference_axis: Optional[tuple] = None,
    solver_type: str = "Modal",
    load_nodeset: Optional[str] = None,
    load_vector: Optional[list] = None,
    dt: float = 0.01,
    t_end: float = 1.0,
    quadratic: bool = False,
    nl_initial_increment: Optional[float] = None,
    nl_min_increment: Optional[float] = None,
    nl_max_increment: Optional[float] = None,
    nl_max_increments: Optional[int] = None,
    angle_bucket_sets: Optional[Dict[str, Dict[int, list[int]]]] = None,
    quadratic_data: Optional[Dict] = None,
) -> None:
    """Write the main .inp file with *INCLUDE statements and optional material/step."""
    with open(filename, "wt") as f:
        f.write("**\n")
        f.write("** CalculiX input file generated by aeroelast\n")
        f.write("** Following CGX file structure conventions\n")
        f.write("**\n")
        f.write("** ===========================================\n")
        f.write("**              MESH DEFINITION\n")
        f.write("** ===========================================\n")
        f.write("**\n")
        f.write(f"*INCLUDE, INPUT={base_name}.msh\n")
        f.write("**\n")
        f.write("** ===========================================\n")
        f.write("**         NODE AND ELEMENT SETS\n")
        f.write("** ===========================================\n")
        f.write("**\n")
        f.write(f"*INCLUDE, INPUT={base_name}.nam\n")

        if has_surfaces:
            f.write("**\n")
            f.write("** ===========================================\n")
            f.write("**            SURFACE DEFINITIONS\n")
            f.write("** ===========================================\n")
            f.write("**\n")
            f.write(f"*INCLUDE, INPUT={base_name}.sur\n")

        # ---- Material & section definitions (when properties provided) ----
        if properties is not None:
            _write_ccx_materials(f, properties)
            _write_ccx_orientations(
                f,
                properties,
                span_direction=span_direction,
                orientation_reference_axis=orientation_reference_axis,
                angle_bucket_sets=angle_bucket_sets,
            )
            _write_ccx_sections(
                f,
                mesh,
                properties,
                span_direction=span_direction,
                quadratic=quadratic,
                angle_bucket_sets=angle_bucket_sets,
            )
            if load_nodeset:
                n_load_nodes = len(mesh.get_node_set(load_nodeset).nodes)
                if quadratic_data is not None:
                    n_load_nodes += len(
                        quadratic_data.get("node_set_extra", {}).get(load_nodeset, [])
                    )
            else:
                n_load_nodes = 1
            if solver_type == "LinearStatic":
                _write_ccx_static_step(
                    f,
                    mesh,
                    boundary_nodeset,
                    load_nodeset,
                    load_vector,
                    n_load_nodes=n_load_nodes,
                )
            elif solver_type == "NonlinearStatic":
                _write_ccx_nonlinear_static_step(
                    f,
                    mesh,
                    boundary_nodeset,
                    load_nodeset,
                    load_vector,
                    n_load_nodes=n_load_nodes,
                    initial_increment=nl_initial_increment,
                    min_increment=nl_min_increment,
                    max_increment=nl_max_increment,
                    max_increments=nl_max_increments,
                )
            elif solver_type == "LinearDynamic":
                _write_ccx_dynamic_step(
                    f,
                    mesh,
                    boundary_nodeset,
                    load_nodeset,
                    load_vector,
                    dt,
                    t_end,
                    n_load_nodes=n_load_nodes,
                )
            else:
                _write_ccx_modal_step(f, mesh, boundary_nodeset, num_modes)
        else:
            # Legacy: placeholder comments
            _write_ccx_placeholder_comments(f, mesh)


def _write_ccx_materials(f, properties: Dict) -> None:
    """Write *MATERIAL blocks for every unique material found in properties."""
    try:
        from _aeroelast import Laminate as _RL, OrthotropicMaterial as _RMat  # noqa: PLC0415

        _has_rust = True
    except ImportError:
        _has_rust = False
    try:
        from aeroelast.core.material import IsotropicMaterial, OrthotropicMaterial as PyOrtho  # noqa: PLC0415
        from aeroelast.core.properties import CompositeShellProperty, ShellProperty  # noqa: PLC0415

        _has_py = True
    except ImportError:
        _has_py = False

    # Collect unique materials — (name, writer_callable) pairs
    # writer_callable(f, mat) writes the *ELASTIC and *DENSITY blocks
    materials_out: Dict[str, object] = {}

    for set_name, prop in properties.items():
        if _has_rust and isinstance(prop, _RL):
            # Rust Laminate: collect per-ply orthotropic materials.
            # Each ply may share the same material constants as another ply —
            # we deduplicate by a hash of the material constants so that plies
            # with identical properties share a single *MATERIAL block in CCX.
            plies = getattr(prop, "plies", None)
            if plies is None and hasattr(prop, "plies_data"):
                plies = prop.plies_data()
            if plies is None:
                # Extension without ply accessors: fall back to a single
                # equivalent laminate material so section references MAT_{set}
                # still resolve (non-quadratic exports rely on this name).
                mat_name = f"MAT_{set_name}"
                if mat_name not in materials_out:
                    materials_out[mat_name] = ("rust_laminate_equiv", prop)
                continue
            for i, ply in enumerate(plies):
                m = ply["material"]
                # Build a stable key from the material constants (rounded to
                # avoid float noise).  Using 4 significant figures is enough
                # to distinguish physically different materials.
                key = (
                    round(m["e1"], -3),
                    round(m["e2"], -3),
                    round(m["e3"], -3),
                    round(m["g12"], -3),
                    round(m["g23"], -3),
                    round(m["g13"], -3),
                    round(m["nu12"], 4),
                    round(m["nu23"], 4),
                    round(m["nu31"], 4),
                    round(m["rho"], 2),
                )
                mat_name = f"MAT_{set_name}_P{i}"
                # Use deduplication: if same constants already stored under a
                # different name we still need a mapping from (set_name, ply_i)
                # to the canonical name.  Store by (set_name, ply_i) key.
                mat_key = (set_name, i)
                if mat_key not in materials_out:
                    materials_out[mat_key] = ("rust_ply", m, mat_name, ply["thickness"])
            # Also register the equivalent laminate material MAT_{set_name} for
            # non-quadratic sections that reference MATERIAL=MAT_{set_name}.
            equiv_name = f"MAT_{set_name}"
            if equiv_name not in materials_out:
                materials_out[equiv_name] = ("rust_laminate_equiv", prop)
        elif _has_py and isinstance(prop, CompositeShellProperty):
            # Keep per-ply materials for COMPOSITE sections (quadratic shells)
            for ply in prop.laminate.plies:
                mat = ply.material
                if mat.name not in materials_out:
                    materials_out[mat.name] = ("py_ortho", mat)

            # Also provide an equivalent laminate material MAT_{set_name} for
            # non-quadratic exports where sections reference MATERIAL=MAT_{set}.
            mat_name = f"MAT_{set_name}"
            if mat_name not in materials_out:
                materials_out[mat_name] = ("py_laminate_equiv", prop.laminate)
        elif isinstance(prop, dict) and prop.get("type") == "isotropic":
            mat_name = prop.get("name", f"MAT_{set_name}")
            if mat_name not in materials_out:
                materials_out[mat_name] = ("iso_dict", prop, set_name)
        elif _has_py and isinstance(prop, ShellProperty):
            mat = prop.material
            if mat.name not in materials_out:
                materials_out[mat.name] = ("py_iso", mat)

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           MATERIAL DEFINITIONS\n")
    f.write("** ===========================================\n")

    for mat_key, entry in materials_out.items():
        kind = entry[0]
        mat_name = (
            entry[2]
            if kind == "rust_ply"
            else (mat_key if isinstance(mat_key, str) else f"MAT_{mat_key}")
        )
        f.write("**\n")
        f.write(f"*MATERIAL, NAME={mat_name}\n")
        if kind == "rust_ply":
            _, m, _mat_name, _t = entry
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            nu12 = max(-0.99, min(0.99, m["nu12"]))
            nu13 = max(-0.99, min(0.99, m["nu31"]))  # nu13 = nu31 in CCX convention
            nu23 = max(-0.99, min(0.99, m["nu23"]))
            f.write(
                f"{m['e1']:.6E}, {m['e2']:.6E}, {m['e3']:.6E}, "
                f"{nu12:.6f}, {nu13:.6f}, {nu23:.6f}, "
                f"{m['g12']:.6E}, {m['g13']:.6E}\n"
            )
            f.write(f"{m['g23']:.6E}\n")
            f.write("*DENSITY\n")
            f.write(f"{m['rho']:.6E}\n")
        elif kind == "rust_laminate_equiv":
            prop = entry[1]
            t = prop.total_thickness
            a = prop.abd_matrix()
            import numpy as np  # noqa: PLC0415

            D = np.array([
                [float(a[3, 3]), float(a[3, 4]), float(a[3, 5])],
                [float(a[4, 3]), float(a[4, 4]), float(a[4, 5])],
                [float(a[5, 3]), float(a[5, 4]), float(a[5, 5])],
            ])
            A = np.array([
                [float(a[0, 0]), float(a[0, 1]), float(a[0, 2])],
                [float(a[1, 0]), float(a[1, 1]), float(a[1, 2])],
                [float(a[2, 0]), float(a[2, 1]), float(a[2, 2])],
            ])
            if t > 0 and abs(np.linalg.det(D)) > 1e-30:
                S_D = np.linalg.inv(D) * (t**3 / 12.0)
                E1 = 1.0 / S_D[0, 0]
                E2 = 1.0 / S_D[1, 1]
                nu12 = -S_D[0, 1] * E1
                G12 = 1.0 / S_D[2, 2]
            elif t > 0 and abs(np.linalg.det(A)) > 1e-30:
                S_A = np.linalg.inv(A) * t
                E1 = 1.0 / S_A[0, 0]
                E2 = 1.0 / S_A[1, 1]
                nu12 = -S_A[0, 1] * E1
                G12 = 1.0 / S_A[2, 2]
            else:
                E1 = E2 = G12 = 1.0
                nu12 = 0.0
            E3 = E2
            G13 = G12
            G23 = G12
            nu12 = max(-0.99, min(0.99, nu12))
            nu13 = nu12
            nu23 = max(-0.99, min(0.99, float(a[1, 1]) / max(float(a[0, 0]), 1e-30) * nu12))
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            f.write(
                f"{E1:.6E}, {E2:.6E}, {E3:.6E}, {nu12:.6f}, {nu13:.6f}, {nu23:.6f}, {G12:.6E}, {G13:.6E}\n"
            )
            f.write(f"{G23:.6E}\n")
            t_eff = prop.total_thickness if prop.total_thickness > 0.0 else 1.0
            rho_equiv = prop.areal_density / t_eff
            f.write("*DENSITY\n")
            f.write(f"{rho_equiv:.6E}\n")
        elif kind == "py_ortho":
            mat = entry[1]
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            f.write(
                f"{mat.E1:.6E}, {mat.E2:.6E}, {mat.E3:.6E}, "
                f"{mat.nu12:.6f}, {mat.nu31:.6f}, {mat.nu23:.6f}, "
                f"{mat.G12:.6E}, {mat.G31:.6E}\n"
            )
            f.write(f"{mat.G23:.6E}\n")
            f.write("*DENSITY\n")
            f.write(f"{mat.rho:.6E}\n")
        elif kind == "py_laminate_equiv":
            lam = entry[1]
            eq = lam.get_equivalent_properties()
            E1 = float(eq["Ex_membrane"])
            E2 = float(eq["Ey_membrane"])
            E3 = E2
            nu12 = float(eq["nuxy_membrane"])
            nu13 = nu12
            nu23 = nu12
            G12 = float(eq["Gxy_membrane"])
            G13 = G12
            G23 = G12
            nu12 = max(-0.99, min(0.99, nu12))
            nu13 = max(-0.99, min(0.99, nu13))
            nu23 = max(-0.99, min(0.99, nu23))
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            f.write(
                f"{E1:.6E}, {E2:.6E}, {E3:.6E}, "
                f"{nu12:.6f}, {nu13:.6f}, {nu23:.6f}, "
                f"{G12:.6E}, {G13:.6E}\n"
            )
            f.write(f"{G23:.6E}\n")
            t = float(lam.total_thickness)
            rho_areal = float(sum(p.material.rho * p.thickness for p in lam.plies))
            rho = rho_areal / max(t, 1e-16)
            f.write("*DENSITY\n")
            f.write(f"{rho:.6E}\n")
        elif kind == "iso_dict":
            d = entry[1]
            f.write("*ELASTIC\n")
            f.write(f"{d['e']:.6E}, {d['nu']:.6f}\n")
            f.write("*DENSITY\n")
            f.write(f"{d['rho']:.6E}\n")
        elif kind == "py_iso":
            mat = entry[1]
            f.write("*ELASTIC\n")
            f.write(f"{mat.E:.6E}, {mat.nu:.6f}\n")
            f.write("*DENSITY\n")
            f.write(f"{mat.rho:.6E}\n")


def _write_ccx_orientations(
    f,
    properties: Dict[str, "ShellPropertyType"],
    span_direction: Optional[tuple] = None,
    orientation_reference_axis: Optional[tuple] = None,
    angle_bucket_sets: Optional[Dict[str, Dict[int, list[int]]]] = None,
) -> None:
    """Write *ORIENTATION blocks for each unique ply angle.

    When *span_direction* is supplied (e.g. ``(0, 0, 1)`` for a blade along
    Z), every ply angle – including 0° – gets an orientation whose reference
    axis is the projected span direction on the element surface.  This
    corrects the common bug where the element's first edge (often chordwise)
    is used as the 0° fibre reference instead of the structural span.

    Without *span_direction* the previous behaviour is preserved: the global
    X axis is used as the reference and only non-zero angles are emitted.
    """
    try:
        from _aeroelast import Laminate as _RL  # noqa: PLC0415
    except ImportError:
        _RL = None
    try:
        from aeroelast.core.properties import CompositeShellProperty  # noqa: PLC0415
    except ImportError:
        CompositeShellProperty = None

    # Collect unique ply angles; when a span direction is given, include 0°.
    angles: set = set()
    for prop in properties.values():
        if _RL is not None and isinstance(prop, _RL):
            # Rust Laminate: now ply angles are available via prop.plies.
            # Always add the real ply angles (needed for COMPOSITE sections).
            plies = getattr(prop, "plies", None)
            if plies is None and hasattr(prop, "plies_data"):
                plies = prop.plies_data()
            if plies is None:
                continue
            for ply in plies:
                if span_direction is not None or abs(ply["angle"]) > 1e-10:
                    angles.add(ply["angle"])
        elif _prop_is_composite(prop):
            if CompositeShellProperty is not None and isinstance(prop, CompositeShellProperty):
                for ply in prop.laminate.plies:
                    if span_direction is not None or abs(ply.angle) > 1e-10:
                        angles.add(ply.angle)

    # Include bucket-derived angles (used by Rust laminate per-element orientation)
    if angle_bucket_sets:
        for buckets in angle_bucket_sets.values():
            for bucket_tenths in buckets.keys():
                angles.add(bucket_tenths / 10.0)

    if not angles:
        return

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          ORIENTATION DEFINITIONS\n")
    f.write("** ===========================================\n")

    if span_direction is not None:
        # Normalise the span vector to use as the orientation 1-axis.
        sd = span_direction
        sd_len = (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) ** 0.5
        if sd_len < 1e-12:
            raise ValueError("span_direction must be a non-zero vector")
        s1, s2, s3 = sd[0] / sd_len, sd[1] / sd_len, sd[2] / sd_len
        # Choose a second-axis reference that defines the 1-2 plane.
        # Default keeps current behavior (axis with smallest span component).
        if orientation_reference_axis is not None:
            if len(orientation_reference_axis) != 3:
                raise ValueError("orientation_reference_axis must have 3 components")
            qv = np.asarray(orientation_reference_axis, dtype=float)
            qn = float(np.linalg.norm(qv))
            if qn < 1e-12:
                raise ValueError("orientation_reference_axis must be non-zero")
            qv = qv / qn
            # Must not be near-parallel to span axis (otherwise orientation plane is ill-defined)
            if abs(float(np.dot(qv, np.asarray([s1, s2, s3], dtype=float)))) > 0.999:
                raise ValueError(
                    "orientation_reference_axis is nearly parallel to span_direction; choose a different axis"
                )
            q1, q2, q3 = float(qv[0]), float(qv[1]), float(qv[2])
        else:
            abs_sd = (abs(s1), abs(s2), abs(s3))
            min_idx = abs_sd.index(min(abs_sd))
            q = [0.0, 0.0, 0.0]
            q[min_idx] = 1.0
            q1, q2, q3 = q[0], q[1], q[2]
        # CCX RECTANGULAR orientation: first 3 numbers = direction of local 1-axis;
        # next 3 = a point in the 1-2 plane (used to define local 2-axis).
        ref_line = f"{s1:.4f}, {s2:.4f}, {s3:.4f}, {q1:.4f}, {q2:.4f}, {q3:.4f}"
    else:
        # Legacy behaviour: base system aligned with global X, Y.
        ref_line = "1.0, 0.0, 0.0, 0.0, 1.0, 0.0"

    for angle in sorted(angles):
        ori_name = _ccx_orientation_name(angle)
        f.write("**\n")
        f.write(f"*ORIENTATION, NAME={ori_name}, SYSTEM=RECTANGULAR\n")
        f.write(ref_line + "\n")
        if abs(angle) > 1e-10:
            # Additional rotation around axis 3 (shell normal)
            f.write(f"3, {angle:.4f}\n")


def _ccx_orientation_name(angle: float) -> str:
    """Generate a CalculiX orientation name from a ply angle."""
    # e.g. 45.0 -> ORI_P45_0, -45.0 -> ORI_N45_0
    sign = "N" if angle < 0 else "P"
    a = abs(angle)
    integer_part = int(a)
    decimal_part = int(round((a - integer_part) * 10))
    return f"ORI_{sign}{integer_part}_{decimal_part}"


def _write_ccx_sections(
    f,
    mesh: "MeshModel",
    properties: Dict[str, "ShellPropertyType"],
    span_direction: Optional[tuple] = None,
    quadratic: bool = False,
    angle_bucket_sets: Optional[Dict[str, Dict[int, list[int]]]] = None,
) -> None:
    """Write *SHELL SECTION or *SOLID SECTION blocks for each element set.

    Detects element types in the mesh to determine whether to use:
    - *SOLID SECTION for 3D solid elements (C3D*)
    - *SHELL SECTION for 2D shell elements (S3/S4/S6/S8R)

    When *quadratic* is True, composite laminates use ``*SHELL SECTION,
    COMPOSITE`` with per-ply thickness/material/orientation — requires S8R/S6.
    When *quadratic* is False (default), composite laminates use a single
    ``*SHELL SECTION, MATERIAL=`` with the orthotropic equivalent derived from
    the A-matrix inversion — compatible with S4/S3 first-order elements.

    When *span_direction* is given every ply (including 0°) references the
    corresponding ORI_Pxx_x orientation so that CCX measures ply angles from
    the span axis rather than from the element's default local axis.
    """
    try:
        from _aeroelast import Laminate as _RL  # noqa: PLC0415
    except ImportError:
        _RL = None
    try:
        from aeroelast.core.properties import CompositeShellProperty, ShellProperty  # noqa: PLC0415
    except ImportError:
        CompositeShellProperty = None
        ShellProperty = None

    # Detect if mesh contains shell or solid elements
    has_solid_elements = any(
        el.element_type.name
        in (
            "hexahedron",
            "tetra",
            "wedge",
            "pyramid",
            "hexahedron20",
            "tetra10",
            "wedge15",
            "pyramid13",
        )
        for el in mesh.elements
    )

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           SECTION DEFINITIONS\n")
    f.write("** ===========================================\n")

    for set_name, prop in properties.items():
        elset_name = f"E{set_name.upper()}"
        f.write("**\n")

        if _RL is not None and isinstance(prop, _RL):
            mat_name = f"MAT_{set_name}"
            t = prop.total_thickness
            buckets_for_set = (
                (angle_bucket_sets or {}).get(set_name, {}) if span_direction is not None else {}
            )

            if quadratic:
                # S8R/S6 — *SHELL SECTION, COMPOSITE with per-ply real data.
                # Each ply gets its own material name (MAT_{set_name}_P{i}) and
                # real angle.  This exactly mirrors the AeroElast CLT assembly.
                f.write(f"*SHELL SECTION, ELSET={elset_name}, COMPOSITE\n")
                plies = getattr(prop, "plies", None)
                if plies is None and hasattr(prop, "plies_data"):
                    plies = prop.plies_data()
                if plies is None:
                    # Fallback: keep previous behavior to avoid crashing export.
                    t_ply = t / max(prop.n_plies, 1)
                    for i in range(prop.n_plies):
                        ply_mat_name = f"MAT_{set_name}_P{i}"
                        if span_direction is not None:
                            ori_name = _ccx_orientation_name(0.0)
                            f.write(f"{t_ply:.6E}, , {ply_mat_name}, {ori_name}\n")
                        else:
                            f.write(f"{t_ply:.6E}, , {ply_mat_name}\n")
                    continue
                for i, ply in enumerate(plies):
                    ply_mat_name = f"MAT_{set_name}_P{i}"
                    ply_angle = ply["angle"]
                    ply_t = ply["thickness"]
                    if span_direction is not None or abs(ply_angle) > 1e-10:
                        ori_name = _ccx_orientation_name(ply_angle)
                        f.write(f"{ply_t:.6E}, , {ply_mat_name}, {ori_name}\n")
                    else:
                        f.write(f"{ply_t:.6E}, , {ply_mat_name}\n")
            else:
                # S4/S3 — *SHELL SECTION, MATERIAL= with orthotropic equivalent
                if span_direction is not None and buckets_for_set:
                    # Match Rust assembler behavior: one section per angle bucket for this set
                    for bucket_tenths, elem_ids in sorted(buckets_for_set.items()):
                        if not elem_ids:
                            continue
                        angle = bucket_tenths / 10.0
                        ori_name = _ccx_orientation_name(angle)
                        bucket_elset = _bucket_elset_name(set_name, bucket_tenths)
                        f.write(
                            f"*SHELL SECTION, ELSET=E{bucket_elset.upper()}, MATERIAL={mat_name}, ORIENTATION={ori_name}\n"
                        )
                        f.write(f"{t:.6E}\n")
                else:
                    if span_direction is not None:
                        ori_name = _ccx_orientation_name(0.0)
                        f.write(
                            f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}, ORIENTATION={ori_name}\n"
                        )
                    else:
                        f.write(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}\n")
                    f.write(f"{t:.6E}\n")
        elif CompositeShellProperty is not None and isinstance(prop, CompositeShellProperty):
            mat_name = f"MAT_{set_name}"
            t = (
                prop.laminate.total_thickness
                if hasattr(prop.laminate, "total_thickness")
                else sum(p.thickness for p in prop.laminate.plies)
            )
            if quadratic:
                # S8R/S6 — *SHELL SECTION, COMPOSITE with per-ply detail
                f.write(f"*SHELL SECTION, ELSET={elset_name}, COMPOSITE\n")
                for ply in prop.laminate.plies:
                    if span_direction is not None or abs(ply.angle) > 1e-10:
                        ori_name = _ccx_orientation_name(ply.angle)
                        f.write(f"{ply.thickness:.6E}, , {ply.material.name}, {ori_name}\n")
                    else:
                        f.write(f"{ply.thickness:.6E}, , {ply.material.name}\n")
            else:
                # S4/S3 — *SHELL SECTION, MATERIAL= with orthotropic equivalent
                if span_direction is not None:
                    ori_name = _ccx_orientation_name(0.0)
                    f.write(
                        f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}, ORIENTATION={ori_name}\n"
                    )
                else:
                    f.write(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}\n")
                f.write(f"{t:.6E}\n")
        elif isinstance(prop, dict) and prop.get("type") == "isotropic":
            mat_name = prop.get("name", f"MAT_{set_name}")
            if has_solid_elements:
                # Use *SOLID SECTION for 3D solid elements (C3D*)
                f.write(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={mat_name}\n")
                # No thickness needed for solids - CCX infers from element geometry
            else:
                # Use *SHELL SECTION for 2D shell elements (S3/S4/S6/S8R)
                thickness = prop.get("thickness", 1.0)
                f.write(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={mat_name}\n")
                f.write(f"{thickness:.6E}\n")
        elif ShellProperty is not None and isinstance(prop, ShellProperty):
            if has_solid_elements:
                f.write(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={prop.material.name}\n")
            else:
                f.write(f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={prop.material.name}\n")
                f.write(f"{prop.thickness:.6E}\n")


def _write_ccx_modal_step(
    f, mesh: "MeshModel", boundary_nodeset: Optional[str], num_modes: int
) -> None:
    """Write boundary conditions and *FREQUENCY step."""
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          BOUNDARY CONDITIONS\n")
    f.write("** ===========================================\n")
    f.write("**\n")

    if boundary_nodeset:
        nset_name = f"N{boundary_nodeset.upper()}"
        f.write(f"*BOUNDARY\n")
        f.write(f"{nset_name}, 1, 6, 0.0\n")
    else:
        # Fall back to first available node set
        for name in mesh.node_sets:
            f.write(f"*BOUNDARY\n")
            f.write(f"N{name.upper()}, 1, 6, 0.0\n")
            break

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**              MODAL ANALYSIS\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("*STEP\n")
    f.write("*FREQUENCY\n")
    f.write(f"{num_modes}\n")
    f.write("**\n")
    f.write("*NODE FILE, OUTPUT=2D\n")
    f.write("U\n")
    f.write("*EL FILE, OUTPUT=2D\n")
    f.write("S, E\n")
    f.write("**\n")
    f.write("*END STEP\n")


def _write_ccx_cload(
    f,
    load_nodeset: Optional[str],
    load_vector: Optional[list],
    n_nodes: int = 1,
) -> None:
    """Write *CLOAD lines from a force vector.

    CCX format: one line per non-zero component — ``nset, dof_1based, magnitude``.
    DOF indices in CCX are 1-based (1=X, 2=Y, 3=Z, 4=RX, 5=RY, 6=RZ).

    ``load_vector`` is the **total** resultant force [N].  CCX applies *CLOAD
    to every node in the set individually, so we divide by ``n_nodes`` so that
    the sum across all nodes equals the intended resultant.
    """
    if not load_nodeset or not load_vector:
        return
    n = max(n_nodes, 1)
    load_nset = f"N{load_nodeset.upper()}"
    f.write("*CLOAD\n")
    for dof_0based, magnitude in enumerate(load_vector):
        if magnitude != 0.0:
            f.write(f"{load_nset}, {dof_0based + 1}, {magnitude / n:.6E}\n")
    f.write("**\n")


def _write_ccx_static_step(
    f,
    mesh: "MeshModel",
    boundary_nodeset: Optional[str],
    load_nodeset: Optional[str],
    load_vector: Optional[list],
    n_load_nodes: int = 1,
) -> None:
    """Write boundary conditions and *STATIC step with cload."""
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          BOUNDARY CONDITIONS\n")
    f.write("** ===========================================\n")
    f.write("**\n")

    if boundary_nodeset:
        nset_name = f"N{boundary_nodeset.upper()}"
        f.write("*BOUNDARY\n")
        f.write(f"{nset_name}, 1, 6, 0.0\n")
    else:
        for name in mesh.node_sets:
            f.write("*BOUNDARY\n")
            f.write(f"N{name.upper()}, 1, 6, 0.0\n")
            break

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           LINEAR STATIC ANALYSIS\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("*STEP\n")
    f.write("*STATIC\n")
    f.write("**\n")

    _write_ccx_cload(f, load_nodeset, load_vector, n_nodes=n_load_nodes)

    f.write("*NODE FILE, OUTPUT=2D\n")
    f.write("U, RF\n")
    f.write("*EL FILE, OUTPUT=2D\n")
    f.write("S, E\n")
    f.write("**\n")
    f.write("*END STEP\n")


def _write_ccx_nonlinear_static_step(
    f,
    mesh: "MeshModel",
    boundary_nodeset: Optional[str],
    load_nodeset: Optional[str],
    load_vector: Optional[list],
    n_load_nodes: int = 1,
    initial_increment: Optional[float] = None,
    min_increment: Optional[float] = None,
    max_increment: Optional[float] = None,
    max_increments: Optional[int] = None,
) -> None:
    """Write boundary conditions and *STATIC, NLGEOM step with cload."""

    def _resolve_nl_ctrl(
        initial: Optional[float],
        min_inc: Optional[float],
        max_inc: Optional[float],
        max_incs: Optional[int],
    ) -> tuple[float, float, float, int]:
        init = 1.0 if initial is None else float(initial)
        min_i = 1.0e-5 if min_inc is None else float(min_inc)
        max_i = 1.0 if max_inc is None else float(max_inc)
        max_n = 100 if max_incs is None else int(max_incs)

        if init <= 0.0:
            raise ValueError(f"nl_initial_increment must be positive, got {init}")
        if min_i <= 0.0:
            raise ValueError(f"nl_min_increment must be positive, got {min_i}")
        if max_i <= 0.0:
            raise ValueError(f"nl_max_increment must be positive, got {max_i}")
        if max_n <= 0:
            raise ValueError(f"nl_max_increments must be > 0, got {max_n}")
        if min_i > init:
            raise ValueError(
                f"nl_min_increment ({min_i}) cannot exceed nl_initial_increment ({init})"
            )
        if init > max_i:
            raise ValueError(
                f"nl_initial_increment ({init}) cannot exceed nl_max_increment ({max_i})"
            )
        return init, min_i, max_i, max_n

    init_inc, min_inc, max_inc, max_incs = _resolve_nl_ctrl(
        initial_increment,
        min_increment,
        max_increment,
        max_increments,
    )

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          BOUNDARY CONDITIONS\n")
    f.write("** ===========================================\n")
    f.write("**\n")

    if boundary_nodeset:
        nset_name = f"N{boundary_nodeset.upper()}"
        f.write("*BOUNDARY\n")
        f.write(f"{nset_name}, 1, 6, 0.0\n")
    else:
        for name in mesh.node_sets:
            f.write("*BOUNDARY\n")
            f.write(f"N{name.upper()}, 1, 6, 0.0\n")
            break

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**        NONLINEAR STATIC ANALYSIS\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write(f"*STEP, NLGEOM, INC={max_incs}\n")
    f.write("*STATIC\n")
    f.write(f"{init_inc:.6E}, 1.000000E+00, {min_inc:.6E}, {max_inc:.6E}\n")
    f.write("**\n")

    _write_ccx_cload(f, load_nodeset, load_vector, n_nodes=n_load_nodes)

    f.write("*NODE FILE, OUTPUT=2D\n")
    f.write("U, RF\n")
    f.write("*EL FILE, OUTPUT=2D\n")
    f.write("S, E\n")
    f.write("**\n")
    f.write("*END STEP\n")


def _write_ccx_dynamic_step(
    f,
    mesh: "MeshModel",
    boundary_nodeset: Optional[str],
    load_nodeset: Optional[str],
    load_vector: Optional[list],
    dt: float,
    t_end: float,
    n_load_nodes: int = 1,
) -> None:
    """Write boundary conditions and *DYNAMIC step with cload."""
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          BOUNDARY CONDITIONS\n")
    f.write("** ===========================================\n")
    f.write("**\n")

    if boundary_nodeset:
        nset_name = f"N{boundary_nodeset.upper()}"
        f.write("*BOUNDARY\n")
        f.write(f"{nset_name}, 1, 6, 0.0\n")
    else:
        for name in mesh.node_sets:
            f.write("*BOUNDARY\n")
            f.write(f"N{name.upper()}, 1, 6, 0.0\n")
            break

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          LINEAR DYNAMIC ANALYSIS\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("*STEP\n")
    f.write("*DYNAMIC\n")
    f.write(f"{dt:.6E}, {t_end:.6E}\n")
    f.write("**\n")

    _write_ccx_cload(f, load_nodeset, load_vector, n_nodes=n_load_nodes)

    f.write("*NODE FILE, FREQUENCY=10\n")
    f.write("U, V\n")
    f.write("*EL FILE, FREQUENCY=10\n")
    f.write("S\n")
    f.write("**\n")
    f.write("*END STEP\n")


def _write_ccx_placeholder_comments(f, mesh: "MeshModel") -> None:
    """Write placeholder comments (legacy behavior when no properties given)."""
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           MATERIAL DEFINITION\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("** Define your material here, for example:\n")
    f.write("** *MATERIAL, NAME=steel\n")
    f.write("** *ELASTIC\n")
    f.write("** 210000, 0.3\n")
    f.write("** *DENSITY\n")
    f.write("** 7.85E-9\n")
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           SECTION DEFINITION\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("** Define shell section, for example:\n")
    f.write("** *SHELL SECTION, ELSET=Eall, MATERIAL=steel\n")
    f.write("** 0.01\n")
    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**              STEP DEFINITION\n")
    f.write("** ===========================================\n")
    f.write("**\n")
    f.write("** *STEP\n")
    f.write("** *STATIC\n")
    f.write("**\n")
    f.write("** Boundary conditions:\n")

    for name in mesh.node_sets:
        f.write("** *BOUNDARY\n")
        f.write(f"** N{name.upper()}, 1, 6, 0.0\n")
        break

    f.write("**\n")
    f.write("** Loads:\n")
    f.write("** *CLOAD or *DLOAD definitions here\n")
    f.write("**\n")
    f.write("** Output requests:\n")
    f.write("** *NODE FILE\n")
    f.write("** U\n")
    f.write("** *EL FILE\n")
    f.write("** S, E\n")
    f.write("**\n")
    f.write("** *END STEP\n")
    f.write("**\n")


# ============================================================================
# Gmsh MSH writer
# ============================================================================


def write_gmsh_mesh(mesh: "MeshModel", filename: str) -> None:
    """
    Write the mesh to a Gmsh MSH file format.

    This method exports the mesh in Gmsh format (.msh) including physical groups
    for node sets and element sets.
    """
    if not mesh.nodes:
        raise ValueError("Mesh has no nodes.")
    if not mesh.elements:
        raise ValueError("Mesh has no elements.")

    # Create node ID mapping
    node_id_to_gmsh = {node.id: i + 1 for i, node in enumerate(mesh.nodes)}

    with open(filename, "wt") as f:
        # Write MeshFormat section
        f.write("$MeshFormat\n")
        f.write("4.1 0 8\n")
        f.write("$EndMeshFormat\n")

        # Write PhysicalNames section
        physical_names = []
        physical_tag = 1

        elset_tags = {}
        for name in mesh.element_sets.keys():
            physical_names.append((2, physical_tag, name))
            elset_tags[name] = physical_tag
            physical_tag += 1

        nset_tags = {}
        for name in mesh.node_sets.keys():
            physical_names.append((0, physical_tag, name))
            nset_tags[name] = physical_tag
            physical_tag += 1

        if physical_names:
            f.write("$PhysicalNames\n")
            f.write(f"{len(physical_names)}\n")
            for dim, tag, name in physical_names:
                f.write(f'{dim} {tag} "{name}"\n')
            f.write("$EndPhysicalNames\n")

        # Write Entities section
        num_points = len(mesh.node_sets)
        num_surfaces = max(1, len(mesh.element_sets))

        f.write("$Entities\n")
        f.write(f"{num_points} 0 {num_surfaces} 0\n")

        # Write point entities
        point_tag = 1
        nset_entity_tags = {}
        for name, node_set in mesh.node_sets.items():
            if node_set.nodes:
                coords = np.array([node.coords for node in node_set.nodes.values()])
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                center = (min_coords + max_coords) / 2
                f.write(f"{point_tag} {center[0]} {center[1]} {center[2]} 1 {nset_tags[name]}\n")
                nset_entity_tags[name] = point_tag
                point_tag += 1

        # Write surface entities
        surface_tag = 1
        elset_entity_tags = {}
        coords = mesh.coords_array
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        if mesh.element_sets:
            for name, element_set in mesh.element_sets.items():
                f.write(f"{surface_tag} {min_coords[0]} {min_coords[1]} {min_coords[2]} ")
                f.write(f"{max_coords[0]} {max_coords[1]} {max_coords[2]} ")
                f.write(f"1 {elset_tags[name]} 0\n")
                elset_entity_tags[name] = surface_tag
                surface_tag += 1
        else:
            f.write(f"1 {min_coords[0]} {min_coords[1]} {min_coords[2]} ")
            f.write(f"{max_coords[0]} {max_coords[1]} {max_coords[2]} 0 0\n")

        f.write("$EndEntities\n")

        # Write Nodes section
        f.write("$Nodes\n")
        f.write(f"1 {len(mesh.nodes)} 1 {len(mesh.nodes)}\n")
        f.write(f"2 1 0 {len(mesh.nodes)}\n")
        for i, node in enumerate(mesh.nodes):
            f.write(f"{i + 1}\n")
        for node in mesh.nodes:
            f.write(f"{node.x} {node.y} {node.z}\n")
        f.write("$EndNodes\n")

        # Write Elements section
        f.write("$Elements\n")

        element_blocks = []

        if mesh.element_sets:
            for name, element_set in mesh.element_sets.items():
                elements_by_type: Dict[ElementType, list] = {}
                for el in element_set.elements:
                    if el.element_type not in elements_by_type:
                        elements_by_type[el.element_type] = []
                    elements_by_type[el.element_type].append(el)

                for el_type, elements in elements_by_type.items():
                    gmsh_type = ELEMENT_TYPE_TO_GMSH.get(el_type)
                    if gmsh_type:
                        element_blocks.append({
                            "entity_tag": elset_entity_tags[name],
                            "gmsh_type": gmsh_type,
                            "elements": elements,
                        })

            elements_in_sets = set()
            for el_set in mesh.element_sets.values():
                elements_in_sets.update(el.id for el in el_set.elements)

            elements_not_in_sets = [el for el in mesh.elements if el.id not in elements_in_sets]
            if elements_not_in_sets:
                elements_by_type = {}
                for el in elements_not_in_sets:
                    if el.element_type not in elements_by_type:
                        elements_by_type[el.element_type] = []
                    elements_by_type[el.element_type].append(el)

                for el_type, elements in elements_by_type.items():
                    gmsh_type = ELEMENT_TYPE_TO_GMSH.get(el_type)
                    if gmsh_type:
                        element_blocks.append({
                            "entity_tag": 1,
                            "gmsh_type": gmsh_type,
                            "elements": elements,
                        })
        else:
            elements_by_type = {}
            for el in mesh.elements:
                if el.element_type not in elements_by_type:
                    elements_by_type[el.element_type] = []
                elements_by_type[el.element_type].append(el)

            for el_type, elements in elements_by_type.items():
                gmsh_type = ELEMENT_TYPE_TO_GMSH.get(el_type)
                if gmsh_type:
                    element_blocks.append({
                        "entity_tag": 1,
                        "gmsh_type": gmsh_type,
                        "elements": elements,
                    })

        total_elements = sum(len(block["elements"]) for block in element_blocks)
        f.write(f"{len(element_blocks)} {total_elements} 1 {total_elements}\n")

        element_tag = 1
        for block in element_blocks:
            f.write(f"2 {block['entity_tag']} {block['gmsh_type']} {len(block['elements'])}\n")
            for el in block["elements"]:
                node_ids_gmsh = " ".join(str(node_id_to_gmsh[nid]) for nid in el.node_ids)
                f.write(f"{element_tag} {node_ids_gmsh}\n")
                element_tag += 1

        f.write("$EndElements\n")

        # Write NodeData section
        if mesh.node_sets:
            f.write("$NodeData\n")
            f.write("1\n")
            f.write('"NodeSets"\n')
            f.write("1\n")
            f.write("0.0\n")
            f.write("3\n")
            f.write("0\n")
            f.write("1\n")
            f.write(f"{len(mesh.nodes)}\n")

            node_values = {}
            for i, node in enumerate(mesh.nodes):
                node_values[node.id] = 0

            set_value = 1
            for name, node_set in mesh.node_sets.items():
                for node_id in node_set.node_ids:
                    node_values[node_id] = set_value
                set_value += 1

            for i, node in enumerate(mesh.nodes):
                f.write(f"{i + 1} {node_values[node.id]}\n")

            f.write("$EndNodeData\n")

    print(f"Gmsh mesh written to {filename}")


# ============================================================================
# HDF5 writer
# ============================================================================


def write_hdf5(mesh: "MeshModel", filepath, compression: str = "gzip") -> None:
    """Write mesh to HDF5 format."""
    import h5py

    comp_opts = {"compression": compression} if compression else {}

    with h5py.File(filepath, "w") as f:
        # Metadata
        f.attrs["mesh_format_version"] = "1.0"
        f.attrs["node_count"] = len(mesh.nodes)
        f.attrs["element_count"] = len(mesh.elements)

        # Nodes
        nodes_grp = f.create_group("nodes")
        coords = np.array([n.coords for n in mesh.nodes], dtype=np.float64)
        nodes_grp.create_dataset("coords", data=coords, **comp_opts)
        node_ids = np.array([n.id for n in mesh.nodes], dtype=np.int64)
        nodes_grp.create_dataset("ids", data=node_ids, **comp_opts)
        geometric_flags = np.array([n.geometric_node for n in mesh.nodes], dtype=np.bool_)
        nodes_grp.create_dataset("geometric_node", data=geometric_flags, **comp_opts)

        # Elements
        elements_grp = f.create_group("elements")
        element_ids = np.array([e.id for e in mesh.elements], dtype=np.int64)
        elements_grp.create_dataset("ids", data=element_ids, **comp_opts)
        element_types = np.array([int(e.element_type) for e in mesh.elements], dtype=np.int32)
        elements_grp.create_dataset("types", data=element_types, **comp_opts)

        all_node_ids = []
        offsets = [0]
        for e in mesh.elements:
            all_node_ids.extend(e.node_ids)
            offsets.append(len(all_node_ids))

        elements_grp.create_dataset(
            "connectivity", data=np.array(all_node_ids, dtype=np.int64), **comp_opts
        )
        elements_grp.create_dataset(
            "connectivity_offsets", data=np.array(offsets, dtype=np.int64), **comp_opts
        )

        # Node Sets
        if mesh.node_sets:
            nsets_grp = f.create_group("node_sets")
            for name, nset in mesh.node_sets.items():
                set_grp = nsets_grp.create_group(name)
                set_grp.create_dataset(
                    "node_ids", data=np.array(list(nset.node_ids), dtype=np.int64), **comp_opts
                )

        # Element Sets
        if mesh.element_sets:
            esets_grp = f.create_group("element_sets")
            for name, eset in mesh.element_sets.items():
                set_grp = esets_grp.create_group(name)
                set_grp.create_dataset(
                    "element_ids", data=np.array(eset.element_ids, dtype=np.int64), **comp_opts
                )

    print(f"Mesh saved to {filepath} (HDF5 format)")


# ============================================================================
# Pickle writer
# ============================================================================


def write_pickle(mesh: "MeshModel", filepath) -> None:
    """Write mesh to pickle format."""
    import pickle

    data = {
        "format_version": "1.0",
        "nodes": [
            {"id": n.id, "coords": n.coords.tolist(), "geometric_node": n.geometric_node}
            for n in mesh.nodes
        ],
        "elements": [
            {"id": e.id, "node_ids": list(e.node_ids), "element_type": int(e.element_type)}
            for e in mesh.elements
        ],
        "node_sets": {name: list(nset.node_ids) for name, nset in mesh.node_sets.items()},
        "element_sets": {name: eset.element_ids for name, eset in mesh.element_sets.items()},
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Mesh saved to {filepath} (pickle format)")
