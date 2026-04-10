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
    from fem_shell.core.mesh.model import MeshModel

from fem_shell.core.mesh.entities import ElementType

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
            el.element_type.name in ('tetra', 'tetra10', 'hexahedron', 'hexahedron20', 'hexahedron27', 'wedge', 'wedge15', 'pyramid', 'pyramid13')
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

                    cap_tris, centroid = _cap_tip_loop(
                        tip_loop, points, id_to_idx
                    )
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
    """

    def split_list(arr, chunk_size: int = 7):
        """Split list into chunks for formatted output."""
        arr = list(arr)
        return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

    # Extract base path and name
    base_path = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Define file paths
    msh_file = os.path.join(base_path, f"{base_name}.msh") if base_path else f"{base_name}.msh"
    nam_file = os.path.join(base_path, f"{base_name}.nam") if base_path else f"{base_name}.nam"
    sur_file = os.path.join(base_path, f"{base_name}.sur") if base_path else f"{base_name}.sur"
    inp_file = filename

    # Check if composite properties require quadratic elements
    has_composite = False
    if properties is not None:
        from fem_shell.core.properties import CompositeShellProperty
        has_composite = any(
            isinstance(p, CompositeShellProperty) for p in properties.values()
        )

    # Build quadratic mesh data if needed (CalculiX requires S8R/S6 for composite)
    quadratic_data = None
    if has_composite:
        quadratic_data = _build_quadratic_mesh_data(mesh)
        print(f"  Converted to quadratic: {quadratic_data['n_nodes']} nodes "
              f"({quadratic_data['n_midside']} midside nodes added)")

    # Write files
    _write_ccx_msh_file(mesh, msh_file, quadratic_data=quadratic_data)
    _write_ccx_nam_file(mesh, nam_file, split_list, quadratic_data=quadratic_data)
    has_surfaces = _write_ccx_sur_file(mesh, sur_file, split_list)
    _write_ccx_inp_file(
        mesh, inp_file, base_name, has_surfaces,
        properties=properties,
        boundary_nodeset=boundary_nodeset,
        num_modes=num_modes,
        span_direction=span_direction,
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
                [nids[0]+1, nids[1]+1, nids[2]+1,
                 m01+1, m12+1, m20+1],
            ))
        elif etype == "quad":
            m01 = _get_midside(nids[0], nids[1])
            m12 = _get_midside(nids[1], nids[2])
            m23 = _get_midside(nids[2], nids[3])
            m30 = _get_midside(nids[3], nids[0])
            elements.append((
                "S8R",
                [nids[0]+1, nids[1]+1, nids[2]+1, nids[3]+1,
                 m01+1, m12+1, m23+1, m30+1],
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
    mesh: "MeshModel", filename: str,
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
    mesh: "MeshModel", filename: str, split_func,
    quadratic_data: Optional[Dict] = None,
) -> None:
    """Write the .nam file containing node sets and element sets."""
    with open(filename, "wt") as f:
        # Write element sets
        for name, element_set in mesh.element_sets.items():
            f.write(f"*ELSET, ELSET=E{name.upper()}\n")
            labels = [el_id + 1 for el_id in element_set.element_ids]
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
) -> None:
    """Write the main .inp file with *INCLUDE statements and optional material/step."""
    from fem_shell.core.material import IsotropicMaterial, OrthotropicMaterial
    from fem_shell.core.properties import CompositeShellProperty, ShellProperty

    with open(filename, "wt") as f:
        f.write("**\n")
        f.write("** CalculiX input file generated by fem_shell\n")
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
            _write_ccx_orientations(f, properties, span_direction=span_direction)
            _write_ccx_sections(f, properties, span_direction=span_direction)
            _write_ccx_modal_step(f, mesh, boundary_nodeset, num_modes)
        else:
            # Legacy: placeholder comments
            _write_ccx_placeholder_comments(f, mesh)


def _write_ccx_materials(f, properties: Dict[str, "ShellPropertyType"]) -> None:
    """Write *MATERIAL blocks for every unique material found in properties."""
    from fem_shell.core.material import IsotropicMaterial, OrthotropicMaterial
    from fem_shell.core.properties import CompositeShellProperty, ShellProperty

    # Collect unique materials (by name) across all element sets
    materials: Dict[str, Union[IsotropicMaterial, OrthotropicMaterial]] = {}
    for prop in properties.values():
        if isinstance(prop, CompositeShellProperty):
            for ply in prop.laminate.plies:
                mat = ply.material
                materials[mat.name] = mat
        elif isinstance(prop, ShellProperty):
            materials[prop.material.name] = prop.material

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           MATERIAL DEFINITIONS\n")
    f.write("** ===========================================\n")

    for mat_name, mat in materials.items():
        f.write("**\n")
        f.write(f"*MATERIAL, NAME={mat_name}\n")

        if isinstance(mat, OrthotropicMaterial):
            # *ELASTIC, TYPE=ENGINEERING CONSTANTS
            # E1, E2, E3, nu12, nu13, nu23, G12, G13  (line 1)
            # G23                                       (line 2)
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            f.write(
                f"{mat.E1:.6E}, {mat.E2:.6E}, {mat.E3:.6E}, "
                f"{mat.nu12:.6f}, {mat.nu31:.6f}, {mat.nu23:.6f}, "
                f"{mat.G12:.6E}, {mat.G31:.6E}\n"
            )
            f.write(f"{mat.G23:.6E}\n")
        elif isinstance(mat, IsotropicMaterial):
            f.write("*ELASTIC\n")
            f.write(f"{mat.E:.6E}, {mat.nu:.6f}\n")

        f.write("*DENSITY\n")
        f.write(f"{mat.rho:.6E}\n")


def _write_ccx_orientations(
    f,
    properties: Dict[str, "ShellPropertyType"],
    span_direction: Optional[tuple] = None,
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
    from fem_shell.core.properties import CompositeShellProperty

    # Collect unique ply angles; when a span direction is given, include 0°.
    angles: set = set()
    for prop in properties.values():
        if isinstance(prop, CompositeShellProperty):
            for ply in prop.laminate.plies:
                if span_direction is not None or abs(ply.angle) > 1e-10:
                    angles.add(ply.angle)

    if not angles:
        return

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**          ORIENTATION DEFINITIONS\n")
    f.write("** ===========================================\n")

    if span_direction is not None:
        # Normalise the span vector to use as the orientation 1-axis.
        sd = span_direction
        sd_len = (sd[0]**2 + sd[1]**2 + sd[2]**2) ** 0.5
        if sd_len < 1e-12:
            raise ValueError("span_direction must be a non-zero vector")
        s1, s2, s3 = sd[0]/sd_len, sd[1]/sd_len, sd[2]/sd_len
        # Choose a second-axis vector that is not parallel to span.
        # Use the global axis with the *smallest* component of span.
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
    properties: Dict[str, "ShellPropertyType"],
    span_direction: Optional[tuple] = None,
) -> None:
    """Write *SHELL SECTION blocks for each element set.

    When *span_direction* is given every ply (including 0°) references the
    corresponding ORI_Pxx_x orientation so that CCX measures ply angles from
    the span axis rather than from the element's default local axis.
    """
    from fem_shell.core.properties import CompositeShellProperty, ShellProperty

    f.write("**\n")
    f.write("** ===========================================\n")
    f.write("**           SECTION DEFINITIONS\n")
    f.write("** ===========================================\n")

    for set_name, prop in properties.items():
        elset_name = f"E{set_name.upper()}"
        f.write("**\n")

        if isinstance(prop, CompositeShellProperty):
            f.write(f"*SHELL SECTION, ELSET={elset_name}, COMPOSITE\n")
            for ply in prop.laminate.plies:
                # Data line: thickness, nip, material_name, orientation_name
                if span_direction is not None or abs(ply.angle) > 1e-10:
                    ori_name = _ccx_orientation_name(ply.angle)
                    f.write(
                        f"{ply.thickness:.6E}, , {ply.material.name}, {ori_name}\n"
                    )
                else:
                    # No span_direction and angle==0: use element-default orientation
                    f.write(
                        f"{ply.thickness:.6E}, , {ply.material.name}\n"
                    )
        elif isinstance(prop, ShellProperty):
            f.write(
                f"*SHELL SECTION, ELSET={elset_name}, MATERIAL={prop.material.name}\n"
            )
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
    f.write("*NODE FILE\n")
    f.write("U\n")
    f.write("*EL FILE\n")
    f.write("S, E\n")
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
                        element_blocks.append(
                            {
                                "entity_tag": elset_entity_tags[name],
                                "gmsh_type": gmsh_type,
                                "elements": elements,
                            }
                        )

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
                        element_blocks.append(
                            {
                                "entity_tag": 1,
                                "gmsh_type": gmsh_type,
                                "elements": elements,
                            }
                        )
        else:
            elements_by_type = {}
            for el in mesh.elements:
                if el.element_type not in elements_by_type:
                    elements_by_type[el.element_type] = []
                elements_by_type[el.element_type].append(el)

            for el_type, elements in elements_by_type.items():
                gmsh_type = ELEMENT_TYPE_TO_GMSH.get(el_type)
                if gmsh_type:
                    element_blocks.append(
                        {
                            "entity_tag": 1,
                            "gmsh_type": gmsh_type,
                            "elements": elements,
                        }
                    )

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
