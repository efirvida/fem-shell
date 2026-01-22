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
from typing import TYPE_CHECKING, Dict

import meshio
import numpy as np

if TYPE_CHECKING:
    from fem_shell.core.mesh.model import MeshModel

from fem_shell.core.mesh.entities import ElementType

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
        write_ccx_mesh(mesh, filename)
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


def write_meshio(mesh: "MeshModel", filename: str, **kwargs) -> None:
    """Write mesh using meshio library (geometry only, no metadata)."""
    points = mesh.coords_array

    cells = []
    for el in mesh.elements:
        indices = tuple(mesh.node_id_to_index[nid] for nid in el.node_ids)
        cells.append((el.element_type.name, np.array([indices])))

    mesh_io = meshio.Mesh(points=points, cells=cells)
    meshio.write(filename, mesh_io, **kwargs)
    print(f"Mesh written to {filename}.")


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


def write_ccx_mesh(mesh: "MeshModel", filename: str) -> None:
    """
    Write the mesh to CalculiX format following CGX conventions.

    This method generates multiple files as recommended by CalculiX documentation:
    - {basename}.msh: Mesh file containing nodes and elements
    - {basename}.nam: Node sets and element sets definitions
    - {basename}.sur: Surface definitions (if applicable)
    - {basename}.inp: Main input file with *INCLUDE statements
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

    # Write files
    _write_ccx_msh_file(mesh, msh_file)
    _write_ccx_nam_file(mesh, nam_file, split_list)
    has_surfaces = _write_ccx_sur_file(mesh, sur_file, split_list)
    _write_ccx_inp_file(mesh, inp_file, base_name, has_surfaces)

    print("CalculiX mesh files written:")
    print(f"  - Mesh file: {msh_file}")
    print(f"  - Names file: {nam_file}")
    if has_surfaces:
        print(f"  - Surface file: {sur_file}")
    print(f"  - Main input file: {inp_file}")


def _write_ccx_msh_file(mesh: "MeshModel", filename: str) -> None:
    """Write the .msh file containing nodes and elements."""
    with open(filename, "wt") as f:
        # Write nodes
        f.write("*NODE, NSET=Nall\n")
        for i, nd in enumerate(mesh.nodes):
            f.write(f"{i + 1:8d}, {nd.x:14.6E}, {nd.y:14.6E}, {nd.z:14.6E}\n")

        # Group elements by type
        elements_by_type: Dict[str, list] = {}
        for i, el in enumerate(mesh.elements):
            el_type_name = el.element_type.name
            if el_type_name not in elements_by_type:
                elements_by_type[el_type_name] = []
            elements_by_type[el_type_name].append((i, el))

        # Write elements grouped by type
        for el_type_name, elements in elements_by_type.items():
            if el_type_name in ELEMENTS_TO_CALCULIX:
                ccx_type = ELEMENTS_TO_CALCULIX[el_type_name]
                f.write(f"*ELEMENT, TYPE={ccx_type}, ELSET=Eall\n")
                for i, el in elements:
                    node_ids_str = ", ".join(str(n + 1) for n in el.node_ids)
                    f.write(f"{i + 1:8d}, {node_ids_str}\n")


def _write_ccx_nam_file(mesh: "MeshModel", filename: str, split_func) -> None:
    """Write the .nam file containing node sets and element sets."""
    with open(filename, "wt") as f:
        # Write element sets
        for name, element_set in mesh.element_sets.items():
            f.write(f"*ELSET, ELSET=E{name.upper()}\n")
            labels = [el_id + 1 for el_id in element_set.element_ids]
            for chunk in split_func(labels):
                f.write(", ".join(f"{e:8d}" for e in chunk) + "\n")

        # Write node sets
        for name, node_set in mesh.node_sets.items():
            f.write(f"*NSET, NSET=N{name.upper()}\n")
            labels = sorted([n_id + 1 for n_id in node_set.node_ids])
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
    mesh: "MeshModel", filename: str, base_name: str, has_surfaces: bool
) -> None:
    """Write the main .inp file with *INCLUDE statements."""
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
