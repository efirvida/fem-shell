"""
Mesh I/O readers module.

This module contains functions for loading meshes from various file formats:
- HDF5 format
- Pickle format
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fem_shell.core.mesh.model import MeshModel

from fem_shell.core.mesh.entities import ElementSet, ElementType, MeshElement, Node, NodeSet


def load_mesh(filepath: str, format: str = "auto") -> "MeshModel":
    """
    Load a mesh from disk.

    Parameters
    ----------
    filepath : str
        Path to the mesh file.
    format : str, optional
        File format: "auto", "hdf5", or "pickle". Default is "auto".

    Returns
    -------
    MeshModel
        A new MeshModel instance with all data restored.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If format is not recognized.
    """
    # Import here to avoid circular imports

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    # Determine format from extension if auto
    if format == "auto":
        ext = path.suffix.lower()
        if ext in (".h5", ".hdf5"):
            format = "hdf5"
        elif ext in (".pkl", ".pickle"):
            format = "pickle"
        else:
            raise ValueError(
                f"Cannot determine format from extension '{ext}'. "
                "Use .h5/.hdf5 for HDF5 or .pkl/.pickle for pickle, "
                "or specify format explicitly."
            )

    if format == "hdf5":
        return load_hdf5(filepath)
    elif format == "pickle":
        return load_pickle(filepath)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'hdf5' or 'pickle'.")


def load_hdf5(filepath) -> "MeshModel":
    """Load mesh from HDF5 format."""
    import h5py

    # Import here to avoid circular imports
    from fem_shell.core.mesh.model import MeshModel

    with h5py.File(filepath, "r") as f:
        # Load Nodes
        nodes_grp = f["nodes"]
        coords = nodes_grp["coords"][:]
        node_ids = nodes_grp["ids"][:]
        geometric_flags = nodes_grp["geometric_node"][:]

        # Temporarily store original counter
        original_counter = Node._id_counter

        nodes = []
        for i in range(len(coords)):
            node = Node(coords[i], geometric_node=bool(geometric_flags[i]))
            node.id = int(node_ids[i])  # Override auto-assigned ID
            nodes.append(node)

        # Build node lookup by ID
        node_lookup = {n.id: n for n in nodes}

        # Load Elements
        elements_grp = f["elements"]
        element_ids = elements_grp["ids"][:]
        element_types = elements_grp["types"][:]
        connectivity = elements_grp["connectivity"][:]
        offsets = elements_grp["connectivity_offsets"][:]

        original_elem_counter = MeshElement._id_counter

        elements = []
        for i in range(len(element_ids)):
            start = offsets[i]
            end = offsets[i + 1]
            elem_node_ids = connectivity[start:end]

            # Get node objects from IDs
            elem_nodes = [node_lookup[int(nid)] for nid in elem_node_ids]
            element_type = ElementType(int(element_types[i]))

            elem = MeshElement(elem_nodes, element_type)
            elem.id = int(element_ids[i])  # Override auto-assigned ID
            elements.append(elem)

        # Create mesh
        mesh = MeshModel(nodes=nodes, elements=elements)

        # Load Node Sets
        if "node_sets" in f:
            for name in f["node_sets"]:
                set_node_ids = f["node_sets"][name]["node_ids"][:]
                set_nodes = {node_lookup[int(nid)] for nid in set_node_ids}
                mesh.add_node_set(NodeSet(name, set_nodes))

        # Load Element Sets
        if "element_sets" in f:
            elem_lookup = {e.id: e for e in elements}
            for name in f["element_sets"]:
                set_elem_ids = f["element_sets"][name]["element_ids"][:]
                set_elements = {elem_lookup[int(eid)] for eid in set_elem_ids}
                mesh.add_element_set(ElementSet(name, set_elements))

        # Update class counters to avoid ID conflicts
        max_node_id = max(n.id for n in nodes) if nodes else -1
        max_elem_id = max(e.id for e in elements) if elements else -1
        Node._id_counter = max(max_node_id + 1, original_counter)
        MeshElement._id_counter = max(max_elem_id + 1, original_elem_counter)

    print(f"Mesh loaded from {filepath} (HDF5 format)")
    return mesh


def load_pickle(filepath) -> "MeshModel":
    """Load mesh from pickle format."""
    import pickle

    # Import here to avoid circular imports
    from fem_shell.core.mesh.model import MeshModel

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    original_node_counter = Node._id_counter
    original_elem_counter = MeshElement._id_counter

    # Reconstruct nodes
    nodes = []
    for n_data in data["nodes"]:
        node = Node(n_data["coords"], geometric_node=n_data["geometric_node"])
        node.id = n_data["id"]
        nodes.append(node)

    node_lookup = {n.id: n for n in nodes}

    # Reconstruct elements
    elements = []
    for e_data in data["elements"]:
        elem_nodes = [node_lookup[nid] for nid in e_data["node_ids"]]
        element_type = ElementType(e_data["element_type"])
        elem = MeshElement(elem_nodes, element_type)
        elem.id = e_data["id"]
        elements.append(elem)

    # Create mesh
    mesh = MeshModel(nodes=nodes, elements=elements)

    # Reconstruct node sets
    for name, node_ids in data["node_sets"].items():
        set_nodes = {node_lookup[nid] for nid in node_ids}
        mesh.add_node_set(NodeSet(name, set_nodes))

    # Reconstruct element sets
    elem_lookup = {e.id: e for e in elements}
    for name, elem_ids in data["element_sets"].items():
        set_elements = {elem_lookup[eid] for eid in elem_ids}
        mesh.add_element_set(ElementSet(name, set_elements))

    # Update class counters
    max_node_id = max(n.id for n in nodes) if nodes else -1
    max_elem_id = max(e.id for e in elements) if elements else -1
    Node._id_counter = max(max_node_id + 1, original_node_counter)
    MeshElement._id_counter = max(max_elem_id + 1, original_elem_counter)

    print(f"Mesh loaded from {filepath} (pickle format)")
    return mesh
