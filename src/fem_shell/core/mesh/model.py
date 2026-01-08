"""
MeshModel class module.

This module contains the main MeshModel class that represents a complete mesh
with nodes, elements, node sets, and element sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from fem_shell.core.mesh.entities import ElementSet, MeshElement, Node, NodeSet
from fem_shell.core.mesh.io import load_mesh, write_hdf5, write_mesh, write_pickle
from fem_shell.core.viewer import plot_mesh


class MeshModel:
    """
    Represents a mesh composed of nodes and connectivity elements.

    This class caches nodes and elements for fast lookup and supports node
    and element sets for applying specific boundary conditions or loads.

    Attributes
    ----------
    nodes : list of Node
        List of nodes in the mesh. Use add_node() to add new nodes.
    elements : list of MeshElement
        List of connectivity elements in the mesh. Use add_element() to add new elements.
    node_map : dict
        Dictionary mapping node IDs to Node instances.
    element_map : dict
        Dictionary mapping element IDs to MeshElement instances.
    node_sets : dict
        Dictionary mapping node set names to NodeSet instances.
    element_sets : dict
        Dictionary mapping element set names to ElementSet instances.
    """

    def __init__(
        self,
        nodes: Iterable[Node] | None = None,
        elements: Optional[List[MeshElement]] = None,
    ):
        """
        Initialize a MeshModel instance.

        Parameters
        ----------
        nodes : list of Node, optional
            Initial list of nodes (default is None, which creates an empty list).
        elements : list of MeshElement, optional
            Initial list of connectivity elements (default is None, which creates an empty list).

        Raises
        ------
        ValueError
            If duplicate node or element IDs are found in the initial lists.
        """
        # Use copies of input lists to avoid external modifications
        self.nodes = list(nodes) if nodes is not None else []
        self.elements = list(elements) if elements is not None else []

        self.node_map: Dict[int, Node] = {}
        self.element_map: Dict[int, MeshElement] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, ElementSet] = {}

        # Validate and build node_map
        seen_node_ids = set()
        for node in self.nodes:
            if node.id in seen_node_ids:
                raise ValueError(f"Duplicate node ID {node.id} in initial nodes list.")
            seen_node_ids.add(node.id)
            self.node_map[node.id] = node

        # Validate and build element_map
        seen_element_ids = set()
        for element in self.elements:
            if element.id in seen_element_ids:
                raise ValueError(f"Duplicate element ID {element.id} in initial elements list.")
            seen_element_ids.add(element.id)
            self.element_map[element.id] = element

        # Cache for ID-to-index mappings (invalidated when nodes/elements change)
        self._node_id_to_index_cache: Optional[Dict[int, int]] = None
        self._element_id_to_index_cache: Optional[Dict[int, int]] = None

    # =========================================================================
    # ID-to-Index Mapping Properties
    # =========================================================================

    @property
    def node_id_to_index(self) -> Dict[int, int]:
        """
        Mapping from node IDs to consecutive array indices (0-based).

        This is essential for FEM assembly and I/O operations when node IDs
        are not consecutive (e.g., after merging meshes from multiple blades).

        Returns
        -------
        Dict[int, int]
            Dictionary mapping node.id -> array index in self.nodes
        """
        if self._node_id_to_index_cache is None:
            self._node_id_to_index_cache = {node.id: idx for idx, node in enumerate(self.nodes)}
        return self._node_id_to_index_cache

    @property
    def element_id_to_index(self) -> Dict[int, int]:
        """
        Mapping from element IDs to consecutive array indices (0-based).

        This is essential for FEM operations when element IDs are not
        consecutive (e.g., after merging meshes).

        Returns
        -------
        Dict[int, int]
            Dictionary mapping element.id -> array index in self.elements
        """
        if self._element_id_to_index_cache is None:
            self._element_id_to_index_cache = {
                elem.id: idx for idx, elem in enumerate(self.elements)
            }
        return self._element_id_to_index_cache

    def get_node_index(self, node_id: int) -> int:
        """Get the array index for a node given its ID."""
        return self.node_id_to_index[node_id]

    def get_element_index(self, element_id: int) -> int:
        """Get the array index for an element given its ID."""
        return self.element_id_to_index[element_id]

    # =========================================================================
    # Renumbering
    # =========================================================================

    @property
    def needs_renumbering(self) -> bool:
        """
        Check if the mesh needs renumbering.

        Returns True if node IDs or element IDs don't match their array indices.
        """
        for idx, node in enumerate(self.nodes):
            if node.id != idx:
                return True
        return any(element.id != idx for idx, element in enumerate(self.elements))

    def renumber_mesh(self, algorithm: str = "simple", verbose: bool = False) -> "MeshModel":
        """
        Renumber node and element IDs to match their array indices.

        This operation optimizes the mesh for FEM assembly by ensuring that
        node.id == index and element.id == index.

        Parameters
        ----------
        algorithm : str, optional
            Renumbering algorithm to use:
            - "simple": Direct index assignment (default)
            - "rcm": Reverse Cuthill-McKee for bandwidth reduction
        verbose : bool, optional
            If True, print detailed progress information. Default is False.

        Returns
        -------
        MeshModel
            Self, for method chaining.
        """
        algorithms = {
            "simple": self._renumber_simple,
            "rcm": self._renumber_rcm,
        }

        if algorithm not in algorithms:
            raise ValueError(
                f"Unknown renumbering algorithm: {algorithm}. Available: {list(algorithms.keys())}"
            )

        algorithms[algorithm](verbose=verbose)
        return self

    def _renumber_simple(self, verbose: bool = False) -> None:
        """Simple renumbering: assign IDs equal to array indices."""
        if not self.needs_renumbering:
            if verbose:
                print(
                    "Mesh renumbering (simple): IDs already match indices, no renumbering needed."
                )
            return

        old_node_id_min = min(n.id for n in self.nodes) if self.nodes else 0
        old_node_id_max = max(n.id for n in self.nodes) if self.nodes else 0
        old_elem_id_min = min(e.id for e in self.elements) if self.elements else 0
        old_elem_id_max = max(e.id for e in self.elements) if self.elements else 0

        for new_id, node in enumerate(self.nodes):
            node.id = new_id

        for new_id, element in enumerate(self.elements):
            element.id = new_id

        self._finalize_renumbering()

        if verbose:
            print("Mesh renumbering completed (algorithm: simple):")
            print(f"  Nodes: {len(self.nodes):,} renumbered")
            if self.nodes:
                print(
                    f"    ID range: [{old_node_id_min:,} - {old_node_id_max:,}] -> [0 - {len(self.nodes) - 1:,}]"
                )
            print(f"  Elements: {len(self.elements):,} renumbered")
            if self.elements:
                print(
                    f"    ID range: [{old_elem_id_min:,} - {old_elem_id_max:,}] -> [0 - {len(self.elements) - 1:,}]"
                )
            print(f"  Node sets updated: {len(self.node_sets)}")
            print(f"  Element sets updated: {len(self.element_sets)}")

    def _renumber_rcm(self, verbose: bool = False) -> None:
        """Reverse Cuthill-McKee renumbering for bandwidth reduction."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        if not self.nodes or not self.elements:
            self._renumber_simple(verbose=verbose)
            return

        n_nodes = len(self.nodes)
        n_elements = len(self.elements)

        if verbose:
            print("Mesh renumbering (algorithm: rcm):")
            print(f"  Building adjacency graph ({n_nodes:,} nodes, {n_elements:,} elements)...")

        def compute_bandwidth():
            max_diff = 0
            for elem in self.elements:
                node_ids = [n.id for n in elem.nodes]
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        diff = abs(node_ids[i] - node_ids[j])
                        max_diff = max(max_diff, diff)
            return max_diff

        bandwidth_before = compute_bandwidth()

        row_indices = []
        col_indices = []

        for element in self.elements:
            node_ids = element.node_ids
            for i, nid_i in enumerate(node_ids):
                idx_i = self.node_id_to_index[nid_i]
                for j, nid_j in enumerate(node_ids):
                    if i != j:
                        idx_j = self.node_id_to_index[nid_j]
                        row_indices.append(idx_i)
                        col_indices.append(idx_j)

        data = np.ones(len(row_indices), dtype=np.int8)
        adjacency = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.int8,
        )

        if verbose:
            print(f"  Adjacency matrix: {adjacency.nnz:,} non-zeros")
            print("  Applying Reverse Cuthill-McKee algorithm...")

        rcm_order = reverse_cuthill_mckee(adjacency, symmetric_mode=True)

        old_node_id_to_new: Dict[int, int] = {}
        for new_idx, old_idx in enumerate(rcm_order):
            old_node = self.nodes[old_idx]
            old_node_id_to_new[old_node.id] = new_idx

        element_sort_key = []
        for element in self.elements:
            min_new_idx = min(old_node_id_to_new[nid] for nid in element.node_ids)
            element_sort_key.append((min_new_idx, element))

        old_nodes = self.nodes.copy()
        self.nodes = [old_nodes[old_idx] for old_idx in rcm_order]

        for new_id, node in enumerate(self.nodes):
            node.id = new_id

        element_sort_key.sort(key=lambda x: x[0])
        self.elements = [elem for _, elem in element_sort_key]

        for new_id, element in enumerate(self.elements):
            element.id = new_id

        self._finalize_renumbering()

        bandwidth_after = compute_bandwidth()
        reduction = (
            100 * (bandwidth_before - bandwidth_after) / bandwidth_before
            if bandwidth_before > 0
            else 0
        )

        if verbose:
            print(f"  Nodes: {n_nodes:,} reordered")
            print(f"  Elements: {n_elements:,} reordered")
            print(
                f"  Bandwidth: {bandwidth_before:,} -> {bandwidth_after:,} ({reduction:.1f}% reduction)"
            )
            print(f"  Node sets updated: {len(self.node_sets)}")
            print(f"  Element sets updated: {len(self.element_sets)}")

    def _finalize_renumbering(self) -> None:
        """Common finalization steps after renumbering."""
        self.node_map = {node.id: node for node in self.nodes}
        self.element_map = {element.id: element for element in self.elements}

        for node_set in self.node_sets.values():
            node_set.nodes = {node.id: node for node in node_set.nodes.values()}

        self._node_id_to_index_cache = None
        self._element_id_to_index_cache = None

        Node._id_counter = len(self.nodes)
        MeshElement._id_counter = len(self.elements)

    # =========================================================================
    # Add/Get Methods
    # =========================================================================

    def add_node(self, node: Node):
        """Add a node to the mesh and update the cache."""
        if node.id in self.node_map:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes.append(node)
        self.node_map[node.id] = node
        self._node_id_to_index_cache = None

    def add_element(self, element: MeshElement):
        """Add a connectivity element to the mesh and update the cache."""
        if element.id in self.element_map:
            raise ValueError(f"Element with id {element.id} already exists.")
        self.elements.append(element)
        self.element_map[element.id] = element
        self._element_id_to_index_cache = None

    def add_node_set(self, node_set: NodeSet):
        """Add a node set to the mesh."""
        if node_set.name in self.node_sets:
            raise ValueError(f"NodeSet '{node_set.name}' already exists.")
        self.node_sets[node_set.name] = node_set

    def add_element_set(self, element_set: ElementSet):
        """Add an element set to the mesh."""
        if element_set.name in self.element_sets:
            raise ValueError(f"ElementSet '{element_set.name}' already exists.")
        self.element_sets[element_set.name] = element_set

    def get_node_by_id(self, node_id: int) -> Node:
        """Retrieve a node by its ID."""
        try:
            return self.node_map[node_id]
        except KeyError:
            raise ValueError(f"Node with id {node_id} not found.")

    def get_element_by_id(self, element_id: int) -> MeshElement:
        """Retrieve an element by its ID."""
        try:
            return self.element_map[element_id]
        except KeyError:
            raise ValueError(f"Element with id {element_id} not found.")

    def get_node_set(self, name: str) -> NodeSet:
        """Retrieve a node set by its name."""
        try:
            return self.node_sets[name]
        except KeyError:
            raise ValueError(f"NodeSet '{name}' not found.")

    def get_element_set(self, name: str) -> ElementSet:
        """Retrieve an element set by its name."""
        try:
            return self.element_sets[name]
        except KeyError:
            raise ValueError(f"ElementSet '{name}' not found.")

    def get_element_associated_set(self, element: Union[MeshElement, int]):
        """Get all element sets that contain the given element."""
        if not isinstance(element, MeshElement):
            element = self.get_element_by_id(element)
        set_names = []
        for e_set in self.element_sets.values():
            if e_set.has_element(element):
                set_names.append(e_set)
        return set_names

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def write_mesh(self, filename: str, **kwargs) -> None:
        """
        Write the mesh to a file.

        Dispatches to the appropriate writer based on file extension.

        Parameters
        ----------
        filename : str
            The path to the output file.
        **kwargs
            Additional arguments passed to the underlying writer.
        """
        write_mesh(self, filename, **kwargs)

    def save(self, filepath: str, format: str = "auto", compression: str = "gzip") -> None:
        """
        Save the mesh to disk for later loading.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        format : str, optional
            File format to use: "auto", "hdf5", or "pickle". Default is "auto".
        compression : str, optional
            Compression algorithm for HDF5 format. Default is "gzip".
        """
        path = Path(filepath)

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
            write_hdf5(self, path, compression)
        elif format == "pickle":
            write_pickle(self, path)
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'hdf5' or 'pickle'.")

    @classmethod
    def load(cls, filepath: str, format: str = "auto") -> "MeshModel":
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
        """
        return load_mesh(filepath, format)

    # =========================================================================
    # Transformations
    # =========================================================================

    def translate_mesh(
        self,
        vector: Tuple[float, float, float],
        distance: float,
    ) -> None:
        """Translates the mesh along a specified direction vector."""
        coords = self.coords_array
        vector_arr = np.array(vector)
        norm = np.linalg.norm(vector_arr)
        if np.isclose(norm, 0):
            raise ValueError("Translation vector cannot be zero.")
        unit_vector = vector_arr / norm
        translation = unit_vector * distance
        coords += translation
        self.coords_array = coords

    def rotate_mesh(
        self,
        axis: Tuple[float, float, float],
        angle: float,
    ) -> None:
        """Rotates the mesh about a specified axis using Rodrigues' rotation formula."""
        axis_arr = np.array(axis)
        norm = np.linalg.norm(axis_arr)
        if np.isclose(norm, 0):
            raise ValueError("Rotation axis cannot be zero.")
        axis_arr = axis_arr / norm
        ux, uy, uz = axis_arr
        cosθ = np.cos(angle)
        sinθ = np.sin(angle)

        cross_prod_mat = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        outer_prod_mat = np.outer(axis_arr, axis_arr)
        identity = np.eye(3)

        R = cosθ * identity + (1 - cosθ) * outer_prod_mat + sinθ * cross_prod_mat
        self.coords_array = self.coords_array @ R.T

    # =========================================================================
    # Displacement Field Application
    # =========================================================================

    def apply_displacement_field(
        self,
        displacements: np.ndarray,
        scale: float = 1.0,
        dofs_per_node: int = None,
        displacement_indices: Tuple[int, ...] = None,
        inplace: bool = True,
    ) -> "MeshModel":
        """
        Apply a displacement field to the mesh coordinates.

        Parameters
        ----------
        displacements : np.ndarray
            Displacement field from solver.
        scale : float, optional
            Scale factor for displacements. Default is 1.0.
        dofs_per_node : int, optional
            Number of DOFs per node in the displacement array.
        displacement_indices : tuple of int, optional
            Indices of the translational DOFs (u, v, w).
        inplace : bool, optional
            If True (default), modify this mesh in place.

        Returns
        -------
        MeshModel
            The deformed mesh (self if inplace=True, new mesh otherwise).
        """
        n_nodes = len(self.nodes)

        if hasattr(displacements, "array"):
            displacements = displacements.array

        displacements = np.asarray(displacements, dtype=np.float64)

        if displacements.ndim == 1:
            if dofs_per_node is None:
                if displacements.size % n_nodes != 0:
                    raise ValueError(
                        f"Displacement array size ({displacements.size}) is not divisible "
                        f"by number of nodes ({n_nodes}). Please specify dofs_per_node."
                    )
                dofs_per_node = displacements.size // n_nodes
            displacements = displacements.reshape(n_nodes, dofs_per_node)
        elif displacements.ndim == 2:
            if dofs_per_node is None:
                dofs_per_node = displacements.shape[1]
            if displacements.shape[0] != n_nodes:
                raise ValueError(
                    f"Displacement array has {displacements.shape[0]} rows, "
                    f"but mesh has {n_nodes} nodes."
                )
        else:
            raise ValueError(f"Displacement array must be 1D or 2D, got {displacements.ndim}D.")

        if displacement_indices is None:
            n_trans = min(dofs_per_node, 3)
            displacement_indices = tuple(range(n_trans))

        if dofs_per_node == 3 and displacement_indices == (0, 1, 2):
            uvw = displacements
        else:
            valid_indices = [i for i in displacement_indices if i < dofs_per_node]
            uvw = displacements[:, valid_indices]

        if uvw.shape[1] < 3:
            padding = np.zeros((n_nodes, 3 - uvw.shape[1]))
            uvw = np.hstack([uvw, padding])

        uvw = uvw * scale
        current_coords = self.coords_array
        deformed_coords = current_coords + uvw

        if inplace:
            self.coords_array = deformed_coords
            return self
        else:
            import copy

            new_mesh = copy.deepcopy(self)
            new_mesh.coords_array = deformed_coords
            return new_mesh

    def get_deformed_copy(
        self,
        displacements: np.ndarray,
        scale: float = 1.0,
        dofs_per_node: int = None,
        displacement_indices: Tuple[int, ...] = None,
    ) -> "MeshModel":
        """
        Create a copy of the mesh with applied displacements.

        Convenience method equivalent to `apply_displacement_field(..., inplace=False)`.
        """
        return self.apply_displacement_field(
            displacements,
            scale=scale,
            dofs_per_node=dofs_per_node,
            displacement_indices=displacement_indices,
            inplace=False,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def node_sets_names(self) -> List[str]:
        """List of node set names."""
        return list(self.node_sets.keys())

    @property
    def element_sets_names(self) -> List[str]:
        """List of element set names."""
        return list(self.element_sets.keys())

    @property
    def node_count(self) -> int:
        """Number of nodes in the mesh."""
        return len(self.nodes)

    @property
    def elements_count(self) -> int:
        """Number of elements in the mesh."""
        return len(self.elements)

    @property
    def coords_array(self) -> np.ndarray:
        """Numpy array containing all nodal coordinates in the mesh."""
        return np.array([node.coords for node in self.nodes])

    @coords_array.setter
    def coords_array(self, value: np.ndarray) -> None:
        """Sets nodal coordinates from a numpy array."""
        if value.shape != (len(self.nodes), 3):
            raise ValueError("Array must have shape (N, 3)")
        for i, node in enumerate(self.nodes):
            node.coords = value[i]
            node.x = value[i, 0]
            node.y = value[i, 1]
            node.z = value[i, 2]

    # =========================================================================
    # Visualization
    # =========================================================================

    def view(self) -> None:
        """Visualize the mesh using the viewer."""
        plot_mesh(self)

    def __repr__(self) -> str:
        return (
            f"<MeshModel: {self.node_count} nodes, {self.elements_count} elements, "
            f"{len(self.node_sets_names)} node sets, {len(self.element_sets_names)} element sets>"
        )
