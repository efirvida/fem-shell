"""
Mesh entities module.

This module contains the fundamental building blocks for mesh representation:
- Node: A point in 3D space
- MeshElement: A connectivity element defined by nodes
- NodeSet: A collection of nodes
- ElementSet: A collection of elements
"""

from enum import IntEnum
from typing import Iterable, List, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pyvista.core.celltype import CellType


class ElementType(IntEnum):
    """Enumeration of supported element types.

    Includes 2D surface elements and 3D volumetric elements.
    Values correspond to PyVista/VTK cell type constants.
    """

    # 2D Surface elements
    triangle = CellType.TRIANGLE
    triangle6 = CellType.QUADRATIC_TRIANGLE
    quad = CellType.QUAD
    quad8 = CellType.QUADRATIC_QUAD
    quad9 = CellType.LAGRANGE_QUADRILATERAL

    # 3D Volumetric elements
    tetra = CellType.TETRA
    tetra10 = CellType.QUADRATIC_TETRA
    hexahedron = CellType.HEXAHEDRON
    hexahedron20 = CellType.QUADRATIC_HEXAHEDRON
    hexahedron27 = CellType.TRIQUADRATIC_HEXAHEDRON
    wedge = CellType.WEDGE
    wedge15 = CellType.QUADRATIC_WEDGE
    pyramid = CellType.PYRAMID
    pyramid13 = CellType.QUADRATIC_PYRAMID


# Mapping from node count to element type (for automatic element type detection)
# Note: This is ambiguous for some counts (e.g., 6 could be triangle6 or wedge)
# Use element_type parameter when constructing MeshElement for volumetric elements
ELEMENT_NODES_MAP = {
    # 2D elements (default for ambiguous counts)
    3: ElementType.triangle,
    4: ElementType.quad,  # Could be tetra, use explicit type
    8: ElementType.quad8,  # Could be hexahedron, use explicit type
    9: ElementType.quad9,
    # Unambiguous 2D
    6: ElementType.triangle6,  # Could be wedge, use explicit type
}

# Mapping for 3D volumetric elements by node count
SOLID_ELEMENT_NODES_MAP = {
    4: ElementType.tetra,
    5: ElementType.pyramid,
    6: ElementType.wedge,
    8: ElementType.hexahedron,
    10: ElementType.tetra10,
    13: ElementType.pyramid13,
    15: ElementType.wedge15,
    20: ElementType.hexahedron20,
    27: ElementType.hexahedron27,
}


class Node:
    """
    Represents a node with 3D coordinates.

    This class ensures that coordinates always include a z-value.
    If fewer than 3 coordinates are provided, zeros are appended.

    Attributes
    ----------
    coords : np.ndarray
        Array of coordinates in the form [x, y, z].
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float
        Z coordinate.
    id : int
        Unique identifier for the node.
    geometric_node : bool
        Whether this is a geometric (corner) node or a mid-side node.
    """

    _id_counter = 0

    def __init__(self, coords: Union[Iterable[float], np.ndarray], geometric_node: bool = True):
        """
        Initialize a Node instance.

        Parameters
        ----------
        coords : list of float or np.ndarray
            Coordinates of the node. If fewer than 3 values are provided,
            the z-coordinate is set to 0.0.
        geometric_node : bool, optional
            Whether this is a geometric (corner) node. Default is True.
        """
        coords_arr = np.array(coords, dtype=float)
        if coords_arr.size < 3:
            coords_arr = np.concatenate((coords_arr, np.zeros(3 - coords_arr.size)))
        self.coords = coords_arr
        self.x = coords_arr[0]
        self.y = coords_arr[1]
        self.z = coords_arr[2]
        self.id = Node._id_counter
        self.geometric_node = geometric_node
        Node._id_counter += 1

    def __repr__(self):
        return f"<Node id={self.id} coords={self.coords.tolist()}>"


class MeshElement:
    """
    Represents a mesh element defined solely by node connectivity.

    This class stores the connectivity as a list of node IDs.

    Attributes
    ----------
    nodes : list of Nodes
        List of nodes that form the element.
    node_ids : list of int
        List of node IDs that form the element.
    id : int
        Unique identifier for the element.
    element_type : ElementType
        Type of the element (triangle, quad, etc.).
    """

    _id_counter = 0

    def __init__(self, nodes: Sequence[Node], element_type: ElementType):
        """
        Initialize a MeshElement instance.

        Parameters
        ----------
        nodes : Sequence[Node]
            List of nodes defining the element connectivity.
        element_type : ElementType
            Type of the element.
        """
        self.id = MeshElement._id_counter
        self.nodes = nodes
        self.element_type = element_type
        MeshElement._id_counter += 1

    @property
    def node_ids(self) -> Tuple:
        """Get tuple of node IDs for this element."""
        return tuple([node.id for node in self.nodes])

    @property
    def node_count(self) -> int:
        """Get the number of nodes in this element."""
        return len(self.nodes)

    @property
    def node_coords(self) -> np.ndarray:
        """Get array of node coordinates for this element."""
        return np.array([node.coords for node in self.nodes])

    def visualize(self, show_node_ids: bool = True, show_mid_nodes: bool = True):
        """
        Visualize the mesh element using matplotlib with improved quadrilateral support.

        Parameters
        ----------
        show_node_ids : bool, optional
            Whether to show node IDs in the visualization. Default is True.
        show_mid_nodes : bool, optional
            Whether to show mid-side nodes. Default is True.
        """
        plt.figure()
        plt.title(f"Element {self.id} (Type: {self.element_type.name.upper()})")
        plt.gca().set_aspect("equal", adjustable="box")

        # Common configuration
        node_size = 50 if self.element_type == ElementType.quad else 30
        edge_color = "navy"
        mid_node_color = "darkorange"
        center_node_color = "green"

        # Node coordinates
        coords = self.node_coords
        x = coords[:, 0]
        y = coords[:, 1]

        # Draw element edges
        if self.element_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
            corners = [0, 1, 2, 3, 0]
            plt.plot(
                x[corners],
                y[corners],
                color=edge_color,
                linestyle="-",
                linewidth=1.5,
                label="Element edges",
            )

        if self.element_type == ElementType.quad9:
            plt.plot(
                [x[4], x[5], x[6], x[7], x[4]],
                [y[4], y[5], y[6], y[7], y[4]],
                color=edge_color,
                linestyle="--",
                linewidth=1,
            )

        if self.element_type == ElementType.triangle:
            corners = [0, 1, 2, 0]
            plt.plot(x[corners], y[corners], color=edge_color, linewidth=1.5)

        # Draw corner nodes
        plt.scatter(
            x[:4], y[:4], color=edge_color, s=node_size + 20, zorder=3, label="Corner nodes"
        )

        # Draw mid and center nodes (for QUAD9)
        if self.element_type == ElementType.quad9 and show_mid_nodes:
            plt.scatter(
                x[4:8],
                y[4:8],
                color=mid_node_color,
                s=node_size,
                marker="s",
                zorder=3,
                label="Mid nodes",
            )
            plt.scatter(
                x[8],
                y[8],
                color=center_node_color,
                s=node_size,
                marker="*",
                zorder=3,
                label="Center node",
            )

        # Node labels
        if show_node_ids:
            for i, (xi, yi) in enumerate(zip(x, y)):
                va = "bottom" if i in [0, 4, 7] else "top" if i in [2, 3, 6] else "center"
                ha = "right" if i in [0, 3, 7] else "left" if i in [1, 2, 5] else "center"
                plt.text(
                    xi,
                    yi,
                    f"({i}) - {self.nodes[i].id}",
                    color="white",
                    fontsize=16,
                    ha=ha,
                    va=va,
                    bbox={
                        "facecolor": edge_color if i < 4 else mid_node_color,
                        "alpha": 0.7,
                        "boxstyle": "round,pad=0.3",
                    },
                )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(linestyle="--", alpha=0.5)
        if len(self.nodes) >= 9 or show_mid_nodes:
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<MeshElement id={self.id} type={self.element_type.name} node_ids={self.node_ids}>"


class NodeSet:
    """
    Represents a set of nodes within the mesh.

    This can be used to group nodes for applying boundary conditions,
    loads, or other constraints.

    Attributes
    ----------
    name : str
        Name of the node set.
    nodes : dict
        Dictionary mapping node IDs to Node instances.
    """

    def __init__(self, name: str, nodes: Set[Node] | None = None):
        """
        Initialize a NodeSet instance.

        Parameters
        ----------
        name : str
            Name of the node set.
        nodes : set of Node, optional
            Initial set of nodes (default is an empty set).
        """
        self.name = name
        self.nodes = {node.id: node for node in nodes} if nodes is not None else {}

    def add_node(self, node: Node):
        """
        Add a node to the set if it is not already included.

        Parameters
        ----------
        node : Node
            The node to add.
        """
        self.nodes[node.id] = node

    def remove_node(self, node: Node):
        """
        Remove a node from the set if it exists.

        Parameters
        ----------
        node : Node
            The node to remove.
        """
        self.nodes.pop(node.id)

    @property
    def node_ids(self) -> Set[int]:
        """
        Get the set of node IDs in this set.

        Returns
        -------
        Set[int]
            The IDs of the nodes in the set.
        """
        return set(self.nodes.keys())

    @property
    def node_count(self) -> int:
        """
        Get the number of nodes in the set.

        Returns
        -------
        int
            The number of nodes in the set.
        """
        return len(self.nodes)

    def __repr__(self):
        return f"<NodeSet '{self.name}': {self.node_count} nodes>"


class ElementSet:
    """
    Represents a set of mesh elements.

    This is used to group elements that share similar properties or boundary
    conditions.

    Attributes
    ----------
    name : str
        Name of the element set.
    elements : set of MeshElement
        Set of mesh elements in the set.
    """

    def __init__(self, name: str, elements: Set[MeshElement] | None = None):
        """
        Initialize an ElementSet instance.

        Parameters
        ----------
        name : str
            Name of the element set.
        elements : set of MeshElement, optional
            Initial set of elements (default is an empty set).
        """
        self.name = name
        self.elements = elements if elements is not None else set()

    def add_element(self, element: MeshElement):
        """
        Add an element to the set if it is not already included.

        Parameters
        ----------
        element : MeshElement
            The element to add.
        """
        self.elements.add(element)

    def remove_element(self, element: MeshElement):
        """
        Remove an element from the set if it exists.

        Parameters
        ----------
        element : MeshElement
            The element to remove.
        """
        self.elements.remove(element)

    @property
    def element_ids(self) -> List[int]:
        """
        Get the list of element IDs in the set.

        Returns
        -------
        List[int]
            The IDs of the elements in the set.
        """
        return [element.id for element in self.elements]

    def has_element(self, element: MeshElement) -> bool:
        """Check if an element is in this set."""
        return element in self.elements

    def __repr__(self):
        return f"<ElementSet '{self.name}': {len(self.elements)} elements>"
