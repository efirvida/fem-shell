from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from pyvista.core.celltype import CellType

from fem_shell.core.viewer import plot_mesh


class ElementType(IntEnum):
    triangle = CellType.TRIANGLE
    triangle6 = CellType.QUADRATIC_TRIANGLE
    quad = CellType.QUAD
    quad8 = CellType.QUADRATIC_QUAD
    quad9 = CellType.LAGRANGE_QUADRILATERAL


ELEMENT_NODES_MAP = {
    3: ElementType.triangle,
    6: ElementType.triangle6,
    4: ElementType.quad,
    8: ElementType.quad8,
    9: ElementType.quad9,
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
    """

    _id_counter = 0

    def __init__(self, nodes: Sequence[Node], element_type: ElementType):
        """
        Initialize a MeshElement instance.

        Parameters
        ----------
        node_ids : list of int
            List of node IDs defining the connectivity.
        """
        self.id = MeshElement._id_counter
        self.nodes = nodes
        self.element_type = element_type
        MeshElement._id_counter += 1

    @property
    def node_ids(self) -> Tuple:
        return tuple([node.id for node in self.nodes])

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def node_coords(self):
        return np.array([node.coords for node in self.nodes])

    def visualize(self, show_node_ids=True, show_mid_nodes=True):
        """
        Visualize the mesh element using matplotlib with improved quadrilateral support.
        """

        plt.figure()
        plt.title(f"Element {self.id} (Type: {self.element_type.name.upper()})")
        plt.gca().set_aspect("equal", adjustable="box")

        # Configuración común
        node_size = 50 if self.element_type == ElementType.quad else 30
        edge_color = "navy"
        mid_node_color = "darkorange"
        center_node_color = "green"

        # Coordenadas de los nodos
        coords = self.node_coords
        x = coords[:, 0]
        y = coords[:, 1]

        # Dibujar bordes del elemento
        if self.element_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
            # Conexiones para QUAD4/QUAD9
            corners = [0, 1, 2, 3, 0]
            plt.plot(
                x[corners],
                y[corners],
                color=edge_color,
                linestyle="-",
                linewidth=1.5,
                label="Element edges",
            )

            # Si es QUAD9, dibujar líneas internas
        if self.element_type == ElementType.quad9:
            # Líneas horizontales/verticales centrales
            plt.plot(
                [x[4], x[5], x[6], x[7], x[4]],
                [y[4], y[5], y[6], y[7], y[4]],
                color=edge_color,
                linestyle="--",
                linewidth=1,
            )

        if self.element_type == ElementType.triangle:
            # Conexiones originales para triángulos
            corners = [0, 1, 2, 0]
            plt.plot(x[corners], y[corners], color=edge_color, linewidth=1.5)

        # Dibujar nodos con diferentes estilos
        # Nodos esquina
        plt.scatter(
            x[:4], y[:4], color=edge_color, s=node_size + 20, zorder=3, label="Corner nodes"
        )

        # Nodos medios (para QUAD9)
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

        # Etiquetas de nodos
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

        # Configuración del gráfico
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(linestyle="--", alpha=0.5)
        if len(self.nodes) >= 9 or show_mid_nodes:
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<MeshElement id={self.id}  type={self.element_type.name} node_ids={self.nodes}>"


class NodeSet:
    """
    Represents a set of nodes within the mesh.

    This can be used to group nodes for applying boundary conditions,
    loads, or other constraints.

    Attributes
    ----------
    name : str
        Name of the node set.
    nodes : list of Node
        List of nodes included in the set.
    """

    def __init__(self, name: str, nodes: Set[Node] | None = None):
        """
        Initialize a NodeSet instance.

        Parameters
        ----------
        name : str
            Name of the node set.
        nodes : list of Node, optional
            Initial list of nodes (default is an empty list).
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
        Get the list of node IDs in the set.

        Returns
        -------
        list of int
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
    elements : list of MeshElement
        List of mesh elements in the set.
    """

    def __init__(self, name: str, elements: Set[MeshElement] | None = None):
        """
        Initialize an ElementSet instance.

        Parameters
        ----------
        name : str
            Name of the element set.
        elements : list of MeshElement, optional
            Initial list of elements (default is an empty list).
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
        list of int
            The IDs of the elements in the set.
        """
        return [element.id for element in self.elements]

    def has_element(self, element: MeshElement) -> bool:
        return element in self.elements

    def __repr__(self):
        return f"<ElementSet '{self.name}': {len(self.elements)} elements>"


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

    def add_node(self, node: Node):
        """
        Add a node to the mesh and update the cache.

        Parameters
        ----------
        node : Node
            The node to add.

        Raises
        ------
        ValueError
            If a node with the same ID already exists.
        """
        if node.id in self.node_map:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes.append(node)
        self.node_map[node.id] = node

    def add_element(self, element: MeshElement):
        """
        Add a connectivity element to the mesh and update the cache.

        Parameters
        ----------
        element : MeshElement
            The element to add.

        Raises
        ------
        ValueError
            If an element with the same ID already exists.
        """
        if element.id in self.element_map:
            raise ValueError(f"Element with id {element.id} already exists.")
        self.elements.append(element)
        self.element_map[element.id] = element

    def add_node_set(self, node_set: NodeSet):
        """
        Add a node set to the mesh.

        Parameters
        ----------
        node_set : NodeSet
            The node set to add.

        Raises
        ------
        ValueError
            If a node set with the same name already exists.
        """
        if node_set.name in self.node_sets:
            raise ValueError(f"NodeSet '{node_set.name}' already exists.")
        self.node_sets[node_set.name] = node_set

    def add_element_set(self, element_set: ElementSet):
        """
        Add an element set to the mesh.

        Parameters
        ----------
        element_set : ElementSet
            The element set to add.

        Raises
        ------
        ValueError
            If an element set with the same name already exists.
        """
        if element_set.name in self.element_sets:
            raise ValueError(f"ElementSet '{element_set.name}' already exists.")
        self.element_sets[element_set.name] = element_set

    def get_node_by_id(self, node_id: int) -> Node:
        """
        Retrieve a node by its ID.

        Parameters
        ----------
        node_id : int
            The ID of the node.

        Returns
        -------
        Node
            The node with the specified ID.

        Raises
        ------
        ValueError
            If no node with the given ID is found.
        """
        try:
            return self.node_map[node_id]
        except KeyError:
            raise ValueError(f"Node with id {node_id} not found.")

    def get_element_by_id(self, element_id: int) -> MeshElement:
        """
        Retrieve an element by its ID.

        Parameters
        ----------
        element_id : int
            The ID of the element.

        Returns
        -------
        MeshElement
            The element with the specified ID.

        Raises
        ------
        ValueError
            If no element with the given ID is found.
        """
        try:
            return self.element_map[element_id]
        except KeyError:
            raise ValueError(f"Element with id {element_id} not found.")

    def get_node_set(self, name: str) -> NodeSet:
        """
        Retrieve a node set by its name.

        Parameters
        ----------
        name : str
            The name of the node set.

        Returns
        -------
        NodeSet
            The node set with the specified name.

        Raises
        ------
        ValueError
            If no node set with the given name is found.
        """
        try:
            return self.node_sets[name]
        except KeyError:
            raise ValueError(f"NodeSet '{name}' not found.")

    def get_element_set(self, name: str) -> ElementSet:
        """
        Retrieve an element set by its name.

        Parameters
        ----------
        name : str
            The name of the element set.

        Returns
        -------
        ElementSet
            The element set with the specified name.

        Raises
        ------
        ValueError
            If no element set with the given name is found.
        """
        try:
            return self.element_sets[name]
        except KeyError:
            raise ValueError(f"ElementSet '{name}' not found.")

    def get_element_associated_set(self, element: Union[MeshElement, int]):
        if not isinstance(element, MeshElement):
            element = self.get_element_by_id(element)
        set_names = []
        for e_set in self.element_sets.values():
            if e_set.has_element(element):
                set_names.append(e_set)
        return set_names

    def write_mesh(self, filename: str, **kwaargs) -> None:
        """
        Write the mesh to a file using meshio.

        Parameters
        ----------
        filename : str
            The path to the output file (e.g., "mesh.vtk").
        file_format : str, optional
            The file format to use (default is "vtk").
            Supported formats: vtk, vtu, msh, etc.

        Raises
        ------
        ValueError
            If the mesh has no nodes or elements.
        ValueError
            If the mesh has unsupported ElementType.
        """
        if not self.nodes:
            raise ValueError("Mesh has no nodes.")
        if not self.elements:
            raise ValueError("Mesh has no elements.")

        if filename.endswith("inp"):
            self.write_ccx_mesh(filename)
        else:
            points = self.coords_array
            cells = []

            cells = [(el.element_type.name, np.array([el.node_ids])) for el in self.elements]

            # Create mesh object and write file
            mesh = meshio.Mesh(points=points, cells=cells)
            meshio.write(filename, mesh, **kwaargs)
            print(f"Mesh written to {filename}.")

    def write_ccx_mesh(self, filename: str):
        ELEMENTS_TO_CALCULIX = {"quad": "S4", "triangle": "S3"}

        def split(arr):
            chunk_size = 8
            arr = list(arr)
            return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

        with open(filename, "wt") as f:
            f.write("**************++***********\n")
            f.write("**      MESH NODES       **\n")
            f.write("***************************\n")
            f.write("*NODE, NSET=N_ALL\n")
            for i, nd in enumerate(self.nodes):
                ln = f"{i + 1}, {nd.x:3e}, {nd.y:3e}, {nd.z:3e} \n"
                f.write(ln)

            f.write("\n**************++**++++********\n")
            f.write("**      MESH ELEMENTS       **\n")
            f.write("***********************+++****\n")
            for element_type in ElementType.__members__:
                if element_type in ELEMENTS_TO_CALCULIX:
                    f.write(f"*ELEMENT, TYPE={ELEMENTS_TO_CALCULIX[element_type]}, ELSET=E_ALL\n")
                    for i, el in enumerate(self.elements):
                        if el.element_type.name == element_type:
                            ln = f"{i + 1}, {', '.join(str(n + 1) for n in el.node_ids)} \n"
                            f.write(ln)

            f.write("\n**************+++*****+**********++**************\n")
            f.write("**      ELEMENT SETS DEFINITION SECTION       **\n")
            f.write("*******************+++++************************\n")

            for name, elements in self.element_sets.items():
                ln = f"*ELSET, ELSET=E_{name}\n"
                f.write(ln)
                labels = elements.element_ids
                for el in split(labels):
                    ln = ", ".join(str(e + 1) for e in el) + "\n"
                    f.write(ln)
                f.write("\n")

            f.write("\n**************************+****++*************\n")
            f.write("**       NODE SETS DEFINITION SECTION       **\n")
            f.write("*************************+++******************\n")

            for name, nodes in self.node_sets.items():
                ln = f"*NSET, NSET=N_{name}\n"
                f.write(ln)
                labels = nodes.node_ids
                for nd in split(labels):
                    ln = ", ".join(str(n + 1) for n in nd) + "\n"
                    f.write(ln)
                f.write("\n")

            f.write("\n********************************\n")
            f.write("**        STEPS SECTION       **\n")
            f.write("********************************\n")

    def translate_mesh(
        self,
        vector: Tuple[float, float, float],
        distance: float,
    ) -> None:
        """Translates the mesh along a specified direction vector.

        Parameters
        ----------
        vector : Tuple[float, float, float]
            Direction vector for translation (does not need to be normalized)
        distance : float
            Translation distance along the vector direction

        Raises
        ------
        ValueError
            If input vector is a zero vector

        Notes
        -----
        The actual translation is calculated as:
        `unit_vector * distance` where `unit_vector` is the normalized input vector.
        Modifies node coordinates in-place.
        """
        coords = self.coords_array
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0):
            raise ValueError("Translation vector cannot be zero.")
        unit_vector = vector / norm
        translation = unit_vector * distance
        coords += translation

    def rotate_mesh(
        self,
        axis: Tuple[float, float, float],
        angle: float,
    ) -> None:
        """Rotates the mesh about a specified axis using Rodrigues' rotation formula.

        Parameters
        ----------
        axis : Tuple[float, float, float]
            Rotation axis vector (does not need to be normalized)
        angle : float
            Rotation angle in radians (follows right-hand rule)

        Raises
        ------
        ValueError
            If input axis is a zero vector

        Notes
        -----
        Rotation matrix construction follows Rodrigues' formula:
        R = I*cosθ + (1 - cosθ)*(a⊗a) + sinθ*K
        Where:
        - I is identity matrix
        - a is unit axis vector
        - K is cross-product matrix of a
        - ⊗ denotes outer product
        Modifies node coordinates in-place.
        """
        axis = np.array(axis)
        norm = np.linalg.norm(axis)
        if np.isclose(norm, 0):
            raise ValueError("Rotation axis cannot be zero.")
        axis = axis / norm
        ux, uy, uz = axis
        cosθ = np.cos(angle)
        sinθ = np.sin(angle)

        # Cross product matrix for unit axis
        cross_prod_mat = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])

        # Rotation matrix components
        outer_prod_mat = np.outer(axis, axis)
        identity = np.eye(3)

        # Construct rotation matrix (Rodrigues' formula)
        R = cosθ * identity + (1 - cosθ) * outer_prod_mat + sinθ * cross_prod_mat

        # Apply rotation to all coordinates
        self.coords_array = self.coords_array @ R.T

    @property
    def node_sets_names(self) -> List[str]:
        return list(self.node_sets.keys())

    @property
    def element_sets_names(self) -> List[str]:
        return list(self.element_sets.keys())

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def elements_count(self) -> int:
        return len(self.elements)

    @property
    def coords_array(self) -> np.ndarray:
        """Numpy array containing all nodal coordinates in the mesh.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) containing (x, y, z) coordinates for each node,
            where N is the number of nodes in the mesh.

        Notes
        -----
        This property is dynamically generated from the node coordinates each time
        it's accessed. Any modifications to the returned array will NOT affect
        the actual node coordinates unless using the corresponding setter.
        """
        return np.array([node.coords for node in self.nodes])

    @coords_array.setter
    def coords_array(self, value: np.ndarray) -> None:
        """Sets nodal coordinates from a numpy array.

        Parameters
        ----------
        value : np.ndarray
            Array of shape (N, 3) containing new (x, y, z) coordinates for each node,
            must match the number of nodes in the mesh.

        Raises
        ------
        ValueError
            If input array has incorrect shape or dimensions
        """
        if value.shape != (len(self.nodes), 3):
            raise ValueError("Array must have shape (N, 3)")
        for i, node in enumerate(self.nodes):
            node.coords = value[i]

    def view(self) -> None:
        plot_mesh(self)

    def __repr__(self) -> str:
        return (
            f"<MeshModel: {self.node_count} nodes, {self.elements_count} elements, "
            f"{len(self.node_sets_names)} node sets, {len(self.element_sets_names)} element sets>"
        )


class SquareShapeMesh:
    """
    Generates structured 2D meshes using Gmsh and returns a MeshModel instance.

    Attributes
    ----------
    width : float
        Domain width (x-direction)
    height : float
        Domain height (y-direction)
    nx : int
        Number of divisions in x-direction
    ny : int
        Number of divisions in y-direction
    quadratic : bool
        Use quadratic elements
    triangular : bool
        Use triangular elements
    """

    def __init__(
        self,
        width: float,
        height: float,
        nx: int,
        ny: int,
        quadratic: bool = False,
        triangular: bool = False,
    ):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.quadratic = quadratic
        self.triangular = triangular

    def generate(self) -> "MeshModel":
        """Generates and returns a MeshModel with the structured mesh"""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("rectangle")

            # Create geometry with explicit corner tags
            self._create_geometry()

            # Configure mesh parameters
            self._configure_mesh()

            # Generate mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize()

            # Create and return MeshModel
            return self._create_mesh_model()
        finally:
            gmsh.finalize()

    def _create_geometry(self):
        """Create geometric entities with proper boundary handling"""
        x0 = -self.width / 2
        x1 = self.width / 2
        y0 = 0.0
        y1 = self.height

        # Add points with explicit tags
        self.p1 = gmsh.model.geo.addPoint(x0, y0, 0)
        self.p2 = gmsh.model.geo.addPoint(x1, y0, 0)
        self.p3 = gmsh.model.geo.addPoint(x1, y1, 0)
        self.p4 = gmsh.model.geo.addPoint(x0, y1, 0)

        # Create boundary lines with physical groups
        self.bottom = gmsh.model.geo.addLine(self.p1, self.p2)
        self.right = gmsh.model.geo.addLine(self.p2, self.p3)
        self.top = gmsh.model.geo.addLine(self.p3, self.p4)
        self.left = gmsh.model.geo.addLine(self.p4, self.p1)

        # Create surface
        loop = gmsh.model.geo.addCurveLoop([self.bottom, self.right, self.top, self.left])
        self.surface = gmsh.model.geo.addPlaneSurface([loop])

        # Add physical groups including corners
        self._add_physical_groups()

    def _add_physical_groups(self):
        """Add physical groups with proper corner handling"""
        # Physical groups for boundaries (include corner nodes)
        gmsh.model.addPhysicalGroup(1, [self.top], name="top")
        gmsh.model.addPhysicalGroup(1, [self.bottom], name="bottom")
        gmsh.model.addPhysicalGroup(1, [self.left], name="left")
        gmsh.model.addPhysicalGroup(1, [self.right], name="right")

        # Physical groups for corners
        gmsh.model.addPhysicalGroup(0, [self.p1], name="corner_p1")
        gmsh.model.addPhysicalGroup(0, [self.p2], name="corner_p2")
        gmsh.model.addPhysicalGroup(0, [self.p3], name="corner_p3")
        gmsh.model.addPhysicalGroup(0, [self.p4], name="corner_p4")

    def _configure_mesh(self):
        """Configure meshing parameters"""
        # Transfinite parameters
        for curve in [self.bottom, self.top]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, self.nx + 1)
        for curve in [self.left, self.right]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, self.ny + 1)

        gmsh.model.geo.mesh.setTransfiniteSurface(
            self.surface, "Right", [self.p1, self.p2, self.p3, self.p4]
        )

        # Element type configuration
        if self.triangular:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
        else:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)

        if self.quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

    def _create_mesh_model(self) -> "MeshModel":
        """Converts Gmsh mesh to MeshModel instance with proper index handling"""
        mesh_model = MeshModel()

        # Obtener elementos 2D de la malla
        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        # Construir un conjunto con las etiquetas de los nodos geométricos (esquinas)
        # Asumimos que para elementos quad o quad9 las 4 primeras entradas de la conectividad son las esquinas.
        geometric_node_tags = set()
        for et, conn in zip(elementTypes, nodeTags):
            # Obtener las propiedades del elemento: (elementType, elementName, numNodes, numNodesBoundary, minTag, maxTag)
            props = gmsh.model.mesh.getElementProperties(et)
            total_nodes = props[3]  # cantidad total de nodos en el elemento
            # Para un elemento cuadrilateral (4 o 9 nodos), los 4 primeros nodos corresponden a las esquinas
            e_type = ELEMENT_NODES_MAP[total_nodes]
            if e_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                num_corners = 4
            elif e_type in (ElementType.triangle6, ElementType.triangle6):
                num_corners = 3
            else:
                raise ValueError
            # Recorrer la conectividad (que es un arreglo plano) dividiéndola en grupos de 'total_nodes'
            geometric_node_tags.update(
                nodeTags[0].reshape(-1, total_nodes)[:, :num_corners].flatten()
            )

        # Obtener todos los nodos de la malla.
        # gmsh.model.mesh.getNodes() devuelve (nodeTags, coords, parametricCoords)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)

        # Agregar nodos al modelo: se marca como "geometric_node" solo si su tag está en el conjunto anterior
        for tag, coord in zip(node_tags, coords):
            if tag in geometric_node_tags:
                mesh_model.add_node(Node(coord, geometric_node=True))
            else:
                mesh_model.add_node(Node(coord, geometric_node=False))

        # Agregar elementos con la conectividad corregida
        self._add_elements(mesh_model)

        # Crear conjuntos de nodos (node sets) basados en los nodos de contorno u otras condiciones
        self._create_node_sets(mesh_model)

        return mesh_model

    def _add_elements(self, mesh_model: "MeshModel"):
        """Add elements with corrected node indices"""
        elem_types = gmsh.model.mesh.getElementTypes()
        for elem_type in elem_types:
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 2:  # 2D elements
                _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
                num_nodes = elem_props[3]
                connectivity = elem_node_tags.reshape(-1, num_nodes)

                for nodes in connectivity:
                    # Convert to 0-based indices and get node objects
                    node_objs = [mesh_model.get_node_by_id(int(nt - 1)) for nt in nodes]
                    e_type = ELEMENT_NODES_MAP.get(len(nodes), ElementType.quad)
                    mesh_model.add_element(MeshElement(nodes=node_objs, element_type=e_type))

    def _create_node_sets(self, mesh_model: "MeshModel"):
        """Creates node sets including boundary corners"""
        physical_groups = gmsh.model.getPhysicalGroups()
        boundary_nodes = set()

        boundary_sets = {"top": set(), "bottom": set(), "left": set(), "right": set()}

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            node_ids = []

            if dim == 0:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                for e in entities:
                    nt, _, _ = gmsh.model.mesh.getNodes(dim=0, tag=e)
                    node_ids.extend([int(n - 1) for n in nt])

                if "corner" in name:
                    if "p1" in name:
                        boundary_sets["bottom"].update(node_ids)
                        boundary_sets["left"].update(node_ids)
                    elif "p2" in name:
                        boundary_sets["bottom"].update(node_ids)
                        boundary_sets["right"].update(node_ids)
                    elif "p3" in name:
                        boundary_sets["top"].update(node_ids)
                        boundary_sets["right"].update(node_ids)
                    elif "p4" in name:
                        boundary_sets["top"].update(node_ids)
                        boundary_sets["left"].update(node_ids)

            elif dim == 1:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                for e in entities:
                    nt, _, _ = gmsh.model.mesh.getNodes(dim=1, tag=e)
                    node_ids.extend([int(n - 1) for n in nt])

                if name in boundary_sets:
                    boundary_sets[name].update(node_ids)

            boundary_nodes.update(node_ids)

        for name, node_ids in boundary_sets.items():
            if node_ids:
                node_objs = {node for node in [mesh_model.get_node_by_id(nid) for nid in node_ids]}
                mesh_model.add_node_set(NodeSet(name=name, nodes=node_objs))

        all_nodes = {node for node in mesh_model.nodes}
        mesh_model.add_node_set(NodeSet(name="all", nodes=all_nodes))

        surface_nodes = {n for n in all_nodes if n.id not in boundary_nodes}
        mesh_model.add_node_set(NodeSet(name="surface", nodes=surface_nodes))

    @classmethod
    def create_rectangle(
        cls,
        width: float,
        height: float,
        nx: int,
        ny: int,
        quadratic: bool = False,
        triangular: bool = False,
    ) -> "MeshModel":
        """Helper method to create rectangular mesh"""
        return cls(width, height, int(nx), int(ny), quadratic, triangular).generate()

    @classmethod
    def create_unit_square(
        cls,
        nx: int,
        ny: int,
        quadratic: bool = False,
        triangular: bool = False,
    ) -> "MeshModel":
        """Helper method to create unit square mesh"""
        return cls(1.0, 1.0, nx, ny, quadratic, triangular).generate()


class BoxSurfaceMesh:
    """
    Generates structured 3D box surface meshes using Gmsh's classic geo API.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Center coordinates of the box (x, y, z)
    dims : Tuple[float, float, float]
        Total box dimensions (dx, dy, dz)
    nx : int
        Number of divisions in x-direction
    ny : int
        Number of divisions in y-direction
    nz : int
        Number of divisions in z-direction
    quadratic : bool, optional
        Use quadratic elements (default: False)
    triangular : bool, optional
        Generate triangular mesh (default: False)
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        dims: Tuple[float, float, float],
        nx: int,
        ny: int,
        nz: int,
        quadratic: bool = False,
        triangular: bool = False,
    ):
        self.center = center
        self.dims = dims
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.quadratic = quadratic
        self.triangular = triangular

    def generate(self) -> "MeshModel":
        """Generates and returns the MeshModel instance"""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("box_surface")

            self._create_geometry()
            gmsh.model.geo.synchronize()

            self._configure_mesh()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize()

            return self._create_mesh_model()
        finally:
            gmsh.finalize()

    def _create_geometry(self):
        cx, cy, cz = self.center
        dx, dy, dz = self.dims

        # Ajuste de coordenadas: Y es ahora el eje vertical
        x = (cx - dx / 2, cx + dx / 2)
        y = (cy - dy / 2, cy + dy / 2)  # y ahora es el eje vertical
        z = (cz - dz / 2, cz + dz / 2)

        # Puntos redefinidos con Y como eje vertical
        self.points = {
            "p1": gmsh.model.geo.addPoint(x[0], y[0], z[0]),
            "p2": gmsh.model.geo.addPoint(x[1], y[0], z[0]),
            "p3": gmsh.model.geo.addPoint(x[1], y[0], z[1]),
            "p4": gmsh.model.geo.addPoint(x[0], y[0], z[1]),
            "p5": gmsh.model.geo.addPoint(x[0], y[1], z[0]),
            "p6": gmsh.model.geo.addPoint(x[1], y[1], z[0]),
            "p7": gmsh.model.geo.addPoint(x[1], y[1], z[1]),
            "p8": gmsh.model.geo.addPoint(x[0], y[1], z[1]),
        }

        # Create edges
        self._create_edges()
        # Create faces
        self._create_faces()

        gmsh.model.geo.synchronize()

    def _create_edges(self):
        p = self.points
        self.edges = {
            # Cara inferior (y=y0)
            "l1": gmsh.model.geo.addLine(p["p1"], p["p2"]),  # X-direction
            "l2": gmsh.model.geo.addLine(p["p2"], p["p3"]),  # Z-direction
            "l3": gmsh.model.geo.addLine(p["p3"], p["p4"]),  # X-direction
            "l4": gmsh.model.geo.addLine(p["p4"], p["p1"]),  # Z-direction
            # Cara superior (y=y1)
            "l5": gmsh.model.geo.addLine(p["p5"], p["p6"]),  # X-direction
            "l6": gmsh.model.geo.addLine(p["p6"], p["p7"]),  # Z-direction
            "l7": gmsh.model.geo.addLine(p["p7"], p["p8"]),  # X-direction
            "l8": gmsh.model.geo.addLine(p["p8"], p["p5"]),  # Z-direction
            # Aristas verticales (Y-direction)
            "l9": gmsh.model.geo.addLine(p["p1"], p["p5"]),
            "l10": gmsh.model.geo.addLine(p["p2"], p["p6"]),
            "l11": gmsh.model.geo.addLine(p["p3"], p["p7"]),
            "l12": gmsh.model.geo.addLine(p["p4"], p["p8"]),
        }

    def _create_faces(self):
        """Creates box faces with proper orientation and physical groups"""
        edges = self.edges
        self.faces = {
            "bottom": self._create_face_loop([edges["l1"], edges["l2"], edges["l3"], edges["l4"]]),
            "top": self._create_face_loop([edges["l5"], edges["l6"], edges["l7"], edges["l8"]]),
            "front": self._create_face_loop([
                edges["l3"],
                edges["l12"],
                -edges["l7"],
                -edges["l11"],
            ]),
            "back": self._create_face_loop([edges["l1"], edges["l10"], -edges["l5"], -edges["l9"]]),
            "left": self._create_face_loop([-edges["l4"], edges["l12"], edges["l8"], -edges["l9"]]),
            "right": self._create_face_loop([
                edges["l2"],
                edges["l11"],
                -edges["l6"],
                -edges["l10"],
            ]),
        }

        # Add physical groups
        for name, face_tag in self.faces.items():
            gmsh.model.addPhysicalGroup(2, [face_tag], name=name)

    def _create_face_loop(self, curves):
        """Helper to create face from oriented curves"""
        loop = gmsh.model.geo.addCurveLoop(curves)
        return gmsh.model.geo.addPlaneSurface([loop])

    def _configure_mesh(self):
        """Configures mesh parameters and transfinite settings"""
        # Transfinite settings
        self._set_transfinite_curves()
        self._set_transfinite_surfaces()

        # Mesh algorithm
        if self.triangular:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # Element order
        if self.quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

    def _set_transfinite_curves(self):
        """Configuración corregida de curvas transfinas"""
        # X-direction: l1, l3, l5, l7
        for e in ["l1", "l3", "l5", "l7"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.nx + 1)

        # Z-direction: l2, l4, l6, l8
        for e in ["l2", "l4", "l6", "l8"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.nz + 1)

        # Y-direction (vertical): l9, l10, l11, l12
        for e in ["l9", "l10", "l11", "l12"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.ny + 1)

    def _set_transfinite_surfaces(self):
        """Configures transfinite surfaces"""
        for face in self.faces.values():
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)

    def _create_mesh_model(self) -> "MeshModel":
        """Converts Gmsh mesh to MeshModel instance with proper index handling"""
        mesh_model = MeshModel()

        # Obtener elementos 2D de la malla
        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        # Construir un conjunto con las etiquetas de los nodos geométricos (esquinas)
        # Asumimos que para elementos quad o quad9 las 4 primeras entradas de la conectividad son las esquinas.
        geometric_node_tags = set()
        for et, conn in zip(elementTypes, nodeTags):
            # Obtener las propiedades del elemento: (elementType, elementName, numNodes, numNodesBoundary, minTag, maxTag)
            props = gmsh.model.mesh.getElementProperties(et)
            total_nodes = props[3]  # cantidad total de nodos en el elemento
            # Para un elemento cuadrilateral (4 o 9 nodos), los 4 primeros nodos corresponden a las esquinas
            e_type = ELEMENT_NODES_MAP[total_nodes]
            if e_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                num_corners = 4
            elif e_type in (ElementType.triangle6, ElementType.triangle6):
                num_corners = 3
            else:
                raise ValueError
            # Recorrer la conectividad (que es un arreglo plano) dividiéndola en grupos de 'total_nodes'
            geometric_node_tags.update(
                nodeTags[0].reshape(-1, total_nodes)[:, :num_corners].flatten()
            )

        # Obtener todos los nodos de la malla.
        # gmsh.model.mesh.getNodes() devuelve (nodeTags, coords, parametricCoords)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)

        # Agregar nodos al modelo: se marca como "geometric_node" solo si su tag está en el conjunto anterior
        for tag, coord in zip(node_tags, coords):
            if tag in geometric_node_tags:
                mesh_model.add_node(Node(coord, geometric_node=True))
            else:
                mesh_model.add_node(Node(coord, geometric_node=False))

        # Agregar elementos con la conectividad corregida
        self._add_elements(mesh_model)
        self._create_node_sets(mesh_model)
        # Crear conjuntos de nodos (node sets) basados en los nodos de contorno u otras condiciones
        # self._create_node_sets(mesh_model)

        return mesh_model

    def _add_elements(self, mesh_model: "MeshModel"):
        """Add elements with corrected node indices"""
        elem_types = gmsh.model.mesh.getElementTypes()
        for elem_type in elem_types:
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 2:  # 2D elements
                _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
                num_nodes = elem_props[3]
                connectivity = elem_node_tags.reshape(-1, num_nodes)

                for nodes in connectivity:
                    # Convert to 0-based indices and get node objects
                    node_objs = [mesh_model.get_node_by_id(int(nt - 1)) for nt in nodes]
                    e_type = ELEMENT_NODES_MAP.get(len(nodes), ElementType.quad)
                    mesh_model.add_element(MeshElement(nodes=node_objs, element_type=e_type))

    def _create_node_sets(self, mesh_model: "MeshModel"):
        """Creates node sets for all boundary faces"""
        physical_groups = gmsh.model.getPhysicalGroups()

        face_sets = {
            "top": set(),
            "bottom": set(),
            "front": set(),
            "back": set(),
            "left": set(),
            "right": set(),
        }

        for dim, tag in physical_groups:
            if dim != 2:  # Solo procesar superficies
                continue

            name = gmsh.model.getPhysicalName(dim, tag)
            if name not in face_sets:
                continue

            # Obtener nodos en 1-based indexing
            node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0]

            # Convertir a 0-based y agregar
            face_sets[name].update((tag - 1 for tag in node_tags))

        # Crear los NodeSets
        for name, node_ids in face_sets.items():
            if node_ids:
                node_objs = {mesh_model.get_node_by_id(nid) for nid in node_ids}
                mesh_model.add_node_set(NodeSet(name=name, nodes=node_objs))

        # Añadir conjunto de todos los nodos
        all_nodes = {node for node in mesh_model.nodes}
        mesh_model.add_node_set(NodeSet(name="all", nodes=all_nodes))

    @classmethod
    def create_box(
        cls,
        center: Tuple[float, float, float],
        dims: Tuple[float, float, float],
        nx: int,
        ny: int,
        nz: int,
        quadratic: bool = False,
        triangular: bool = False,
    ) -> "MeshModel":
        """Creates a box with specified dimensions"""
        return cls(center, dims, nx, ny, nz, quadratic, triangular).generate()

    @classmethod
    def create_unit_box(
        cls,
        divisions: int = 10,
        quadratic: bool = False,
        triangular: bool = False,
    ) -> "MeshModel":
        """Creates a unit cube (1x1x1) centered at origin"""
        return cls(
            (0, 0, 0), (1, 1, 1), divisions, divisions, divisions, quadratic, triangular
        ).generate()


if __name__ == "__main__":
    # Example usage
    unit_mesh = BoxSurfaceMesh.create_box((0, 0, 5), (1, 1, 10), 5, 5, 50)
    unit_mesh.rotate_mesh((1, 0, 0), 20)
    unit_mesh.view()
