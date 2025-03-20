from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

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

    def __init__(self, coords: Union[Iterable[float], np.ndarray]):
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

    def __init__(self, nodes: Iterable[Node], element_type: ElementType):
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

    def visualize(self):
        """
        Visualize the mesh element using matplotlib.
        """
        if self.element_type == ElementType.triangle:
            coords = self.node_coords
            plt.figure()
            plt.title(f"Mesh Element {self.id} (Type: {self.element_type.name.upper()})")
            plt.gca().set_aspect("equal", adjustable="box")

            plt.scatter(coords[:, 0], coords[:, 1], color="red")
            for node in self.nodes:
                plt.text(node.x, node.y, f"{node.id}", fontsize=12, ha="right", va="bottom")

            edges = [(self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1)] + [
                [
                    self.nodes[-1],
                    self.nodes[0],
                ]
            ]

            for edge in edges:
                start = edge[0]
                end = edge[1]
                plt.plot([start.x, end.x], [start.y, end.y], color="blue")

            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
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
    def coords_array(self):
        return np.array([node.coords for node in self.nodes])

    def view(self) -> None:
        plot_mesh(self)

    def __repr__(self):
        return (
            f"<MeshModel: {len(self.node_count)} nodes, {len(self.elements_count)} elements, "
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

        # Add nodes with correct 0-based IDs
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        for _, coord in zip(node_tags, np.array(coords).reshape(-1, 3)):
            mesh_model.add_node(Node(coord))

        # Add elements with corrected connectivity
        self._add_elements(mesh_model)

        # Create node sets with proper boundary nodes
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

                for elem_id, nodes in enumerate(connectivity):
                    # Convert to 0-based indices and get node objects
                    node_objs = [mesh_model.get_node_by_id(int(nt - 1)) for nt in nodes]
                    e_type = ElementType.quad
                    if self.triangular and not self.quadratic:
                        e_type = ElementType.triangle
                    elif self.triangular and self.quadratic:
                        e_type = ElementType.triangle6
                    elif not self.triangular and self.quadratic:
                        e_type = ElementType.quad9

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
        return cls(width, height, nx, ny, quadratic, triangular).generate()

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


if __name__ == "__main__":
    mesh = SquareShapeMesh.create_rectangle(
        width=1.0, height=1.0, nx=10, ny=10, triangular=False, quadratic=False
    )
    mesh.view()
    mesh.write_mesh("mesh_file.vtk")

    print(f"Malla contiene {len(mesh.nodes)} nodos y {len(mesh.elements)} elementos")
    print("Primer nodo:", mesh.nodes[0])
    print("Primer elemento:", mesh.elements[0].node_ids)
    print("Node Sets:", mesh.node_sets_names)
    print("Element Sets:", mesh.element_sets_names)
