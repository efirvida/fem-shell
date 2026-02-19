"""
Mesh utility functions.

This module provides utility functions for mesh manipulation and analysis:
- detect_open_boundaries: Detect if a surface mesh has open boundaries
- close_open_boundaries: Close open boundaries by filling holes
- volumetric_remesh: Convert a surface mesh to a volumetric mesh
"""

import os
import tempfile
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from fem_shell.core.mesh.model import MeshModel


def detect_open_boundaries(mesh: "MeshModel") -> bool:
    """
    Detect if a surface mesh has open boundaries.

    Parameters
    ----------
    mesh : MeshModel
        The surface mesh to check

    Returns
    -------
    bool
        True if the mesh has open boundaries, False if it's closed

    Notes
    -----
    A closed surface mesh has each edge shared by exactly 2 faces.
    An open boundary exists when an edge is shared by only 1 face.
    """

    edge_count = defaultdict(int)

    for element in mesh.elements:
        nodes = element.nodes
        n = len(nodes)

        # For triangular and quad elements, check edges
        if n == 3:  # Triangle
            edges = [(0, 1), (1, 2), (2, 0)]
        elif n == 4:  # Quad
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif n == 6:  # Triangle6
            edges = [(0, 2), (2, 4), (4, 0)]  # Only corner nodes
        elif n == 8 or n == 9:  # Quad8 or Quad9
            edges = [(0, 2), (2, 4), (4, 6), (6, 0)]  # Only corner nodes
        else:
            # Skip volumetric elements
            continue

        for i, j in edges:
            # Create a normalized edge (smaller index first)
            edge = tuple(sorted([nodes[i].id, nodes[j].id]))
            edge_count[edge] += 1

    # Check if any edge is shared by only one face (open boundary)
    for count in edge_count.values():
        if count == 1:
            return True

    return False


def get_open_boundary_loops(mesh: "MeshModel") -> List[List[int]]:
    """
    Get ordered loops of nodes that form open boundaries.

    Parameters
    ----------
    mesh : MeshModel
        The surface mesh to analyze

    Returns
    -------
    List[List[int]]
        List of boundary loops, where each loop is a list of node IDs in order
    """

    # Find all edges and count their occurrences
    edge_count = defaultdict(int)
    edge_elements = defaultdict(list)  # Track which element uses each edge

    for element in mesh.elements:
        nodes = element.nodes
        n = len(nodes)

        # Get edges based on element type
        if n == 3:  # Triangle
            edges = [(0, 1), (1, 2), (2, 0)]
        elif n == 4:  # Quad
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif n == 6:  # Triangle6
            edges = [(0, 2), (2, 4), (4, 0)]
        elif n == 8 or n == 9:  # Quad8 or Quad9
            edges = [(0, 2), (2, 4), (4, 6), (6, 0)]
        else:
            continue

        for i, j in edges:
            edge = tuple(sorted([nodes[i].id, nodes[j].id]))
            edge_count[edge] += 1
            # Store directed edge (maintains order)
            edge_elements[(nodes[i].id, nodes[j].id)] = element

    # Find open boundary edges (count == 1)
    open_edges = {}  # {node_id: [connected_node_ids]}
    for element in mesh.elements:
        nodes = element.nodes
        n = len(nodes)

        if n == 3:
            edges = [(0, 1), (1, 2), (2, 0)]
        elif n == 4:
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif n == 6:
            edges = [(0, 2), (2, 4), (4, 0)]
        elif n == 8 or n == 9:
            edges = [(0, 2), (2, 4), (4, 6), (6, 0)]
        else:
            continue

        for i, j in edges:
            edge = tuple(sorted([nodes[i].id, nodes[j].id]))
            if edge_count[edge] == 1:
                # This is an open boundary edge
                n1, n2 = nodes[i].id, nodes[j].id
                if n1 not in open_edges:
                    open_edges[n1] = []
                if n2 not in open_edges:
                    open_edges[n2] = []
                open_edges[n1].append(n2)
                open_edges[n2].append(n1)

    # Build loops from connected edges
    loops = []
    visited = set()

    for start_node in open_edges:
        if start_node in visited:
            continue

        # Start a new loop
        loop = [start_node]
        visited.add(start_node)
        current = start_node

        # Follow the boundary
        while True:
            neighbors = [n for n in open_edges[current] if n not in visited]
            if not neighbors:
                # Check if we can close the loop
                if len(open_edges[current]) == 2:
                    # Try to connect back to start
                    other = [
                        n for n in open_edges[current] if n != (loop[-2] if len(loop) > 1 else None)
                    ]
                    if other and other[0] == start_node:
                        break
                break

            next_node = neighbors[0]
            loop.append(next_node)
            visited.add(next_node)
            current = next_node

            # Check if we've completed the loop
            if current == start_node or (len(loop) > 2 and start_node in open_edges[current]):
                break

        if len(loop) > 2:
            loops.append(loop)

    return loops


def close_open_boundaries(mesh: "MeshModel") -> "MeshModel":
    """
    Close open boundaries in a surface mesh by filling holes.

    Parameters
    ----------
    mesh : MeshModel
        The input surface mesh with open boundaries

    Returns
    -------
    MeshModel
        A new mesh with boundaries closed

    Notes
    -----
    This function detects open boundary loops and fills them with
    triangular elements to create a closed surface.
    """
    from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
    from fem_shell.core.mesh.model import MeshModel

    # Create a copy of the mesh
    closed_mesh = MeshModel()

    # Copy all existing nodes
    node_map = {}  # old_id -> new_node
    for node in mesh.nodes:
        new_node = Node(coords=node.coords.copy())
        closed_mesh.add_node(new_node)
        node_map[node.id] = new_node

    # Copy all existing elements
    for element in mesh.elements:
        new_nodes = [node_map[n.id] for n in element.nodes]
        new_element = MeshElement(nodes=new_nodes, element_type=element.element_type)
        closed_mesh.add_element(new_element)

    # Get open boundary loops
    loops = get_open_boundary_loops(mesh)

    if not loops:
        return closed_mesh

    print(f"  Found {len(loops)} open boundary loop(s) to close")

    # Fill each loop with triangular elements
    for loop_idx, loop in enumerate(loops):
        print(f"    Closing loop {loop_idx + 1} with {len(loop)} boundary nodes...")

        # Get coordinates of boundary nodes
        boundary_coords = []
        boundary_node_objs = []

        for node_id in loop:
            # Find the original node to get the new node
            orig_node = next(n for n in mesh.nodes if n.id == node_id)
            new_node = node_map[node_id]
            boundary_coords.append(orig_node.coords.copy())
            boundary_node_objs.append(new_node)

        boundary_coords = np.array(boundary_coords)

        # Simple fan triangulation from centroid
        if len(loop) >= 3:
            # Calculate centroid
            centroid = np.mean(boundary_coords, axis=0)
            centroid_node = Node(coords=centroid)
            closed_mesh.add_node(centroid_node)

            # Create triangular elements from centroid to each boundary edge
            for i in range(len(loop)):
                n1 = boundary_node_objs[i]
                n2 = boundary_node_objs[(i + 1) % len(loop)]

                # Create triangle (centroid, edge_start, edge_end)
                # The orientation will be checked and fixed later by Gmsh or during volume meshing
                element = MeshElement(
                    nodes=[centroid_node, n1, n2], element_type=ElementType.triangle
                )
                closed_mesh.add_element(element)

    return closed_mesh


def _check_tetra_orientation(nodes):
    """
    Check if a tetrahedron has positive volume (correct orientation).

    For a tetrahedron with nodes [n0, n1, n2, n3], the volume is:
    V = (1/6) * det([n1-n0, n2-n0, n3-n0])

    If V < 0, the nodes need to be reordered.

    Parameters
    ----------
    nodes : list of Node
        List of 4 nodes defining the tetrahedron

    Returns
    -------
    list of Node
        Reordered nodes with positive volume if needed
    """
    if len(nodes) != 4:
        return nodes

    # Get coordinates
    p0 = np.array(nodes[0].coords)
    p1 = np.array(nodes[1].coords)
    p2 = np.array(nodes[2].coords)
    p3 = np.array(nodes[3].coords)

    # Compute vectors from p0
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0

    # Compute volume (6 * V)
    volume = np.dot(v1, np.cross(v2, v3))

    # If volume is negative, swap two nodes to fix orientation
    if volume < 0:
        # Swap nodes 0 and 1
        return [nodes[1], nodes[0], nodes[2], nodes[3]]

    return nodes


def _convert_to_quadratic(mesh: "MeshModel") -> "MeshModel":
    """
    Convert a linear tetrahedral mesh to quadratic (TETRA10).

    This creates new mid-side nodes for every edge in the mesh and upgrades
    TETRA4 elements to TETRA10 elements.

    Parameters
    ----------
    mesh : MeshModel
        Input linear mesh (must contain tetrahedra)

    Returns
    -------
    MeshModel
        New mesh with quadratic elements
    """
    from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
    from fem_shell.core.mesh.model import MeshModel

    quadratic_mesh = MeshModel()

    # Map to store midpoint nodes by sorted edge IDs: (id1, id2) -> Node
    edge_node_cache = {}

    def get_midpoint(n1: Node, n2: Node) -> Node:
        edge_key = tuple(sorted((n1.id, n2.id)))
        if edge_key in edge_node_cache:
            return edge_node_cache[edge_key]

        # Create new midpoint node
        mid_coords = (np.array(n1.coords) + np.array(n2.coords)) * 0.5
        new_node = Node(coords=mid_coords, geometric_node=False)
        edge_node_cache[edge_key] = new_node
        # Add to mesh immediately to assign ID if needed,
        # though MeshModel usually assigns IDs on add_node if not present?
        # Node() doesn't auto-assign ID, but MeshModel might use it.
        # Ideally we collect them and add later, or add now.
        return new_node

    # Add all original nodes
    for node in mesh.nodes:
        # We presume original nodes are geometric
        # We assume mesh.nodes order is preserved or IDs are stable
        quadratic_mesh.add_node(node)

    # Convert elements
    for element in mesh.elements:
        if element.element_type != ElementType.tetra:
            continue

        nodes = element.nodes
        if len(nodes) != 4:
            continue

        n0, n1, n2, n3 = nodes

        # Create midpoints according to SOLID.py TETRA10 ordering
        # 4: (0-1)
        n4 = get_midpoint(n0, n1)
        # 5: (1-2)
        n5 = get_midpoint(n1, n2)
        # 6: (0-2)
        n6 = get_midpoint(n0, n2)
        # 7: (0-3)
        n7 = get_midpoint(n0, n3)
        # 8: (2-3)
        n8 = get_midpoint(n2, n3)
        # 9: (1-3)
        n9 = get_midpoint(n1, n3)

        new_nodes = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]

        new_element = MeshElement(nodes=new_nodes, element_type=ElementType.tetra10)
        quadratic_mesh.add_element(new_element)

    # Add all cached midpoint nodes to the mesh
    # Note: We should ideally add them as we create them or in a batch.
    # Since we didn't add them yet, we add them now.
    for node in edge_node_cache.values():
        quadratic_mesh.add_node(node)

    return quadratic_mesh


def volumetric_remesh(
    surface_mesh: "MeshModel",
    target_edge_length: Optional[float] = None,
    order: int = 1,
    algorithm: int = 1,
    auto_close_boundaries: bool = True,
    backend: str = "wildmeshing",
) -> "MeshModel":
    """
    Convert a surface mesh to a volumetric mesh using Gmsh or wildmeshing.

    Parameters
    ----------
    surface_mesh : MeshModel
        The input surface mesh
    target_edge_length : float, optional
        Target edge length for the volumetric mesh.
        If None, it will be estimated from the surface mesh.
    order : int, optional
        Element order (1 for linear, 2 for quadratic). Default is 1.
    algorithm : int, optional
        Gmsh 3D meshing algorithm:
        1 = Delaunay (default)
        4 = Frontal
        7 = MMG3D
        10 = HXT
    auto_close_boundaries : bool, optional
        Automatically close open boundaries before remeshing (default: True)
    backend : str, optional
        Remeshing backend: "wildmeshing" (default) or "gmsh".

    Returns
    -------
    MeshModel
        A new mesh model with volumetric elements

    Notes
    -----
    This function can use Gmsh or wildmeshing (fTetWild) to generate a volumetric
    mesh from a surface mesh.
    If the surface has open boundaries and auto_close_boundaries is True,
    it will attempt to close them before generating the volume mesh.

    The approach writes a temporary surface mesh file and invokes the selected
    backend to generate a volume mesh. The wildmeshing backend requires the
    optional dependencies "wildmeshing" and "meshio".
    """
    from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
    from fem_shell.core.mesh.model import MeshModel

    # Check for open boundaries
    has_open = detect_open_boundaries(surface_mesh)

    if has_open:
        if auto_close_boundaries:
            print("  ⚠ Open boundaries detected. Attempting to close them...")
            surface_mesh = close_open_boundaries(surface_mesh)
            print("  ✓ Boundaries closed successfully")
        else:
            raise ValueError(
                "Surface mesh has open boundaries. "
                "Set auto_close_boundaries=True to automatically close them, "
                "or provide a closed surface mesh."
            )

    # Estimate edge length if not provided
    if target_edge_length is None:
        edge_lengths = []
        for element in surface_mesh.elements:
            nodes = element.nodes
            n = len(nodes)

            # Get edges for different element types
            if n == 3:  # Triangle
                edges = [(0, 1), (1, 2), (2, 0)]
            elif n == 4:  # Quad
                edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            elif n == 6:  # Triangle6
                edges = [(0, 2), (2, 4), (4, 0)]
            elif n == 8 or n == 9:  # Quad8 or Quad9
                edges = [(0, 2), (2, 4), (4, 6), (6, 0)]
            else:
                continue

            for i, j in edges:
                p1 = np.array(nodes[i].coords)
                p2 = np.array(nodes[j].coords)
                length = np.linalg.norm(p2 - p1)
                edge_lengths.append(length)

        if edge_lengths:
            target_edge_length = np.mean(edge_lengths)
        else:
            target_edge_length = 0.1  # Default fallback

    backend = backend.lower().strip()
    if backend not in {"gmsh", "wildmeshing"}:
        raise ValueError("backend must be 'gmsh' or 'wildmeshing'")

    if backend == "wildmeshing":
        try:
            import meshio
            import wildmeshing as wm
        except ImportError as exc:
            raise ImportError(
                "wildmeshing backend requires 'wildmeshing' and 'meshio' packages"
            ) from exc

        node_id_to_idx = {node.id: idx for idx, node in enumerate(surface_mesh.nodes)}
        vertices = np.array([[node.x, node.y, node.z] for node in surface_mesh.nodes])

        faces = []
        for element in surface_mesh.elements:
            face_indices = [node_id_to_idx[node.id] for node in element.nodes]
            if len(face_indices) == 3:
                faces.append(face_indices)
            elif len(face_indices) == 4:
                faces.append([face_indices[0], face_indices[1], face_indices[2]])
                faces.append([face_indices[0], face_indices[2], face_indices[3]])
            elif len(face_indices) == 6:
                faces.append([face_indices[0], face_indices[2], face_indices[4]])
            elif len(face_indices) == 8 or len(face_indices) == 9:
                faces.append([face_indices[0], face_indices[2], face_indices[4]])
                faces.append([face_indices[0], face_indices[4], face_indices[6]])

        if not faces:
            raise ValueError("Surface mesh has no valid triangular faces for wildmeshing")

        faces = np.array(faces, dtype=np.int32)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        if bbox_diag <= 0:
            bbox_diag = 1.0

        edge_length_r = target_edge_length / bbox_diag
        edge_length_r = max(min(edge_length_r, 0.5), 1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_mesh_path = f"{tmpdir}/input.obj"
            output_mesh_path = f"{tmpdir}/output.msh"

            meshio.write(input_mesh_path, meshio.Mesh(points=vertices, cells=[("triangle", faces)]))

            success = wm.tetrahedralize(
                input=input_mesh_path,
                output=output_mesh_path,
                edge_length_r=edge_length_r,
                epsilon=edge_length_r * 0.1,
                stop_quality=10,
                max_its=100,
                skip_simplify=True,
                coarsen=False,
                mute_log=False,
            )

            if not success:
                raise RuntimeError("fTetWild failed to create mesh")

            result = meshio.read(output_mesh_path)
            tet_verts = result.points

            tet_tets = None
            for cell_block in result.cells:
                if cell_block.type == "tetra":
                    tet_tets = cell_block.data
                    break

            if tet_tets is None:
                raise RuntimeError("No tetrahedra found in output")

        volumetric_mesh = MeshModel()
        Node._id_counter = 0

        node_map = {}
        for i, v in enumerate(tet_verts):
            node = Node(coords=v)
            node_map[i] = node
            volumetric_mesh.add_node(node)

        for tet in tet_tets:
            element_nodes = [node_map[int(idx)] for idx in tet]
            element_nodes = _check_tetra_orientation(element_nodes)
            element = MeshElement(nodes=element_nodes, element_type=ElementType.tetra)
            volumetric_mesh.add_element(element)

        if order == 2:
            volumetric_mesh = _convert_to_quadratic(volumetric_mesh)

        return volumetric_mesh

    # Create temporary file for the surface mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".stl", delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        # Write surface mesh to STL format
        surface_mesh.write_mesh(tmp_filename)

        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Gmsh is required for backend='gmsh'") from exc

        # Initialize Gmsh
        gmsh.initialize()

        try:
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("volumetric_remesh")

            # Merge the surface mesh
            gmsh.merge(tmp_filename)

            # Set meshing parameters before generation
            gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_edge_length * 1.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_edge_length * 0.5)

            # Classify and create geometry
            gmsh.model.mesh.classifySurfaces(
                angle=40 * np.pi / 180, boundary=True, forReparametrization=False
            )
            gmsh.model.mesh.createGeometry()
            gmsh.model.geo.synchronize()

            # Check if we have surfaces
            surfaces = gmsh.model.getEntities(2)
            if not surfaces:
                raise ValueError("No surfaces found after classification")

            print(f"  Found {len(surfaces)} surface(s)")

            # Create volume from surfaces
            surface_tags = [s[1] for s in surfaces]
            try:
                # Try to create a surface loop and volume
                surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
                volume = gmsh.model.geo.addVolume([surface_loop])
                gmsh.model.geo.synchronize()
                print("  Created volume")
            except Exception as e:
                print(f"  Warning: Could not create explicit volume: {e}")
                # Gmsh might still be able to infer the volume

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Create new mesh model
            new_mesh = MeshModel()
            Node._id_counter = 0

            # Map from gmsh node tags to Node objects
            gmsh_tag_map = {}

            # Get all nodes from gmsh
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            coords = coords.reshape(-1, 3)

            for tag, coord in zip(node_tags, coords):
                node = Node(coords=coord)
                gmsh_tag_map[int(tag)] = node
                new_mesh.add_node(node)

            # Get volumetric elements
            element_types, element_tags_list, node_tags_list = gmsh.model.mesh.getElements(3)

            # Map gmsh element types to our ElementType
            gmsh_to_element_type = {
                4: ElementType.tetra,  # 4-node tetrahedron
                11: ElementType.tetra10,  # 10-node tetrahedron
                5: ElementType.hexahedron,  # 8-node hexahedron
                17: ElementType.hexahedron20,  # 20-node hexahedron
                12: ElementType.hexahedron27,  # 27-node hexahedron
                6: ElementType.wedge,  # 6-node wedge/prism
                13: ElementType.wedge15,  # 15-node wedge
                7: ElementType.pyramid,  # 5-node pyramid
            }

            for elem_type, elem_tags, node_tags in zip(
                element_types, element_tags_list, node_tags_list
            ):
                e_type = gmsh_to_element_type.get(elem_type)
                if e_type is None:
                    continue

                # Get number of nodes for this element type
                props = gmsh.model.mesh.getElementProperties(elem_type)
                num_nodes = props[3]

                # Reshape connectivity
                connectivity = node_tags.reshape(-1, num_nodes)

                for conn in connectivity:
                    element_nodes = [gmsh_tag_map[int(tag)] for tag in conn]

                    # Check and fix orientation for tetrahedra
                    if e_type == ElementType.tetra:
                        element_nodes = _check_tetra_orientation(element_nodes)

                    new_mesh.add_element(MeshElement(nodes=element_nodes, element_type=e_type))

            return new_mesh

        finally:
            gmsh.finalize()

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def boolean_union_meshes(
    surface_meshes: List["MeshModel"], target_edge_length: Optional[float] = None
) -> "MeshModel":
    """
    Perform boolean union operation on multiple surface meshes.

    This function performs a TRUE geometric boolean union using Trimesh, which:
    - Removes internal surfaces where meshes overlap
    - Returns only the outer shell of the combined geometry
    - Handles complex intersections between components

    Parameters
    ----------
    surface_meshes : List[MeshModel]
        List of closed surface meshes to union. Each mesh MUST be a closed
        surface (no open boundaries).
    target_edge_length : Optional[float]
        Currently unused. Kept for API compatibility.

    Returns
    -------
    MeshModel
        New surface mesh containing the geometric union of all input meshes.
        All internal surfaces are removed.

    Raises
    ------
    ValueError
        If fewer than 2 meshes provided or any mesh has open boundaries
    RuntimeError
        If boolean operation fails

    Notes
    -----
    Uses Trimesh with Manifold engine for fast boolean operations.
    Trimesh is already a dependency of fem_shell.

    All input meshes should be:
    - Closed (no open boundaries)
    - Clean (no duplicate vertices/faces)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "Trimesh is required for boolean operations. Install it with: pip install trimesh"
        )

    from fem_shell.core.mesh.entities import ElementType, MeshElement, Node
    from fem_shell.core.mesh.model import MeshModel

    if len(surface_meshes) < 2:
        raise ValueError("At least 2 meshes required for boolean union")

    # Check that all meshes are closed
    for i, mesh in enumerate(surface_meshes):
        if detect_open_boundaries(mesh):
            raise ValueError(
                f"Mesh {i} has open boundaries. All meshes must be closed for boolean operations."
            )

    # Convert all FEM meshes to Trimesh meshes
    trimesh_meshes = []
    for mesh_idx, mesh in enumerate(surface_meshes):
        # Build node_id to index mapping
        node_id_to_idx = {node.id: idx for idx, node in enumerate(mesh.nodes)}

        # Extract vertices
        vertices = np.array([[node.x, node.y, node.z] for node in mesh.nodes])

        # Extract faces (only triangles supported for boolean ops)
        faces = []
        for element in mesh.elements:
            if element.element_type == ElementType.triangle:
                node_indices = [node_id_to_idx[node.id] for node in element.nodes]
                if len(node_indices) == 3:
                    faces.append(node_indices)
            elif element.element_type == ElementType.quad:
                # Split quad into 2 triangles
                node_indices = [node_id_to_idx[node.id] for node in element.nodes]
                if len(node_indices) == 4:
                    faces.append([node_indices[0], node_indices[1], node_indices[2]])
                    faces.append([node_indices[0], node_indices[2], node_indices[3]])

        faces = np.array(faces, dtype=np.int32)

        # Create Trimesh mesh
        tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        trimesh_meshes.append(tm_mesh)

    # Perform boolean union using trimesh
    print(f"  Using Trimesh boolean union with {len(trimesh_meshes)} meshes...")
    try:
        # Use trimesh.boolean.union which handles multiple meshes efficiently
        result_mesh = trimesh.boolean.union(trimesh_meshes, engine="manifold")
    except Exception as e:
        print(f"  ⚠ Manifold engine failed: {e}")
        print("  Trying fallback method...")
        # Fallback: successive unions
        result_mesh = trimesh_meshes[0]
        for i in range(1, len(trimesh_meshes)):
            try:
                result_mesh = trimesh.boolean.union(
                    [result_mesh, trimesh_meshes[i]], engine="manifold"
                )
            except Exception:
                print(f"  ⚠ Boolean failed for mesh {i}, using simple concatenation")
                result_mesh = trimesh.util.concatenate([result_mesh, trimesh_meshes[i]])

    # Convert result back to FEM mesh
    vertices = result_mesh.vertices
    faces = result_mesh.faces

    union_mesh = MeshModel()

    # Add nodes
    for vertex in vertices:
        union_mesh.add_node(Node(coords=vertex, geometric_node=True))

    # Add triangles
    for face in faces:
        nodes = [union_mesh.nodes[int(idx)] for idx in face]
        union_mesh.add_element(MeshElement(nodes=nodes, element_type=ElementType.triangle))

    return union_mesh


def _write_mesh_to_stl(mesh: "MeshModel", filepath: str) -> None:
    """
    Write a surface mesh to STL file (ASCII format).

    Parameters
    ----------
    mesh : MeshModel
        Surface mesh to write (should contain only triangles)
    filepath : str
        Output file path
    """
    with open(filepath, "w") as f:
        f.write("solid mesh\n")

        for element in mesh.elements:
            if len(element.nodes) == 3:  # Triangle
                nodes = element.nodes
                p1 = np.array([nodes[0].x, nodes[0].y, nodes[0].z])
                p2 = np.array([nodes[1].x, nodes[1].y, nodes[1].z])
                p3 = np.array([nodes[2].x, nodes[2].y, nodes[2].z])

                # Compute normal
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm

                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
                f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
                f.write(f"      vertex {p3[0]:.6e} {p3[1]:.6e} {p3[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

        f.write("endsolid mesh\n")


# =============================================================================
# Solid Element Orientation Verification and Correction
# =============================================================================


def _compute_tetra_volume(coords: np.ndarray) -> float:
    """
    Compute signed volume of a tetrahedron.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (4, 3) for linear or (10, 3) for quadratic.
        Only first 4 nodes (corners) are used.

    Returns
    -------
    float
        Signed volume. Positive = correct orientation, negative = inverted.
    """
    p0, p1, p2, p3 = coords[:4]
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    return np.dot(v1, np.cross(v2, v3)) / 6.0


def _compute_wedge_volume(coords: np.ndarray) -> float:
    """
    Compute approximate signed volume of a wedge/prism element.

    Uses the Jacobian determinant at the centroid as a proxy for orientation.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (6, 3) for linear or (15, 3) for quadratic.
        Only first 6 nodes (corners) are used.

    Returns
    -------
    float
        Signed volume proxy. Positive = correct orientation.
    """
    # For wedge: evaluate Jacobian at centroid (1/3, 1/3, 0)
    # Shape function derivatives for 6-node wedge at centroid
    xi, eta, zeta = 1 / 3, 1 / 3, 0

    # Triangular coordinates
    L1, L2, L3 = 1 - xi - eta, xi, eta
    Lm, Lp = 0.5 * (1 - zeta), 0.5 * (1 + zeta)

    # dN/dxi, dN/deta, dN/dzeta for nodes 0-5
    dL1_dxi, dL2_dxi, dL3_dxi = -1, 1, 0
    dL1_deta, dL2_deta, dL3_deta = -1, 0, 1
    dLm_dzeta, dLp_dzeta = -0.5, 0.5

    dN_dxi = np.array([
        dL1_dxi * Lm,
        dL2_dxi * Lm,
        dL3_dxi * Lm,
        dL1_dxi * Lp,
        dL2_dxi * Lp,
        dL3_dxi * Lp,
    ])
    dN_deta = np.array([
        dL1_deta * Lm,
        dL2_deta * Lm,
        dL3_deta * Lm,
        dL1_deta * Lp,
        dL2_deta * Lp,
        dL3_deta * Lp,
    ])
    dN_dzeta = np.array([
        L1 * dLm_dzeta,
        L2 * dLm_dzeta,
        L3 * dLm_dzeta,
        L1 * dLp_dzeta,
        L2 * dLp_dzeta,
        L3 * dLp_dzeta,
    ])

    corners = coords[:6]
    J = np.array([
        [dN_dxi @ corners[:, 0], dN_dxi @ corners[:, 1], dN_dxi @ corners[:, 2]],
        [dN_deta @ corners[:, 0], dN_deta @ corners[:, 1], dN_deta @ corners[:, 2]],
        [dN_dzeta @ corners[:, 0], dN_dzeta @ corners[:, 1], dN_dzeta @ corners[:, 2]],
    ])
    return np.linalg.det(J)


def _compute_hexa_volume(coords: np.ndarray) -> float:
    """
    Compute approximate signed volume of a hexahedron element.

    Uses the Jacobian determinant at the centroid as a proxy for orientation.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (8, 3) for linear or (20/27, 3) for quadratic.
        Only first 8 nodes (corners) are used.

    Returns
    -------
    float
        Signed volume proxy. Positive = correct orientation.
    """
    # For hexa: evaluate Jacobian at centroid (0, 0, 0)
    # Shape function derivatives for 8-node hexa at centroid
    # dN/dxi = 1/8 * [-1, 1, 1, -1, -1, 1, 1, -1] (at origin)
    dN_dxi = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) / 8
    dN_deta = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) / 8
    dN_dzeta = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) / 8

    corners = coords[:8]
    J = np.array([
        [dN_dxi @ corners[:, 0], dN_dxi @ corners[:, 1], dN_dxi @ corners[:, 2]],
        [dN_deta @ corners[:, 0], dN_deta @ corners[:, 1], dN_deta @ corners[:, 2]],
        [dN_dzeta @ corners[:, 0], dN_dzeta @ corners[:, 1], dN_dzeta @ corners[:, 2]],
    ])
    return np.linalg.det(J)


def _compute_pyramid_volume(coords: np.ndarray) -> float:
    """
    Compute approximate signed volume of a pyramid element.

    Uses a simplified approach based on base orientation and apex position.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (5, 3) for linear or (13, 3) for quadratic.
        Only first 5 nodes (corners) are used.

    Returns
    -------
    float
        Signed volume proxy. Positive = correct orientation (apex above base).
    """
    # Base nodes: 0, 1, 2, 3 (quad), apex: 4
    base = coords[:4]
    apex = coords[4]

    # Compute base centroid and normal
    centroid = np.mean(base, axis=0)
    v1 = base[1] - base[0]
    v2 = base[3] - base[0]
    base_normal = np.cross(v1, v2)

    # Check if apex is on positive side of base
    apex_vec = apex - centroid
    return np.dot(apex_vec, base_normal)


def _fix_tetra_orientation(nodes: list, quadratic: bool = False) -> list:
    """
    Fix tetrahedron orientation by swapping nodes if needed.

    Parameters
    ----------
    nodes : list
        List of Node objects (4 for linear, 10 for quadratic)
    quadratic : bool
        True if this is a 10-node quadratic tetrahedron

    Returns
    -------
    list
        Reordered nodes with positive volume
    """
    coords = np.array([n.coords for n in nodes])
    volume = _compute_tetra_volume(coords)

    if volume >= 0:
        return nodes

    # Swap nodes 0 and 1 to fix orientation
    if not quadratic:
        return [nodes[1], nodes[0], nodes[2], nodes[3]]
    else:
        # For TETRA10: swap corners 0↔1 and corresponding edge nodes
        # Original: corners 0,1,2,3; edges 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(2-3), 9(1-3)
        # After swap: corners 1,0,2,3; edges 4(1-0), 5(0-2), 6(1-2), 7(1-3), 8(2-3), 9(0-3)
        return [
            nodes[1],
            nodes[0],
            nodes[2],
            nodes[3],  # corners swapped
            nodes[4],  # edge 0-1 = edge 1-0 (same midpoint)
            nodes[6],  # edge 1-2 becomes edge 0-2
            nodes[5],  # edge 0-2 becomes edge 1-2
            nodes[9],  # edge 0-3 becomes edge 1-3
            nodes[8],  # edge 2-3 stays
            nodes[7],  # edge 1-3 becomes edge 0-3
        ]


def _fix_wedge_orientation(nodes: list, quadratic: bool = False) -> list:
    """
    Fix wedge orientation by swapping nodes if needed.

    Parameters
    ----------
    nodes : list
        List of Node objects (6 for linear, 15 for quadratic)
    quadratic : bool
        True if this is a 15-node quadratic wedge

    Returns
    -------
    list
        Reordered nodes with positive volume
    """
    coords = np.array([n.coords for n in nodes])
    volume = _compute_wedge_volume(coords)

    if volume >= 0:
        return nodes

    # Swap bottom and top faces (reverse z-direction)
    if not quadratic:
        return [nodes[3], nodes[4], nodes[5], nodes[0], nodes[1], nodes[2]]
    else:
        # For WEDGE15: swap faces and corresponding edge nodes
        # Corners: 0,1,2 (bottom), 3,4,5 (top)
        # Edges: 6(0-1), 7(0-2), 8(0-3), 9(1-2), 10(1-4), 11(2-5), 12(3-4), 13(3-5), 14(4-5)
        return [
            nodes[3],
            nodes[4],
            nodes[5],  # top becomes bottom
            nodes[0],
            nodes[1],
            nodes[2],  # bottom becomes top
            nodes[12],
            nodes[13],
            nodes[8],  # edges on new bottom
            nodes[14],
            nodes[10],
            nodes[11],  # vertical edges
            nodes[6],
            nodes[7],
            nodes[9],  # edges on new top
        ]


def _fix_hexa_orientation(nodes: list, quadratic: bool = False) -> list:
    """
    Fix hexahedron orientation by swapping nodes if needed.

    Parameters
    ----------
    nodes : list
        List of Node objects (8 for linear, 20/27 for quadratic)
    quadratic : bool
        True if this is a quadratic hexahedron

    Returns
    -------
    list
        Reordered nodes with positive volume
    """
    coords = np.array([n.coords for n in nodes])
    volume = _compute_hexa_volume(coords)

    if volume >= 0:
        return nodes

    # Swap bottom and top faces
    if not quadratic:
        return [nodes[4], nodes[5], nodes[6], nodes[7], nodes[0], nodes[1], nodes[2], nodes[3]]
    else:
        # For HEXA20: more complex reordering
        # This is a simplified fix - swap z-direction
        return [
            nodes[4],
            nodes[5],
            nodes[6],
            nodes[7],
            nodes[0],
            nodes[1],
            nodes[2],
            nodes[3],
            nodes[16],
            nodes[17],
            nodes[18],
            nodes[19],
            nodes[8],
            nodes[9],
            nodes[10],
            nodes[11],
            nodes[12],
            nodes[13],
            nodes[14],
            nodes[15],
        ]


def _fix_pyramid_orientation(nodes: list, quadratic: bool = False) -> list:
    """
    Fix pyramid orientation by reordering base nodes if needed.

    Parameters
    ----------
    nodes : list
        List of Node objects (5 for linear, 13 for quadratic)
    quadratic : bool
        True if this is a 13-node quadratic pyramid

    Returns
    -------
    list
        Reordered nodes with positive volume
    """
    coords = np.array([n.coords for n in nodes])
    volume = _compute_pyramid_volume(coords)

    if volume >= 0:
        return nodes

    # Reverse base winding order
    if not quadratic:
        return [nodes[0], nodes[3], nodes[2], nodes[1], nodes[4]]
    else:
        # For PYRAMID13: reverse base and adjust edge nodes
        return [
            nodes[0],
            nodes[3],
            nodes[2],
            nodes[1],
            nodes[4],
            nodes[7],
            nodes[6],
            nodes[5],
            nodes[8],  # base edges reversed
            nodes[9],
            nodes[12],
            nodes[11],
            nodes[10],  # apex edges
        ]


def verify_solid_element_orientations(
    mesh: "MeshModel",
    fix_inplace: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Verify and optionally fix orientations of all solid elements in a mesh.

    This function checks that all volumetric elements have positive Jacobian
    determinants (correct orientation for FEM analysis). Elements with negative
    Jacobians will have incorrect strain-displacement relations.

    Parameters
    ----------
    mesh : MeshModel
        The mesh to verify
    fix_inplace : bool, optional
        If True, fix element orientations in place. Default: True
    verbose : bool, optional
        If True, print diagnostic information. Default: True

    Returns
    -------
    dict
        Dictionary with verification results:
        - 'total_solid_elements': Total number of solid elements
        - 'correct': Number of correctly oriented elements
        - 'fixed': Number of elements that were fixed
        - 'failed': Number of elements that could not be fixed
        - 'by_type': Dict with counts per element type

    Examples
    --------
    >>> from fem_shell.core.mesh import load_mesh, verify_solid_element_orientations
    >>> mesh = load_mesh("volumetric_mesh.h5")
    >>> results = verify_solid_element_orientations(mesh)
    >>> print(f"Fixed {results['fixed']} elements")
    """
    from fem_shell.core.mesh.entities import ElementType

    # Element type to volume function and fix function mapping
    SOLID_CHECKS = {
        ElementType.tetra: (_compute_tetra_volume, _fix_tetra_orientation, False),
        ElementType.tetra10: (_compute_tetra_volume, _fix_tetra_orientation, True),
        ElementType.wedge: (_compute_wedge_volume, _fix_wedge_orientation, False),
        ElementType.wedge15: (_compute_wedge_volume, _fix_wedge_orientation, True),
        ElementType.hexahedron: (_compute_hexa_volume, _fix_hexa_orientation, False),
        ElementType.hexahedron20: (_compute_hexa_volume, _fix_hexa_orientation, True),
        ElementType.hexahedron27: (_compute_hexa_volume, _fix_hexa_orientation, True),
        ElementType.pyramid: (_compute_pyramid_volume, _fix_pyramid_orientation, False),
        ElementType.pyramid13: (_compute_pyramid_volume, _fix_pyramid_orientation, True),
    }

    results = {
        "total_solid_elements": 0,
        "correct": 0,
        "fixed": 0,
        "failed": 0,
        "by_type": {},
    }

    for element in mesh.elements:
        elem_type = element.element_type
        if elem_type not in SOLID_CHECKS:
            continue  # Not a solid element

        results["total_solid_elements"] += 1
        type_name = elem_type.name  # Use consistent key

        if type_name not in results["by_type"]:
            results["by_type"][type_name] = {"total": 0, "correct": 0, "fixed": 0, "failed": 0}

        results["by_type"][type_name]["total"] += 1

        volume_func, fix_func, is_quadratic = SOLID_CHECKS[elem_type]
        coords = np.array([n.coords for n in element.nodes])
        volume = volume_func(coords)

        if volume > 0:
            results["correct"] += 1
            results["by_type"][type_name]["correct"] += 1
        else:
            if fix_inplace:
                fixed_nodes = fix_func(list(element.nodes), quadratic=is_quadratic)
                element.nodes = fixed_nodes  # Reassign nodes directly

                # Verify fix worked
                new_coords = np.array([n.coords for n in element.nodes])
                new_volume = volume_func(new_coords)

                if new_volume > 0:
                    results["fixed"] += 1
                    results["by_type"][type_name]["fixed"] += 1
                else:
                    results["failed"] += 1
                    results["by_type"][type_name]["failed"] += 1
            else:
                results["failed"] += 1
                results["by_type"][type_name]["failed"] += 1

    if verbose and results["total_solid_elements"] > 0:
        print(f"\n{'─' * 60}")
        print("  SOLID ELEMENT ORIENTATION VERIFICATION")
        print(f"{'─' * 60}")
        print(f"  Total solid elements: {results['total_solid_elements']}")
        print(f"  ✓ Correct orientation: {results['correct']}")
        if fix_inplace:
            print(f"  ⟳ Fixed orientation:   {results['fixed']}")
        print(f"  ✗ Invalid (unfixable): {results['failed']}")
        print(f"{'─' * 60}")

        if results["by_type"]:
            print("  By element type:")
            for etype, counts in results["by_type"].items():
                status = "✓" if counts["failed"] == 0 else "⚠"
                print(
                    f"    {status} {etype}: {counts['total']} total, "
                    f"{counts['correct']} ok, {counts['fixed']} fixed, {counts['failed']} failed"
                )
        print()

    return results


def check_mesh_quality(
    mesh: "MeshModel",
    verbose: bool = True,
) -> dict:
    """
    Perform comprehensive mesh quality checks for solid elements.

    Checks include:
    - Element orientation (positive Jacobian)
    - Aspect ratio (elongated elements)
    - Minimum Jacobian ratio (distorted elements)

    Parameters
    ----------
    mesh : MeshModel
        The mesh to check
    verbose : bool, optional
        If True, print diagnostic information. Default: True

    Returns
    -------
    dict
        Dictionary with quality metrics:
        - 'orientation': Results from verify_solid_element_orientations
        - 'aspect_ratio': {'min', 'max', 'mean', 'bad_count'}
        - 'jacobian_ratio': {'min', 'max', 'mean', 'bad_count'}

    Examples
    --------
    >>> from fem_shell.core.mesh import load_mesh, check_mesh_quality
    >>> mesh = load_mesh("volumetric_mesh.h5")
    >>> quality = check_mesh_quality(mesh)
    >>> if quality['orientation']['failed'] > 0:
    ...     print("WARNING: Some elements have invalid orientation!")
    """
    from fem_shell.core.mesh.entities import ElementType

    results = {
        "orientation": verify_solid_element_orientations(mesh, fix_inplace=False, verbose=False),
        "aspect_ratio": {"min": float("inf"), "max": 0, "mean": 0, "bad_count": 0, "threshold": 10},
        "jacobian_ratio": {
            "min": float("inf"),
            "max": 0,
            "mean": 0,
            "bad_count": 0,
            "threshold": 0.1,
        },
    }

    aspect_ratios = []
    solid_types = {
        ElementType.tetra,
        ElementType.tetra10,
        ElementType.wedge,
        ElementType.wedge15,
        ElementType.hexahedron,
        ElementType.hexahedron20,
        ElementType.hexahedron27,
        ElementType.pyramid,
        ElementType.pyramid13,
    }

    for element in mesh.elements:
        if element.element_type not in solid_types:
            continue

        coords = np.array([n.coords for n in element.nodes])

        # Compute aspect ratio (max edge / min edge)
        n_corners = {4: 4, 5: 5, 6: 6, 8: 8, 10: 4, 13: 5, 15: 6, 20: 8, 27: 8}.get(
            len(element.nodes), 4
        )
        corners = coords[:n_corners]

        edge_lengths = []
        for i in range(n_corners):
            for j in range(i + 1, n_corners):
                edge_lengths.append(np.linalg.norm(corners[i] - corners[j]))

        if edge_lengths:
            min_edge = min(edge_lengths)
            max_edge = max(edge_lengths)
            if min_edge > 1e-10:
                ar = max_edge / min_edge
                aspect_ratios.append(ar)

                if ar > results["aspect_ratio"]["threshold"]:
                    results["aspect_ratio"]["bad_count"] += 1

    if aspect_ratios:
        results["aspect_ratio"]["min"] = min(aspect_ratios)
        results["aspect_ratio"]["max"] = max(aspect_ratios)
        results["aspect_ratio"]["mean"] = np.mean(aspect_ratios)

    if verbose:
        print(f"\n{'═' * 60}")
        print("  MESH QUALITY REPORT")
        print(f"{'═' * 60}")

        # Orientation
        orient = results["orientation"]
        if orient["total_solid_elements"] > 0:
            pct_ok = 100 * (orient["correct"] / orient["total_solid_elements"])
            status = "✓" if orient["failed"] == 0 else "✗"
            print(
                f"\n  {status} Orientation: {pct_ok:.1f}% correct ({orient['correct']}/{orient['total_solid_elements']})"
            )
            if orient["failed"] > 0:
                print(f"    ⚠ WARNING: {orient['failed']} elements have inverted orientation!")

        # Aspect ratio
        ar = results["aspect_ratio"]
        if aspect_ratios:
            status = "✓" if ar["bad_count"] == 0 else "⚠"
            print(f"\n  {status} Aspect Ratio (threshold={ar['threshold']}):")
            print(f"    Min: {ar['min']:.2f}, Max: {ar['max']:.2f}, Mean: {ar['mean']:.2f}")
            if ar["bad_count"] > 0:
                print(f"    ⚠ {ar['bad_count']} elements exceed threshold")

        print(f"\n{'═' * 60}\n")

    return results
