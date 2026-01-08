"""
Mesh generators module.

This module contains classes for generating various types of structured meshes:
- SquareShapeMesh: 2D rectangular meshes
- BoxSurfaceMesh: 3D box surface meshes
- MultiFlapMesh: Multi-flap structures for FSI simulations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import gmsh
import numpy as np

from fem_shell.core.mesh.entities import ELEMENT_NODES_MAP, ElementType, MeshElement, Node, NodeSet

if TYPE_CHECKING:
    from fem_shell.core.mesh.model import MeshModel


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
        # Import here to avoid circular imports
        from fem_shell.core.mesh.model import MeshModel

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("rectangle")

            self._create_geometry()
            self._configure_mesh()

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize()

            return self._create_mesh_model(MeshModel)
        finally:
            gmsh.finalize()

    def _create_geometry(self):
        """Create geometric entities with proper boundary handling"""
        x0 = -self.width / 2
        x1 = self.width / 2
        y0 = 0.0
        y1 = self.height

        self.p1 = gmsh.model.geo.addPoint(x0, y0, 0)
        self.p2 = gmsh.model.geo.addPoint(x1, y0, 0)
        self.p3 = gmsh.model.geo.addPoint(x1, y1, 0)
        self.p4 = gmsh.model.geo.addPoint(x0, y1, 0)

        self.bottom = gmsh.model.geo.addLine(self.p1, self.p2)
        self.right = gmsh.model.geo.addLine(self.p2, self.p3)
        self.top = gmsh.model.geo.addLine(self.p3, self.p4)
        self.left = gmsh.model.geo.addLine(self.p4, self.p1)

        loop = gmsh.model.geo.addCurveLoop([self.bottom, self.right, self.top, self.left])
        self.surface = gmsh.model.geo.addPlaneSurface([loop])

        self._add_physical_groups()

    def _add_physical_groups(self):
        """Add physical groups with proper corner handling"""
        gmsh.model.addPhysicalGroup(1, [self.top], name="top")
        gmsh.model.addPhysicalGroup(1, [self.bottom], name="bottom")
        gmsh.model.addPhysicalGroup(1, [self.left], name="left")
        gmsh.model.addPhysicalGroup(1, [self.right], name="right")

        gmsh.model.addPhysicalGroup(0, [self.p1], name="corner_p1")
        gmsh.model.addPhysicalGroup(0, [self.p2], name="corner_p2")
        gmsh.model.addPhysicalGroup(0, [self.p3], name="corner_p3")
        gmsh.model.addPhysicalGroup(0, [self.p4], name="corner_p4")

    def _configure_mesh(self):
        """Configure meshing parameters"""
        for curve in [self.bottom, self.top]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, self.nx + 1)
        for curve in [self.left, self.right]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, self.ny + 1)

        gmsh.model.geo.mesh.setTransfiniteSurface(
            self.surface, "Right", [self.p1, self.p2, self.p3, self.p4]
        )

        if self.triangular:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
        else:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)

        if self.quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

    def _create_mesh_model(self, MeshModelClass) -> "MeshModel":
        """Converts Gmsh mesh to MeshModel instance"""
        mesh_model = MeshModelClass()

        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        geometric_node_tags = set()
        for et, conn in zip(elementTypes, nodeTags):
            props = gmsh.model.mesh.getElementProperties(et)
            total_nodes = props[3]
            e_type = ELEMENT_NODES_MAP[total_nodes]
            if e_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                num_corners = 4
            elif e_type in (ElementType.triangle6, ElementType.triangle6):
                num_corners = 3
            else:
                raise ValueError
            geometric_node_tags.update(
                nodeTags[0].reshape(-1, total_nodes)[:, :num_corners].flatten()
            )

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)

        for tag, coord in zip(node_tags, coords):
            if tag in geometric_node_tags:
                mesh_model.add_node(Node(coord, geometric_node=True))
            else:
                mesh_model.add_node(Node(coord, geometric_node=False))

        self._add_elements(mesh_model)
        self._create_node_sets(mesh_model)

        return mesh_model

    def _add_elements(self, mesh_model: "MeshModel"):
        """Add elements with corrected node indices"""
        elem_types = gmsh.model.mesh.getElementTypes()
        for elem_type in elem_types:
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 2:
                _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
                num_nodes = elem_props[3]
                connectivity = elem_node_tags.reshape(-1, num_nodes)

                for nodes in connectivity:
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
        from fem_shell.core.mesh.model import MeshModel

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("box_surface")

            self._create_geometry()
            gmsh.model.geo.synchronize()

            self._configure_mesh()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize()

            return self._create_mesh_model(MeshModel)
        finally:
            gmsh.finalize()

    def _create_geometry(self):
        cx, cy, cz = self.center
        dx, dy, dz = self.dims

        x = (cx - dx / 2, cx + dx / 2)
        y = (cy - dy / 2, cy + dy / 2)
        z = (cz - dz / 2, cz + dz / 2)

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

        self._create_edges()
        self._create_faces()

        gmsh.model.geo.synchronize()

    def _create_edges(self):
        p = self.points
        self.edges = {
            "l1": gmsh.model.geo.addLine(p["p1"], p["p2"]),
            "l2": gmsh.model.geo.addLine(p["p2"], p["p3"]),
            "l3": gmsh.model.geo.addLine(p["p3"], p["p4"]),
            "l4": gmsh.model.geo.addLine(p["p4"], p["p1"]),
            "l5": gmsh.model.geo.addLine(p["p5"], p["p6"]),
            "l6": gmsh.model.geo.addLine(p["p6"], p["p7"]),
            "l7": gmsh.model.geo.addLine(p["p7"], p["p8"]),
            "l8": gmsh.model.geo.addLine(p["p8"], p["p5"]),
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

        for name, face_tag in self.faces.items():
            gmsh.model.addPhysicalGroup(2, [face_tag], name=name)

    def _create_face_loop(self, curves):
        """Helper to create face from oriented curves"""
        loop = gmsh.model.geo.addCurveLoop(curves)
        return gmsh.model.geo.addPlaneSurface([loop])

    def _configure_mesh(self):
        """Configures mesh parameters and transfinite settings"""
        self._set_transfinite_curves()
        self._set_transfinite_surfaces()

        if self.triangular:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)

        if self.quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

    def _set_transfinite_curves(self):
        """Configure transfinite curves"""
        for e in ["l1", "l3", "l5", "l7"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.nx + 1)

        for e in ["l2", "l4", "l6", "l8"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.nz + 1)

        for e in ["l9", "l10", "l11", "l12"]:
            gmsh.model.mesh.setTransfiniteCurve(self.edges[e], self.ny + 1)

    def _set_transfinite_surfaces(self):
        """Configures transfinite surfaces"""
        for face in self.faces.values():
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)

    def _create_mesh_model(self, MeshModelClass) -> "MeshModel":
        """Converts Gmsh mesh to MeshModel instance"""
        mesh_model = MeshModelClass()

        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        geometric_node_tags = set()
        for et, conn in zip(elementTypes, nodeTags):
            props = gmsh.model.mesh.getElementProperties(et)
            total_nodes = props[3]
            e_type = ELEMENT_NODES_MAP[total_nodes]
            if e_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                num_corners = 4
            elif e_type in (ElementType.triangle6, ElementType.triangle6):
                num_corners = 3
            else:
                raise ValueError
            geometric_node_tags.update(
                nodeTags[0].reshape(-1, total_nodes)[:, :num_corners].flatten()
            )

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)

        for tag, coord in zip(node_tags, coords):
            if tag in geometric_node_tags:
                mesh_model.add_node(Node(coord, geometric_node=True))
            else:
                mesh_model.add_node(Node(coord, geometric_node=False))

        self._add_elements(mesh_model)
        self._create_node_sets(mesh_model)

        return mesh_model

    def _add_elements(self, mesh_model: "MeshModel"):
        """Add elements with corrected node indices"""
        elem_types = gmsh.model.mesh.getElementTypes()
        for elem_type in elem_types:
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 2:
                _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
                num_nodes = elem_props[3]
                connectivity = elem_node_tags.reshape(-1, num_nodes)

                for nodes in connectivity:
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
            if dim != 2:
                continue

            name = gmsh.model.getPhysicalName(dim, tag)
            if name not in face_sets:
                continue

            node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0]
            face_sets[name].update((tag - 1 for tag in node_tags))

        for name, node_ids in face_sets.items():
            if node_ids:
                node_objs = {mesh_model.get_node_by_id(nid) for nid in node_ids}
                mesh_model.add_node_set(NodeSet(name=name, nodes=node_objs))

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


class MultiFlapMesh:
    """
    Generates a structured 2D mesh for multiple flaps connected to a common base.

    The geometry consists of:
    - A horizontal base (rectangle) at y=0 with fixed boundary condition
    - Multiple vertical flaps extending upward from the base

    Parameters
    ----------
    n_flaps : int
        Number of flaps
    flap_width : float
        Width of each flap (X direction)
    flap_height : float
        Height of each flap (Y direction)
    x_spacing : float
        Spacing between consecutive flaps (edge to edge)
    base_height : float
        Height of the base strip (Y direction)
    nx_flap : int
        Number of divisions in X direction for each flap
    ny_flap : int
        Number of divisions in Y direction for each flap
    nx_base_segment : int
        Number of divisions in X for base segments between flaps
    ny_base : int
        Number of divisions in Y for the base
    quadratic : bool
        Use quadratic elements (default: False)
    """

    def __init__(
        self,
        n_flaps: int,
        flap_width: float,
        flap_height: float,
        x_spacing: float,
        base_height: float = 0.05,
        nx_flap: int = 4,
        ny_flap: int = 20,
        nx_base_segment: int = 10,
        ny_base: int = 2,
        quadratic: bool = False,
    ):
        self.n_flaps = n_flaps
        self.flap_width = flap_width
        self.flap_height = flap_height
        self.x_spacing = x_spacing
        self.base_height = base_height
        self.nx_flap = nx_flap
        self.ny_flap = ny_flap
        self.nx_base_segment = nx_base_segment
        self.ny_base = ny_base
        self.quadratic = quadratic

        self.points = {}
        self.lines = {}
        self.surfaces = {}
        self.flap_surfaces = []
        self.base_surfaces = []

    def _calculate_positions(self) -> List[dict]:
        """Calculate X positions of all flaps centered around x=0"""
        total_width = self.n_flaps * self.flap_width + (self.n_flaps - 1) * self.x_spacing
        x_start = -total_width / 2

        positions = []
        for i in range(self.n_flaps):
            x_left = x_start + i * (self.flap_width + self.x_spacing)
            positions.append({
                "index": i + 1,
                "x_left": x_left,
                "x_right": x_left + self.flap_width,
            })
        return positions

    def generate(self) -> "MeshModel":
        """Generates and returns the MeshModel instance"""
        from fem_shell.core.mesh.model import MeshModel

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("multiflap")

            self._create_geometry()
            gmsh.model.geo.synchronize()

            self._configure_mesh()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize()

            return self._create_mesh_model(MeshModel)
        finally:
            gmsh.finalize()

    def _create_geometry(self):
        """Create the multi-flap geometry with base"""
        flap_positions = self._calculate_positions()

        y_base_bottom = -self.base_height
        y_base_top = 0.0
        y_flap_top = self.flap_height

        x_left_total = flap_positions[0]["x_left"]
        x_right_total = flap_positions[-1]["x_right"]

        point_id = 1

        base_bottom_points = []
        x_coords = [x_left_total]
        for fp in flap_positions:
            x_coords.extend([fp["x_left"], fp["x_right"]])
        x_coords.append(x_right_total)
        x_coords = sorted(set(x_coords))

        for x in x_coords:
            self.points[f"bb_{point_id}"] = gmsh.model.geo.addPoint(x, y_base_bottom, 0)
            base_bottom_points.append((x, self.points[f"bb_{point_id}"]))
            point_id += 1

        base_top_points = []
        for x in x_coords:
            self.points[f"bt_{point_id}"] = gmsh.model.geo.addPoint(x, y_base_top, 0)
            base_top_points.append((x, self.points[f"bt_{point_id}"]))
            point_id += 1

        flap_top_points = []
        for fp in flap_positions:
            p_left = gmsh.model.geo.addPoint(fp["x_left"], y_flap_top, 0)
            p_right = gmsh.model.geo.addPoint(fp["x_right"], y_flap_top, 0)
            self.points[f"ft_{fp['index']}_left"] = p_left
            self.points[f"ft_{fp['index']}_right"] = p_right
            flap_top_points.append({
                "index": fp["index"],
                "x_left": fp["x_left"],
                "x_right": fp["x_right"],
                "p_left": p_left,
                "p_right": p_right,
            })

        self._create_base_structure(base_bottom_points, base_top_points, flap_positions)
        self._create_flaps(base_top_points, flap_top_points, flap_positions)
        self._add_physical_groups(flap_positions)

    def _create_base_structure(self, base_bottom_points, base_top_points, flap_positions):
        """Create the base horizontal strip"""
        bottom_lines = []
        top_lines = []

        for i in range(len(base_bottom_points) - 1):
            l_bottom = gmsh.model.geo.addLine(
                base_bottom_points[i][1], base_bottom_points[i + 1][1]
            )
            l_top = gmsh.model.geo.addLine(base_top_points[i][1], base_top_points[i + 1][1])
            bottom_lines.append(l_bottom)
            top_lines.append(l_top)

        vertical_lines = []
        for i in range(len(base_bottom_points)):
            l_vert = gmsh.model.geo.addLine(base_bottom_points[i][1], base_top_points[i][1])
            vertical_lines.append(l_vert)

        for i in range(len(bottom_lines)):
            loop = gmsh.model.geo.addCurveLoop([
                bottom_lines[i],
                vertical_lines[i + 1],
                -top_lines[i],
                -vertical_lines[i],
            ])
            surf = gmsh.model.geo.addPlaneSurface([loop])
            self.base_surfaces.append(surf)

        self.lines["base_bottom"] = bottom_lines
        self.lines["base_top"] = top_lines
        self.lines["base_vertical"] = vertical_lines

    def _create_flaps(self, base_top_points, flap_top_points, flap_positions):
        """Create the vertical flaps extending from the base"""
        base_top_lookup = {round(x, 10): p for x, p in base_top_points}

        for ftp in flap_top_points:
            idx = ftp["index"]
            x_left = round(ftp["x_left"], 10)
            x_right = round(ftp["x_right"], 10)

            p_base_left = base_top_lookup[x_left]
            p_base_right = base_top_lookup[x_right]

            p_top_left = ftp["p_left"]
            p_top_right = ftp["p_right"]

            l_left = gmsh.model.geo.addLine(p_base_left, p_top_left)
            l_top = gmsh.model.geo.addLine(p_top_left, p_top_right)
            l_right = gmsh.model.geo.addLine(p_top_right, p_base_right)
            l_bottom = gmsh.model.geo.addLine(p_base_right, p_base_left)

            loop = gmsh.model.geo.addCurveLoop([l_left, l_top, l_right, l_bottom])
            surf = gmsh.model.geo.addPlaneSurface([loop])

            self.flap_surfaces.append({
                "index": idx,
                "surface": surf,
                "l_left": l_left,
                "l_top": l_top,
                "l_right": l_right,
                "l_bottom": l_bottom,
            })

    def _add_physical_groups(self, flap_positions):
        """Add physical groups for boundary conditions"""
        all_bottom_lines = self.lines["base_bottom"]
        gmsh.model.addPhysicalGroup(1, all_bottom_lines, name="base_bottom")

        gmsh.model.addPhysicalGroup(1, [self.lines["base_vertical"][0]], name="base_left")
        gmsh.model.addPhysicalGroup(1, [self.lines["base_vertical"][-1]], name="base_right")

        for flap_data in self.flap_surfaces:
            idx = flap_data["index"]
            gmsh.model.addPhysicalGroup(1, [flap_data["l_left"]], name=f"flap{idx}_left")
            gmsh.model.addPhysicalGroup(1, [flap_data["l_top"]], name=f"flap{idx}_top")
            gmsh.model.addPhysicalGroup(1, [flap_data["l_right"]], name=f"flap{idx}_right")
            gmsh.model.addPhysicalGroup(2, [flap_data["surface"]], name=f"flap{idx}")

        all_flap_left = [f["l_left"] for f in self.flap_surfaces]
        all_flap_top = [f["l_top"] for f in self.flap_surfaces]
        all_flap_right = [f["l_right"] for f in self.flap_surfaces]
        gmsh.model.addPhysicalGroup(1, all_flap_left, name="flaps_left")
        gmsh.model.addPhysicalGroup(1, all_flap_top, name="flaps_top")
        gmsh.model.addPhysicalGroup(1, all_flap_right, name="flaps_right")

        all_surfaces = self.base_surfaces + [f["surface"] for f in self.flap_surfaces]
        gmsh.model.addPhysicalGroup(2, all_surfaces, name="all_surfaces")

    def _configure_mesh(self):
        """Configure mesh parameters"""
        for line in self.lines["base_bottom"]:
            gmsh.model.mesh.setTransfiniteCurve(line, self.nx_base_segment + 1)

        for line in self.lines["base_top"]:
            gmsh.model.mesh.setTransfiniteCurve(line, self.nx_base_segment + 1)

        for line in self.lines["base_vertical"]:
            gmsh.model.mesh.setTransfiniteCurve(line, self.ny_base + 1)

        for surf in self.base_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        for flap_data in self.flap_surfaces:
            gmsh.model.mesh.setTransfiniteCurve(flap_data["l_left"], self.ny_flap + 1)
            gmsh.model.mesh.setTransfiniteCurve(flap_data["l_right"], self.ny_flap + 1)
            gmsh.model.mesh.setTransfiniteCurve(flap_data["l_top"], self.nx_flap + 1)
            gmsh.model.mesh.setTransfiniteCurve(flap_data["l_bottom"], self.nx_flap + 1)

            gmsh.model.mesh.setTransfiniteSurface(flap_data["surface"])
            gmsh.model.mesh.setRecombine(2, flap_data["surface"])

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        if self.quadratic:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

    def _create_mesh_model(self, MeshModelClass) -> "MeshModel":
        """Converts Gmsh mesh to MeshModel instance"""
        mesh_model = MeshModelClass()

        elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

        geometric_node_tags = set()
        for et, conn in zip(elementTypes, nodeTags):
            props = gmsh.model.mesh.getElementProperties(et)
            total_nodes = props[3]
            e_type = ELEMENT_NODES_MAP[total_nodes]
            if e_type in (ElementType.quad, ElementType.quad8, ElementType.quad9):
                num_corners = 4
            elif e_type in (ElementType.triangle, ElementType.triangle6):
                num_corners = 3
            else:
                raise ValueError(f"Unsupported element type: {e_type}")
            geometric_node_tags.update(
                nodeTags[0].reshape(-1, total_nodes)[:, :num_corners].flatten()
            )

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)

        self._gmsh_tag_to_id = {}
        for i, (tag, coord) in enumerate(zip(node_tags, coords)):
            is_geometric = tag in geometric_node_tags
            mesh_model.add_node(Node(coord, geometric_node=is_geometric))
            self._gmsh_tag_to_id[int(tag)] = i

        self._add_elements(mesh_model)
        self._create_node_sets(mesh_model)

        return mesh_model

    def _add_elements(self, mesh_model: "MeshModel"):
        """Add elements to the mesh model"""
        elem_types = gmsh.model.mesh.getElementTypes()
        for elem_type in elem_types:
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 2:
                _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
                num_nodes = elem_props[3]
                connectivity = elem_node_tags.reshape(-1, num_nodes)

                for nodes in connectivity:
                    node_objs = [
                        mesh_model.get_node_by_id(self._gmsh_tag_to_id[int(nt)]) for nt in nodes
                    ]

                    if len(node_objs) >= 4:
                        coords = [n.coords[:2] for n in node_objs[:4]]
                        area = 0.0
                        for i in range(4):
                            j = (i + 1) % 4
                            area += coords[i][0] * coords[j][1]
                            area -= coords[j][0] * coords[i][1]
                        if area < 0:
                            if len(node_objs) == 4:
                                node_objs = [node_objs[0], node_objs[3], node_objs[2], node_objs[1]]
                            elif len(node_objs) == 8:
                                node_objs = [
                                    node_objs[0],
                                    node_objs[3],
                                    node_objs[2],
                                    node_objs[1],
                                    node_objs[7],
                                    node_objs[6],
                                    node_objs[5],
                                    node_objs[4],
                                ]
                            elif len(node_objs) == 9:
                                node_objs = [
                                    node_objs[0],
                                    node_objs[3],
                                    node_objs[2],
                                    node_objs[1],
                                    node_objs[7],
                                    node_objs[6],
                                    node_objs[5],
                                    node_objs[4],
                                    node_objs[8],
                                ]

                    e_type = ELEMENT_NODES_MAP.get(len(nodes), ElementType.quad)
                    mesh_model.add_element(MeshElement(nodes=node_objs, element_type=e_type))

    def _create_node_sets(self, mesh_model: "MeshModel"):
        """Create node sets for boundaries"""
        physical_groups = gmsh.model.getPhysicalGroups()

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            if dim == 1 and name:
                node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0]
                node_ids = [self._gmsh_tag_to_id[int(nt)] for nt in node_tags]
                node_objs = {mesh_model.get_node_by_id(nid) for nid in node_ids}
                mesh_model.add_node_set(NodeSet(name=name, nodes=node_objs))

        all_nodes = {node for node in mesh_model.nodes}
        mesh_model.add_node_set(NodeSet(name="all", nodes=all_nodes))

        if "base_bottom" in mesh_model.node_sets:
            bottom_nodes = mesh_model.node_sets["base_bottom"].nodes
            mesh_model.add_node_set(NodeSet(name="bottom", nodes=set(bottom_nodes.values())))

    @classmethod
    def create_from_config(
        cls,
        n_flaps: int,
        flap_width: float,
        flap_height: float,
        x_spacing: float,
        base_height: float = 0.05,
        nx_flap: int = 4,
        ny_flap: int = 20,
        nx_base_segment: int = 10,
        ny_base: int = 2,
        quadratic: bool = False,
    ) -> "MeshModel":
        """Create multi-flap mesh from parameters"""
        return cls(
            n_flaps=n_flaps,
            flap_width=flap_width,
            flap_height=flap_height,
            x_spacing=x_spacing,
            base_height=base_height,
            nx_flap=nx_flap,
            ny_flap=ny_flap,
            nx_base_segment=nx_base_segment,
            ny_base=ny_base,
            quadratic=quadratic,
        ).generate()


class BladeMesh:
    """
    Generates shell meshes for wind turbine blades from YAML definitions.

    This generator uses the numad library to create parametric blade geometries
    and mesh them with shell elements. The resulting mesh includes element sets
    for different blade regions and sections.

    Attributes
    ----------
    yaml_file : str
        Path to the blade YAML definition file
    element_size : float
        Target element size for meshing
    n_samples : int
        Number of samples for airfoil discretization

    Examples
    --------
    >>> blade_mesh = BladeMesh("blade_definition.yaml", element_size=0.05)
    >>> mesh = blade_mesh.generate(renumber="rcm")
    >>> mesh.write_mesh("blade.vtk")
    """

    def __init__(
        self,
        yaml_file: str,
        element_size: float = 0.1,
        n_samples: int = 300,
    ):
        self.yaml_file = yaml_file
        self.element_size = element_size
        self.n_samples = n_samples
        self._numad_blade = None
        self._numad_mesh = None

    def generate(self, renumber: str | None = None, verbose: bool = True) -> "MeshModel":
        """
        Generate the blade mesh from the YAML definition.

        Parameters
        ----------
        renumber : str, optional
            Renumbering algorithm to apply after mesh generation.
            - None (default): No renumbering
            - "simple": Direct index assignment
            - "rcm": Reverse Cuthill-McKee for bandwidth reduction
        verbose : bool, optional
            Whether to print progress messages (default: True)

        Returns
        -------
        MeshModel
            The generated mesh model with nodes, elements, and sets
        """
        # Import here to avoid circular imports and optional dependency
        from fem_shell.core.mesh.model import MeshModel

        try:
            from fem_shell.models.blade.numad import Blade as numadBlade
            from fem_shell.models.blade.numad.mesh_gen import get_shell_mesh
        except ImportError as e:
            raise ImportError(
                "BladeMesh requires the numad module. "
                "Make sure fem_shell.models.blade.numad is available."
            ) from e

        mesh_model = MeshModel()

        # Initialize numad blade
        if verbose:
            print("  Loading blade definition...")
        self._numad_blade = numadBlade()
        self._numad_blade.read_yaml(self.yaml_file)

        # Resample airfoils
        for stat in self._numad_blade.definition.stations:
            stat.airfoil.resample(n_samples=self.n_samples)

        # Update geometry
        if verbose:
            print("  Updating blade geometry...")
        self._numad_blade.update_blade()

        # Expand trailing edge
        n_stations = self._numad_blade.geometry.coordinates.shape[2]
        min_TE_lengths = 0.001 * np.ones(n_stations)
        self._numad_blade.expand_blade_geometry_te(min_TE_lengths)

        # Generate shell mesh using numad
        if verbose:
            print("  Generating shell mesh...")
        self._numad_mesh = get_shell_mesh(self._numad_blade, self.element_size)

        num_raw_nodes = len(self._numad_mesh["nodes"])
        num_elements = len(self._numad_mesh["elements"])
        if verbose:
            print(f"  Raw mesh: {num_raw_nodes} nodes, {num_elements} elements")

        # Deduplicate nodes
        self._deduplicate_and_create_mesh(mesh_model, verbose)

        # Renumber mesh if requested
        if renumber is not None:
            if verbose:
                print(f"  Renumbering mesh using {renumber} algorithm...")
            mesh_model.renumber_mesh(algorithm=renumber)

        if verbose:
            print(
                f"  Blade mesh generated: {mesh_model.node_count} nodes, {mesh_model.elements_count} elements"
            )

        return mesh_model

    def _deduplicate_and_create_mesh(self, mesh_model: "MeshModel", verbose: bool = True):
        """
        Deduplicate nodes and create mesh elements.

        Numad may create duplicate nodes at region boundaries, so we need to
        deduplicate them using spatial tolerance matching.
        """
        from scipy.spatial import cKDTree

        from fem_shell.core.mesh.entities import ElementSet

        node_coords = self._numad_mesh["nodes"]
        num_raw_nodes = len(node_coords)
        num_elements = len(self._numad_mesh["elements"])

        if verbose:
            print("  Deduplicating nodes...")

        tolerance = 1e-6
        numad_to_unique = {}  # Maps numad node index to unique node index
        unique_nodes = []

        # Build KDTree for efficient nearest neighbor search
        all_coords = np.array(node_coords)
        tree = cKDTree(all_coords)

        # Find duplicate groups using the tree
        processed = set()
        for idx in range(num_raw_nodes):
            if idx in processed:
                continue

            neighbors = tree.query_ball_point(all_coords[idx], r=tolerance)
            unique_idx = len(unique_nodes)
            unique_nodes.append(all_coords[idx])

            for neighbor_idx in neighbors:
                numad_to_unique[neighbor_idx] = unique_idx
                processed.add(neighbor_idx)

        num_unique_nodes = len(unique_nodes)
        if verbose:
            reduction_pct = 100 * (num_raw_nodes - num_unique_nodes) // max(1, num_raw_nodes)
            print(
                f"  Deduplicated: {num_raw_nodes} -> {num_unique_nodes} nodes ({reduction_pct}% reduction)"
            )

        # Create nodes
        node_index_to_node = {}
        for unique_idx, unique_coord in enumerate(unique_nodes):
            node = Node(unique_coord)
            mesh_model.add_node(node)
            node_index_to_node[unique_idx] = node

        # Create elements
        if verbose:
            print(f"  Creating {num_elements} elements...")
        for node_ids in self._numad_mesh["elements"]:
            if node_ids[3] == -1:
                element_type = ElementType.triangle
                node_ids = node_ids[:3]
            else:
                element_type = ElementType.quad
            node_objs = [node_index_to_node[numad_to_unique[n_id]] for n_id in node_ids]
            mesh_model.add_element(MeshElement(nodes=node_objs, element_type=element_type))

        # Create element sets
        if verbose:
            print("  Creating element and node sets...")
        for element_set in self._numad_mesh["sets"]["element"]:
            name = element_set["name"]
            elements = {mesh_model.get_element_by_id(i) for i in element_set["labels"]}
            mesh_model.add_element_set(ElementSet(name=name, elements=elements))

        # Create node sets
        for node_set in self._numad_mesh["sets"]["node"]:
            name = node_set["name"]
            nodes = {node_index_to_node[numad_to_unique[i]] for i in node_set["labels"]}
            mesh_model.add_node_set(NodeSet(name=name, nodes=nodes))

        # Create node sets from element sets with "all" in name
        for element_set in [
            eset for eset in mesh_model.element_sets.values() if "all" in eset.name.lower()
        ]:
            name = element_set.name.replace("Els", "Nods")
            nodes = {node for element in element_set.elements for node in element.nodes}
            mesh_model.add_node_set(NodeSet(name=name, nodes=nodes))

    @property
    def numad_mesh_data(self):
        """
        Access the raw numad mesh data dictionary.

        This includes materials, sections, and other metadata from the blade definition.
        Only available after calling generate().

        Returns
        -------
        dict
            Raw mesh data from numad including nodes, elements, sets, materials, sections
        """
        if self._numad_mesh is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate() first.")
        return self._numad_mesh

    @property
    def numad_blade(self):
        """
        Access the numad blade object.

        Only available after calling generate().

        Returns
        -------
        numadBlade
            The numad Blade object with geometry and definition data
        """
        if self._numad_blade is None:
            raise RuntimeError("Blade has not been loaded yet. Call generate() first.")
        return self._numad_blade

    @classmethod
    def create_from_yaml(
        cls,
        yaml_file: str,
        element_size: float = 0.1,
        n_samples: int = 300,
        renumber: str = None,
    ) -> "MeshModel":
        """
        Convenience method to create blade mesh directly from YAML file.

        Parameters
        ----------
        yaml_file : str
            Path to the blade YAML definition file
        element_size : float, optional
            Target element size (default: 0.1)
        n_samples : int, optional
            Number of samples for airfoil discretization (default: 300)
        renumber : str, optional
            Renumbering algorithm ("simple", "rcm", or None)

        Returns
        -------
        MeshModel
            The generated mesh model
        """
        return cls(
            yaml_file=yaml_file,
            element_size=element_size,
            n_samples=n_samples,
        ).generate(renumber=renumber)


class RotorMesh:
    """
    Generates shell meshes for wind turbine rotors from blade YAML definitions.

    This generator creates a multi-blade rotor by generating a base blade mesh
    and then copying, translating, and rotating it for each blade position.

    The coordinate system follows wind turbine conventions:
    - Y-axis: Rotor rotation axis (blade span direction)
    - Z-axis: Radial direction (blades are offset in this direction)
    - X-axis: Tangential direction

    Attributes
    ----------
    yaml_file : str
        Path to the blade YAML definition file
    n_blades : int
        Number of blades in the rotor
    hub_radius : float or None
        Radial distance from rotation axis to blade root
    element_size : float
        Target element size for meshing
    n_samples : int
        Number of samples for airfoil discretization

    Examples
    --------
    >>> rotor = RotorMesh("blade.yaml", n_blades=3, hub_radius=1.5)
    >>> mesh = rotor.generate(renumber="rcm")
    >>> mesh.write_mesh("rotor.vtk")
    """

    def __init__(
        self,
        yaml_file: str,
        n_blades: int,
        hub_radius: float | None = None,
        element_size: float = 0.1,
        n_samples: int = 300,
    ):
        self.yaml_file = yaml_file
        self.n_blades = n_blades
        self.hub_radius = hub_radius
        self.element_size = element_size
        self.n_samples = n_samples
        self._blade_generator: BladeMesh = None

    def generate(self, renumber: str | None = None, verbose: bool = True) -> "MeshModel":
        """
        Generate the rotor mesh by creating and transforming blade meshes.

        Parameters
        ----------
        renumber : str, optional
            Renumbering algorithm to apply after mesh generation.
            - None (default): No renumbering
            - "simple": Direct index assignment
            - "rcm": Reverse Cuthill-McKee for bandwidth reduction
        verbose : bool, optional
            Whether to print progress messages (default: True)

        Returns
        -------
        MeshModel
            The generated rotor mesh model
        """
        from fem_shell.core.mesh.model import MeshModel

        rotor_mesh = MeshModel()

        # Generate base blade mesh
        if verbose:
            print("\nGenerating base blade mesh...")
        self._blade_generator = BladeMesh(
            yaml_file=self.yaml_file,
            element_size=self.element_size,
            n_samples=self.n_samples,
        )
        base_mesh = self._blade_generator.generate(renumber=None, verbose=verbose)

        if verbose:
            print(
                f"  Base blade: {base_mesh.node_count} nodes, {base_mesh.elements_count} elements"
            )

        # Get hub_radius from blade definition if not provided
        actual_hub_radius = self.hub_radius
        if actual_hub_radius is None:
            actual_hub_radius = self._blade_generator.numad_blade.definition.hub_diameter / 2
            if verbose:
                print(f"  Using hub radius from blade definition: {actual_hub_radius}")

        if verbose:
            print(f"\nGenerating rotor mesh with {self.n_blades} blades...")

        for i in range(self.n_blades):
            if verbose:
                print(f"  Processing blade {i + 1}/{self.n_blades}...")

            # Create a copy of the base blade mesh with renamed sets
            blade_mesh = self._create_blade_copy(base_mesh, i)

            # Apply radial translation (along Z axis) to offset blade from center
            if actual_hub_radius > 0:
                blade_mesh.translate_mesh(vector=(0, 0, 1), distance=actual_hub_radius)
                if verbose:
                    print(f"    Translated by Z={actual_hub_radius}")

            # Apply rotation around Y axis (rotor rotation axis) to distribute blades
            angle_rad = i * 2 * np.pi / self.n_blades
            if angle_rad > 0:
                blade_mesh.rotate_mesh(axis=(0, 1, 0), angle=angle_rad)
                if verbose:
                    print(f"    Rotated by {np.degrees(angle_rad):.1f}°")

            # Merge transformed blade mesh into rotor mesh
            self._merge_mesh(rotor_mesh, blade_mesh)

            if verbose:
                print(
                    f"    Rotor now has: {rotor_mesh.node_count} nodes, {rotor_mesh.elements_count} elements"
                )

        # Renumber mesh if requested
        if renumber is not None:
            if verbose:
                print(f"\nRenumbering mesh using {renumber} algorithm...")
            rotor_mesh.renumber_mesh(algorithm=renumber)

        if verbose:
            print(
                f"\nRotor mesh generated: {rotor_mesh.node_count} nodes, {rotor_mesh.elements_count} elements"
            )

        return rotor_mesh

    def _create_blade_copy(self, base_mesh: "MeshModel", blade_index: int) -> "MeshModel":
        """Create a copy of the base blade mesh with renamed sets."""
        from fem_shell.core.mesh.entities import ElementSet
        from fem_shell.core.mesh.model import MeshModel

        blade_mesh = MeshModel()
        node_map = {}

        # Copy nodes
        for node in base_mesh.nodes:
            new_node = Node([node.x, node.y, node.z])
            blade_mesh.add_node(new_node)
            node_map[node.id] = new_node

        # Copy elements
        element_map = {}
        for element in base_mesh.elements:
            new_nodes = [node_map[n.id] for n in element.nodes]
            new_element = MeshElement(new_nodes, element.element_type)
            blade_mesh.add_element(new_element)
            element_map[element.id] = new_element

        # Copy node sets with blade identifier
        for name, node_set in base_mesh.node_sets.items():
            new_set_name = f"{name}_blade_{blade_index + 1}"
            new_set_nodes = {node_map[n_id] for n_id in node_set.node_ids}
            blade_mesh.add_node_set(NodeSet(new_set_name, new_set_nodes))

        # Copy element sets with blade identifier
        for name, element_set in base_mesh.element_sets.items():
            new_set_name = f"{name}_blade_{blade_index + 1}"
            new_set_elements = {element_map[e.id] for e in element_set.elements}
            blade_mesh.add_element_set(ElementSet(new_set_name, new_set_elements))

        return blade_mesh

    def _merge_mesh(self, target_mesh: "MeshModel", source_mesh: "MeshModel") -> None:
        """Merge a source mesh into the target mesh."""
        from fem_shell.core.mesh.entities import ElementSet

        node_map = {}

        for node in source_mesh.nodes:
            new_node = Node([node.x, node.y, node.z])
            target_mesh.add_node(new_node)
            node_map[node.id] = new_node

        element_map = {}
        for element in source_mesh.elements:
            new_nodes = [node_map[n.id] for n in element.nodes]
            new_element = MeshElement(new_nodes, element.element_type)
            target_mesh.add_element(new_element)
            element_map[element.id] = new_element

        for name, node_set in source_mesh.node_sets.items():
            new_set_nodes = {node_map[n_id] for n_id in node_set.node_ids}
            target_mesh.add_node_set(NodeSet(name, new_set_nodes))

        for name, element_set in source_mesh.element_sets.items():
            new_set_elements = {element_map[e.id] for e in element_set.elements}
            target_mesh.add_element_set(ElementSet(name, new_set_elements))

    @property
    def numad_mesh_data(self):
        """Access raw numad mesh data (only available after generate())."""
        if self._blade_generator is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate() first.")
        return self._blade_generator.numad_mesh_data

    @property
    def numad_blade(self):
        """Access the numad blade object (only available after generate())."""
        if self._blade_generator is None:
            raise RuntimeError("Mesh has not been generated yet. Call generate() first.")
        return self._blade_generator.numad_blade

    @classmethod
    def create_from_yaml(
        cls,
        yaml_file: str,
        n_blades: int,
        hub_radius: float = None,
        element_size: float = 0.1,
        n_samples: int = 300,
        renumber: str = None,
    ) -> "MeshModel":
        """
        Convenience method to create rotor mesh directly from YAML file.

        Parameters
        ----------
        yaml_file : str
            Path to the blade YAML definition file
        n_blades : int
            Number of blades in the rotor
        hub_radius : float, optional
            Radial distance from rotation axis to blade root
        element_size : float, optional
            Target element size (default: 0.1)
        n_samples : int, optional
            Number of samples for airfoil discretization (default: 300)
        renumber : str, optional
            Renumbering algorithm ("simple", "rcm", or None)

        Returns
        -------
        MeshModel
            The generated rotor mesh model
        """
        return cls(
            yaml_file=yaml_file,
            n_blades=n_blades,
            hub_radius=hub_radius,
            element_size=element_size,
            n_samples=n_samples,
        ).generate(renumber=renumber)
