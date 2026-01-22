from enum import IntEnum
from typing import Dict, Iterable, List, Literal, Sequence, Tuple, Union

import numpy as np

from fem_shell.core.material import MaterialType as Material
from fem_shell.core.mesh import MeshElement


class ElementFamily(IntEnum):
    SHELL = 2
    PLANE = 3
    SOLID = 4


class FemElement:
    _id_counter: int = 0
    vector_form: Dict = {}

    def __init__(
        self,
        name: str,
        node_coords: Iterable[Union[float, np.ndarray]],
        node_ids: List[int],
        material: Material,
        dofs_per_node: int,
    ):
        self.name = name
        self.node_coords = node_coords
        self.node_ids = node_ids
        self.material = material
        self.dofs_per_node = dofs_per_node
        self.node_count = len(self.node_coords)
        self.dofs_count = self.node_count * self.dofs_per_node
        self.element_family: ElementFamily = None
        self.id = FemElement._id_counter
        self.integration_order = 1
        FemElement._id_counter += 1

    @property
    def spatial_dimmension(self) -> Literal[2] | Literal[3]:
        if self.element_family == ElementFamily.PLANE:
            return 2
        elif self.element_family == ElementFamily.SHELL:
            return 3
        elif self.element_family == ElementFamily.SOLID:
            return 3
        return 2

    @property
    def global_dof_indices(self) -> Dict[int, Tuple[int]]:
        """
        Devuelve los índices globales de los grados de libertad (DOFs) asociados a los nodos del elemento.

        Returns:
            List[int]: Lista de índices globales de los DOFs.
        """
        global_dof_indices = {}
        for node_id in self.node_ids:
            start_dof = node_id * self.dofs_per_node
            end_dof = start_dof + self.dofs_per_node
            global_dof_indices[node_id] = tuple(range(start_dof, end_dof))

        return global_dof_indices

    def __repr__(self):
        return f"<Element id={self.id} name={self.name}>"


class ShellElement(FemElement):
    def __init__(
        self,
        name: str,
        node_coords: Union[Sequence[float], np.ndarray],
        node_ids: Tuple[int, int, int, int],
        material: Material,
        dofs_per_node: int,
        thickness: float,
        nonlinear: bool,
    ):
        super().__init__(name, node_coords, node_ids, material, dofs_per_node)
        self.thickness = thickness
        self.nonlinear = nonlinear
        self.element_family = ElementFamily.SHELL

    def __repr__(self):
        return f"<ShellElement id={self.id} name={self.name} thickens={self.thickens}>"


class PlaneElement(FemElement):
    def __init__(
        self,
        name: str,
        node_coords: Union[Sequence[float], np.ndarray],
        node_ids: List[int],
        material: Material,
        dofs_per_node: int,
    ):
        super().__init__(name, node_coords, node_ids, material, dofs_per_node)
        self.element_family = ElementFamily.PLANE

    def __repr__(self):
        return f"<PlaneElement id={self.id} name={self.name}>"


class ElementFactory:
    """
    Factory for creating finite element instances based on mesh and configuration.

    The factory automatically selects the appropriate element type based on:
    - Element family (SHELL, PLANE, or SOLID)
    - Number of nodes (3 for triangular, 4 for quadrilateral, etc.)
    - Material type (isotropic Material or Laminate for composites)
    - Analysis type (linear or nonlinear)

    Examples
    --------
    >>> # Isotropic shell element
    >>> elem = ElementFactory.get_element(
    ...     element_family=ElementFamily.SHELL,
    ...     mesh_element=mesh_elem,
    ...     material=material,
    ...     thickness=0.01
    ... )

    >>> # Composite shell element (auto-detected from laminate)
    >>> elem = ElementFactory.get_element(
    ...     element_family=ElementFamily.SHELL,
    ...     mesh_element=mesh_elem,
    ...     laminate=laminate  # MITC4Composite or MITC3Composite used
    ... )

    >>> # Solid element (3D volumetric)
    >>> elem = ElementFactory.get_element(
    ...     element_family=ElementFamily.SOLID,
    ...     mesh_element=mesh_elem,
    ...     material=material  # Isotropic or Orthotropic
    ... )

    >>> # Nonlinear analysis
    >>> elem = ElementFactory.get_element(
    ...     element_family=ElementFamily.SHELL,
    ...     mesh_element=mesh_elem,
    ...     material=material,
    ...     thickness=0.01,
    ...     nonlinear=True
    ... )
    """

    @staticmethod
    def get_element(
        element_family: ElementFamily, mesh_element: MeshElement, **kwargs
    ) -> FemElement | bool:
        """
        Create a finite element instance for the given mesh element.

        Parameters
        ----------
        element_family : ElementFamily
            The element family (SHELL, PLANE, or SOLID)
        mesh_element : MeshElement
            The mesh element containing node coordinates and IDs
        **kwargs : dict
            Additional parameters passed to element constructor:

            For isotropic shell elements:
            - material : Material
                Isotropic material properties
            - thickness : float
                Shell thickness
            - nonlinear : bool, optional
                Enable geometric nonlinear analysis (default False)

            For composite shell elements:
            - laminate : Laminate
                Laminate definition (auto-selects composite element)
            - nonlinear : bool, optional
                Enable geometric nonlinear analysis (default False)

            For plane elements:
            - material : Material
                Material properties

            For solid elements:
            - material : Material
                Isotropic or Orthotropic material properties
            - orientation : np.ndarray, optional
                3×3 rotation matrix for orthotropic material orientation

        Returns
        -------
        FemElement or False
            The created element instance, or False if element type not supported

        Notes
        -----
        Element selection logic:

        **Shell Elements (by node count):**
        - 3 nodes: MITC3 (isotropic) or MITC3Composite (laminate)
        - 4 nodes: MITC4 (isotropic) or MITC4Composite (laminate)

        **Plane Elements (by node count):**
        - 4 nodes: QUAD4
        - 8 nodes: QUAD8
        - 9 nodes: QUAD9

        **Solid Elements (by node count):**
        - 4 nodes: TETRA4
        - 5 nodes: PYRAMID5
        - 6 nodes: WEDGE6
        - 8 nodes: HEXA8
        - 10 nodes: TETRA10
        - 13 nodes: PYRAMID13
        - 15 nodes: WEDGE15
        - 20 nodes: HEXA20

        The composite variant is automatically selected when `laminate`
        parameter is provided instead of `material` + `thickness`.
        """
        from .MITC3 import MITC3
        from .MITC3_composite import MITC3Composite
        from .MITC4 import MITC4
        from .MITC4_composite import MITC4Composite
        from .QUAD import QUAD4, QUAD8, QUAD9
        from .SOLID import HEXA8, HEXA20, PYRAMID5, PYRAMID13, TETRA4, TETRA10, WEDGE6, WEDGE15

        # Check if this is a composite element (laminate provided)
        laminate = kwargs.pop("laminate", None)
        is_composite = laminate is not None

        # Shell element maps
        SHELL_ELEMENT_MAP = {3: MITC3, 4: MITC4}
        SHELL_COMPOSITE_MAP = {3: MITC3Composite, 4: MITC4Composite}

        # Plane element map
        PLANE_ELEMENT_MAP = {4: QUAD4, 8: QUAD8, 9: QUAD9}

        # Solid element map
        SOLID_ELEMENT_MAP = {
            4: TETRA4,
            5: PYRAMID5,
            6: WEDGE6,
            8: HEXA8,
            10: TETRA10,
            13: PYRAMID13,
            15: WEDGE15,
            20: HEXA20,
        }

        node_ids = mesh_element.node_ids
        node_coords = mesh_element.node_coords
        node_count = mesh_element.node_count

        try:
            if element_family == ElementFamily.SHELL:
                if is_composite:
                    # Composite shell element
                    element_class = SHELL_COMPOSITE_MAP[node_count]
                    return element_class(
                        node_coords=node_coords, node_ids=node_ids, laminate=laminate, **kwargs
                    )
                else:
                    # Isotropic shell element
                    element_class = SHELL_ELEMENT_MAP[node_count]
                    return element_class(node_coords=node_coords, node_ids=node_ids, **kwargs)

            elif element_family == ElementFamily.PLANE:
                element_class = PLANE_ELEMENT_MAP[node_count]
                return element_class(node_coords=node_coords, node_ids=node_ids, **kwargs)

            elif element_family == ElementFamily.SOLID:
                element_class = SOLID_ELEMENT_MAP[node_count]
                return element_class(node_coords=node_coords, node_ids=node_ids, **kwargs)
            else:
                raise NotImplementedError(f"Element family {element_family} not implemented")

        except KeyError:
            return False
