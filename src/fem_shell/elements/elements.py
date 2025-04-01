from enum import IntEnum
from typing import Dict, Iterable, List, Literal, Sequence, Tuple, Union

import numpy as np

from fem_shell.core.material import Material
from fem_shell.core.mesh import MeshElement


class ElementFamily(IntEnum):
    SHELL = 2
    PLANE = 3


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
    ):
        super().__init__(name, node_coords, node_ids, material, dofs_per_node)
        self.thickness = thickness
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
    @staticmethod
    def get_element(
        element_family: ElementFamily, mesh_element: MeshElement, **kwargs
    ) -> FemElement | bool:
        from .MITC4 import MITC4, MITC4Layered
        from .QUAD import QUAD4, QUAD8, QUAD9

        SHELL_ELEMENT_MAP = {4: MITC4}
        LAYERED_SHELL_ELEMENT_MAP = {4: MITC4Layered}
        PLANE_ELEMENT_MAP = {4: QUAD4, 8: QUAD8, 9: QUAD9}

        node_ids = mesh_element.node_ids
        node_coords = mesh_element.node_coords
        try:
            if element_family == ElementFamily.SHELL:
                if "layers" in kwargs:
                    element = LAYERED_SHELL_ELEMENT_MAP[mesh_element.node_count]
                else:
                    element = SHELL_ELEMENT_MAP[mesh_element.node_count]
            elif element_family == ElementFamily.PLANE:
                element = PLANE_ELEMENT_MAP[mesh_element.node_count]
            else:
                raise NotImplementedError
            return element(node_coords=node_coords, node_ids=node_ids, **kwargs)
        except KeyError:
            return False
