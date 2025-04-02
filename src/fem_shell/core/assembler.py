from typing import Dict

import numpy as np
from cython_assembler import assemble_global_matrix, assemble_global_vector

from fem_shell.core.mesh import MeshModel
from fem_shell.elements import ElementFactory, FemElement


class MeshAssembler:
    def __init__(self, mesh: MeshModel, model: Dict):
        self.mesh = mesh
        self.model = model["elements"]
        self.element_family = self.model["element_family"]
        self._element_map: Dict[int, FemElement] = {}
        self._dofs_array: np.ndarray = None
        self._ke_array: np.ndarray = None  # Matrices de rigidez locales
        self._me_array: np.ndarray = None  # Matrices de masa locales
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.dofs_count: int = 0
        self.vector_form: Dict = {}
        self._precompute_elements()

    def _precompute_elements(self):
        """Precomputa elementos y almacena datos en arrays."""
        elements = self.mesh.elements
        if not elements:
            return

        # Inicialización de arrays
        dofs_list = []
        ke_list = []
        me_list = []

        # Llenar datos
        for element in elements:
            fem_element = ElementFactory.get_element(mesh_element=element, **self.model)
            if not fem_element:
                continue

            dofs = np.array(
                [dof for node in fem_element.global_dof_indices.values() for dof in node],
                dtype=np.int64,
            )

            self._element_map[element.id] = fem_element
            dofs_list.append(dofs)
            ke_list.append(fem_element.K)
            me_list.append(fem_element.M)

            # Actualizar parámetros una vez
            if not self.dofs_per_node:
                self.dofs_per_node = fem_element.dofs_per_node
                self.spatial_dim = fem_element.spatial_dimmension
                self.vector_form = fem_element.vector_form

        # Convertir a arrays numpy
        self._dofs_array = np.array(dofs_list, dtype=np.int64)
        self._ke_array = np.array(ke_list, dtype=np.float64)
        self._me_array = np.array(me_list, dtype=np.float64)
        self.dofs_count = self.mesh.node_count * self.dofs_per_node

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """Ensambla matriz de rigidez usando Cython."""
        K = np.zeros((self.dofs_count, self.dofs_count))
        assemble_global_matrix(K, self._dofs_array, self._ke_array)
        return K

    def assemble_mass_matrix(self) -> np.ndarray:
        """Ensambla matriz de masa usando Cython."""
        M = np.zeros((self.dofs_count, self.dofs_count))
        assemble_global_matrix(M, self._dofs_array, self._me_array)
        return M

    def assemble_load_vector(self, load_condition) -> np.ndarray:
        """Ensambla vector de carga usando Cython."""
        f_global = np.zeros(self.dofs_count)
        load_vector = load_condition.value
        fe_list = [self._element_map[eid].body_load(load_vector) for eid in self._element_map]
        fe_array = np.array(fe_list, dtype=np.float64)
        assemble_global_vector(f_global, self._dofs_array, fe_array)
        return f_global
