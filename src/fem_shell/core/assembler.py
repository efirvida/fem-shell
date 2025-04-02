from __future__ import absolute_import

from typing import Dict, Sequence

import numpy as np

from fem_shell.core.mesh import MeshModel
from fem_shell.elements import ElementFactory


class MeshAssembler:
    """Optimized assembler for global matrices and vectors."""

    __slots__ = [
        "_element_global_dofs_map",
        "_element_ix_indices",
        "_element_map",
        "_nodes_global_dofs_map",
        "dofs_count",
        "dofs_per_node",
        "element_family",
        "mesh",
        "model",
        "simple_element_definition",
        "spatial_dim",
        "vector_form",
    ]

    def __init__(self, mesh: MeshModel, model: Dict):
        self.mesh = mesh
        self.element_family = model["elements"]["element_family"]
        self.model = model["elements"]
        self.simple_element_definition = self._validate_model()

        # Inicialización optimizada de atributos
        self._element_map: Dict[int, FemElement] = {}
        self._element_global_dofs_map: Dict[int, np.ndarray] = {}
        self._nodes_global_dofs_map: Dict[int, Sequence[int]] = {}
        self.dofs_count: int = 0
        self.dofs_per_node: int = 0
        self.spatial_dim: int = 0
        self.vector_form: Dict = {}
        self._element_ix_indices: Dict[int, tuple] = {}

        self._precompute_elements()

    def _validate_model(self):
        """Validación optimizada del modelo."""
        if not isinstance(self.model, dict):
            raise ValueError("Model must be a dictionary.")

        if "material" in self.model and "element_family" in self.model:
            return True
        elif all(isinstance(k, int) for k in self.model):
            if len(self.model) != self.mesh.elements_count:
                raise ValueError("Model/Element count mismatch.")
            return False
        raise ValueError("Invalid model structure.")

    def _precompute_elements(self) -> None:
        """Precomputación optimizada de elementos y DOFs."""
        if not self.mesh.elements:
            return

        # Variables locales para acceso rápido
        simple_model = self.simple_element_definition
        element_models = (
            self.model if simple_model else {e.id: self.model[e.id] for e in self.mesh.elements}
        )

        dofs_per_node = spatial_dim = 0
        element_map = {}
        element_dofs_map = {}
        nodes_dofs_map = {}

        for element in self.mesh.elements:
            # Creación optimizada de elementos
            fem_element = ElementFactory.get_element(
                mesh_element=element,
                **element_models[element.id] if not simple_model else self.model,
            )

            if not fem_element:
                continue

            # Actualización de parámetros una sola vez
            if not dofs_per_node:
                dofs_per_node = fem_element.dofs_per_node
                spatial_dim = fem_element.spatial_dimmension
                self.vector_form = fem_element.vector_form

            # Almacenamiento optimizado de DOFs
            element_dofs = np.array(
                [dof for node in fem_element.global_dof_indices.values() for dof in node],
                dtype=np.int64,
            )

            element_map[element.id] = fem_element
            element_dofs_map[element.id] = element_dofs

            # Actualización no redundante de nodos
            for nid, dofs in fem_element.global_dof_indices.items():
                if nid not in nodes_dofs_map:
                    nodes_dofs_map[nid] = dofs

        # Asignación final de atributos
        self._element_map = element_map
        self._element_global_dofs_map = element_dofs_map
        self._nodes_global_dofs_map = nodes_dofs_map
        self.dofs_per_node = dofs_per_node
        self.spatial_dim = spatial_dim
        self.dofs_count = self.mesh.node_count * dofs_per_node

        # Precomputación de índices para operaciones matriciales
        self._precompute_ix_indices()

    def _precompute_ix_indices(self):
        """Precalcula índices para operaciones matriciales."""
        for eid, dofs in self._element_global_dofs_map.items():
            self._element_ix_indices[eid] = np.ix_(dofs, dofs)

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """Ensamblaje optimizado de matriz de rigidez."""
        K = np.zeros((self.dofs_count, self.dofs_count))
        for eid, element in self._element_map.items():
            ix = self._element_ix_indices[eid]
            K[ix] += element.K
        return K

    def assemble_mass_matrix(self) -> np.ndarray:
        """Ensamblaje optimizado de matriz de masa."""
        M = np.zeros((self.dofs_count, self.dofs_count))
        for eid, element in self._element_map.items():
            ix = self._element_ix_indices[eid]
            M[ix] += element.M
        return M

    def assemble_load_vector(self, load_condition) -> np.ndarray:
        """Ensamblaje optimizado de vector de cargas."""
        f_global = np.zeros(self.dofs_count)
        load_vector = load_condition.value

        for eid, element in self._element_map.items():
            dofs = self._element_global_dofs_map[eid]
            f_global[dofs] += element.body_load(load_vector)

        return f_global

    def get_dofs_by_nodeset(self, name: str, only_geometric: bool = True):
        nodes_set = self.mesh.get_node_set(name)
        nodes = nodes_set.nodes.values()  # Acceso local a los nodos
        global_dofs_map = self._nodes_global_dofs_map  # Acceso local al mapeo de DOFs

        if only_geometric:
            return {
                dof for node in nodes if node.geometric_node for dof in global_dofs_map[node.id]
            }
        else:
            return {dof for node in nodes for dof in global_dofs_map[node.id]}
