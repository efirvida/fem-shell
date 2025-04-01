from __future__ import absolute_import

from typing import Dict, Literal, Sequence

import numpy as np

from fem_shell.core.mesh import MeshModel
from fem_shell.elements import ElementFactory


class MeshAssembler:
    """Assembles global matrices and vectors from element contributions.

    Attributes
    ----------
    mesh : MeshModel
        The mesh containing the elements to assemble.
    model : Dict
        Material properties and other model-specific parameters.
    _element_map : Dict[int, FemElement]
        Maps element IDs to their corresponding finite element objects.
    _element_global_dofs_map : Dict[int, list]
        Maps element IDs to their corresponding global DOF indices.
    _dofs_count : int
        Total number of global degrees of freedom.
    """

    def __init__(self, mesh: MeshModel, model: Dict):
        """
        Initializes the MeshAssembler.

        Parameters
        ----------
        mesh : MeshModel
            The mesh containing elements to assemble.
        element_family : ElementFamily
            The family of elements to be used.
        model : Dict
            Model-specific parameters like material properties.
        """
        self.mesh = mesh
        self.element_family = model["elements"]["element_family"]
        self.model = model["elements"]
        self.simple_element_definition = self.simple_model()

        self._element_map = {}
        self._element_global_dofs_map = {}
        self._nodes_global_dofs_map: Dict[int, Sequence[int]] = {}
        self.dofs_count: int = 0
        self._dofs_per_node: int = 0
        self._spatial_dim: int = 0
        self.vector_form: Dict = {}

        self._precompute_elements()

    def simple_model(self):
        """
        Determines if the model is simple (direct material definition) or complex (element-based definition).

        Returns
        -------
        bool
            - True: If the model is simple (direct material definition).
            - False: If the model is complex (element-based definition).

        Raises
        ------
        ValueError
            - If the model has an unknown behavior.
            - If the number of elements in the model does not match the mesh element count.
        """
        if not isinstance(self.model, dict):
            raise ValueError("self.model must be a dictionary.")

        if "material" in self.model and "element_family" in self.model:
            return True
        elif all(isinstance(key, int) for key in self.model):
            if not hasattr(self, "mesh") or not hasattr(self.mesh, "elements_count"):
                raise AttributeError("self.mesh or self.mesh.elements_count is not defined.")
            if len(self.model) != self.mesh.elements_count:
                raise ValueError(
                    "Model with element wise definition does not match element count. "
                    "You must have a model for each element."
                )
            return False
        else:
            raise ValueError("Unknown model behavior.")

    def _precompute_elements(self) -> None:
        """Precomputes the finite element objects and global DOF mappings."""
        if not self._element_map:
            for element in self.mesh.elements:
                if self.simple_element_definition:
                    model = self.model
                else:
                    model = self.model[element.id]

                fem_element = ElementFactory.get_element(mesh_element=element, **model)
                if fem_element:
                    if not self._dofs_per_node:
                        self._dofs_per_node = fem_element.dofs_per_node
                    if not self._spatial_dim:
                        self._spatial_dim = fem_element.spatial_dimmension

                    self.vector_form = fem_element.vector_form
                    self._element_map[element.id] = fem_element
                    self._element_global_dofs_map[element.id] = [
                        dof for node in fem_element.global_dof_indices.values() for dof in node
                    ]
                    self._nodes_global_dofs_map.update(fem_element.global_dof_indices)

            # Count the total number of unique global DOFs
            self.dofs_count = len({
                dof for dofs in self._element_global_dofs_map.values() for dof in dofs
            })

    @property
    def dofs_per_node(self) -> int:
        return self._dofs_per_node

    @property
    def spatial_dim(self) -> Literal[2] | Literal[3]:
        return self._spatial_dim

    def _assemble_matrix(self, attribute: str) -> np.ndarray:
        """
        Generalized assembly function for global matrices or vectors.

        Parameters
        ----------
        attribute : str
            Name of the local attribute to assemble ('K', 'M', or 'F').
        is_matrix : bool, optional
            True if assembling a matrix (default), False if assembling a vector.

        Returns
        -------
        np.ndarray
            Assembled global matrix or vector.
        """
        # Initialize global container (matrix or vector)
        global_container = np.zeros((self.dofs_count, self.dofs_count))

        # Loop over each element
        for element_id, element in self._element_map.items():
            # Get global DOFs for the current element
            global_dofs = self._element_global_dofs_map[element_id]
            # Get local matrix or vector
            local_contribution = getattr(element, attribute)

            for i, global_i in enumerate(global_dofs):
                for j, global_j in enumerate(global_dofs):
                    global_container[global_i, global_j] += local_contribution[i, j]

        return global_container

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """Assembles the global stiffness matrix.

        Returns
        -------
        np.ndarray
            Global stiffness matrix.
        """
        return self._assemble_matrix("K")

    def assemble_mass_matrix(self) -> np.ndarray:
        """Assembles the global mass matrix.

        Returns
        -------
        np.ndarray
            Global mass matrix.
        """
        return self._assemble_matrix("M")

    def assemble_load_vector(self, load_condition) -> np.ndarray:
        """
        Ensambla el vector global de fuerzas integrando la carga en cada elemento.

        Parámetros
        ----------
        f_vec : array, shape (2,)
            Vector de carga volumétrica, por ejemplo [0, -rho * g].

        Retorna
        -------
        f_global : array, shape (ndof,)
        """
        # FIXME Add, support for surface load, this need to have surface load vector in element
        kind = "body_force"

        load_vector = load_condition.value

        global_container = np.zeros(self.dofs_count)
        for element_id, element in self._element_map.items():
            # Get global DOFs for the current element
            global_dofs = self._element_global_dofs_map[element_id]
            # Get local matrix or vector
            if kind == "body_force":
                local_contribution = element.body_load(load_vector)

            global_container[global_dofs] += local_contribution

        return global_container

    def get_dofs_by_nodeset(self, name: str, only_geometric: bool = True):
        nodes_set = self.mesh.get_node_set(name)
        if only_geometric:
            return {
                dof
                for node in nodes_set.nodes.values()
                for dof in self._nodes_global_dofs_map[node.id]
                if node.geometric_node
            }
        else:
            return {
                dof
                for node in nodes_set.nodes.values()
                for dof in self._nodes_global_dofs_map[node.id]
            }
