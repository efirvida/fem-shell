from abc import ABC, abstractmethod
from typing import Dict, List, Set

import meshio
import numpy as np

from fem_shell.core.assembler import MeshAssembler
from fem_shell.core.bc import BodyForce, DirichletCondition
from fem_shell.core.mesh import MeshModel
from fem_shell.core.viewer import plot_results
from fem_shell.elements import ElementFamily


class Solver(ABC):
    """
    Abstract base class for finite element method (FEM) solvers.

    Parameters
    ----------
    mesh : MeshModel
        The mesh model used for the simulation.
    fem_model_properties : dict
        Dictionary containing FEM model properties. Required keys:
            - 'material': Material properties.
            - 'element_family': Element family used in the simulation.
            - 'thickness': Required for SHELL elements.

    Attributes
    ----------
    material : Any
        Material properties.
    element_family : Any
        The element family used.
    thickness : Any, optional
        The thickness of the shell (only defined if element_family is SHELL).
    mesh_obj : MeshModel
        The mesh model.
    model_properties : dict
        FEM model properties.
    domain : MeshAssembler
        Assembled domain based on the mesh and model properties.
    dirichlet_conditions : List[DirichletCondition]
        List of Dirichlet boundary conditions.
    neumann_conditions : List[NeumannCondition]
        List of Neumann boundary conditions.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        # Validate required keys in fem_model_properties
        if "solver" not in fem_model_properties:
            raise KeyError("Missing 'solver' settings in fem_model_properties.")
        self.solver_params = fem_model_properties["solver"]

        if "elements" not in fem_model_properties:
            raise KeyError("Missing 'elements' settings in fem_model_properties.")

        if "material" not in fem_model_properties["elements"]:
            raise KeyError(
                "The key 'material' is missing in fem_model_properties. It is required to define the material properties."
            )
        if "element_family" not in fem_model_properties["elements"]:
            raise KeyError(
                "The key 'element_family' is missing in fem_model_properties. It is required to define the element family."
            )

        self.material = fem_model_properties["elements"]["material"]
        self.element_family = fem_model_properties["elements"]["element_family"]

        # Additional validation for SHELL elements
        if self.element_family == ElementFamily.SHELL:
            if "thickness" not in fem_model_properties["elements"]:
                raise KeyError(
                    "The key 'thickness' is missing in fem_model_properties. It is required for SHELL elements."
                )
            self.thickness = fem_model_properties["elements"]["thickness"]

        self.mesh_obj = mesh
        self.model_properties = fem_model_properties
        self.domain = MeshAssembler(mesh=self.mesh_obj, model=self.model_properties)
        self.dirichlet_conditions: List[DirichletCondition] = []
        self.body_forces: List[BodyForce] = []

    def get_dofs_by_nodeset_name(self, name: str, only_geometric_dofs: bool = True) -> Set[int]:
        """
        Retrieve the degrees of freedom (DOFs) associated with a given node set.

        Parameters
        ----------
        name : str
            Name of the node set.

        Returns
        -------
        Set[int]
            Set of DOFs corresponding to the node set.
        """
        return self.domain.get_dofs_by_nodeset(name, only_geometric_dofs)

    def get_nodeids_by_nodeset_name(self, name: str) -> Set[int]:
        """
        Retrieve the node IDs associated with a given node set.

        Parameters
        ----------
        name : str
            Name of the node set.

        Returns
        -------
        Set[int]
            Set of node IDs corresponding to the node set.
        """
        return self.mesh_obj.get_node_set(name).node_ids

    def add_dirichlet_conditions(self, bcs: List[DirichletCondition]) -> None:
        """
        Add Dirichlet boundary conditions to the solver.

        Parameters
        ----------
        bcs : List[DirichletCondition]
            List of Dirichlet boundary conditions.
        """
        self.dirichlet_conditions = bcs

    def add_body_forces(self, bcs: List[BodyForce]) -> None:
        """
        Add Neumann boundary conditions to the solver.

        Parameters
        ----------
        bcs : List[NeumannCondition]
            List of Neumann boundary conditions.
        """
        self.body_forces = bcs

    def write_results(self, output_file: str) -> None:
        """
        Write the simulation results to a VTK file.

        This method combines vector components from the domain's vector form and calculates
        the magnitude for each vector field. Both the individual components and the vector
        fields (with their magnitude) are written to the VTK file.

        Parameters
        ----------
        output_file : str
            Path to the output VTK file.

        Raises
        ------
        AttributeError
            If `self.u` is not defined. Ensure that the solver has computed the solution
            (i.e., `self.u` is set) before calling this method.
        """
        # 'vector_form' is assumed to be a dict mapping vector field names to lists of component names.
        vector_form = self.domain.vector_form
        vector_components = [comp for vector in vector_form.values() for comp in vector]
        U = self.u.reshape(-1, self.domain.dofs_per_node)

        points = self.mesh_obj.coords_array
        point_data: Dict[str, np.ndarray] = {}

        # Store individual vector components as scalar fields
        for i, comp in enumerate(vector_components):
            point_data[comp] = U[:, i]

        # Combine components to form vector fields and compute their magnitudes
        for vector, components in vector_form.items():
            # Create an array with shape (n_points, n_components)
            vector_array = np.column_stack([point_data[comp] for comp in components])

            if vector_array.shape[1] == 2:
                zeros = np.zeros(vector_array.shape[0])
                vector_array = np.column_stack([vector_array, zeros])

            point_data[vector] = vector_array

        # Assemble cells from the mesh element map
        cells = []
        for element in self.mesh_obj.element_map.values():
            if element:
                cells.append((
                    element.element_type.name,
                    [element.node_ids],
                ))

        mesh_object = meshio.Mesh(points, cells=cells, point_data=point_data)
        mesh_object.write(output_file, file_format="vtk")

    def view_results(self):
        # 'vector_form' is assumed to be a dict mapping vector field names to lists of component names.
        vector_form = self.domain.vector_form
        vector_components = [comp for vector in vector_form.values() for comp in vector]
        U = self.u.reshape(-1, len(vector_components))  # self.u must be defined after solving

        point_data: Dict[str, np.ndarray] = {}

        # Store individual vector components as scalar fields
        for i, comp in enumerate(vector_components):
            point_data[comp] = U[:, i]

        # Combine components to form vector fields and compute their magnitudes
        for vector, components in vector_form.items():
            # Create an array with shape (n_points, n_components)
            vector_array = np.column_stack([point_data[comp] for comp in components])

            # If the vector only has two components, add a third column of zeros.
            if vector_array.shape[1] == 2:
                zeros = np.zeros(vector_array.shape[0])
                vector_array = np.column_stack([vector_array, zeros])

            point_data[vector] = vector_array

        plot_results(self.mesh_obj, point_data)

    @abstractmethod
    def solve(self):
        """
        Solve the FEM problem.

        This abstract method must be implemented by subclasses to perform the solution process.
        """
        ...
