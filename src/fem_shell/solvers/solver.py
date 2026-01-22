from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

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
        self._vector_form: Dict = {}

    @property
    def vector_form(self) -> Dict[str, Tuple]:
        if not self._vector_form:
            self._vector_form = self.domain._element_map[0].vector_form
        return self._vector_form

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

        Notes
        -----
        Uses mesh.node_id_to_index mapping to handle meshes with non-consecutive
        node IDs (e.g., merged rotor meshes).
        """
        dofs_per_node = self.domain.dofs_per_node
        node_ids = self.get_nodeids_by_nodeset_name(name)
        node_id_to_index = self.mesh_obj.node_id_to_index

        # Use node index (not node ID) to calculate DOF positions
        return {
            dof
            for node_id in node_ids
            for dof in range(
                node_id_to_index[node_id] * dofs_per_node,
                (node_id_to_index[node_id] + 1) * dofs_per_node,
            )
        }

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
        self.dirichlet_conditions.extend(bcs)

    def add_body_forces(self, bcs: List[BodyForce]) -> None:
        """
        Add Neumann boundary conditions to the solver.

        Parameters
        ----------
        bcs : List[NeumannCondition]
            List of Neumann boundary conditions.
        """
        self.body_forces = bcs

    # =========================================================================
    # Stress Recovery / Post-processing
    # =========================================================================
    def compute_stresses(
        self,
        location: str = "middle",
        stress_type: str = "total",
        nodal: bool = True,
        smoothing: str = "average",
    ):
        """
        Compute stress field from the displacement solution.

        This method provides post-processing to recover stresses from
        the FEM displacement solution. For shell elements, stresses can
        be computed at different through-thickness locations.

        Parameters
        ----------
        location : str, optional
            Location through the shell thickness:
            - "top": Top surface (z = +h/2)
            - "middle": Mid-surface (z = 0)
            - "bottom": Bottom surface (z = -h/2)
            Default is "middle".
        stress_type : str, optional
            Type of stress contribution:
            - "membrane": Only membrane (in-plane) stresses
            - "bending": Only bending stresses
            - "total": Combined membrane + bending
            Default is "total".
        nodal : bool, optional
            If True, compute smoothed nodal stresses (averaging from adjacent elements).
            If False, compute element-centroid stresses.
            Default is True.
        smoothing : str, optional
            Smoothing method for nodal stresses:
            - "average": Simple averaging
            - "area_weighted": Weight by element area
            Default is "average".

        Returns
        -------
        StressResult
            A dataclass containing:
            - sigma_xx, sigma_yy, sigma_xy: Stress components in Voigt notation
            - von_mises: Von Mises equivalent stress
            - sigma_1, sigma_2: Principal stresses
            - tau_max: Maximum shear stress
            - principal_angle: Angle of principal direction from x-axis

        Raises
        ------
        AttributeError
            If the solution (self.u) has not been computed yet.
        ValueError
            If an invalid location or stress_type is specified.

        Examples
        --------
        >>> solver.solve()
        >>> stress_result = solver.compute_stresses(location="top")
        >>> print(stress_result.von_mises.max())

        Notes
        -----
        The von Mises stress for plane stress is computed as:
            σ_vm = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

        For shell elements, total stress at location z is:
            σ_total = σ_membrane + z · κ · E / (1 - ν²)
        where κ is the curvature.
        """
        from .stress_recovery import StressLocation, StressRecovery, StressType

        if not hasattr(self, "u") or self.u is None:
            raise AttributeError(
                "No displacement solution found. Run solve() before computing stresses."
            )

        # Map string inputs to enums
        location_map = {
            "top": StressLocation.TOP,
            "middle": StressLocation.MIDDLE,
            "bottom": StressLocation.BOTTOM,
        }
        stress_type_map = {
            "membrane": StressType.MEMBRANE,
            "bending": StressType.BENDING,
            "total": StressType.TOTAL,
        }

        if location.lower() not in location_map:
            raise ValueError(
                f"Invalid location '{location}'. Must be one of: {list(location_map.keys())}"
            )
        if stress_type.lower() not in stress_type_map:
            raise ValueError(
                f"Invalid stress_type '{stress_type}'. Must be one of: {list(stress_type_map.keys())}"
            )

        loc = location_map[location.lower()]
        st = stress_type_map[stress_type.lower()]

        recovery = StressRecovery(self.domain, self.u)

        if nodal:
            return recovery.compute_nodal_stresses(
                location=loc, stress_type=st, smoothing=smoothing
            )
        else:
            return recovery.compute_element_stresses(location=loc, stress_type=st)

    def compute_strains(
        self,
        location: str = "middle",
        nodal: bool = True,
        smoothing: str = "average",
    ):
        """
        Compute strain field from the displacement solution.

        Parameters
        ----------
        location : str, optional
            Location through the shell thickness:
            - "top": Top surface (z = +h/2)
            - "middle": Mid-surface (z = 0)
            - "bottom": Bottom surface (z = -h/2)
            Default is "middle".
        nodal : bool, optional
            If True, compute smoothed nodal strains.
            If False, compute element-centroid strains.
            Default is True.
        smoothing : str, optional
            Smoothing method for nodal strains.
            Default is "average".

        Returns
        -------
        StrainResult
            A dataclass containing:
            - epsilon_xx, epsilon_yy, gamma_xy: Strain components
            - epsilon_1, epsilon_2: Principal strains
            - gamma_max: Maximum shear strain
            - principal_angle: Angle of principal direction

        Raises
        ------
        AttributeError
            If the solution (self.u) has not been computed yet.
        """
        from .stress_recovery import StressLocation, StressRecovery

        if not hasattr(self, "u") or self.u is None:
            raise AttributeError(
                "No displacement solution found. Run solve() before computing strains."
            )

        location_map = {
            "top": StressLocation.TOP,
            "middle": StressLocation.MIDDLE,
            "bottom": StressLocation.BOTTOM,
        }

        if location.lower() not in location_map:
            raise ValueError(
                f"Invalid location '{location}'. Must be one of: {list(location_map.keys())}"
            )

        loc = location_map[location.lower()]
        recovery = StressRecovery(self.domain, self.u)

        if nodal:
            return recovery.compute_nodal_strains(location=loc, smoothing=smoothing)
        else:
            return recovery.compute_element_strains(location=loc)

    def compute_von_mises_stress(
        self,
        location: str = "middle",
        nodal: bool = True,
    ) -> np.ndarray:
        """
        Convenience method to compute only the von Mises stress.

        Parameters
        ----------
        location : str, optional
            Location through the shell thickness. Default is "middle".
        nodal : bool, optional
            If True, compute nodal values. Default is True.

        Returns
        -------
        np.ndarray
            Von Mises stress at each node or element centroid.
        """
        stress_result = self.compute_stresses(location=location, nodal=nodal)
        return stress_result.von_mises

    def write_results(self, output_file: str, include_stresses: bool = False) -> None:
        """
        Write the simulation results to a VTK file.

        This method combines vector components from the domain's vector form and calculates
        the magnitude for each vector field. Both the individual components and the vector
        fields (with their magnitude) are written to the VTK file.

        Parameters
        ----------
        output_file : str
            Path to the output VTK file.
        include_stresses : bool, optional
            If True, compute and include stress fields in the output.
            Default is False.

        Raises
        ------
        AttributeError
            If `self.u` is not defined. Ensure that the solver has computed the solution
            (i.e., `self.u` is set) before calling this method.
        """
        # 'vector_form' is assumed to be a dict mapping vector field names to lists of component names.
        vector_form = self.vector_form
        vector_components = [comp for vector in vector_form.values() for comp in vector]
        U = self.u.array.reshape(-1, self.domain.dofs_per_node)

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

        # Include stress fields if requested
        if include_stresses:
            try:
                # Compute stresses at middle surface (total stress)
                stress_result = self.compute_stresses(
                    location="middle", stress_type="total", nodal=True
                )
                point_data["sigma_xx"] = stress_result.sigma_xx
                point_data["sigma_yy"] = stress_result.sigma_yy
                point_data["sigma_xy"] = stress_result.sigma_xy
                point_data["von_mises"] = stress_result.von_mises
                point_data["sigma_1"] = stress_result.sigma_1
                point_data["sigma_2"] = stress_result.sigma_2
                point_data["tau_max"] = stress_result.tau_max

                # Also compute stresses at top and bottom surfaces for shell elements
                stress_top = self.compute_stresses(location="top", stress_type="total", nodal=True)
                stress_bottom = self.compute_stresses(
                    location="bottom", stress_type="total", nodal=True
                )
                point_data["von_mises_top"] = stress_top.von_mises
                point_data["von_mises_bottom"] = stress_bottom.von_mises

            except Exception as e:
                import warnings

                warnings.warn(f"Could not compute stresses: {e}", stacklevel=2)

        # Assemble cells from the mesh element map
        cells = []
        for element in self.mesh_obj.element_map.values():
            if element:
                cells.append(
                    (
                        element.element_type.name,
                        [element.node_ids],
                    )
                )

        mesh_object = meshio.Mesh(points, cells=cells, point_data=point_data)
        mesh_object.write(output_file, file_format="vtk")

    def view_results(self):
        # 'vector_form' is assumed to be a dict mapping vector field names to lists of component names.
        vector_form = self.vector_form
        vector_components = [comp for vector in vector_form.values() for comp in vector]
        U = self.u.array.reshape(-1, len(vector_components))  # self.u must be defined after solving

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
