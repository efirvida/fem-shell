import copy
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import precice
import yaml
from petsc4py import PETSc

from fem_shell.core.assembler import MeshAssembler
from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.linear import LinearDynamicSolver

logging.basicConfig(level=logging.INFO)


class Config:
    """
    Handles reading configuration parameters for the preCICE solver adapter.

    This class reads configuration parameters from a YAML file and stores the data
    in instance attributes. The configuration is then accessible through properties.

    Parameters
    ----------
    adapter_config_filename : str or None
        Path to the YAML configuration file for the adapter.
    """

    def __init__(self, adapter_config_filename: Optional[str] = None):
        self._config_file: Optional[str] = None
        self._participant: Optional[str] = None
        self._coupling_mesh: Optional[str] = None
        self._read_data: Optional[str] = None
        self._write_data: Optional[str] = None

        if adapter_config_filename:
            self.read_yaml(adapter_config_filename)

    def read_yaml(self, adapter_config_filename: str) -> None:
        """
        Reads the YAML configuration file and stores the values in instance attributes.

        Parameters
        ----------
        adapter_config_filename : str
            Path to the YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        ValueError
            If the configuration file is empty or invalid.
        KeyError
            If a required key is missing in the configuration file.
        """
        # Construct the absolute path to the configuration file
        folder = os.path.dirname(
            os.path.abspath(
                os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), adapter_config_filename)
            )
        )
        path = os.path.join(folder, os.path.basename(adapter_config_filename))

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError(f"Configuration file {path} is empty or invalid.")

        # Validate required keys and assign values
        try:
            self._config_file = os.path.join(folder, data["config_file"])
            self._participant = data["participant"]
            interface = data["interface"]
            self._coupling_mesh = interface["coupling_mesh"]
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

        # Optional keys: assign if present, else remain None
        self._write_data = interface.get("write_data")
        self._read_data = interface.get("read_data")

        logging.info("Configuration loaded successfully from %s", path)

    @property
    def config_file(self) -> str:
        """
        Returns the preCICE configuration file path.

        Returns
        -------
        str or None
            Full path to the preCICE configuration file, or None if not set.
        """
        return self._config_file

    @property
    def participant(self) -> Optional[str]:
        """
        Returns the participant name.

        Returns
        -------
        str or None
            Name of the participant, or None if not set.
        """
        return self._participant

    @property
    def mesh_name(self) -> Optional[str]:
        """
        Returns the coupling mesh name.

        Returns
        -------
        str or None
            Name of the coupling mesh, or None if not set.
        """
        return self._coupling_mesh

    @property
    def read_data(self) -> Optional[str]:
        """
        Returns the name of the variable to be read.

        Returns
        -------
        str or None
            Name of the read data variable, or None if it is not specified.
        """
        return self._read_data

    @property
    def write_data(self) -> Optional[str]:
        """
        Returns the name of the variable to be written.

        Returns
        -------
        str or None
            Name of the write data variable, or None if it is not specified.
        """
        return self._write_data

    @property
    def problem_dimension(self) -> str | None:
        """
        Reads the XML file and retrieves the 'dimensions' attribute for the specified mesh.

        If the XML file contains unbound prefixes (e.g., "data:") that are not declared,
        a dummy namespace is injected to allow parsing.

        Parameters
        ----------
        xml_file : str
            Path to the XML file.
        mesh_name : str
            Name of the mesh whose 'dimensions' attribute is to be retrieved.

        Returns
        -------
        Optional[str]
            The value of the 'dimensions' attribute if found, otherwise None.
        """
        # Read the entire file content
        with open(self.config_file, "r", encoding="utf-8") as f:
            xml_string = f.read()

        xml_string = xml_string.replace(":", "_")

        # Parse the modified XML string
        root = ET.fromstring(xml_string)

        # Search for the mesh element with the given name
        for mesh in root.findall("mesh"):
            if mesh.get("name") == self.mesh_name:
                return int(mesh.get("dimensions"))

        return None

        return None

    def __repr__(self) -> str:
        """
        Official string representation of the object for debugging.

        Returns
        -------
        str
            String showing all configuration attributes.
        """
        return f'<Config config_file_name="{self._config_file!r}">'

    def __str__(self) -> str:
        """
        User-friendly string representation of the object.

        Returns
        -------
        str
            String summarizing the main configuration details.
        """
        return (
            f"Config for participant '{self._participant}'\n"
            f"  Configuration file: {self._config_file}\n"
            f"  Coupling mesh: {self._coupling_mesh}\n"
            f"  Read data: {self._read_data}\n"
            f"  Write data: {self._write_data}"
        )


class SolverState:
    def __init__(self, states: Iterable):
        """Store a list of function as states for those
        iteration that not converge
        """
        states_cp = []
        for state in states:
            states_cp.append(copy.deepcopy(state))
        self.__state = states_cp

    def get_state(self):
        """Returns the state of the solver."""
        return self.__state


class Adapter:
    """This adapter class provides an interface to the preCICE v3 coupling library for setting up a
    coupling case with FEniCSx as a participant in 2D and 3D problems.

    The coupling is achieved using the FunctionSpace degrees of freedom (DOFs) on the interface.
    Data interchange between participants (read/write) is performed by accessing the values on
    the interface DOFs. This approach allows us to leverage the full range of FEniCSx functionalities
    while ensuring seamless communication with other participants in the coupling process.

    """

    def __init__(self, adapter_config_filename="precice-adapter-config.yaml"):
        """Constructor of Adapter class.

        Parameters
        ----------
        mpi_comm : mpi4py.MPI.Intercomm
            Communicator used by the adapter. Should be the same one used by FEniCSx, usually MPI.COMM_WORLD
        adapter_config_filename : string
            Name of the JSON adapter configuration file (to be provided by the user)
        """
        self._config = Config(adapter_config_filename)

        self._interface = precice.Participant(
            self._config.participant, self._config.config_file, 0, 1
        )

        # coupling mesh related quantities
        self._solver_vertices = None
        self._precice_vertex_ids = None

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint: SolverState | None = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

    def read_data(self):
        """Read data from preCICE.

        Incoming data is a ndarray where the shape of the array depends on the dimensions of the problem.
        For scalar problems, this will be a 1D array (vector), while for vector problems,
        it will be an Mx2 array (in 2D) or an Mx3 array (in 3D), where M is the number of interface nodes.

        Returns
        -------
        np.ndarray
            The incoming data containing nodal data ordered according to _fenicsx_vertices
        """
        mesh_name = self._config.mesh_name
        data_name = self._config.read_data

        read_data = self._interface.read_data(
            mesh_name, data_name, self._precice_vertex_ids, self.dt
        )
        return copy.deepcopy(read_data.astype(PETSc.ScalarType))

    def write_data(self, write_function) -> None:
        """Writes data to preCICE. Depending on the dimensions of the simulation.
        For scalar problems, this will be a 1D array (vector), while for vector problems,
        it will be an Mx2 array (in 2D) or an Mx3 array (in 3D), where M is the number of interface nodes.


        Parameters
        ----------
        write_function : dolfinx.fem.Function
            A FEniCSx function consisting of the data which this participant will write to preCICE
            in every time step.
        """
        mesh_name = self._config.mesh_name
        write_data_name = self._config.write_data
        write_data = write_function[self.interface_dofs]
        self._interface.write_data(
            mesh_name,
            write_data_name,
            self._precice_vertex_ids,
            write_data,
        )

    def initialize(
        self, domain: MeshAssembler, coupling_boundaries: Sequence[str], fixed_dofs: Tuple[int]
    ) -> float:
        """Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        coupling_subdomain : List
            Indices of entities representing the coupling interface normally face sets tags.
        read_function_space : dolfinx.fem.FunctionSpace
            Function space on which the read function lives. If not provided then the adapter assumes that this
            participant is a write-only participant.
        write_object : dolfinx.fem.Function
            FEniCSx function related to the quantity to be written
            by FEniCSx during each coupling iteration. If not provided then the adapter assumes that this participant is
            a read-only participant.

        Returns
        -------
        dt : float
            Recommended time step value from preCICE.
        """
        self._domain = domain
        self._mesh = domain.mesh
        self._node_sets = []
        for n_set_name in coupling_boundaries:
            self._node_sets.append(self._mesh.node_sets[n_set_name])

        nodes = {node.id: node.coords for _set in self._node_sets for node in _set.nodes.values()}

        self._interface_coords = np.array(list(nodes.values()))[:, : self._domain.spatial_dim]
        self._interface_dofs = np.array([self._domain._node_dofs_map[n] for n in nodes])

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._config.mesh_name, self._interface_coords
        )
        np.savetxt(
            "interface_coords.csv",
            self.interface_coordinates,
            header="X,Y,Z" if self._domain.spatial_dim == 3 else "X,Y",
            delimiter=",",
        )

        if self._interface.requires_initial_data():
            self._interface.write_data(
                self._config.mesh_name,
                self._config.write_data,
                self._precice_vertex_ids,
                np.zeros(self.interface_dofs.shape),
            )
        self._interface.initialize()
        return self._interface.get_max_time_step_size()

    def store_checkpoint(self, states: Sequence) -> None:
        """Defines an object of class SolverState which stores the current states of the variable and the time stamp."""
        if self._first_advance_done:
            assert self.is_time_window_complete
        logging.debug("Store checkpoint")
        self._checkpoint = SolverState(states)

    def retrieve_checkpoint(self) -> List:
        """
        Resets the FEniCSx participant state to the state of the stored checkpoint.

        Returns
        -------
        tuple
            The stored checkpoint state (u, v, a, t).
        """
        assert not self.is_time_window_complete
        logging.debug("Restore solver state")
        if self._checkpoint:
            return self._checkpoint.get_state()

    def advance(self, dt: float) -> float:
        """Advances coupling in preCICE.

        Parameters
        ----------
        dt : double
            Length of timestep used by the solver.

        Notes
        -----
        Refer advance() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        max_dt : double
            Maximum length of timestep to be computed by solver.
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self) -> None:
        """
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/precice.pyx
        """
        self._interface.finalize()

    @property
    def is_coupling_ongoing(self) -> bool:
        """
        Checks if the coupled simulation is still ongoing.

        Notes
        -----
        Refer is_coupling_ongoing() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._interface.is_coupling_ongoing()

    @property
    def is_time_window_complete(self) -> bool:
        """Tag to check if implicit iteration has converged.

        Notes:
        -----
        Refer is_time_window_complete() in https://github.com/precice/python-bindings/blob/develop/precice.pyx

        Returns:
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._interface.is_time_window_complete()

    @property
    def requires_reading_checkpoint(self) -> bool:
        """Checks if reading a checkpoint is required.

        Returns:
        -------
        bool
            True if reading a checkpoint is required, False otherwise.
        """
        return self._interface.requires_reading_checkpoint()

    @property
    def requires_writing_checkpoint(self) -> bool:
        """Checks if writing a checkpoint is required.

        Returns:
        -------
        bool
            True if writing a checkpoint is required, False otherwise.
        """
        return self._interface.requires_writing_checkpoint()

    @property
    def interface_dofs(self):
        """Returns the interface degrees of freedom."""
        if self._interface_dofs.shape[0] > 3:
            return self._interface_dofs[:, :3].astype(PETSc.IntType)
        return self._interface_dofs.astype(PETSc.IntType)

    @property
    def interface_coordinates(self):
        """Returns the interface coordinates."""
        return self._interface_coords

    @property
    def precice(self):
        """Returns the preCICE interface object."""
        return self._interface

    @property
    def dt(self):
        """Returns the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()


class LinearDynamicFSISolver(LinearDynamicSolver):
    """
    Linear dynamic solver for Fluid-Structure Interaction (FSI) problems.

    Inherits from LinearDynamicSolver and adds functionality for FSI problems.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)
        self.precice_participant = Adapter(fem_model_properties["solver"]["adapter_cfg"])
        self._prepared = False

    def _setup_solver(self):
        """Configure PETSc linear solver with residual monitoring support"""
        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setType("cg")

        # Configurar precondicionador para el solucionador principal
        pc = self._solver.getPC()

        opts = PETSc.Options()
        if self.domain.dofs_count > 1e5:
            pc.setType("hypre")
            opts["pc_hypre_type"] = "boomeramg"
            opts["pc_hypre_boomeramg_coarsen_type"] = "Falgout"
            opts["pc_hypre_boomeramg_interp_type"] = "classical"
            opts["pc_hypre_boomeramg_relax_type_all"] = "symmetric-sor/jacobi"
            opts["pc_hypre_boomeramg_strong_threshold"] = 0.5
            opts["pc_hypre_boomeramg_max_levels"] = 5
            opts["pc_hypre_boomeramg_print_statistics"] = 0
            self._solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=500)
        else:
            pc.setType("ilu")
            opts["pc_factor_mat_ordering_type"] = "natural"
            opts["pc_factor_shift_type"] = "inblocks"
            self._solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)

        self._solver.setFromOptions()

        # Configurar solucionador para M_red (precondicionador simple)
        self._m_solver = PETSc.KSP().create(self.comm)
        self._m_solver.setType("preonly")
        m_pc = self._m_solver.getPC()
        m_pc.setType("jacobi")  # Precondicionador rápido para matriz bien condicionada
        self._m_solver.setTolerances(rtol=1e-12, max_it=1)
        self._m_solver.setFromOptions()

    def lump_mass_matrix(self, M: PETSc.Mat) -> PETSc.Mat:
        """Convierte la matriz de masa M en una matriz lumped (diagonal)."""
        diag = PETSc.Vec().createMPI(M.getSize()[0], comm=M.getComm())  # Vector para la diagonal
        M.getRowSum(diag)  # La diagonal será la suma de las filas
        M_lumped = PETSc.Mat().createAIJ(size=M.getSize(), comm=M.getComm())  # Usar AIJ
        M_lumped.setDiagonal(diag)
        M_lumped.assemble()
        return M_lumped

    def solve(self):
        """Perform dynamic analysis using improved Newmark-β method."""
        # Ensamblaje inicial de matrices
        self.K = self.domain.assemble_stiffness_matrix()
        self.M = self.domain.assemble_mass_matrix()
        self.M = self.lump_mass_matrix(self.M)

        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        # Aplicación de condiciones de frontera
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )

        self.dt = self.precice_participant.dt

        K_red, F_red, M_red = bc_manager.reduced_system

        self.free_dofs = bc_manager.free_dofs

        # Configure solver
        if not self._prepared:
            self._setup_solver()
            self._solver.setOperators(K_red)
            self._prepared = True

        # Coeficientes de Newmark-β
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        a0 = 1.0 / (beta * self.dt**2)
        a1 = gamma / (beta * self.dt)  # Coeficiente para velocidad
        a2 = 1.0 / (beta * self.dt)  # Coeficiente para aceleración
        a3 = 1.0 / (2 * beta) - 1.0

        # Matriz de rigidez efectiva
        K_eff = K_red + a0 * M_red
        self._solver.setOperators(K_eff)

        # Condiciones iniciales
        u = K_red.createVecRight()
        v = K_red.createVecRight()

        # 1. Crear vector residual (F_red - K_red*u)
        residual = F_red.duplicate()  # Crea vector del mismo tamaño que F_red
        residual.copy(F_red)  # Copia F_red al residual
        K_red.mult(u, residual)  # residual = K_red*u (sobrescribe)
        residual.scale(-1.0)  # residual = -K_red*u
        residual.axpy(1.0, F_red)  # residual = F_red - K_red*u

        # 2. Resolver el sistema M_red*a = residual
        a = M_red.createVecRight()  # Crea vector para solución
        self._m_solver.setOperators(M_red)  # Establece matriz del sistema
        self._m_solver.solve(residual, a)  # Resuelve M_red*a = residual

        t = 0
        while self.precice_participant.is_coupling_ongoing:
            if self.precice_participant.requires_writing_checkpoint:
                self.precice_participant.store_checkpoint((u, v, a, t))

            # Fuerza efectiva
            data = self.precice_participant.read_data()
            interface_dofs = self.precice_participant.interface_dofs

            F_new = self.F.copy()
            F_new.setValues(interface_dofs, data)
            F_new_red = bc_manager.reduce_vector(F_new)

            # Optimización: evitar operaciones redundantes
            temp_vec = a0 * u  # Precalculo de términos
            temp_vec.axpy(a2, v)  # temp_vec += a2 * v
            temp_vec.axpy(a3, a)  # temp_vec += a3 * a

            # Crear un vector de salida para la multiplicación
            temp_result = M_red.createVecRight()
            M_red.mult(temp_vec, temp_result)  # temp_result = M_red @ temp_vec

            F_eff = F_new_red + temp_result

            # Resolver para el desplazamiento
            u_new = K_eff.createVecRight()
            self._solver.solve(F_eff, u_new)

            self.precice_participant.write_data(bc_manager.expand_solution(u_new).array)
            self.precice_participant.advance(self.dt)

            if self.precice_participant.requires_reading_checkpoint:
                u, v, a, t = self.precice_participant.retrieve_checkpoint()
            else:
                # Actualizar aceleración y velocidad
                delta_u = u_new - u
                a_new = (delta_u - self.dt * v - (0.5 - beta) * self.dt**2 * a) / (
                    beta * self.dt**2
                )
                v_new = v + self.dt * ((1 - gamma) * a + gamma * a_new)
                u, v, a = u_new, v_new, a_new
                t += self.dt

        self.u = bc_manager.expand_solution(u)
        self.v = bc_manager.expand_solution(v)
        self.a = bc_manager.expand_solution(a)

        return self.u, self.v, self.a
