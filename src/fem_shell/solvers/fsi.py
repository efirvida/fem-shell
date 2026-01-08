import copy
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import precice
import yaml
from petsc4py import PETSc

from fem_shell.core.assembler import MeshAssembler
from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.linear import LinearDynamicSolver


class ForceClipper:
    """
    Conservative force clipping to prevent pathological spikes in FSI coupling.

    Only clips force magnitudes that exceed a specified threshold. Does NOT smooth
    or average forces—preserves the actual CFD solution in the normal range.

    Strategy: Detect when nodal force magnitude is excessive (e.g., > 10x typical),
    and scale it down to a reasonable cap. All other forces pass through unchanged.

    Parameters
    ----------
    force_max_cap : Optional[float]
        Hard cap on per-node force magnitude. If None, no clipping is applied.
        Recommended: estimate from steady-state or pre-simulation (e.g., 500 kN for blades).
    """

    def __init__(self, force_max_cap: Optional[float] = None):
        self.force_max_cap = force_max_cap
        self._clipped_count = 0
        self._total_count = 0

    def apply(self, force_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply conservative clipping to force data.

        Parameters
        ----------
        force_data : np.ndarray
            Raw force data from preCICE. Shape: (n_nodes, n_dims) or (n_components,).

        Returns
        -------
        clipped_force : np.ndarray
            Force data with excessive magnitudes clipped, same shape as input.
        diagnostics : Dict[str, float]
            Statistics: mean, max, n_clipped (count of clipped nodes).
        """
        if self.force_max_cap is None:
            # No clipping
            force_mags = (
                np.linalg.norm(force_data, axis=1) if force_data.ndim == 2 else np.abs(force_data)
            )
            return force_data.copy(), {
                "mean": float(np.mean(force_mags)),
                "max": float(np.max(force_mags)),
                "n_clipped": 0,
                "cap": None,
            }

        # Ensure 2D for consistent processing
        if force_data.ndim == 1:
            n_dims = force_data.shape[0] if len(force_data.shape) == 1 else 1
            force_2d = force_data.reshape(-1, max(n_dims, 1))
            reshape_1d = True
        else:
            force_2d = force_data
            reshape_1d = False

        force_clipped = force_2d.copy()

        # Compute per-node magnitude
        force_mags = np.linalg.norm(force_clipped, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-12)  # Avoid division by zero

        # Identify and clip excessive magnitudes
        clip_mask = force_mags[:, 0] > self.force_max_cap
        n_clipped = np.sum(clip_mask)

        if n_clipped > 0:
            scale_factors = self.force_max_cap / force_mags[clip_mask, 0]
            force_clipped[clip_mask] *= scale_factors[:, np.newaxis]

        self._clipped_count += n_clipped
        self._total_count += force_2d.shape[0]

        # Diagnostics
        force_mags_final = np.linalg.norm(force_clipped, axis=1)
        diagnostics = {
            "mean": float(np.mean(force_mags_final)),
            "max": float(np.max(force_mags_final)),
            "n_clipped": int(n_clipped),
            "cap": self.force_max_cap,
        }

        # Reshape to match input if needed
        if reshape_1d:
            return force_clipped.flatten(), diagnostics
        return force_clipped, diagnostics

    def get_statistics(self) -> Dict[str, float]:
        """Return overall clipping statistics."""
        if self._total_count == 0:
            return {"clipped_fraction": 0.0, "total_nodes_processed": 0}
        return {
            "clipped_fraction": self._clipped_count / self._total_count,
            "total_nodes_processed": self._total_count,
        }


class Config:
    """
    Handles reading configuration parameters for the preCICE solver adapter.

    This class reads configuration parameters from a YAML file or dictionary
    and stores the data in instance attributes. The configuration is then
    accessible through properties.

    Parameters
    ----------
    adapter_config : str, dict, or None
        Path to the YAML configuration file for the adapter, or a dictionary
        containing the configuration directly.
    base_path : str, optional
        Base path for resolving relative file paths when using dict config.
    """

    def __init__(
        self,
        adapter_config: Optional[Union[str, Dict[str, Any]]] = None,
        base_path: Optional[str] = None,
    ):
        self._config_file: Optional[str] = None
        self._participant: Optional[str] = None
        self._coupling_mesh: Optional[str] = None
        self._read_data: Optional[str] = None
        self._write_data: Optional[str] = None
        self._base_path = base_path

        if adapter_config:
            if isinstance(adapter_config, dict):
                self.read_dict(adapter_config)
            else:
                self.read_yaml(adapter_config)

    def read_dict(self, data: Dict[str, Any]) -> None:
        """
        Reads configuration from a dictionary and stores the values.

        Parameters
        ----------
        data : dict
            Configuration dictionary with keys: participant, config_file, interface.

        Raises
        ------
        KeyError
            If a required key is missing in the configuration.
        """
        if data is None:
            raise ValueError("Configuration dictionary is empty or invalid.")

        # Determine base path for resolving relative paths
        base_path = self._base_path or os.getcwd()

        # Validate required keys and assign values
        try:
            config_file = data["config_file"]
            # Resolve relative paths
            if not os.path.isabs(config_file):
                config_file = os.path.join(base_path, config_file)
            self._config_file = config_file

            self._participant = data["participant"]
            interface = data["interface"]
            self._coupling_mesh = interface["coupling_mesh"]
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

        # Optional keys: assign if present, else remain None
        self._write_data = interface.get("write_data")
        self._read_data = interface.get("read_data")

        logging.info("Configuration loaded from dictionary")

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
    def __init__(self, states: Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec, float]):
        """Almacena estados (vectores PETSc) para checkpointing eficiente."""
        self._state = []
        for state in states:
            if isinstance(state, PETSc.Vec):
                self._state.append(state.copy())
            else:
                # Para escalares (como el tiempo 't') u otros tipos
                self._state.append(copy.deepcopy(state))

    def get_state(self):
        """Devuelve los vectores clonados."""
        return self._state

    def __del__(self):
        """Liberar memoria de vectores PETSc al destruir el objeto."""
        for vec in self._state:
            if isinstance(vec, PETSc.Vec):
                vec.destroy()


class Adapter:
    """This adapter class provides an interface to the preCICE v3 coupling library for setting up a
    coupling case with FEniCSx as a participant in 2D and 3D problems.

    The coupling is achieved using the FunctionSpace degrees of freedom (DOFs) on the interface.
    Data interchange between participants (read/write) is performed by accessing the values on
    the interface DOFs. This approach allows us to leverage the full range of FEniCSx functionalities
    while ensuring seamless communication with other participants in the coupling process.

    """

    def __init__(
        self,
        adapter_config: Union[str, Dict[str, Any]] = "precice-adapter-config.yaml",
        base_path: Optional[str] = None,
    ):
        """Constructor of Adapter class.

        Parameters
        ----------
        adapter_config : Union[str, Dict[str, Any]]
            Either a path to the YAML adapter configuration file, or a dictionary
            containing the configuration directly (inline config).
        base_path : Optional[str]
            Base path for resolving relative paths in config_file. Only used when
            adapter_config is a dictionary. If None, current working directory is used.
        """
        self._config = Config(adapter_config, base_path=base_path)

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
        adapter_cfg = fem_model_properties["solver"]["adapter_cfg"]
        base_path = fem_model_properties.get("base_path")
        self.precice_participant = Adapter(adapter_cfg, base_path=base_path)
        self._prepared = False

    def _setup_solver(self):
        """Configure PETSc linear solver with residual monitoring support"""
        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setType("cg")

        # Configurar precondicionador para el solucionador principal
        pc = self._solver.getPC()

        opts = PETSc.Options()

        # Para problemas medianos y grandes, usar GAMG (algebraic multigrid de PETSc)
        # GAMG es más robusto y siempre disponible en PETSc
        if self.domain.dofs_count > 1e4:
            pc.setType("gamg")
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_agg_nsmooths"] = 1
            opts["pc_gamg_threshold"] = 0.02
            opts["pc_gamg_square_graph"] = 1
            opts["pc_gamg_sym_graph"] = True
            # Smoothers en cada nivel
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
            opts["mg_levels_ksp_max_it"] = 3
            # Solver del nivel más grueso
            opts["mg_coarse_ksp_type"] = "preonly"
            opts["mg_coarse_pc_type"] = "lu"
            # Tolerancias más relajadas para FSI (preCICE itera)
            self._solver.setTolerances(rtol=1e-5, atol=1e-8, max_it=1000)
        else:
            # Para problemas pequeños, ILU con mejor estabilidad numérica
            pc.setType("ilu")
            opts["pc_factor_mat_ordering_type"] = "rcm"  # Reduce bandwidth
            opts["pc_factor_shift_type"] = "positive_definite"  # Asegura positividad
            opts["pc_factor_shift_amount"] = 1e-10  # Pequeño shift para estabilidad
            opts["pc_factor_levels"] = 1  # ILU(1) para mejor aproximación
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
        logger = logging.getLogger(__name__)

        print("\n" + "═" * 70, flush=True)
        print("  FSI DYNAMIC ANALYSIS - STRUCTURAL SOLVER", flush=True)
        print("═" * 70, flush=True)

        print("  [1/6] Assembling stiffness matrix...", flush=True)
        self.K = self.domain.assemble_stiffness_matrix()

        print("  [2/6] Assembling mass matrix...", flush=True)
        self.M = self.domain.assemble_mass_matrix()
        self.M = self.lump_mass_matrix(self.M)

        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        print("  [3/6] Applying boundary conditions...", flush=True)
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        print(
            f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, Free: {len(bc_manager.free_dofs)} DOFs",
            flush=True,
        )

        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.solver_params.get("time_step", 1.0)))

        print("  [4/6] Initializing preCICE coupling...", flush=True)
        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )

        self.dt = self.precice_participant.dt
        K_red, F_red, M_red = bc_manager.reduced_system

        self.free_dofs = bc_manager.free_dofs

        print("  [5/6] Setting up linear solver...", flush=True)
        if not self._prepared:
            self._setup_solver()
            self._solver.setOperators(K_red)
            self._prepared = True

        # Coeficientes de Newmark-β
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]

        a0 = 1.0 / (beta * self.dt**2)
        a2 = 1.0 / (beta * self.dt)
        a3 = 1.0 / (2 * beta) - 1.0

        # Matriz de rigidez efectiva
        K_eff = K_red + a0 * M_red
        self._solver.setOperators(K_eff)

        print("  [6/6] Setting initial conditions...", flush=True)
        u = K_red.createVecRight()
        v = K_red.createVecRight()

        residual = F_red.duplicate()
        residual.copy(F_red)
        K_red.mult(u, residual)
        residual.scale(-1.0)
        residual.axpy(1.0, F_red)

        a = M_red.createVecRight()
        self._m_solver.setOperators(M_red)
        self._m_solver.solve(residual, a)

        # Try to restore from checkpoint if start_from='latestTime'
        checkpoint_state = self._try_restore_checkpoint()
        starting_from_zero = True  # Flag to track if we're starting fresh
        if checkpoint_state is not None:
            starting_from_zero = False
            print(f"  ✓ Restored from checkpoint at t = {checkpoint_state['t']:.6f} s", flush=True)
            t = checkpoint_state["t"]
            time_step = checkpoint_state["time_step"]
            # Restore reduced vectors if present; otherwise start with zeros
            if "u_red" in checkpoint_state:
                u.array[:] = checkpoint_state["u_red"]
                v.array[:] = checkpoint_state["v_red"]
                a.array[:] = checkpoint_state["a_red"]
            else:
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                print(
                    "  ↳ Using checkpoint time only; state vectors reinitialized to zero",
                    flush=True,
                )

            if self.solver_params.get("reset_state_on_restart", False):
                # Keep temporal continuity but reset state vectors to avoid double-loading
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                print(
                    "  ↳ State vectors reset to zero; time/time_step preserved from checkpoint",
                    flush=True,
                )

        # Write initial state at t=0 with undeformed mesh if configured (OpenFOAM-like)
        # Only write if we're actually starting from t=0 (no checkpoint restored)
        if (
            self._checkpoint_manager is not None
            and self.solver_params.get("write_initial_state", True)
            and starting_from_zero
            and t == 0
        ):
            self._handle_checkpoint(
                t=0.0,
                time_step=0,
                dt=self.dt,
                u_red=u.array.copy(),
                v_red=v.array.copy(),
                a_red=a.array.copy(),
                u_full=bc_manager.expand_solution(u).array.copy(),
                v_full=bc_manager.expand_solution(v).array.copy(),
                a_full=bc_manager.expand_solution(a).array.copy(),
            )

        # Force clipping and ramping configuration
        force_max_cap = self.solver_params.get("force_max_cap", None)
        force_clipper = ForceClipper(force_max_cap=force_max_cap)
        ramp_time = self.solver_params.get("force_ramp_time", 0.0)

        print("═" * 70, flush=True)
        print(f"  dt = {self.dt:.6f} s  │  Newmark β={beta:.2f}, γ={gamma:.2f}", flush=True)
        if force_max_cap:
            print(f"  Force cap: {force_max_cap:.2e} N", flush=True)
        if ramp_time > 0:
            print(f"  Ramp time: {ramp_time:.4f} s", flush=True)
        print("═" * 70, flush=True)

        while self.precice_participant.is_coupling_ongoing:
            step += 1

            if self.precice_participant.requires_writing_checkpoint:
                logger.debug("Step %d: Writing checkpoint at t = %.6f s", step, t)
                self.precice_participant.store_checkpoint((u, v, a, t))

            # Read coupling data from preCICE
            logger.debug("Step %d: Reading coupling data from preCICE...", step)
            data = self.precice_participant.read_data()
            interface_dofs = self.precice_participant.interface_dofs

            # ==============================================================================
            # FORCE ANALYSIS: Compute statistics on raw CFD forces (before any processing)
            # ==============================================================================
            if data.ndim == 2:
                n_nodes = data.shape[0]
                raw_force_x = np.sum(data[:, 0])
                raw_force_y = np.sum(data[:, 1])
                raw_force_z = np.sum(data[:, 2]) if data.shape[1] >= 3 else 0.0
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data, axis=1))
            else:
                n_nodes = len(data) // 3
                data_2d = data.reshape(-1, 3)
                raw_force_x = np.sum(data_2d[:, 0])
                raw_force_y = np.sum(data_2d[:, 1])
                raw_force_z = np.sum(data_2d[:, 2])
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))

            # ==============================================================================
            # Apply conservative force clipping: only clip excessive magnitudes
            # ==============================================================================
            data_clipped, clip_diags = force_clipper.apply(data)

            # ==============================================================================
            # Apply force ramping based on PHYSICAL TIME (not iteration count)
            # This ensures consistent ramping regardless of preCICE sub-iterations
            # Use t_target (end of current time step) to match OpenFOAM convention
            # ==============================================================================
            t_target = t + self.dt  # Time at end of current step (matches OpenFOAM)
            if ramp_time > 0 and t_target < ramp_time:
                # Smooth sine ramp: small value at t=dt, 1 at t=ramp_time
                ramp_factor = 0.5 * (1.0 - np.cos(np.pi * t_target / ramp_time))
            elif ramp_time > 0 and t_target >= ramp_time:
                ramp_factor = 1.0
            else:
                ramp_factor = 1.0

            data_ramped = data_clipped * ramp_factor

            # ==============================================================================
            # Compute final applied forces (after clipping + ramping)
            # ==============================================================================
            if data_ramped.ndim == 2:
                applied_force_x = np.sum(data_ramped[:, 0])
                applied_force_y = np.sum(data_ramped[:, 1])
                applied_force_z = np.sum(data_ramped[:, 2]) if data_ramped.shape[1] >= 3 else 0.0
                applied_force_mag = np.sqrt(
                    applied_force_x**2 + applied_force_y**2 + applied_force_z**2
                )
                applied_max_nodal = np.max(np.linalg.norm(data_ramped, axis=1))
            else:
                data_2d = data_ramped.reshape(-1, 3)
                applied_force_x = np.sum(data_2d[:, 0])
                applied_force_y = np.sum(data_2d[:, 1])
                applied_force_z = np.sum(data_2d[:, 2])
                applied_force_mag = np.sqrt(
                    applied_force_x**2 + applied_force_y**2 + applied_force_z**2
                )
                applied_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))

            # ==============================================================================
            # DETAILED LOGGING OUTPUT WITH BOX FORMAT
            # ==============================================================================
            print(f"\n{'─' * 70}", flush=True)
            print(
                f"  TIME WINDOW {time_step + 1:4d}  │  ITER {step:4d}  │  t → {t_target:.6f} s",
                flush=True,
            )
            print(f"{'─' * 70}", flush=True)

            # CFD Forces section (raw data from fluid solver)
            print("  ┌─ CFD FORCES (mapped from fluid solver)", flush=True)
            print(f"  │  Total:   |F| = {raw_force_mag:12.4e} N", flush=True)
            print(
                f"  │  Components: Fx={raw_force_x:+.4e}  Fy={raw_force_y:+.4e}  Fz={raw_force_z:+.4e}",
                flush=True,
            )
            print(f"  │  Max nodal:  {raw_max_nodal:.4e} N  ({n_nodes} nodes)", flush=True)
            print("  │", flush=True)

            # Processing section
            print("  ├─ PROCESSING", flush=True)
            if clip_diags["n_clipped"] > 0:
                print(
                    f"  │  Clipping: {clip_diags['n_clipped']}/{n_nodes} nodes capped at {force_max_cap:.2e} N",
                    flush=True,
                )
            else:
                print(
                    f"  │  Clipping: None (cap={force_max_cap:.2e} N)"
                    if force_max_cap
                    else "  │  Clipping: Disabled",
                    flush=True,
                )

            if ramp_time > 0:
                print(
                    f"  │  Ramping:  factor = {ramp_factor:.4f}  (t_target={t_target:.4f}s / {ramp_time:.4f}s)",
                    flush=True,
                )
            else:
                print("  │  Ramping:  Disabled", flush=True)
            print("  │", flush=True)

            # Applied Forces section (after all processing)
            print("  └─ APPLIED FORCES (after clipping + ramping)", flush=True)
            print(f"     Total:   |F| = {applied_force_mag:12.4e} N", flush=True)
            print(
                f"     Components: Fx={applied_force_x:+.4e}  Fy={applied_force_y:+.4e}  Fz={applied_force_z:+.4e}",
                flush=True,
            )
            print(f"     Max nodal:  {applied_max_nodal:.4e} N", flush=True)

            # Use ramped data for assembly
            data = data_ramped

            F_new = self.F.copy()
            F_new.setValues(interface_dofs, data)
            F_new_red = bc_manager.reduce_vector(F_new)

            # ==============================================================================
            # Compute effective force with Rayleigh damping effects
            # F_eff = F_external + a0*M*u + a2*M*v + a3*M*a + a1_c*C*v
            # where C = eta_m*M + eta_k*K (damping matrix)
            # ==============================================================================

            # Optimización: evitar operaciones redundantes
            # Contribution from mass and initial conditions
            temp_vec = a0 * u  # Precalculo de términos
            temp_vec.axpy(a2, v)  # temp_vec += a2 * v
            temp_vec.axpy(a3, a)  # temp_vec += a3 * a

            # Crear un vector de salida para la multiplicación
            temp_result = M_red.createVecRight()
            M_red.mult(temp_vec, temp_result)  # temp_result = M_red @ temp_vec

            # Fuerza efectiva (usar operador + de PETSc que funciona correctamente)
            F_eff = F_new_red + temp_result

            # Resolver para el desplazamiento
            logger.debug("Step %d: Solving linear system...", step)
            u_new = K_eff.createVecRight()
            self._solver.solve(F_eff, u_new)

            # Log solver convergence info
            ksp_its = self._solver.getIterationNumber()
            ksp_reason = self._solver.getConvergedReason()

            # Compute solution response for this iteration
            max_disp_iter = u_new.norm(PETSc.NormType.INFINITY)

            # Show structural response for EVERY iteration
            print("  ┌─ SOLVER RESPONSE (this iteration)", flush=True)
            print(f"  │  KSP iterations: {ksp_its}  (reason: {ksp_reason})", flush=True)
            print(f"  │  max|u_new| = {max_disp_iter:.4e} m", flush=True)
            print("  └" + "─" * 67, flush=True)

            logger.debug("Step %d: Writing displacement data to preCICE...", step)
            self.precice_participant.write_data(bc_manager.expand_solution(u_new).array)

            # Visual separator before preCICE output
            print("  ┌─ preCICE " + "─" * 57, flush=True)
            self.precice_participant.advance(self.dt)
            print("  └" + "─" * 67, flush=True)

            if self.precice_participant.requires_reading_checkpoint:
                logger.debug("Step %d: Reading checkpoint (sub-iteration)", step)
                u, v, a, t = self.precice_participant.retrieve_checkpoint()
            else:
                # Time window completed - increment physical time step counter
                time_step += 1

                # Actualizar aceleración y velocidad
                delta_u = u_new - u
                a_new = (delta_u - self.dt * v - (0.5 - beta) * self.dt**2 * a) / (
                    beta * self.dt**2
                )
                v_new = v + self.dt * ((1 - gamma) * a + gamma * a_new)
                u, v, a = u_new, v_new, a_new
                t += self.dt

                # Log final state after time window completion
                max_disp = u.norm(PETSc.NormType.INFINITY)
                max_vel = v.norm(PETSc.NormType.INFINITY)
                max_acc = a.norm(PETSc.NormType.INFINITY)

                print("  ┌─ ✓ TIME WINDOW CONVERGED", flush=True)
                print(f"  │  max|u| = {max_disp:.4e} m", flush=True)
                print(f"  │  max|v| = {max_vel:.4e} m/s", flush=True)
                print(f"  │  max|a| = {max_acc:.4e} m/s²", flush=True)
                print(f"  └─ Advanced to t = {t:.6f} s", flush=True)

                # Handle checkpoint writing if enabled
                self._handle_checkpoint(
                    t=t,
                    time_step=time_step,
                    dt=self.dt,
                    u_red=u.array.copy(),
                    v_red=v.array.copy(),
                    a_red=a.array.copy(),
                    u_full=bc_manager.expand_solution(u).array.copy(),
                    v_full=bc_manager.expand_solution(v).array.copy(),
                    a_full=bc_manager.expand_solution(a).array.copy(),
                )

        # Get final clipping statistics
        clip_stats = force_clipper.get_statistics()

        print("\n" + "═" * 70, flush=True)
        print("  FSI SIMULATION COMPLETED", flush=True)
        print("═" * 70, flush=True)
        print(f"  Final time:        {t:.6f} s", flush=True)
        print(f"  Time steps:        {time_step}", flush=True)
        print(f"  Total iterations:  {step}", flush=True)
        if time_step > 0:
            print(f"  Avg iters/step:    {step / time_step:.2f}", flush=True)
        if clip_stats["clipped_fraction"] > 0:
            print(
                f"  Clipping:          {100 * clip_stats['clipped_fraction']:.2f}% nodes clipped",
                flush=True,
            )
        print("═" * 70 + "\n", flush=True)

        # Flush async checkpoints and create PVD index
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        self.u = bc_manager.expand_solution(u)
        self.v = bc_manager.expand_solution(v)
        self.a = bc_manager.expand_solution(a)

        return self.u, self.v, self.a
