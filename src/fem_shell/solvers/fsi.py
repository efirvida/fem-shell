import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import precice
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
    """preCICE adapter for FSI coupling with structural solvers.

    This adapter class provides an interface to the preCICE coupling library
    for setting up FSI simulations. All configuration is passed directly as
    parameters - no external configuration file is needed.

    Parameters
    ----------
    participant : str
        Name of this participant in the preCICE configuration.
    config_file : str
        Path to the preCICE configuration XML file.
    coupling_mesh : str
        Name of the coupling mesh in preCICE.
    write_data : list of str
        Names of data fields to write to preCICE.
    read_data : list of str
        Names of data fields to read from preCICE.
    """

    def __init__(
        self,
        participant: str,
        config_file: str,
        coupling_mesh: str,
        write_data: List[str],
        read_data: List[str],
    ):
        self._participant = participant
        self._config_file = config_file
        self._coupling_mesh = coupling_mesh
        self._write_data = write_data if isinstance(write_data, list) else [write_data]
        self._read_data = read_data if isinstance(read_data, list) else [read_data]

        self._interface = precice.Participant(participant, config_file, 0, 1)

        # coupling mesh related quantities
        self._solver_vertices = None
        self._precice_vertex_ids = None

        # Solver state used by the Adapter internally to handle checkpointing
        self._checkpoint: SolverState | None = None

        # Necessary bools for enforcing proper control flow / warnings to user
        self._first_advance_done = False

    def read_data(self, data_name: Optional[str] = None) -> np.ndarray:
        """Read data from preCICE.

        Parameters
        ----------
        data_name : str, optional
            Name of the data field to read. If None, reads the first field
            in read_data list.

        Returns
        -------
        np.ndarray
            The incoming data containing nodal data.
        """
        if data_name is None:
            data_name = self._read_data[0]

        read_data = self._interface.read_data(
            self._coupling_mesh, data_name, self._precice_vertex_ids, self.dt
        )
        return copy.deepcopy(read_data.astype(PETSc.ScalarType))

    def read_all_data(self) -> Dict[str, np.ndarray]:
        """Read all configured data fields from preCICE.

        Returns
        -------
        dict
            Dictionary mapping data names to their values.
        """
        return {name: self.read_data(name) for name in self._read_data}

    def write_data(self, write_function, data_name: Optional[str] = None) -> None:
        """Write data to preCICE.

        Parameters
        ----------
        write_function : array-like
            Data to write, indexed by interface DOFs.
        data_name : str, optional
            Name of the data field to write. If None, writes to the first
            field in write_data list.
        """
        if data_name is None:
            data_name = self._write_data[0]

        write_data = write_function[self.interface_dofs]
        self._interface.write_data(
            self._coupling_mesh,
            data_name,
            self._precice_vertex_ids,
            write_data,
        )

    def write_all_data(self, data_dict: Dict[str, np.ndarray]) -> None:
        """Write multiple data fields to preCICE.

        Parameters
        ----------
        data_dict : dict
            Dictionary mapping data names to their values.
        """
        for name, data in data_dict.items():
            self.write_data(data, name)

    def initialize(
        self, domain: MeshAssembler, coupling_boundaries: Sequence[str], fixed_dofs: Tuple[int]
    ) -> float:
        """Initialize the coupling and set up the mesh in preCICE.

        Parameters
        ----------
        domain : MeshAssembler
            The mesh assembler containing the FEM mesh.
        coupling_boundaries : list of str
            Names of node sets representing the coupling interface.
        fixed_dofs : tuple
            Indices of fixed degrees of freedom.

        Returns
        -------
        float
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
            self._coupling_mesh, self._interface_coords
        )
        np.savetxt(
            "interface_coords.csv",
            self.interface_coordinates,
            header="X,Y,Z" if self._domain.spatial_dim == 3 else "X,Y",
            delimiter=",",
        )

        if self._interface.requires_initial_data():
            # Write initial zero data for all write fields
            for data_name in self._write_data:
                self._interface.write_data(
                    self._coupling_mesh,
                    data_name,
                    self._precice_vertex_ids,
                    np.zeros(self.interface_dofs.shape),
                )
        self._interface.initialize()
        return self._interface.get_max_time_step_size()

    def store_checkpoint(self, states: Sequence) -> None:
        """Store current solver state for checkpointing."""
        if self._first_advance_done:
            assert self.is_time_window_complete
        logging.debug("Store checkpoint")
        self._checkpoint = SolverState(states)

    def retrieve_checkpoint(self) -> List:
        """Retrieve stored checkpoint state.

        Returns
        -------
        list
            The stored checkpoint state.
        """
        assert not self.is_time_window_complete
        logging.debug("Restore solver state")
        if self._checkpoint:
            return self._checkpoint.get_state()

    def advance(self, dt: float) -> float:
        """Advance coupling in preCICE.

        Parameters
        ----------
        dt : float
            Length of timestep used by the solver.

        Returns
        -------
        float
            Maximum length of timestep for next iteration.
        """
        self._first_advance_done = True
        max_dt = self._interface.advance(dt)
        return max_dt

    def finalize(self) -> None:
        """Finalize the coupling via preCICE."""
        self._interface.finalize()

    @property
    def is_coupling_ongoing(self) -> bool:
        """Check if the coupled simulation is still ongoing."""
        return self._interface.is_coupling_ongoing()

    @property
    def is_time_window_complete(self) -> bool:
        """Check if implicit iteration has converged."""
        return self._interface.is_time_window_complete()

    @property
    def requires_reading_checkpoint(self) -> bool:
        """Check if reading a checkpoint is required."""
        return self._interface.requires_reading_checkpoint()

    @property
    def requires_writing_checkpoint(self) -> bool:
        """Check if writing a checkpoint is required."""
        return self._interface.requires_writing_checkpoint()

    @property
    def interface_dofs(self):
        """Return the interface degrees of freedom."""
        if self._interface_dofs.shape[0] > 3:
            return self._interface_dofs[:, :3].astype(PETSc.IntType)
        return self._interface_dofs.astype(PETSc.IntType)

    @property
    def interface_coordinates(self):
        """Return the interface coordinates."""
        return self._interface_coords

    @property
    def precice(self):
        """Return the preCICE interface object."""
        return self._interface

    @property
    def dt(self):
        """Return the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()

    @property
    def read_data_names(self) -> List[str]:
        """Return list of read data field names."""
        return self._read_data

    @property
    def write_data_names(self) -> List[str]:
        """Return list of write data field names."""
        return self._write_data


class LinearDynamicFSISolver(LinearDynamicSolver):
    """
    Linear dynamic solver for Fluid-Structure Interaction (FSI) problems.

    Inherits from LinearDynamicSolver and adds functionality for FSI problems.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)

        # Extract coupling configuration
        coupling_cfg = fem_model_properties["solver"]["coupling"]

        self.precice_participant = Adapter(
            participant=coupling_cfg["participant"],
            config_file=coupling_cfg["config_file"],
            coupling_mesh=coupling_cfg["coupling_mesh"],
            write_data=coupling_cfg["write_data"],
            read_data=coupling_cfg["read_data"],
        )
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
