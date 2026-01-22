"""
Linear Dynamic FSI Solver.

This module provides the LinearDynamicFSISolver for fluid-structure interaction
problems using preCICE coupling.
"""

import logging
from typing import Dict

import numpy as np
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.linear import LinearDynamicSolver

from .base import Adapter, ForceClipper


class LinearDynamicFSISolver(LinearDynamicSolver):
    """
    Linear dynamic solver for Fluid-Structure Interaction (FSI) problems.

    Inherits from LinearDynamicSolver and adds functionality for FSI problems
    using preCICE coupling.

    Parameters
    ----------
    mesh : MeshModel
        The finite element mesh.
    fem_model_properties : Dict
        Configuration dictionary containing solver and element properties.

    Configuration
    -------------
    The fem_model_properties dictionary should contain:
        solver.coupling.participant : str
            Name of this participant in preCICE.
        solver.coupling.config_file : str
            Path to preCICE configuration XML.
        solver.coupling.coupling_mesh : str
            Name of the coupling mesh.
        solver.coupling.write_data : str or list
            Data field(s) to write to preCICE.
        solver.coupling.read_data : str or list
            Data field(s) to read from preCICE.
        solver.coupling_boundaries : list
            Node set names for coupling interface.
        solver.force_max_cap : float, optional
            Maximum force magnitude per node (for clipping).
        solver.force_ramp_time : float, optional
            Time over which to ramp forces from 0 to 1.
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
        """Configure PETSc linear solver with residual monitoring support."""
        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setType("cg")

        # Configure preconditioner for main solver
        pc = self._solver.getPC()

        opts = PETSc.Options()

        # For medium and large problems, use GAMG (PETSc's algebraic multigrid)
        # GAMG is more robust and always available in PETSc
        if self.domain.dofs_count > 1e4:
            pc.setType("gamg")
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_agg_nsmooths"] = 1
            opts["pc_gamg_threshold"] = 0.02
            opts["pc_gamg_square_graph"] = 1
            opts["pc_gamg_sym_graph"] = True
            # Smoothers at each level
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
            opts["mg_levels_ksp_max_it"] = 3
            # Coarsest level solver
            opts["mg_coarse_ksp_type"] = "preonly"
            opts["mg_coarse_pc_type"] = "lu"
            # More relaxed tolerances for FSI (preCICE iterates)
            self._solver.setTolerances(rtol=1e-5, atol=1e-8, max_it=1000)
        else:
            # For small problems, ILU with better numerical stability
            pc.setType("ilu")
            opts["pc_factor_mat_ordering_type"] = "rcm"  # Reduce bandwidth
            opts["pc_factor_shift_type"] = "positive_definite"  # Ensure positivity
            opts["pc_factor_shift_amount"] = 1e-10  # Small shift for stability
            opts["pc_factor_levels"] = 1  # ILU(1) for better approximation
            self._solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)

        self._solver.setFromOptions()

        # Configure solver for M_red (simple preconditioner)
        self._m_solver = PETSc.KSP().create(self.comm)
        self._m_solver.setType("preonly")
        m_pc = self._m_solver.getPC()
        m_pc.setType("jacobi")  # Fast preconditioner for well-conditioned matrix
        self._m_solver.setTolerances(rtol=1e-12, max_it=1)
        self._m_solver.setFromOptions()

    def lump_mass_matrix(self, M: PETSc.Mat) -> PETSc.Mat:
        """Convert mass matrix M to lumped (diagonal) form."""
        diag = PETSc.Vec().createMPI(M.getSize()[0], comm=M.getComm())
        M.getRowSum(diag)  # Diagonal will be row sums
        M_lumped = PETSc.Mat().createAIJ(size=M.getSize(), comm=M.getComm())
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

        # Newmark-β coefficients
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]

        a0 = 1.0 / (beta * self.dt**2)
        a2 = 1.0 / (beta * self.dt)
        a3 = 1.0 / (2 * beta) - 1.0

        # Effective stiffness matrix
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

        # Store interface DOFs reference for checkpoint handling
        self._interface_dofs = self.precice_participant.interface_dofs

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

            # Optimization: avoid redundant operations
            # Contribution from mass and initial conditions
            temp_vec = a0 * u  # Precalculate terms
            temp_vec.axpy(a2, v)  # temp_vec += a2 * v
            temp_vec.axpy(a3, a)  # temp_vec += a3 * a

            # Create output vector for multiplication
            temp_result = M_red.createVecRight()
            M_red.mult(temp_vec, temp_result)  # temp_result = M_red @ temp_vec

            # Effective force (use PETSc + operator)
            F_eff = F_new_red + temp_result

            # Solve for displacement
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

                # Update acceleration and velocity
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

                # Prepare for checkpoint
                u_expanded = bc_manager.expand_solution(u)

                # Handle checkpoint writing if enabled
                self._handle_checkpoint(
                    t=t,
                    time_step=time_step,
                    dt=self.dt,
                    u_red=u.array.copy(),
                    v_red=v.array.copy(),
                    a_red=a.array.copy(),
                    u_full=u_expanded.array.copy(),
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
