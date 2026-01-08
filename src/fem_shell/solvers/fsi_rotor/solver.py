"""
FSI Rotor Solver for Wind Turbine Applications.

This module contains the main FSIRotorSolver class that couples structural
dynamics with aerodynamic loads via the preCICE library.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.fsi import Adapter
from fem_shell.solvers.linear import LinearDynamicSolver

from .aerodynamics import AerodynamicsCalculator
from .checkpoint_state import CheckpointState
from .config import RotorConfig
from .generator import GeneratorController
from .inertial_forces import InertialForcesCalculator
from .rotor_logger import RotorLogger
from .transforms import CoordinateTransforms


class FSIRotorSolver(LinearDynamicSolver):
    """
    FSI solver for wind turbine rotor blades with rotational dynamics.

    This solver couples structural finite element analysis with aerodynamic
    loads from CFD, while simultaneously solving rotor rotational dynamics.

    The solver is organized using composition with specialized components:
    - CoordinateTransforms: Frame transformations
    - InertialForcesCalculator: Centrifugal, Coriolis, Euler, gravity forces
    - AerodynamicsCalculator: Torque and coefficient calculations
    - GeneratorController: Load torque models
    - RotorLogger: CSV logging

    Parameters
    ----------
    mesh : MeshModel
        The structural mesh model.
    fem_model_properties : Dict
        Model properties including solver configuration.

    Attributes
    ----------
    _omega : float
        Current angular velocity [rad/s].
    _theta : float
        Accumulated rotation angle [rad].
    _alpha : float
        Current angular acceleration [rad/s²].
    _I_rotor : float
        Total rotor moment of inertia [kg·m²].
    _rotor_radius : float
        Rotor radius computed from mesh [m].

    See Also
    --------
    LinearDynamicSolver : Base class for structural dynamics.
    RotorConfig : Configuration dataclass.
    """

    #: Mapping from axis name to array index
    AXIS_MAP: Dict[str, int] = {"x": 0, "y": 1, "z": 2}

    #: Small number for numerical stability
    _EPS: float = 1e-10

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        """Initialize the FSI rotor solver."""
        super().__init__(mesh, fem_model_properties)

        # Initialize logger
        self._logger = logging.getLogger(__name__)

        # Initialize preCICE adapter
        adapter_cfg = fem_model_properties["solver"]["adapter_cfg"]
        base_path = fem_model_properties.get("base_path")
        self.precice_participant = Adapter(adapter_cfg, base_path=base_path)
        self._prepared = False

        # =====================================================================
        # Configuration
        # =====================================================================
        self._config = RotorConfig.from_solver_params(self.solver_params)

        # =====================================================================
        # Initialize Components (Composition over Inheritance)
        # =====================================================================
        self._rotor_center = np.array(self._config.rotor_center)
        self._rotor_axis = self._config.rotor_axis

        # Coordinate transforms
        self._transforms = CoordinateTransforms(
            self._rotor_axis,
            self._rotor_center,
        )

        # Pre-compute axis indices for convenience
        self._axis_idx, self._perp_indices = self._transforms.get_axis_indices()

        # Generator controller
        self._generator = GeneratorController.from_config(self._config)

        # =====================================================================
        # Angular State Variables
        # =====================================================================
        self._omega_mode = self._config.omega_mode
        self._inertia_mode = self._config.inertia_mode
        self._rotor_inertia_input = self._config.rotor_inertia
        self._hub_inertia = self._config.hub_inertia
        self._hub_fraction = self._config.hub_fraction
        self._rotational_damping = self._config.rotational_damping

        self._omega = self._config.initial_omega
        self._theta = 0.0
        self._alpha = 0.0
        self._I_rotor = None

        # =====================================================================
        # Inertial Forces Configuration
        # =====================================================================
        self._enable_centrifugal = self._config.enable_centrifugal
        self._enable_gravity = self._config.enable_gravity
        self._enable_coriolis = self._config.enable_coriolis
        self._enable_euler = self._config.enable_euler
        self._coriolis_use_rotating_frame = self._config.coriolis_use_rotating_frame
        self._gravity_vector = np.array(self._config.gravity_vector)

        # Inertial forces calculator (initialized after transforms)
        self._inertial_calc = InertialForcesCalculator(
            self._transforms,
            self._gravity_vector,
        )

        # =====================================================================
        # Rotor Geometry (Computed from Mesh)
        # =====================================================================
        self._rotor_radius = None
        self._rotor_area = None

        # =====================================================================
        # Aerodynamic Parameters
        # =====================================================================
        self._air_density = self._config.air_density
        self._wind_speed = self._config.wind_speed

        # Aerodynamics calculator (initialized after transforms)
        self._aero_calc = AerodynamicsCalculator(
            self._transforms,
            self._air_density,
            self._wind_speed,
        )

        # Current aerodynamic state
        self._thrust = 0.0
        self._power = 0.0
        self._tsr = 0.0
        self._Cp = 0.0
        self._Cq = 0.0
        self._Ct = 0.0

        # Cached nodal masses
        self._cached_nodal_masses: Optional[np.ndarray] = None

        # Validate Newmark parameters
        self._validate_newmark_stability()

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_newmark_stability(self) -> None:
        """Validate Newmark-β parameters for numerical stability."""
        beta = self.solver_params.get("beta", 0.25)
        gamma = self.solver_params.get("gamma", 0.5)

        self._logger.debug(f"Validating Newmark-β parameters: β={beta}, γ={gamma}")

        if gamma < 0.5:
            warnings.warn(
                f"Newmark-β parameter γ={gamma} < 0.5 may cause numerical instability. "
                f"Use γ ≥ 0.5 for unconditional stability.",
                UserWarning,
                stacklevel=2,
            )

        beta_min = (gamma + 0.5) ** 2 / 4
        if beta < beta_min:
            warnings.warn(
                f"Newmark-β parameter β={beta} < {beta_min:.4f} may cause numerical instability.",
                UserWarning,
                stacklevel=2,
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _ensure_3d_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Ensure coordinate array is 3D by padding with zeros if needed."""
        return CoordinateTransforms.ensure_3d_coordinates(coords)

    def _compute_position_from_center(self, coords: np.ndarray) -> np.ndarray:
        """Compute position vectors from rotor center."""
        return self._transforms.position_from_center(coords)

    def _compute_perpendicular_distances(self, position_vectors: np.ndarray) -> np.ndarray:
        """Compute perpendicular distances from rotation axis."""
        return self._transforms.perpendicular_distances(position_vectors)

    def _get_rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """Get rotation matrix (legacy interface)."""
        temp_transform = CoordinateTransforms(axis, np.zeros(3))
        return temp_transform.get_rotation_matrix(angle)

    def _rotate_vector_around_axis(self, vector: np.ndarray, axis: str, angle: float) -> np.ndarray:
        """Rotate vector around axis (legacy interface)."""
        temp_transform = CoordinateTransforms(axis, np.zeros(3))
        return temp_transform.rotate_vector(vector, angle)

    def _transform_vectors_to_rotating_frame(self, vectors: np.ndarray) -> np.ndarray:
        """Transform vectors from inertial to rotating frame."""
        return self._transforms.to_rotating_frame(vectors, self._theta)

    def _transform_vectors_to_inertial_frame(self, vectors: np.ndarray) -> np.ndarray:
        """Transform vectors from rotating to inertial frame."""
        return self._transforms.to_inertial_frame(vectors, self._theta)

    # =========================================================================
    # Torque and Force Calculations (Delegated to Components)
    # =========================================================================

    def _compute_torque(self, coords: np.ndarray, forces: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute torque around rotor axis from nodal forces."""
        return self._aero_calc.compute_torque(coords, forces)

    def _compute_centrifugal_forces(self, coords: np.ndarray, nodal_mass: np.ndarray) -> np.ndarray:
        """Compute centrifugal forces at mesh nodes."""
        return self._inertial_calc.compute_centrifugal_forces(coords, nodal_mass, self._omega)

    def _compute_gravity_forces(self, nodal_mass: np.ndarray) -> np.ndarray:
        """Compute gravitational forces in rotating frame."""
        return self._inertial_calc.compute_gravity_forces(nodal_mass, self._theta)

    def _compute_coriolis_forces(
        self,
        coords: np.ndarray,
        nodal_mass: np.ndarray,
        nodal_velocities: np.ndarray,
    ) -> np.ndarray:
        """Compute Coriolis forces at mesh nodes."""
        return self._inertial_calc.compute_coriolis_forces(
            coords, nodal_mass, nodal_velocities, self._omega
        )

    def _compute_euler_forces(self, coords: np.ndarray, nodal_mass: np.ndarray) -> np.ndarray:
        """Compute Euler forces at mesh nodes."""
        return self._inertial_calc.compute_euler_forces(
            coords,
            nodal_mass,
            self._alpha,
            self._theta,
            self._coriolis_use_rotating_frame,
        )

    def _get_nodal_masses(self) -> np.ndarray:
        """Extract nodal masses from the lumped mass matrix."""
        M_diag = self.M.getDiagonal()
        mass_array = M_diag.array

        dofs_per_node = self.domain.dofs_per_node
        n_nodes = len(mass_array) // dofs_per_node

        nodal_mass = np.zeros(n_nodes)
        for i in range(n_nodes):
            start_idx = i * dofs_per_node
            nodal_mass[i] = np.mean(mass_array[start_idx : start_idx + 3])

        return nodal_mass

    def _compute_inertial_forces(self, coords: np.ndarray) -> np.ndarray:
        """Compute total inertial forces (centrifugal + gravitational)."""
        n_nodes = coords.shape[0]

        if (
            not self._enable_centrifugal
            and not self._enable_gravity
            and not self._enable_coriolis
            and not self._enable_euler
        ):
            return np.zeros((n_nodes, 3))

        if self._cached_nodal_masses is None:
            self._cached_nodal_masses = self._get_nodal_masses()

        nodal_mass = self._cached_nodal_masses[:n_nodes]

        return self._inertial_calc.compute_total_forces(
            coords,
            nodal_mass,
            self._omega,
            self._theta,
            enable_centrifugal=self._enable_centrifugal,
            enable_gravity=self._enable_gravity,
            enable_coriolis=False,  # Coriolis handled separately (needs velocities)
            enable_euler=False,  # Euler handled separately
        )

    # =========================================================================
    # Moment of Inertia Calculations
    # =========================================================================

    def _compute_blades_inertia(self) -> float:
        """Compute moment of inertia of blades about the rotor axis."""
        nodal_mass = self._get_nodal_masses()
        coords = self.mesh_obj.coords_array
        r = self._compute_position_from_center(coords)
        r_perp_squared = np.sum(r[:, self._perp_indices] ** 2, axis=1)
        return np.sum(nodal_mass * r_perp_squared)

    def _get_total_inertia(self) -> float:
        """Compute total rotor moment of inertia based on configuration mode."""
        if self._inertia_mode == "total":
            if self._rotor_inertia_input is None:
                raise ValueError("rotor_inertia must be specified when inertia_mode='total'")
            return self._rotor_inertia_input

        elif self._inertia_mode == "hub_plus_blades":
            I_blades = self._compute_blades_inertia()
            I_total = self._hub_inertia + I_blades
            self._logger.info(f"        Blade inertia:  {I_blades:.4e} kg·m²")
            self._logger.info(f"        Hub inertia:    {self._hub_inertia:.4e} kg·m²")
            self._logger.info(f"        Total inertia:  {I_total:.4e} kg·m²")
            return I_total

        elif self._inertia_mode == "fraction":
            I_blades = self._compute_blades_inertia()
            I_total = I_blades * (1.0 + self._hub_fraction)
            self._logger.info(f"        Blade inertia:      {I_blades:.4e} kg·m²")
            self._logger.info(f"        Hub fraction:       {self._hub_fraction:.2%}")
            self._logger.info(f"        Total inertia:      {I_total:.4e} kg·m²")
            return I_total

        else:
            raise ValueError(f"Unknown inertia_mode: {self._inertia_mode}")

    # =========================================================================
    # Angular State Integration
    # =========================================================================

    def _update_angular_state(self, torque_axial: float, dt: float) -> Tuple[float, float, float]:
        """Integrate the rotational equation of motion for one timestep."""
        if self._omega_mode == "constant":
            self._alpha = 0.0
            self._theta += self._omega * dt
            return self._omega, self._alpha, self._theta

        # Dynamic mode: Newmark-β integration
        beta = self.solver_params.get("beta", 0.25)
        gamma = self.solver_params.get("gamma", 0.5)

        tau_load = self._generator.compute_torque(self._omega)
        tau_net = torque_axial - tau_load

        I = self._I_rotor
        c = self._rotational_damping

        theta_old = self._theta
        omega_old = self._omega
        alpha_old = self._alpha

        # Predictor step
        omega_pred = omega_old + dt * (1.0 - gamma) * alpha_old

        # Solve for new acceleration (implicit damping)
        alpha_new = (tau_net - c * omega_pred) / (I + c * dt * gamma)

        # Corrector step
        omega_new = omega_pred + dt * gamma * alpha_new
        theta_new = (
            theta_old + dt * omega_old + dt**2 * ((0.5 - beta) * alpha_old + beta * alpha_new)
        )

        self._alpha = alpha_new
        self._omega = omega_new
        self._theta = theta_new

        return self._omega, self._alpha, self._theta

    def _compute_rotor_radius(self) -> float:
        """Compute rotor radius from mesh."""
        coords = self.mesh_obj.coords_array
        r = self._compute_position_from_center(coords)
        r_perp = self._compute_perpendicular_distances(r)
        radius = np.max(r_perp)

        if radius <= 0:
            raise ValueError(f"Computed rotor radius is non-positive: {radius}")

        return radius

    def _compute_aerodynamic_coefficients(self, torque: float, forces_3d: np.ndarray) -> None:
        """Compute aerodynamic coefficients."""
        state = self._aero_calc.compute_coefficients(
            torque,
            forces_3d,
            self._omega,
            self._rotor_radius,
            self._rotor_area,
        )
        self._thrust = state.thrust
        self._power = state.power
        self._tsr = state.tsr
        self._Cp = state.Cp
        self._Cq = state.Cq
        self._Ct = state.Ct

    # =========================================================================
    # Interface Mapping Optimization
    # =========================================================================

    def _precompute_interface_mapping(self, bc_manager: BoundaryConditionManager) -> None:
        """Pre-compute mapping from interface DOFs to reduced system DOFs."""
        interface_dofs = self.precice_participant.interface_dofs

        if isinstance(bc_manager.free_dofs, dict):
            global_to_reduced = bc_manager.free_dofs
        else:
            global_to_reduced = {dof: i for i, dof in enumerate(bc_manager.free_dofs)}

        self._interface_to_reduced_indices = np.array(
            [global_to_reduced.get(dof, -1) for dof in interface_dofs],
            dtype=int,
        )

    def _extract_interface_velocities(self, v_reduced: PETSc.Vec) -> np.ndarray:
        """Extract interface velocities from reduced velocity vector."""
        v_array = v_reduced.array
        n_dofs = len(self._interface_to_reduced_indices)
        interface_vals = np.zeros(n_dofs)

        mask = self._interface_to_reduced_indices >= 0
        valid_indices = self._interface_to_reduced_indices[mask]
        interface_vals[mask] = v_array[valid_indices]

        return interface_vals.reshape(-1, 3)

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    def _compute_effective_stiffness(
        self,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        a0: float,
        K_geometric: Optional[PETSc.Mat] = None,
    ) -> PETSc.Mat:
        """Compute effective stiffness matrix for Newmark-beta integration."""
        alpha, beta_damp = self._get_rayleigh_coefficients()
        enable_damping = self._config.enable_structural_damping and (alpha > 0 or beta_damp > 0)

        K_eff = K_red.duplicate()
        K_eff.copy(K_red)

        if enable_damping:
            gamma = self.solver_params.get("gamma", 0.5)
            beta_newmark = self.solver_params.get("beta", 0.25)
            a1 = gamma / (beta_newmark * self.dt)

            coef_K = 1.0 + a1 * beta_damp
            coef_M = a0 + a1 * alpha

            K_eff.scale(coef_K)
            K_eff.axpy(coef_M, M_red)
        else:
            K_eff.axpy(a0, M_red)

        if K_geometric is not None:
            K_eff.axpy(1.0, K_geometric)

        return K_eff

    def _get_rayleigh_coefficients(self) -> Tuple[float, float]:
        """Get Rayleigh damping coefficients (alpha, beta)."""
        if not self._config.enable_structural_damping:
            return 0.0, 0.0

        if not self._config.use_damping_ratios:
            return self._config.rayleigh_alpha, self._config.rayleigh_beta

        # Compute from damping ratios
        omega_1 = 2 * np.pi * self._config.damping_freq_1
        omega_2 = 2 * np.pi * self._config.damping_freq_2
        zeta_1 = self._config.damping_ratio_1
        zeta_2 = self._config.damping_ratio_2

        denom = omega_1**2 - omega_2**2
        alpha = 2 * omega_1 * omega_2 * (zeta_2 * omega_1 - zeta_1 * omega_2) / denom
        beta = 2 * (zeta_1 * omega_1 - zeta_2 * omega_2) / denom

        return max(0.0, alpha), max(0.0, beta)

    def _update_solver_operators(self, K_eff: PETSc.Mat) -> None:
        """Update solver with new effective stiffness matrix."""
        self._solver.setOperators(K_eff)
        self._solver.setConvergenceHistory()

    def _assemble_geometric_stiffness(self, bc_manager: BoundaryConditionManager) -> PETSc.Mat:
        """Assemble geometric stiffness matrix from centrifugal prestress."""
        axis_vectors = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        rotation_axis = axis_vectors.get(self._rotor_axis, np.array([1.0, 0.0, 0.0]))
        rotation_center = np.array(self._rotor_center, dtype=np.float64)

        K_G_full = self.domain.assemble_geometric_stiffness(
            omega=self._omega,
            rotation_axis=rotation_axis,
            rotation_center=rotation_center,
        )

        K_G_red = bc_manager.reduce_matrix(K_G_full)

        self._logger.debug(f"Assembled geometric stiffness: ω={self._omega:.4f} rad/s")

        return K_G_red

    # =========================================================================
    # Solver Setup
    # =========================================================================

    def _setup_solver(self) -> None:
        """Configure PETSc linear solver."""
        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setType("cg")

        pc = self._solver.getPC()
        opts = PETSc.Options()

        if self.domain.dofs_count > self._config.solver_dofs_threshold:
            # GAMG for larger problems
            pc.setType("gamg")
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_agg_nsmooths"] = 1
            opts["pc_gamg_threshold"] = 0.02
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
            opts["mg_levels_ksp_max_it"] = 3
            opts["mg_coarse_ksp_type"] = "preonly"
            opts["mg_coarse_pc_type"] = "lu"
            self._solver.setTolerances(
                rtol=self._config.solver_rtol_large,
                atol=self._config.solver_atol_large,
                max_it=self._config.solver_max_it_large,
            )
        else:
            # ILU for smaller problems
            pc.setType("ilu")
            opts["pc_factor_mat_ordering_type"] = "rcm"
            opts["pc_factor_shift_type"] = "positive_definite"
            opts["pc_factor_shift_amount"] = self._config.solver_shift_amount
            opts["pc_factor_levels"] = 1
            self._solver.setTolerances(
                rtol=self._config.solver_rtol_small,
                atol=self._config.solver_atol_small,
                max_it=self._config.solver_max_it_small,
            )

        self._solver.setFromOptions()

        # Mass matrix solver
        self._m_solver = PETSc.KSP().create(self.comm)
        self._m_solver.setType("preonly")
        m_pc = self._m_solver.getPC()
        m_pc.setType("jacobi")
        self._m_solver.setTolerances(rtol=self._config.solver_rtol_mass, max_it=1)
        self._m_solver.setFromOptions()

    def _lump_mass_matrix(self, M: PETSc.Mat) -> PETSc.Mat:
        """Convert mass matrix to lumped (diagonal) form with validation."""
        diag = PETSc.Vec().createMPI(M.getSize()[0], comm=M.getComm())
        M.getRowSum(diag)

        diag_array = diag.getArray()
        negative_count = np.sum(diag_array < 0)
        total_nodes = len(diag_array)

        if negative_count > 0:
            warnings.warn(
                f"Row-sum lumping produced {negative_count} negative masses. "
                f"Applying HRZ lumping correction.",
                UserWarning,
                stacklevel=2,
            )

            total_mass = np.sum(diag_array)
            uniform_mass = total_mass / total_nodes
            negative_mask = diag_array <= self._EPS
            diag_array[negative_mask] = uniform_mass

            corrected_total = np.sum(diag_array)
            if corrected_total > self._EPS:
                diag_array *= total_mass / corrected_total

            diag.setArray(diag_array)

        final_diag = diag.getArray()
        if np.min(final_diag) <= 0:
            raise RuntimeError("Lumped mass matrix contains non-positive entries.")

        M_lumped = PETSc.Mat().createAIJ(size=M.getSize(), comm=M.getComm())
        M_lumped.setDiagonal(diag)
        M_lumped.assemble()

        diag.destroy()
        return M_lumped

    # =========================================================================
    # Simulation Lifecycle
    # =========================================================================

    def _initialize_simulation(self) -> BoundaryConditionManager:
        """Initialize simulation: assembly, inertia, geometry, BCs, preCICE."""
        self._logger.info("\n" + "═" * 70)
        self._logger.info("  FSI ROTOR SOLVER - STRUCTURAL DYNAMICS")
        self._logger.info("═" * 70)

        # Assembly
        self._logger.info("  [1/5] Assembling stiffness matrix...")
        self.K = self.domain.assemble_stiffness_matrix()

        self._logger.info("  [2/5] Assembling mass matrix (lumped)...")
        self.M = self.domain.assemble_mass_matrix()
        self.M = self._lump_mass_matrix(self.M)

        # Inertia
        self._logger.info(f"  [2b/5] Computing rotor inertia (mode: {self._inertia_mode})...")
        self._logger.info(f"        Omega mode: {self._omega_mode}")
        self._I_rotor = self._get_total_inertia()

        # Geometry
        self._logger.info("  [2c/5] Computing rotor geometry from mesh...")
        self._rotor_radius = self._compute_rotor_radius()
        self._rotor_area = np.pi * self._rotor_radius**2
        self._logger.info(f"        Rotor radius: {self._rotor_radius:.4f} m")
        self._logger.info(f"        Rotor area:   {self._rotor_area:.4f} m²")

        # Logger
        self.logger = RotorLogger(
            self._config,
            self._rotor_radius,
            self._rotor_area,
            self._I_rotor,
            self.solver_params,
        )
        self.logger.initialize()

        # Force vector
        self.F = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        self.F.set(0.0)

        # Boundary conditions
        self._logger.info("  [3/5] Applying boundary conditions...")
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        self._logger.info(
            f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, "
            f"Free: {len(bc_manager.free_dofs)} DOFs"
        )

        # Interface mapping
        self._precompute_interface_mapping(bc_manager)

        # preCICE
        self._logger.info("  [4/5] Initializing preCICE coupling...")
        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )
        self.dt = self.precice_participant.dt

        # Solver
        self._logger.info("  [5/5] Setting up linear solver...")
        if not self._prepared:
            self._setup_solver()
            self._prepared = True

        return bc_manager

    def _finalize_simulation(self, t: float, step: int, time_step: int) -> None:
        """Finalize simulation and print summary."""
        self._logger.info("\n" + "═" * 70)
        self._logger.info("  FSI SIMULATION COMPLETED")
        self._logger.info("═" * 70)
        self._logger.info(f"  Final time:       {t:.6f} s")
        self._logger.info(f"  Time steps:       {time_step}")
        self._logger.info(f"  Total iterations: {step}")
        self._logger.info("─" * 70)
        self._logger.info("  ROTOR DYNAMICS RESULTS:")
        self._logger.info(
            f"  Final ω:          {self._omega:.4f} rad/s "
            f"({self._omega * 60 / (2 * np.pi):.2f} RPM)"
        )
        self._logger.info(
            f"  Final θ:          {np.degrees(self._theta):.2f}° "
            f"({self._theta / (2 * np.pi):.2f} rev)"
        )
        self._logger.info(f"  Final α:          {self._alpha:.4e} rad/s²")
        self._logger.info("═" * 70 + "\n")

        self.logger.close()

    # =========================================================================
    # Main Solve Method
    # =========================================================================

    def solve(self):
        """
        Perform FSI dynamic analysis.

        Main coupling loop:
        1. Read forces from CFD (Global Inertial Frame)
        2. Transform forces to Local Rotating Frame
        3. Compute inertial forces
        4. Solve Newmark-β time integration
        5. Transform displacements to Global Frame
        6. Write displacements to CFD
        7. Update rotational dynamics
        8. Advance preCICE coupling
        """
        # Initialize
        bc_manager = self._initialize_simulation()
        K_red, F_red, M_red = bc_manager.reduced_system

        self._K_red = K_red
        self._M_red = M_red

        # Newmark coefficients
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        a0 = 1.0 / (beta * self.dt**2)
        a2 = 1.0 / (beta * self.dt)
        a3 = 1.0 / (2 * beta) - 1.0
        self._a0 = a0

        # Effective stiffness
        K_eff = self._compute_effective_stiffness(K_red, M_red, a0)
        self._K_eff = K_eff
        self._update_solver_operators(K_eff)

        # Dynamic stiffness config
        enable_dynamic_stiffness = self._config.enable_dynamic_stiffness
        stiffness_update_interval = self._config.stiffness_update_interval
        stiffness_update_mode = self._config.stiffness_update_mode
        omega_change_threshold = self._config.omega_change_threshold
        omega_last_update = None

        # Initial conditions
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

        # Time variables
        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.dt))

        # Interface coordinates
        interface_coords = self.precice_participant.interface_coordinates
        interface_coords = self._ensure_3d_coordinates(interface_coords)

        # Log initial state
        self.logger.log_initial_state(
            self.dt,
            beta,
            gamma,
            list(self._rotor_center),
            self._rotor_axis,
            self._air_density,
            self._wind_speed,
            self._rotational_damping,
        )

        # Main coupling loop
        while self.precice_participant.is_coupling_ongoing:
            step += 1

            # Checkpoint
            if self.precice_participant.requires_writing_checkpoint:
                checkpoint_state = CheckpointState(
                    u=u,
                    v=v,
                    a=a,
                    t=t,
                    omega=self._omega,
                    theta=self._theta,
                    alpha=self._alpha,
                )
                self.precice_participant.store_checkpoint(checkpoint_state.to_tuple())

            # Dynamic stiffness update
            if enable_dynamic_stiffness:
                update_stiffness = False
                if stiffness_update_mode == "interval":
                    update_stiffness = time_step % stiffness_update_interval == 0
                elif stiffness_update_mode == "adaptive":
                    if omega_last_update is None:
                        update_stiffness = True
                    else:
                        if abs(omega_last_update) > 1e-10:
                            omega_change = abs(self._omega - omega_last_update) / abs(
                                omega_last_update
                            )
                        else:
                            omega_change = abs(self._omega - omega_last_update)
                        update_stiffness = omega_change >= omega_change_threshold

                if update_stiffness:
                    K_geometric = self._assemble_geometric_stiffness(bc_manager)
                    K_eff = self._compute_effective_stiffness(
                        self._K_red, self._M_red, self._a0, K_geometric
                    )
                    self._K_eff = K_eff
                    self._update_solver_operators(K_eff)
                    omega_last_update = self._omega

            # Read forces from CFD
            data = self.precice_participant.read_data()
            interface_dofs = self.precice_participant.interface_dofs

            if data.ndim == 1:
                forces_global = data.reshape(-1, 3)
            else:
                forces_global = self._ensure_3d_coordinates(data)

            # Transform to rotating frame
            forces_aero = self._transform_vectors_to_rotating_frame(forces_global)

            # Current geometry
            u_full = bc_manager.expand_solution(u).array
            u_interface_flat = u_full[interface_dofs]
            u_interface = u_interface_flat.reshape(-1, 3)
            current_coords = interface_coords + u_interface

            # Inertial forces
            forces_inertial = np.zeros_like(forces_aero)
            if self._enable_centrifugal or self._enable_gravity:
                forces_inertial = self._compute_inertial_forces(current_coords)

            # Coriolis forces (active from step 0 when omega != 0)
            forces_coriolis = np.zeros_like(forces_aero)
            if self._enable_coriolis:
                if self._cached_nodal_masses is None:
                    self._cached_nodal_masses = self._get_nodal_masses()
                interface_velocities = self._extract_interface_velocities(v)
                forces_coriolis = self._compute_coriolis_forces(
                    current_coords,
                    self._cached_nodal_masses[: len(interface_coords)],
                    interface_velocities,
                )

            # Euler forces (active from step 0 when alpha != 0)
            forces_euler = np.zeros_like(forces_aero)
            if self._enable_euler:
                if self._cached_nodal_masses is None:
                    self._cached_nodal_masses = self._get_nodal_masses()
                forces_euler = self._compute_euler_forces(
                    current_coords,
                    self._cached_nodal_masses[: len(interface_coords)],
                )

            # Total forces
            forces_total = forces_aero + forces_inertial + forces_coriolis + forces_euler

            # Torque calculation
            torque_aero, _ = self._compute_torque(current_coords, forces_aero)
            torque_inertial, _ = self._compute_torque(
                current_coords, forces_inertial + forces_coriolis + forces_euler
            )
            torque_net = torque_aero + torque_inertial

            # Update angular state
            self._update_angular_state(torque_net, self.dt)

            # Aerodynamic coefficients
            self._compute_aerodynamic_coefficients(torque_aero, forces_aero)

            # Log timestep
            self.logger.log_timestep(
                t=t + self.dt,
                torque_aero=torque_aero,
                torque_inertial=torque_inertial,
                torque_net=torque_net,
                thrust=self._thrust,
                power=self._power,
                omega=self._omega,
                alpha=self._alpha,
                theta=self._theta,
                tsr=self._tsr,
                cp=self._Cp,
                cq=self._Cq,
                ct=self._Ct,
            )

            # Assemble force vector
            self.F.set(0.0)
            self.domain.apply_nodal_forces(self.F, interface_dofs, forces_total.flatten())
            bc_manager.apply_neumann(self.F)
            F_red_new = bc_manager.reduce_vector(self.F)

            # Effective load
            predictor = u.duplicate()
            predictor.copy(u)
            predictor.scale(a0)
            predictor.axpy(a2, v)
            predictor.axpy(a3, a)

            R_eff = F_red_new.duplicate()
            M_red.mult(predictor, R_eff)
            R_eff.axpy(1.0, F_red_new)

            # Rayleigh damping contribution
            alpha_damp, beta_damp = self._get_rayleigh_coefficients()
            if self._config.enable_structural_damping and (alpha_damp > 0 or beta_damp > 0):
                a1 = gamma / (beta * self.dt)
                a4 = gamma / beta - 1.0
                a5 = self.dt * (gamma / (2.0 * beta) - 1.0)

                damping_predictor = u.duplicate()
                damping_predictor.copy(u)
                damping_predictor.scale(a1)
                damping_predictor.axpy(a4, v)
                damping_predictor.axpy(a5, a)

                temp_vec = damping_predictor.duplicate()
                if alpha_damp > 0:
                    M_red.mult(damping_predictor, temp_vec)
                    R_eff.axpy(alpha_damp, temp_vec)
                if beta_damp > 0:
                    self._K_red.mult(damping_predictor, temp_vec)
                    R_eff.axpy(beta_damp, temp_vec)

                temp_vec.destroy()
                damping_predictor.destroy()

            # Solve
            self._solver.solve(R_eff, u)

            # Update velocity and acceleration
            a_new = u.duplicate()
            a_new.copy(u)
            a_new.scale(a0)
            a_new.axpy(-1.0, predictor)

            v_new = v.duplicate()
            v_new.copy(v)
            v_new.axpy(self.dt * (1.0 - gamma), a)
            v_new.axpy(self.dt * gamma, a_new)

            v.copy(v_new)
            a.copy(a_new)

            # Write displacements to CFD
            u_full = bc_manager.expand_solution(u).array
            u_interface_flat = u_full[interface_dofs]
            u_interface_local = u_interface_flat.reshape(-1, 3)
            u_interface_global = self._transform_vectors_to_inertial_frame(u_interface_local)

            self.precice_participant.write_data(u_interface_global.flatten())
            self.precice_participant.advance(self.dt)

            # Checkpoint restore
            if self.precice_participant.requires_reading_checkpoint:
                checkpoint_data = self.precice_participant.read_checkpoint()
                checkpoint_state = CheckpointState.from_tuple(checkpoint_data)

                checkpoint_state.u.copy(u)
                checkpoint_state.v.copy(v)
                checkpoint_state.a.copy(a)
                t = checkpoint_state.t
                self._omega = checkpoint_state.omega
                self._theta = checkpoint_state.theta
                self._alpha = checkpoint_state.alpha
            else:
                t += self.dt
                time_step += 1

        self._finalize_simulation(t, step, time_step)
