"""
Co-rotational FSI Solver for Rotating Structures.

This module provides the LinearDynamicFSIRotorSolver for rotor/turbine FSI problems
with support for rotating reference frames and inertial forces.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel

from .corotational import (
    ConstantOmega,
    CoordinateTransforms,
    InertialForcesCalculator,
)
from .linear_dynamic import LinearDynamicFSISolver

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as MPIComm

# Type aliases for PETSc types (runtime imports work, but stubs are incomplete)
# Using string annotations to avoid Pylance errors
PETScMat = "PETSc.Mat"
PETScVec = "PETSc.Vec"
PETScKSP = "PETSc.KSP"
PETScPC = "PETSc.PC"

# =============================================================================
# Module Constants
# =============================================================================

# Solver configuration thresholds
_DOF_THRESHOLD_DIRECT_SOLVER = 20_000
_OMEGA_CHANGE_THRESHOLD = 1e-4
_GRAVITY_THRESHOLD = 1e-10
_MIN_DENOMINATOR = 1e-6

# Default physics values
_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
_DEFAULT_ROTATION_AXIS = (0, 0, 1)
_DEFAULT_ROTATION_CENTER = (0, 0, 0)
_DEFAULT_FLUID_DENSITY = 1.225  # kg/m³ (air at sea level)
_DEFAULT_FLOW_VELOCITY = 1.0  # m/s

# Solver tolerances
_DIRECT_SOLVER_RTOL = 1e-12
_DIRECT_SOLVER_ATOL = 1e-14
_ITERATIVE_SOLVER_RTOL = 1e-6
_ITERATIVE_SOLVER_ATOL = 1e-10
_ITERATIVE_SOLVER_MAX_IT = 1000

# Logging
_logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class NewmarkCoefficients:
    """
    Newmark-β time integration coefficients.

    These coefficients are derived from the Newmark parameters (beta, gamma)
    and the time step (dt). They are used in the effective stiffness formulation
    and the state update equations.

    Attributes:
        a0: 1 / (β·dt²) - Mass coefficient for K_eff
        a1: γ / (β·dt) - Damping coefficient for K_eff
        a2: 1 / (β·dt) - Velocity coefficient for F_eff
        a3: 1/(2β) - 1 - Acceleration coefficient for F_eff
        a4: γ/β - 1 - Velocity coefficient for damping contribution
        a5: dt·(γ/(2β) - 1) - Acceleration coefficient for damping
        a6: dt·(1 - γ) - Previous acceleration coefficient for velocity update
        a7: γ·dt - New acceleration coefficient for velocity update
    """

    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float
    a7: float

    @classmethod
    def from_newmark_params(cls, beta: float, gamma: float, dt: float) -> NewmarkCoefficients:
        """Create coefficients from Newmark parameters and time step."""
        return cls(
            a0=1.0 / (beta * dt**2),
            a1=gamma / (beta * dt),
            a2=1.0 / (beta * dt),
            a3=1.0 / (2 * beta) - 1.0,
            a4=gamma / beta - 1.0,
            a5=dt * (gamma / (2 * beta) - 1.0),
            a6=dt * (1 - gamma),
            a7=gamma * dt,
        )


class SolverMatrices(NamedTuple):
    """Container for reduced FEM matrices."""

    K_red: PETSc.Mat
    M_red: PETSc.Mat
    K_eff: PETSc.Mat
    C_red: Optional[PETSc.Mat]
    K_G_red: Optional[PETSc.Mat]


class TimeStepState(NamedTuple):
    """State variables for a single time step."""

    u: PETSc.Vec
    v: PETSc.Vec
    a: PETSc.Vec
    t: float
    theta: float


class LinearDynamicFSIRotorSolver(LinearDynamicFSISolver):
    """
    Co-rotational FSI solver for rotating structures (rotors, blades, turbines).

    This solver operates in a rotating reference frame, which allows the stiffness
    matrix to remain constant while accounting for inertial effects. The key features:

    1. **Co-rotational formulation**: Solves elastic deformation in rotating frame
    2. **Coordinate transforms**: Forces from CFD (global) → local; displacements local → global
    3. **Inertial forces**: Centrifugal, Coriolis, and Euler (when α ≠ 0)
    4. **Geometric stiffness**: Optional stress stiffening from centrifugal loading
    5. **Rayleigh damping**: C = η_m·M + η_k·K

    Mathematical Formulation
    ------------------------
    Position in inertial frame: x_global = R(θ) · (x_ref + u_local)

    Equation of motion in rotating frame:
        M·ü + C·u̇ + (K + K_G)·u = F_aero_local + F_inertial

    Where:
        F_aero_local = R^T · F_aero_global
        F_inertial = F_centrifugal + F_coriolis + F_euler

    Configuration Parameters
    ------------------------
    solver.rotor.omega : float
        Angular velocity in rad/s. Default: 0.0
    solver.rotor.rotation_axis : list[float]
        Rotation axis as unit vector [x, y, z]. Default: [0, 0, 1] (Z-axis)
    solver.rotor.rotation_center : list[float]
        Center of rotation [x, y, z]. Default: [0, 0, 0]
    solver.rotor.include_geometric_stiffness : bool
        Include K_G for stress stiffening. Default: True
    solver.rotor.include_centrifugal : bool
        Include centrifugal forces. Default: True
    solver.rotor.include_coriolis : bool
        Include Coriolis forces. Default: True
    solver.rotor.kg_update_interval : int
        Re-assemble K_G every N steps (0 = never). Default: 0. Reserved for future use.
    solver.rotor.gravity : list[float]
        Gravity acceleration vector [gx, gy, gz] in m/s². Default: [0, 0, -9.81]
        Set to [0, 0, 0] to disable gravity.
    solver.eta_m : float
        Mass-proportional Rayleigh damping. Default: 0.0
    solver.eta_k : float
        Stiffness-proportional Rayleigh damping. Default: 0.0
    solver.solver_type : str
        Linear solver type: "auto", "direct", "iterative". Default: "auto"

    Theoretical Limitations & Risks
    -------------------------------
    1. Explicit Coriolis Force:
       The Coriolis term (-2·M(Ω×v)) is treated as an external force on the RHS.
       This explicit handling may introduce instability for high rotational speeds or
       very flexible structures unless small time steps are used.

    2. Small Strain Assumption:
       Assumes linear elasticity with stress stiffening only. Does not implement a
       full Geometrically Exact Beam or St. Venant-Kirchhoff model for large
       rotations relative to the local frame.

    3. Lumped Mass Approximation:
       Inertial forces are calculated using a diagonal (lumped) mass matrix. This
       simplifies computation but may approximate rotational inertia terms less
       accurately than a consistent mass formulation.

    Future Extensions
    -----------------
    - moment_of_inertia: For computing omega from torque balance
    - Variable omega via OmegaProvider subclasses (TableOmega, FunctionOmega, ComputedOmega)
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict[str, Any]) -> None:
        super().__init__(mesh, fem_model_properties)
        self._init_rotor_config()
        self._init_solver_config()
        self._init_state_tracking()

    def _init_rotor_config(self) -> None:
        """Initialize rotor-specific configuration from model properties."""
        rotor_cfg = self.solver_params.get("rotor", {})

        # Angular velocity configuration
        omega_value = rotor_cfg.get("omega", 0.0)
        self._omega_provider = ConstantOmega(omega=omega_value)

        # Rotation geometry
        rotation_axis = rotor_cfg.get("rotation_axis", list(_DEFAULT_ROTATION_AXIS))
        rotation_center = rotor_cfg.get("rotation_center", list(_DEFAULT_ROTATION_CENTER))

        # Initialize transformation and inertial utilities
        self._coord_transforms = CoordinateTransforms(
            rotation_axis=rotation_axis,
            rotation_center=rotation_center,
        )
        self._inertial_calculator = InertialForcesCalculator(
            rotation_axis=rotation_axis,
            rotation_center=rotation_center,
        )

        # Rotor physics options
        self._include_geometric_stiffness = rotor_cfg.get("include_geometric_stiffness", True)
        self._include_centrifugal = rotor_cfg.get("include_centrifugal", True)
        self._include_coriolis = rotor_cfg.get("include_coriolis", True)
        self._kg_update_interval = rotor_cfg.get("kg_update_interval", 0)  # Reserved

        # Gravity vector (in inertial/global frame)
        self._gravity = np.array(rotor_cfg.get("gravity", list(_DEFAULT_GRAVITY)), dtype=np.float64)
        self._include_gravity = np.linalg.norm(self._gravity) > _GRAVITY_THRESHOLD

        # Aerodynamic performance parameters
        self._fluid_density = float(rotor_cfg.get("fluid_density", _DEFAULT_FLUID_DENSITY))
        self._flow_velocity = float(rotor_cfg.get("flow_velocity", _DEFAULT_FLOW_VELOCITY))
        self._rotor_radius: Optional[float] = rotor_cfg.get("radius")
        if self._rotor_radius is not None:
            self._rotor_radius = float(self._rotor_radius)

        # Reserved for future dynamic omega computation
        self._moment_of_inertia = rotor_cfg.get("moment_of_inertia", None)

    def _init_solver_config(self) -> None:
        """Initialize solver configuration parameters."""
        # Damping parameters (defaults to zero if not specified)
        self._eta_m = self.solver_params.get("eta_m", 0.0)
        self._eta_k = self.solver_params.get("eta_k", 0.0)

        # Solver type configuration
        self._solver_type = self.solver_params.get("solver_type", "auto")

    def _init_state_tracking(self) -> None:
        """Initialize state tracking variables."""
        # Current rotation angle (updated during time stepping)
        self._theta = 0.0

        # State tracking for dynamic updates
        self._prev_omega: Optional[float] = None
        self._v_guess_next: Optional[PETSc.Vec] = None

    # =========================================================================
    # MPI Utilities
    # =========================================================================

    def _get_mpi_comm(self) -> MPIComm:
        """Get the MPI communicator in mpi4py format."""
        if hasattr(self.comm, "tompi4py"):
            return self.comm.tompi4py()
        return self.comm

    def _allreduce_sum(self, local_array: np.ndarray) -> np.ndarray:
        """Perform MPI Allreduce with SUM operation."""
        result = np.zeros_like(local_array)
        try:
            py_comm = self._get_mpi_comm()
            py_comm.Allreduce(local_array, result, op=MPI.SUM)
        except Exception:
            result = local_array.copy()
        return result

    def _allreduce_max(self, local_value: float) -> float:
        """Perform MPI Allreduce with MAX operation."""
        try:
            py_comm = self._get_mpi_comm()
            return py_comm.allreduce(local_value, op=MPI.MAX)
        except Exception:
            return local_value

    # =========================================================================
    # Solver Setup
    # =========================================================================

    def _setup_solver(self) -> None:
        """Configure PETSc linear solver based on solver_type and problem size."""
        self._solver = PETSc.KSP().create(self.comm)
        pc = self._solver.getPC()
        opts = PETSc.Options()

        dof_count = self.domain.dofs_count
        solver_type = self._solver_type

        # Auto-select solver based on problem size
        if solver_type == "auto":
            solver_type = "direct" if dof_count < _DOF_THRESHOLD_DIRECT_SOLVER else "iterative"

        if solver_type == "direct":
            self._configure_direct_solver(pc)
        else:
            self._configure_iterative_solver(pc, opts)

        self._solver.setFromOptions()
        self._setup_mass_solver()

    def _configure_direct_solver(self, pc: PETSc.PC) -> None:
        """Configure direct LU solver - best for small/medium problems."""
        self._solver.setType("preonly")
        pc.setType("lu")
        # Use MUMPS if available for better performance
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass  # Fall back to PETSc default LU
        self._solver.setTolerances(rtol=_DIRECT_SOLVER_RTOL, atol=_DIRECT_SOLVER_ATOL, max_it=1)

    def _configure_iterative_solver(self, pc: PETSc.PC, opts: PETSc.Options) -> None:
        """Configure iterative solver with GAMG preconditioner."""
        self._solver.setType("cg")
        pc.setType("gamg")
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 1
        opts["pc_gamg_threshold"] = 0.02
        opts["pc_gamg_square_graph"] = 1
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_ksp_max_it"] = 3
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "lu"
        self._solver.setTolerances(
            rtol=_ITERATIVE_SOLVER_RTOL,
            atol=_ITERATIVE_SOLVER_ATOL,
            max_it=_ITERATIVE_SOLVER_MAX_IT,
        )

    def _setup_mass_solver(self) -> None:
        """Setup mass solver with simple diagonal preconditioner for lumped mass."""
        self._m_solver = PETSc.KSP().create(self.comm)
        self._m_solver.setType("preonly")
        m_pc = self._m_solver.getPC()
        m_pc.setType("jacobi")
        self._m_solver.setTolerances(rtol=_DIRECT_SOLVER_RTOL, max_it=1)
        self._m_solver.setFromOptions()

    # =========================================================================
    # Matrix Assembly Helpers
    # =========================================================================

    def _create_damping_matrix(self, K: PETSc.Mat, M: PETSc.Mat) -> Optional[PETSc.Mat]:
        """
        Create Rayleigh damping matrix C = η_m·M + η_k·K.

        Returns:
            Damping matrix if either η_m or η_k is non-zero, otherwise None.
        """
        if self._eta_m == 0.0 and self._eta_k == 0.0:
            return None

        C = K.duplicate()
        C.scale(self._eta_k)
        C.axpy(self._eta_m, M)
        return C

    def _build_effective_stiffness(
        self,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        coeffs: NewmarkCoefficients,
        K_G_red: Optional[PETSc.Mat] = None,
        C_red: Optional[PETSc.Mat] = None,
    ) -> PETSc.Mat:
        """Build effective stiffness matrix: K_eff = K + K_G + a0*M + a1*C."""
        K_eff = K_red.duplicate()
        K_eff.axpy(coeffs.a0, M_red)
        if K_G_red is not None:
            K_eff.axpy(1.0, K_G_red)
        if C_red is not None:
            K_eff.axpy(coeffs.a1, C_red)
        return K_eff

    # =========================================================================
    # Interface Data Extraction
    # =========================================================================

    def _extract_nodal_masses(self, M_lumped: PETSc.Mat, interface_dofs: np.ndarray) -> np.ndarray:
        """
        Extract nodal masses from lumped mass matrix diagonal.

        Uses global gather to ensure all processes have access to interface nodal masses
        regardless of domain decomposition.

        Args:
            M_lumped: Lumped mass matrix (diagonal).
            interface_dofs: DOF indices for interface nodes.

        Returns:
            Array of nodal masses for interface nodes.
        """
        diag_local = M_lumped.createVecRight()
        M_lumped.getDiagonal(diag_local)

        # Create a sequential vector to gather the full diagonal
        scatter, diag_full = PETSc.Scatter.toAll(diag_local)
        scatter.begin(
            diag_local, diag_full, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
        )
        scatter.end(
            diag_local, diag_full, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
        )

        diag_array = diag_full.array
        n_nodes = interface_dofs.shape[0]
        nodal_masses = np.zeros(n_nodes, dtype=np.float64)

        for i in range(n_nodes):
            # Take the mass of the first translational DOF (x-direction)
            dof_idx = interface_dofs[i, 0] if interface_dofs.ndim == 2 else interface_dofs[i * 3]
            if dof_idx < len(diag_array):
                nodal_masses[i] = diag_array[dof_idx]

        # Clean up PETSc objects
        diag_local.destroy()
        diag_full.destroy()
        scatter.destroy()

        return nodal_masses

    def _get_interface_nodal_coords(self) -> np.ndarray:
        """Get coordinates of interface nodes in rotating frame."""
        return self.precice_participant.interface_coordinates

    def _extract_interface_vector(
        self, full_vec: PETSc.Vec, interface_dofs: np.ndarray
    ) -> np.ndarray:
        """
        Extract 3D vectors for interface nodes from a full DOF vector.

        Args:
            full_vec: Full DOF vector (displacement, velocity, etc.).
            interface_dofs: DOF indices for interface nodes.

        Returns:
            Array of shape (n_nodes, 3) with vector values for each interface node.
        """
        vec_array = full_vec.array
        n_nodes = interface_dofs.shape[0]
        result = np.zeros((n_nodes, 3), dtype=np.float64)

        for i in range(n_nodes):
            for j in range(3):
                dof = (
                    interface_dofs[i, j] if interface_dofs.ndim == 2 else interface_dofs[i * 3 + j]
                )
                if dof < len(vec_array):
                    result[i, j] = vec_array[dof]

        return result

    def _get_interface_nodal_velocities(
        self, v_full: PETSc.Vec, interface_dofs: np.ndarray
    ) -> np.ndarray:
        """Extract velocity vectors for interface nodes."""
        return self._extract_interface_vector(v_full, interface_dofs)

    def _get_interface_displacements(
        self, u_full: PETSc.Vec, interface_dofs: np.ndarray
    ) -> np.ndarray:
        """Extract displacement vectors for interface nodes."""
        return self._extract_interface_vector(u_full, interface_dofs)

    # =========================================================================
    # Rotor-Specific Calculations
    # =========================================================================

    def _compute_rotor_radius(
        self,
        interface_coords: np.ndarray,
        interface_disps: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute rotor radius as max perpendicular distance from rotation axis.

        The radius is computed from the deformed configuration when displacements
        are provided, which gives a more accurate representation during operation.

        Args:
            interface_coords: Reference coordinates of interface nodes.
            interface_disps: Displacement vectors for interface nodes (optional).
                            If None, computes radius from reference configuration.

        Returns:
            Maximum perpendicular distance from rotation axis.
        """
        # Use deformed coordinates if displacements are provided
        if interface_disps is not None:
            coords = interface_coords + interface_disps
        else:
            coords = interface_coords

        rel_pos = coords - self._coord_transforms.center
        axis = self._coord_transforms.axis
        parallel = np.outer(np.dot(rel_pos, axis), axis)
        perp = rel_pos - parallel
        local_max_r = np.max(np.linalg.norm(perp, axis=1)) if len(perp) > 0 else 0.0

        return self._allreduce_max(local_max_r)

    def _auto_detect_radius(self, interface_coords: np.ndarray) -> float:
        """
        Auto-detect rotor radius as max perpendicular distance from rotation axis.

        Note: This computes the initial/reference radius. For deformed radius,
        use _compute_rotor_radius() with displacement data.

        Args:
            interface_coords: Coordinates of interface nodes.

        Returns:
            Maximum perpendicular distance from rotation axis.
        """
        return self._compute_rotor_radius(interface_coords, interface_disps=None)

    def _compute_gravity_forces(
        self, interface_masses: np.ndarray, n_nodes: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gravity force vectors for interface nodes.

        Args:
            interface_masses: Nodal masses for interface nodes.
            n_nodes: Number of interface nodes.

        Returns:
            Tuple of (gravity forces array, magnitude of total gravity force).
        """
        if self._include_gravity:
            F_gravity = np.outer(interface_masses, self._gravity)
            total_force = np.sum(F_gravity, axis=0)
            magnitude = np.linalg.norm(total_force)
            return F_gravity, magnitude
        return np.zeros((n_nodes, 3), dtype=np.float64), 0.0

    def _compute_performance_coefficients(
        self,
        thrust: float,
        power_watts: float,
        omega: float,
        radius: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute rotor performance coefficients (Ct, Cp, TSR).

        Args:
            thrust: Thrust force in Newtons.
            power_watts: Power in Watts.
            omega: Angular velocity in rad/s.
            radius: Rotor radius in meters. If None, uses the stored reference radius.
                   For accurate coefficients, pass the deformed radius.

        Returns:
            Tuple of (Ct, Cp, TSR).
        """
        # Use provided radius (deformed) or fall back to stored radius
        effective_radius = radius if radius is not None else self._rotor_radius
        if effective_radius is None:
            effective_radius = 1.0

        area = np.pi * effective_radius**2
        q_dynamic = 0.5 * self._fluid_density * self._flow_velocity**2

        denom_force = q_dynamic * area
        denom_power = q_dynamic * area * self._flow_velocity

        ct = thrust / denom_force if abs(denom_force) > _MIN_DENOMINATOR else 0.0
        cp = power_watts / denom_power if abs(denom_power) > _MIN_DENOMINATOR else 0.0
        tsr = (
            (abs(omega) * effective_radius) / self._flow_velocity
            if abs(self._flow_velocity) > _MIN_DENOMINATOR
            else 0.0
        )

        return ct, cp, tsr

    # =========================================================================
    # Logging and Output
    # =========================================================================

    def _is_primary_rank(self) -> bool:
        """Check if current process is the primary MPI rank."""
        return self.comm.getRank() == 0

    def _print_header(self, title: str) -> None:
        """Print a formatted header section."""
        if self._is_primary_rank():
            print("\n" + "═" * 70, flush=True)
            print(f"  {title}", flush=True)
            print("═" * 70, flush=True)

    def _print_separator(self) -> None:
        """Print a section separator."""
        if self._is_primary_rank():
            print("═" * 70, flush=True)

    def _print_phase(self, phase: int, total: int, message: str) -> None:
        """Print a phase progress message."""
        if self._is_primary_rank():
            print(f"  [{phase}/{total}] {message}", flush=True)

    def _print_info(self, message: str) -> None:
        """Print an info message."""
        if self._is_primary_rank():
            print(f"  [Info] {message}", flush=True)

    def _log_rotor_performance(
        self,
        t: float,
        omega_rpm: float,
        angle_deg: float,
        driving_torque: float,
        torque_vector: np.ndarray,
        thrust: float,
        power: float,
        cp: float,
        ct: float,
        tsr: float,
        deformed_radius: Optional[float] = None,
    ) -> None:
        """
        Write rotor performance metrics to CSV log (rank 0 only).

        Args:
            t: Current time in seconds.
            omega_rpm: Angular velocity in RPM.
            angle_deg: Rotation angle in degrees.
            driving_torque: Driving torque in N·m.
            torque_vector: Full torque vector [Tx, Ty, Tz] in N·m.
            thrust: Thrust force in N.
            power: Power in W.
            cp: Power coefficient.
            ct: Thrust coefficient.
            tsr: Tip speed ratio.
            deformed_radius: Current deformed radius in meters (optional).
        """
        if not self._is_primary_rank():
            return

        log_path = os.path.join(self.solver_params["output_folder"], "rotor_performance.csv")
        file_exists = os.path.exists(log_path)

        try:
            with open(log_path, "a") as f:
                if not file_exists:
                    header = (
                        "Time [s],RPM,Angle [deg],Driving Torque [Nm],Thrust [N],"
                        "Power [W],Cp,Ct,TSR,Torque X [Nm],Torque Y [Nm],Torque Z [Nm],"
                        "Deformed Radius [m]\n"
                    )
                    f.write(header)

                radius_str = f"{deformed_radius:.6f}" if deformed_radius is not None else ""
                line = (
                    f"{t:.6f},{omega_rpm:.4f},{angle_deg:.4f},{driving_torque:.6e},"
                    f"{thrust:.6e},{power:.6e},{cp:.6f},{ct:.6f},{tsr:.6f},"
                    f"{torque_vector[0]:.6e},{torque_vector[1]:.6e},{torque_vector[2]:.6e},"
                    f"{radius_str}\n"
                )
                f.write(line)
        except Exception as e:
            _logger.warning("Failed to write rotor log: %s", e)

    def _log_configuration_summary(
        self,
        beta: float,
        gamma: float,
        omega_initial: float,
        K_G_enabled: bool,
        C_enabled: bool,
        gravity_force_mag: float,
    ) -> None:
        """Log the solver configuration summary."""
        if not self._is_primary_rank():
            return

        self._print_separator()
        print(f"  dt = {self.dt:.6f} s  │  Newmark β={beta:.2f}, γ={gamma:.2f}", flush=True)
        print(
            f"  ω = {omega_initial:.4f} rad/s  │  axis = {self._coord_transforms.axis}",
            flush=True,
        )
        print(
            f"  K_G: {'enabled' if K_G_enabled else 'disabled'}  │  "
            f"Damping: {'enabled' if C_enabled else 'disabled'}  │  "
            f"Solver: {self._solver_type}",
            flush=True,
        )
        print(
            f"  Centrifugal: {self._include_centrifugal}  │  Coriolis: {self._include_coriolis}",
            flush=True,
        )
        if self._include_gravity:
            print(
                f"  Gravity: {self._gravity} m/s²  │  |F_g| = {gravity_force_mag:.4e} N",
                flush=True,
            )
        else:
            print("  Gravity: disabled", flush=True)
        self._print_separator()

    def _log_time_step_header(
        self, time_step: int, step: int, t_target: float, theta_deg: float
    ) -> None:
        """Log time step header information."""
        if not self._is_primary_rank():
            return

        print(f"\n{'─' * 70}", flush=True)
        print(
            f"  TIME WINDOW {time_step + 1:4d}  │  ITER {step:4d}  │  "
            f"t → {t_target:.6f} s  │  θ_est = {theta_deg:.1f}°",
            flush=True,
        )
        print(f"{'─' * 70}", flush=True)

    def _log_force_statistics(
        self,
        raw_force: np.ndarray,
        raw_max_nodal: float,
        n_nodes: int,
        local_force_mag: float,
        omega: float,
        inertial_diags: Dict[str, Any],
        F_gravity_local: Optional[np.ndarray],
        torque_global: np.ndarray,
        torque_power: float,
    ) -> None:
        """Log force statistics for current time step."""
        if not self._is_primary_rank():
            return

        raw_force_mag = np.linalg.norm(raw_force)

        print("  ┌─ CFD FORCES (global frame)", flush=True)
        print(f"  │  Total:   |F| = {raw_force_mag:12.4e} N", flush=True)
        print(
            f"  │  Components: Fx={raw_force[0]:+.4e}  "
            f"Fy={raw_force[1]:+.4e}  Fz={raw_force[2]:+.4e}",
            flush=True,
        )
        print(f"  │  Max nodal:  {raw_max_nodal:.4e} N  ({n_nodes} nodes)", flush=True)
        print("  │", flush=True)
        print("  ├─ TRANSFORMED TO ROTATING FRAME", flush=True)
        print(f"  │  Total:   |F| = {local_force_mag:12.4e} N", flush=True)
        print("  │", flush=True)
        print(f"  ├─ INERTIAL FORCES (ω={omega:.4f} rad/s)", flush=True)

        if "centrifugal" in inertial_diags:
            print(
                f"  │  Centrifugal: {inertial_diags['centrifugal']['total']:.4e} N",
                flush=True,
            )
        if "coriolis" in inertial_diags:
            print(
                f"  │  Coriolis:    {inertial_diags['coriolis']['total']:.4e} N",
                flush=True,
            )
        if "euler" in inertial_diags:
            print(
                f"  │  Euler:       {inertial_diags['euler']['total']:.4e} N",
                flush=True,
            )
        print(
            f"  │  Total inertial: {inertial_diags['total_inertial']['total']:.4e} N",
            flush=True,
        )

        if self._include_gravity and F_gravity_local is not None:
            gravity_local_total = np.linalg.norm(np.sum(F_gravity_local, axis=0))
            print(f"  │  Gravity (local): {gravity_local_total:.4e} N", flush=True)

        print("  │", flush=True)
        print("  ├─ ROTOR TORQUE (Rotating Frame)", flush=True)
        print(
            f"  │  Total Vector: [{torque_global[0]:.4e}, "
            f"{torque_global[1]:.4e}, {torque_global[2]:.4e}] N·m",
            flush=True,
        )
        print(
            f"  │  Driving Torque: {torque_power:.4e} N·m (Projected on axis)",
            flush=True,
        )
        print("  └" + "─" * 67, flush=True)

    def _log_solver_response(self, ksp_its: int, ksp_reason: int, max_disp: float) -> None:
        """Log linear solver response."""
        if not self._is_primary_rank():
            return

        print("  ┌─ SOLVER RESPONSE", flush=True)
        print(f"  │  KSP iterations: {ksp_its}  (reason: {ksp_reason})", flush=True)
        print(f"  │  max|u_new| = {max_disp:.4e} m", flush=True)
        print("  └" + "─" * 67, flush=True)

    def _log_time_window_converged(
        self, t: float, max_disp: float, max_vel: float, max_acc: float
    ) -> None:
        """Log time window convergence."""
        if not self._is_primary_rank():
            return

        print("  ┌─ ✓ TIME WINDOW CONVERGED", flush=True)
        print(f"  │  max|u| = {max_disp:.4e} m", flush=True)
        print(f"  │  max|v| = {max_vel:.4e} m/s", flush=True)
        print(f"  │  max|a| = {max_acc:.4e} m/s²", flush=True)
        print(
            f"  └─ Advanced to t = {t:.6f} s  │  θ = {np.degrees(self._theta):.1f}°",
            flush=True,
        )

    def _log_finalization(self, t: float, time_step: int, step: int) -> None:
        """Log simulation finalization."""
        if not self._is_primary_rank():
            return

        self._print_header("FSI SIMULATION COMPLETED (CO-ROTATIONAL)")
        print(f"  Final time:        {t:.6f} s", flush=True)
        print(f"  Final θ:           {np.degrees(self._theta):.1f}°", flush=True)
        print(f"  Time steps:        {time_step}", flush=True)
        print(f"  Total iterations:  {step}", flush=True)
        if time_step > 0:
            print(f"  Avg iters/step:    {step / time_step:.2f}", flush=True)
        print("═" * 70 + "\n", flush=True)

    # =========================================================================
    # Time Integration Helpers
    # =========================================================================

    def _compute_effective_force(
        self,
        F_new_red: PETSc.Vec,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        M_red: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        coeffs: NewmarkCoefficients,
    ) -> PETSc.Vec:
        """
        Compute effective force vector for Newmark formulation.

        F_eff = F + M*(a0*u + a2*v + a3*a) + C*(a1*u + a4*v + a5*a)
        """
        # Mass contribution
        temp_vec = u.duplicate()
        u.copy(temp_vec)
        temp_vec.scale(coeffs.a0)
        temp_vec.axpy(coeffs.a2, v)
        temp_vec.axpy(coeffs.a3, a)

        temp_result = M_red.createVecRight()
        M_red.mult(temp_vec, temp_result)

        F_eff = F_new_red.duplicate()
        F_new_red.copy(F_eff)
        F_eff.axpy(1.0, temp_result)

        # Damping contribution
        if C_red is not None:
            temp_vec_c = u.duplicate()
            u.copy(temp_vec_c)
            temp_vec_c.scale(coeffs.a1)
            temp_vec_c.axpy(coeffs.a4, v)
            temp_vec_c.axpy(coeffs.a5, a)

            temp_result_c = C_red.createVecRight()
            C_red.mult(temp_vec_c, temp_result_c)
            F_eff.axpy(1.0, temp_result_c)

            temp_vec_c.destroy()
            temp_result_c.destroy()

        temp_vec.destroy()
        temp_result.destroy()

        return F_eff

    def _newmark_update(
        self,
        u: PETSc.Vec,
        u_new: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        coeffs: NewmarkCoefficients,
        beta: float,
    ) -> Tuple[PETSc.Vec, PETSc.Vec]:
        """
        Perform Newmark state update.

        Args:
            u: Current displacement.
            u_new: New displacement.
            v: Current velocity.
            a: Current acceleration.
            coeffs: Newmark coefficients.
            beta: Newmark beta parameter.

        Returns:
            Tuple of (new velocity, new acceleration).
        """
        delta_u = u_new - u
        a_new = (delta_u - self.dt * v - (0.5 - beta) * self.dt**2 * a) / (beta * self.dt**2)
        v_new = v + coeffs.a6 * a + coeffs.a7 * a_new

        return v_new, a_new

    # =========================================================================
    # Main Solve Method
    # =========================================================================

    def solve(self) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """
        Perform co-rotational dynamic FSI analysis.

        The solver operates in a rotating reference frame:
        1. Assemble K, M, C, K_G (constant throughout simulation)
        2. Form effective stiffness K_eff and factorize (once)
        3. Time loop:
           a. Get omega, alpha from OmegaProvider
           b. Update rotation angle theta
           c. Read forces from preCICE and transform to rotating frame
           d. Compute inertial forces (centrifugal, Coriolis)
           e. Solve for displacement in rotating frame
           f. Transform displacement to inertial frame and write to preCICE
           g. Update state variables

        Returns:
            Tuple of (displacement, velocity, acceleration) PETSc vectors.
        """
        omega_initial, _ = self._omega_provider.get_omega(0.0)

        self._print_header("FSI DYNAMIC ANALYSIS - CO-ROTATIONAL ROTOR SOLVER")

        # Phase 1: Matrix Assembly
        matrices, bc_manager, C, K_G = self._assemble_system_matrices(omega_initial)

        # Phase 2: preCICE Initialization
        t, step, time_step = self._initialize_precice(bc_manager)

        # Phase 3: Reduce matrices and setup solver
        K_red, F_red, M_red = bc_manager.reduced_system
        self.free_dofs = bc_manager.free_dofs

        C_red = bc_manager.reduce_matrix(C) if C is not None else None
        K_G_red = bc_manager.reduce_matrix(K_G) if K_G is not None else None

        # Phase 4: Newmark coefficients and effective stiffness
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        coeffs = NewmarkCoefficients.from_newmark_params(beta, gamma, self.dt)

        K_eff = self._build_effective_stiffness(K_red, M_red, coeffs, K_G_red, C_red)

        if not self._prepared:
            self._setup_solver()
            self._prepared = True

        self._solver.setOperators(K_eff)
        self._m_solver.setOperators(M_red)

        # Phase 5: Initial conditions
        u, v, a, t, time_step = self._setup_initial_conditions(
            K_red, F_red, M_red, bc_manager, t, time_step, omega_initial
        )

        # Phase 6: Interface setup
        interface_coords = self._get_interface_nodal_coords()
        interface_dofs = self.precice_participant.interface_dofs
        n_interface_nodes = interface_coords.shape[0]

        # Auto-detect radius if not provided
        if self._rotor_radius is None:
            self._rotor_radius = self._auto_detect_radius(interface_coords)
            self._print_info(f"Auto-detected rotor radius: {self._rotor_radius:.4f} m")

        # Extract nodal masses
        interface_masses = self._extract_nodal_masses(self.M, interface_dofs)
        if np.sum(interface_masses) == 0.0:
            _logger.warning(
                "Extracted interface masses are ZERO. Falling back to average approximation."
            )
            total_mass = self.M.getDiagonal().sum() / self.domain.dofs_per_node
            avg_mass = total_mass / self.domain.mesh.node_count
            interface_masses[:] = avg_mass

        # Compute gravity forces
        F_gravity_global, gravity_force_mag = self._compute_gravity_forces(
            interface_masses, n_interface_nodes
        )

        # Log configuration
        self._log_configuration_summary(
            beta,
            gamma,
            omega_initial,
            K_G_red is not None,
            C_red is not None,
            gravity_force_mag,
        )

        # Time stepping loop
        t, time_step, step, u, v, a = self._time_stepping_loop(
            u,
            v,
            a,
            t,
            time_step,
            step,
            K_red,
            M_red,
            K_eff,
            C_red,
            K_G_red,
            bc_manager,
            coeffs,
            beta,
            gamma,
            interface_coords,
            interface_dofs,
            interface_masses,
            F_gravity_global,
        )

        # Finalization
        self._log_finalization(t, time_step, step)

        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        self.u = bc_manager.expand_solution(u)
        self.v = bc_manager.expand_solution(v)
        self.a = bc_manager.expand_solution(a)

        return self.u, self.v, self.a

    # =========================================================================
    # Solve Sub-Methods
    # =========================================================================

    def _assemble_system_matrices(
        self, omega_initial: float
    ) -> Tuple[
        Tuple[PETSc.Mat, PETSc.Mat],
        BoundaryConditionManager,
        Optional[PETSc.Mat],
        Optional[PETSc.Mat],
    ]:
        """Assemble stiffness, mass, damping, and geometric stiffness matrices."""
        self._print_phase(1, 7, "Assembling stiffness matrix...")
        self.K = self.domain.assemble_stiffness_matrix()

        self._print_phase(2, 7, "Assembling mass matrix (lumped)...")
        M_consistent = self.domain.assemble_mass_matrix()
        self.M = self.lump_mass_matrix(M_consistent)

        # Geometric stiffness
        K_G = None
        if self._include_geometric_stiffness and omega_initial != 0.0:
            self._print_phase(3, 7, "Assembling geometric stiffness (centrifugal)...")
            try:
                K_G = self.domain.assemble_geometric_stiffness(
                    omega=omega_initial,
                    rotation_axis=self._coord_transforms.axis,
                    rotation_center=self._coord_transforms.center,
                )
            except Exception as e:
                _logger.warning("Could not assemble K_G: %s. Proceeding without.", e)
                K_G = None
        else:
            self._print_phase(3, 7, "Geometric stiffness: skipped")

        # Damping matrix
        C = None
        if self._eta_m != 0.0 or self._eta_k != 0.0:
            self._print_phase(
                4, 7, f"Creating Rayleigh damping (η_m={self._eta_m}, η_k={self._eta_k})..."
            )
            C = self._create_damping_matrix(self.K, self.M)
        else:
            self._print_phase(4, 7, "Rayleigh damping: disabled")

        # Force vector
        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        # Boundary conditions
        self._print_phase(5, 7, "Applying boundary conditions...")
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        if self._is_primary_rank():
            print(
                f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, "
                f"Free: {len(bc_manager.free_dofs)} DOFs",
                flush=True,
            )

        return (self.K, self.M), bc_manager, C, K_G

    def _initialize_precice(self, bc_manager: BoundaryConditionManager) -> Tuple[float, int, int]:
        """Initialize preCICE coupling and return timing state."""
        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.solver_params.get("time_step", 1.0)))

        self._print_phase(6, 7, "Initializing preCICE coupling...")
        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )
        self.dt = self.precice_participant.dt

        return t, step, time_step

    def _setup_initial_conditions(
        self,
        K_red: PETSc.Mat,
        F_red: PETSc.Vec,
        M_red: PETSc.Mat,
        bc_manager: BoundaryConditionManager,
        t: float,
        time_step: int,
        omega_initial: float,
    ) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec, float, int]:
        """Setup initial conditions and restore from checkpoint if available."""
        self._print_phase(7, 7, "Setting up linear solver...")

        u = K_red.createVecRight()
        v = K_red.createVecRight()
        a = M_red.createVecRight()

        # Compute initial acceleration
        residual = F_red.duplicate()
        residual.copy(F_red)
        K_red.mult(u, residual)
        residual.scale(-1.0)
        residual.axpy(1.0, F_red)
        self._m_solver.solve(residual, a)

        # Try checkpoint restoration
        checkpoint_state = self._try_restore_checkpoint()
        starting_from_zero = True

        if checkpoint_state is not None:
            starting_from_zero = False
            if self._is_primary_rank():
                print(
                    f"  ✓ Restored from checkpoint at t = {checkpoint_state['t']:.6f} s",
                    flush=True,
                )
            t = checkpoint_state["t"]
            time_step = checkpoint_state["time_step"]
            self._theta = checkpoint_state.get("theta", omega_initial * t)

            if "u_red" in checkpoint_state:
                u.array[:] = checkpoint_state["u_red"]
                v.array[:] = checkpoint_state["v_red"]
                a.array[:] = checkpoint_state["a_red"]
            else:
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                if self._is_primary_rank():
                    print("  ↳ State vectors reinitialized to zero", flush=True)

            if self.solver_params.get("reset_state_on_restart", False):
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                if self._is_primary_rank():
                    print("  ↳ State vectors reset to zero", flush=True)

        # Write initial state
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
                theta=self._theta,
                u_red=u.array.copy(),
                v_red=v.array.copy(),
                a_red=a.array.copy(),
                u_full=bc_manager.expand_solution(u).array.copy(),
                v_full=bc_manager.expand_solution(v).array.copy(),
                a_full=bc_manager.expand_solution(a).array.copy(),
            )

        return u, v, a, t, time_step

    def _time_stepping_loop(
        self,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        t: float,
        time_step: int,
        step: int,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        K_eff: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        K_G_red: Optional[PETSc.Mat],
        bc_manager: BoundaryConditionManager,
        coeffs: NewmarkCoefficients,
        beta: float,
        gamma: float,
        interface_coords: np.ndarray,
        interface_dofs: np.ndarray,
        interface_masses: np.ndarray,
        F_gravity_global: np.ndarray,
    ) -> Tuple[float, int, int, PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """
        Main time stepping loop for the FSI simulation.

        Returns:
            Tuple of (final_t, final_time_step, final_step, u, v, a).
        """
        while self.precice_participant.is_coupling_ongoing:
            step += 1

            # Handle preCICE checkpointing
            if self.precice_participant.requires_writing_checkpoint:
                _logger.debug("Step %d: Writing checkpoint at t = %.6f s", step, t)
                self.precice_participant.store_checkpoint((u, v, a, t, self._theta))

            if not self.precice_participant.requires_reading_checkpoint:
                self._v_guess_next = None

            # Get angular velocity and update K_G if needed
            t_target = t + self.dt
            omega, alpha = self._omega_provider.get_omega(t_target)

            K_eff, K_G_red = self._update_geometric_stiffness_if_needed(
                omega, K_red, M_red, K_eff, C_red, K_G_red, bc_manager, coeffs
            )
            self._prev_omega = omega

            theta_target = self._theta + omega * self.dt

            # Read and process forces
            data_global, raw_force, raw_max_nodal, n_nodes = self._read_and_reduce_forces()

            # Transform to rotating frame
            data_local_2d = self._coord_transforms.transform_force_to_rotating(
                data_global.flatten(), theta_target
            ).reshape(-1, 3)
            local_force_mag = np.linalg.norm(np.sum(data_local_2d, axis=0))

            # Compute inertial forces
            v_for_inertial = self._v_guess_next if self._v_guess_next is not None else v
            v_full = bc_manager.expand_solution(v_for_inertial)
            interface_velocities = self._get_interface_nodal_velocities(v_full, interface_dofs)

            F_inertial, inertial_diags = self._inertial_calculator.compute_all_inertial_forces(
                nodal_coords=interface_coords,
                nodal_velocities=interface_velocities,
                nodal_masses=interface_masses,
                omega=omega,
                alpha=alpha,
                include_centrifugal=self._include_centrifugal,
                include_coriolis=self._include_coriolis,
                include_euler=True,
            )

            # Transform gravity to rotating frame
            F_gravity_local = self._transform_gravity_to_local(
                F_gravity_global, theta_target, data_local_2d.shape
            )

            # Combine forces
            data_combined = data_local_2d + F_inertial + F_gravity_local

            # Compute torque and performance metrics
            u_full_disp = bc_manager.expand_solution(u)
            interface_disps = self._get_interface_displacements(u_full_disp, interface_dofs)
            torque_global, torque_power = self._compute_rotor_torque(
                interface_coords, interface_disps, data_combined
            )

            # Update rotor radius with deformation for accurate performance coefficients
            current_radius = self._compute_rotor_radius(interface_coords, interface_disps)

            thrust = np.dot(raw_force, self._coord_transforms.axis)
            power_watts = torque_power * omega
            ct, cp, tsr = self._compute_performance_coefficients(
                thrust, power_watts, omega, radius=current_radius
            )

            # Log performance
            self._log_rotor_performance(
                t_target,
                omega * 30.0 / np.pi,
                np.degrees(theta_target),
                torque_power,
                torque_global,
                thrust,
                power_watts,
                cp,
                ct,
                tsr,
                deformed_radius=current_radius,
            )

            # Log time step info
            self._log_time_step_header(time_step, step, t_target, np.degrees(theta_target))
            self._log_force_statistics(
                raw_force,
                raw_max_nodal,
                n_nodes,
                local_force_mag,
                omega,
                inertial_diags,
                F_gravity_local if self._include_gravity else None,
                torque_global,
                torque_power,
            )

            # Solve linear system
            u_new, ksp_its, ksp_reason = self._solve_time_step(
                u,
                v,
                a,
                data_combined,
                interface_dofs,
                K_eff,
                M_red,
                C_red,
                bc_manager,
                coeffs,
            )

            max_disp_iter = u_new.norm(PETSc.NormType.INFINITY)
            self._log_solver_response(ksp_its, ksp_reason, max_disp_iter)

            # Update velocity guess for next sub-iteration
            self._update_velocity_guess(u, u_new, v, a, beta, gamma)

            # Write displacement to preCICE
            self._write_displacement_to_precice(u_new, interface_dofs, theta_target, bc_manager)

            # Advance preCICE
            if self._is_primary_rank():
                print("  ┌─ preCICE " + "─" * 57, flush=True)
            self.precice_participant.advance(self.dt)
            if self._is_primary_rank():
                print("  └" + "─" * 67, flush=True)

            # Update state or restore checkpoint
            if self.precice_participant.requires_reading_checkpoint:
                _logger.debug("Step %d: Reading checkpoint (sub-iteration)", step)
                u_new.destroy()
                u_old, v_old, a_old, t_old, theta_old = (
                    self.precice_participant.retrieve_checkpoint()
                )
                u.destroy()
                v.destroy()
                a.destroy()
                u, v, a, t, self._theta = u_old, v_old, a_old, t_old, theta_old
            else:
                time_step += 1
                self._theta += omega * self.dt

                v_new, a_new = self._newmark_update(u, u_new, v, a, coeffs, beta)

                u.destroy()
                v.destroy()
                a.destroy()
                u, v, a = u_new, v_new, a_new
                t += self.dt

                max_disp = u.norm(PETSc.NormType.INFINITY)
                max_vel = v.norm(PETSc.NormType.INFINITY)
                max_acc = a.norm(PETSc.NormType.INFINITY)

                self._log_time_window_converged(t, max_disp, max_vel, max_acc)

                self._handle_checkpoint(
                    t=t,
                    time_step=time_step,
                    dt=self.dt,
                    theta=self._theta,
                    omega=omega,
                    u_red=u.array.copy(),
                    v_red=v.array.copy(),
                    a_red=a.array.copy(),
                    u_full=bc_manager.expand_solution(u).array.copy(),
                    v_full=bc_manager.expand_solution(v).array.copy(),
                    a_full=bc_manager.expand_solution(a).array.copy(),
                )

        return t, time_step, step, u, v, a

    # =========================================================================
    # Time Step Helper Methods
    # =========================================================================

    def _update_geometric_stiffness_if_needed(
        self,
        omega: float,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        K_eff: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        K_G_red: Optional[PETSc.Mat],
        bc_manager: BoundaryConditionManager,
        coeffs: NewmarkCoefficients,
    ) -> Tuple[PETSc.Mat, Optional[PETSc.Mat]]:
        """Update geometric stiffness if omega changed significantly."""
        if self._prev_omega is not None and abs(omega - self._prev_omega) > _OMEGA_CHANGE_THRESHOLD:
            if self._include_geometric_stiffness and K_G_red is not None:
                self._print_info(f"Updating K_G (ω changed: {self._prev_omega:.4f} -> {omega:.4f})")
                try:
                    K_G_new = self.domain.assemble_geometric_stiffness(
                        omega=omega,
                        rotation_axis=self._coord_transforms.axis,
                        rotation_center=self._coord_transforms.center,
                    )
                    K_G_red = bc_manager.reduce_matrix(K_G_new)
                    K_eff = self._build_effective_stiffness(K_red, M_red, coeffs, K_G_red, C_red)
                    self._solver.setOperators(K_eff)
                except Exception as e:
                    _logger.warning("Failed to update K_G: %s", e)

        return K_eff, K_G_red

    def _read_and_reduce_forces(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Read forces from preCICE and compute global reductions."""
        _logger.debug("Reading coupling data from preCICE...")
        data_global = self.precice_participant.read_data()

        if data_global.ndim == 1:
            data_global = data_global.reshape(-1, 3)

        n_nodes = data_global.shape[0]
        raw_force_local = np.sum(data_global, axis=0)
        raw_force = self._allreduce_sum(raw_force_local)
        raw_max_nodal = np.max(np.linalg.norm(data_global, axis=1))

        return data_global, raw_force, raw_max_nodal, n_nodes

    def _transform_gravity_to_local(
        self, F_gravity_global: np.ndarray, theta: float, shape: Tuple[int, int]
    ) -> np.ndarray:
        """Transform gravity forces to rotating frame."""
        if self._include_gravity:
            return self._coord_transforms.transform_force_to_rotating(
                F_gravity_global.flatten(), theta
            ).reshape(shape)
        return np.zeros(shape, dtype=np.float64)

    def _compute_rotor_torque(
        self,
        interface_coords: np.ndarray,
        interface_disps: np.ndarray,
        data_combined: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute rotor torque from combined forces."""
        r_vec = (interface_coords + interface_disps) - self._coord_transforms.center
        nodal_torques = np.cross(r_vec, data_combined)
        torque_local_sum = np.sum(nodal_torques, axis=0)
        torque_global = self._allreduce_sum(torque_local_sum)
        torque_power = np.dot(torque_global, self._coord_transforms.axis)

        return torque_global, torque_power

    def _solve_time_step(
        self,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        data_combined: np.ndarray,
        interface_dofs: np.ndarray,
        K_eff: PETSc.Mat,
        M_red: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        bc_manager: BoundaryConditionManager,
        coeffs: NewmarkCoefficients,
    ) -> Tuple[PETSc.Vec, int, int]:
        """Solve the linear system for a single time step."""
        # Assemble force vector
        F_new = self.F.duplicate()
        self.F.copy(F_new)
        F_new.setValues(interface_dofs, data_combined.flatten())
        F_new_red = bc_manager.reduce_vector(F_new)

        # Compute effective force
        F_eff = self._compute_effective_force(F_new_red, u, v, a, M_red, C_red, coeffs)

        # Solve
        _logger.debug("Solving linear system...")
        u_new = K_eff.createVecRight()
        self._solver.solve(F_eff, u_new)

        # Cleanup
        F_new.destroy()
        F_new_red.destroy()
        F_eff.destroy()

        ksp_its = self._solver.getIterationNumber()
        ksp_reason = self._solver.getConvergedReason()

        return u_new, ksp_its, ksp_reason

    def _update_velocity_guess(
        self,
        u: PETSc.Vec,
        u_new: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        beta: float,
        gamma: float,
    ) -> None:
        """Update velocity guess for next sub-iteration (Coriolis)."""
        a_new_iter = u_new.duplicate()
        u_new.copy(a_new_iter)
        a_new_iter.axpy(-1.0, u)
        a_new_iter.axpy(-self.dt, v)
        a_new_iter.axpy(-(0.5 - beta) * self.dt**2, a)
        a_new_iter.scale(1.0 / (beta * self.dt**2))

        v_new_iter = v.duplicate()
        v.copy(v_new_iter)
        v_new_iter.axpy(self.dt * (1.0 - gamma), a)
        v_new_iter.axpy(self.dt * gamma, a_new_iter)

        if self._v_guess_next is not None:
            self._v_guess_next.destroy()
        self._v_guess_next = v_new_iter

        a_new_iter.destroy()

    def _write_displacement_to_precice(
        self,
        u_new: PETSc.Vec,
        interface_dofs: np.ndarray,
        theta_target: float,
        bc_manager: BoundaryConditionManager,
    ) -> None:
        """Transform displacement to global frame and write to preCICE."""
        u_full_new = bc_manager.expand_solution(u_new)

        u_interface_local = u_full_new.array[interface_dofs].reshape(-1, 3)
        u_interface_global = self._coord_transforms.transform_displacement_to_inertial(
            u_interface_local.flatten(), theta_target
        )

        _logger.debug("Writing displacement data to preCICE...")
        u_full_global = u_full_new.array.copy()
        u_full_global[interface_dofs.flatten()] = u_interface_global
        self.precice_participant.write_data(u_full_global)
        u_full_new.destroy()
