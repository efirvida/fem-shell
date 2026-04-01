"""
Co-rotational FSI Solver for Rotating Structures.

This module provides the LinearDynamicFSIRotorSolver for rotor/turbine FSI problems
with support for rotating reference frames and inertial forces.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel

from .corotational import (
    ComputedOmega,
    ConstantOmega,
    CoordinateTransforms,
    InertialForcesCalculator,
    RampedComputedOmega,
    RampedOmega,
)
from .linear_dynamic import LinearDynamicFSISolver

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
    solver.rotor.include_euler : bool
        Include Euler forces (tangential acceleration). Default: True
        Only applies when alpha != 0 (during ramp or dynamic omega).
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
        omega_value = float(rotor_cfg.get("omega", 0.0))
        omega_ramp_time = float(rotor_cfg.get("omega_ramp_time", 0.0))

        # Check for dynamic omega (ComputedOmega)
        moment_of_inertia = rotor_cfg.get("moment_of_inertia")
        resistive_torque = rotor_cfg.get("resistive_torque", 0.0)
        self._auto_inertia = False

        # Priority: auto-inertia > explicit inertia > ramp-only > constant
        if isinstance(moment_of_inertia, str) and moment_of_inertia.lower() == "auto":
            # Auto-compute inertia from mesh - will be resolved in solve()
            self._auto_inertia = True
            self._auto_inertia_params = {
                "target_omega": omega_value,
                "ramp_time": omega_ramp_time,
                "resistive_torque": resistive_torque,
            }
            # Temporary provider until inertia is computed
            if omega_ramp_time > 0.0:
                self._omega_provider = RampedOmega(
                    target_omega=omega_value, ramp_time=omega_ramp_time
                )
            else:
                self._omega_provider = ConstantOmega(omega=omega_value)
        elif moment_of_inertia is not None:
            # Explicit moment of inertia provided
            inertia_val = float(moment_of_inertia)
            if omega_ramp_time > 0.0:
                # Ramp + dynamic: use combined provider
                self._omega_provider = RampedComputedOmega(
                    target_omega=omega_value,
                    ramp_time=omega_ramp_time,
                    moment_of_inertia=inertia_val,
                    resistive_torque=resistive_torque,
                )
            else:
                # No ramp: pure dynamic from start
                self._omega_provider = ComputedOmega(
                    moment_of_inertia=inertia_val,
                    initial_omega=omega_value,
                    resistive_torque=resistive_torque,
                )
        elif omega_ramp_time > 0.0:
            # Ramp only, no dynamic computation
            self._omega_provider = RampedOmega(target_omega=omega_value, ramp_time=omega_ramp_time)
        else:
            # Constant omega
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
        self._include_euler = rotor_cfg.get("include_euler", True)
        self._kg_update_interval = rotor_cfg.get("kg_update_interval", 0)  # Reserved
        self._force_ramp_time = float(rotor_cfg.get("force_ramp_time", 0.0))
        self._send_omega_to_precice = rotor_cfg.get("send_omega_to_precice", True)

        # Force sanity checks: detect diverged forces from CFD before they
        # contaminate the structural solve.
        # force_max_magnitude: absolute upper bound on |F_total| [N]; None = disabled.
        # force_jump_factor:   if |F_new| > factor * max_seen, treat as CFD divergence.
        _fmax = rotor_cfg.get("force_max_magnitude", None)
        self._force_max_magnitude: Optional[float] = float(_fmax) if _fmax is not None else None
        self._force_jump_factor: float = float(rotor_cfg.get("force_jump_factor", 1000.0))
        self._max_force_seen: float = 0.0  # running maximum |F_total| observed

        # Coordinate transformation options for preCICE data exchange
        # transform_displacement_to_inertial: If True, transforms displacement
        # from rotating frame to inertial frame using R(θ)·u before sending to preCICE.
        # Default is False because the FEM solver uses a STATIC mesh in global coordinates,
        # so the displacement is already in the inertial frame.
        # Only set to True if your formulation computes displacement in body-fixed coordinates.
        # transform_displacement_to_inertial: If True, transforms displacement
        # from rotating frame to inertial frame using R(θ)·u before sending to preCICE.
        # Default is set to True to ensure compatibility with OpenFOAM Dynamic Mesh
        # which expects displacements in the global frame to apply on top of mesh motion.
        self._transform_displacement = rotor_cfg.get("transform_displacement_to_inertial", True)

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
        Extract nodal masses from lumped mass matrix diagonal in serial.
        """
        diag_vec = M_lumped.createVecRight()
        M_lumped.getDiagonal(diag_vec)

        # In serial, we have the full array locally
        diag_array = diag_vec.getArray(readonly=True)

        n_nodes = interface_dofs.shape[0]
        nodal_masses = np.zeros(n_nodes, dtype=np.float64)

        for i in range(n_nodes):
            # Take the mass of the first translational DOF (x-direction)
            if interface_dofs.ndim == 2:
                dof_idx = interface_dofs[i, 0]
            else:
                dof_idx = interface_dofs[i * 3]

            if dof_idx < len(diag_array):
                nodal_masses[i] = diag_array[dof_idx]

        diag_vec.destroy()
        return nodal_masses

    def _compute_estimated_inertia(self) -> float:
        """
        Estimate total moment of inertia about rotation axis from mass matrix.

        Uses the lumped mass matrix diagonal and nodal coordinates.
        """
        if self.M is None:
            _logger.warning("Mass matrix not available for inertia estimation.")
            return 1.0

        # Get diagonal mass vector
        diag_vec = self.M.getDiagonal()
        mass_array = diag_vec.getArray(readonly=True)

        nodes = self.domain.mesh.nodes
        dofs_per_node = self.domain.dofs_per_node
        n_nodes = len(nodes)

        total_inertia = 0.0

        center = self._coord_transforms.center
        axis = self._coord_transforms.axis

        # Iterate only over available local data
        # Assumes nodes array aligns with local matrix partition if parallel
        limit_idx = len(mass_array)

        for i in range(n_nodes):
            idx = i * dofs_per_node
            if idx < limit_idx:
                mass = mass_array[idx]

                # Perpendicular distance squared
                pos = nodes[i].coords
                r_vec = pos - center
                proj = np.dot(r_vec, axis)
                r_perp_sq = np.dot(r_vec, r_vec) - proj * proj

                total_inertia += mass * r_perp_sq

        diag_vec.destroy()

        return total_inertia

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

    def _expand_interface_forces_to_full(
        self,
        interface_forces: np.ndarray,
    ) -> np.ndarray:
        """
        Expand interface forces to full mesh array for visualization.

        Parameters
        ----------
        interface_forces : np.ndarray
            Forces at interface nodes, shape (n_interface_indices, dim).

        Returns
        -------
        np.ndarray
            Full force array, shape (n_nodes, 3) for VTU visualization.
        """
        n_nodes = self.domain.mesh.node_count
        full_forces = np.zeros((n_nodes, 3), dtype=np.float64)
        dim = interface_forces.shape[1]

        # Get the proper node indices from the adapter
        interface_node_indices = self.precice_participant.interface_node_indices

        # Map interface forces to their corresponding mesh node indices
        for i, node_idx in enumerate(interface_node_indices):
            if node_idx < n_nodes:
                full_forces[node_idx, :dim] = interface_forces[i, :]

        return full_forces

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

        return local_max_r

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
        """Always True in serial."""
        return True

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
            f"  Centrifugal: {self._include_centrifugal}  │  Coriolis: {self._include_coriolis}  │  Euler: {self._include_euler}",
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
        applied_force: np.ndarray,
        applied_max_nodal: float,
        n_nodes: int,
        local_force_mag: float,
        omega: float,
        inertial_diags: Dict[str, Any],
        F_gravity_local: Optional[np.ndarray],
        torque_global: np.ndarray,
        torque_power: float,
        ramp_factor: Optional[float] = None,
        force_ramp_time: Optional[float] = None,
    ) -> None:
        """Log force statistics for current time step."""
        if not self._is_primary_rank():
            return

        raw_force_mag = np.linalg.norm(raw_force)
        applied_force_mag = np.linalg.norm(applied_force)

        print("  ┌─ CFD FORCES (global frame)", flush=True)
        print("  │  Source: preCICE adapter (raw)", flush=True)
        print(f"  │  Total:   |F| = {raw_force_mag:12.4e} N", flush=True)
        print(
            f"  │  Components: Fx={raw_force[0]:+.4e}  "
            f"Fy={raw_force[1]:+.4e}  Fz={raw_force[2]:+.4e}",
            flush=True,
        )
        print(f"  │  Max nodal:  {raw_max_nodal:.4e} N  ({n_nodes} nodes)", flush=True)
        if ramp_factor is not None and ramp_factor < 1.0:
            print(
                f"  │  ► RAMP ACTIVE: {ramp_factor * 100:5.1f}% applied (Target: {force_ramp_time:.3f}s)",
                flush=True,
            )
            print("  │  Applied to structure (after ramp)", flush=True)
            print(f"  │  Total:   |F| = {applied_force_mag:12.4e} N", flush=True)
            print(
                f"  │  Components: Fx={applied_force[0]:+.4e}  "
                f"Fy={applied_force[1]:+.4e}  Fz={applied_force[2]:+.4e}",
                flush=True,
            )
            print(f"  │  Max nodal:  {applied_max_nodal:.4e} N  ({n_nodes} nodes)", flush=True)
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
        self,
        t: float,
        max_disp: float,
        max_vel: float,
        max_acc: float,
        driving_torque: Optional[float] = None,
        new_alpha: Optional[float] = None,
        new_omega: Optional[float] = None,
        phase_info: Optional[str] = None,
    ) -> None:
        """Log time window convergence."""
        if not self._is_primary_rank():
            return

        print("  ┌─ ✓ TIME WINDOW CONVERGED", flush=True)
        print(f"  │  max|u| = {max_disp:.4e} m", flush=True)
        print(f"  │  max|v| = {max_vel:.4e} m/s", flush=True)
        print(f"  │  max|a| = {max_acc:.4e} m/s²", flush=True)
        if driving_torque is not None and new_alpha is not None and new_omega is not None:
            print(f"  │  τ_drive = {driving_torque:.4e} N·m", flush=True)
            print(f"  │  α = {new_alpha:.4f} rad/s²", flush=True)
            phase_str = f" [{phase_info}]" if phase_info else ""
            print(f"  │  ω_next = {new_omega:.4f} rad/s{phase_str}", flush=True)
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

        # Auto-compute inertia if requested
        if getattr(self, "_auto_inertia", False):
            estimated_inertia = self._compute_estimated_inertia()
            _logger.info("Auto-computed Moment of Inertia: %.4e kg·m²", estimated_inertia)

            # Re-initialize provider with computed inertia
            ramp_time = self._auto_inertia_params.get("ramp_time", 0.0)
            target_omega = self._auto_inertia_params["target_omega"]
            resistive_torque = self._auto_inertia_params["resistive_torque"]

            if ramp_time > 0.0:
                # Use combined ramp + computed provider
                self._omega_provider = RampedComputedOmega(
                    target_omega=target_omega,
                    ramp_time=ramp_time,
                    moment_of_inertia=estimated_inertia,
                    resistive_torque=resistive_torque,
                )
                _logger.info(
                    "Omega mode: Ramp (%.3f s) → Dynamic (I=%.4e kg·m²)",
                    ramp_time,
                    estimated_inertia,
                )
            else:
                # Pure dynamic mode from start
                self._omega_provider = ComputedOmega(
                    moment_of_inertia=estimated_inertia,
                    initial_omega=target_omega,
                    resistive_torque=resistive_torque,
                )
                _logger.info("Omega mode: Dynamic (I=%.4e kg·m²)", estimated_inertia)

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
        u, v, a, t, time_step, starting_from_zero = self._setup_initial_conditions(
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

        # Compute centrifugal forces at initial state (for debugging)
        F_centrifugal_initial, _ = self._inertial_calculator.compute_all_inertial_forces(
            nodal_coords=interface_coords,
            nodal_velocities=np.zeros_like(interface_coords),  # Zero velocity initially
            nodal_masses=interface_masses,
            omega=omega_initial,
            alpha=0.0,
            include_centrifugal=self._include_centrifugal,
            include_coriolis=False,  # No Coriolis at zero velocity
            include_euler=False,  # No Euler at zero alpha
        )

        # Write initial state checkpoint with all debugging fields
        if (
            self._checkpoint_manager is not None
            and self.solver_params.get("write_initial_state", True)
            and starting_from_zero
            and t == 0
        ):
            # Prepare nodal mass field for visualization
            n_mesh_nodes = self.domain.mesh.node_count
            nodal_mass_field = np.zeros(n_mesh_nodes, dtype=np.float64)
            interface_node_indices = self.precice_participant.interface_node_indices
            for i, node_idx in enumerate(interface_node_indices):
                if node_idx < n_mesh_nodes:
                    nodal_mass_field[node_idx] = interface_masses[i]

            # Extract element masses from mass matrix diagonal
            M_diag = self.M.getDiagonal().array
            element_mass_field = np.zeros(n_mesh_nodes, dtype=np.float64)
            dofs_per_node = self.domain.dofs_per_node
            for node_idx in range(n_mesh_nodes):
                dof_start = node_idx * dofs_per_node
                if dof_start < len(M_diag):
                    # Sum mass contributions for this node (first 3 translational DOFs)
                    element_mass_field[node_idx] = M_diag[dof_start]

            # Reference coordinates for interface nodes
            ref_coords_field = np.zeros((n_mesh_nodes, 3), dtype=np.float64)
            for i, node_idx in enumerate(interface_node_indices):
                if node_idx < n_mesh_nodes:
                    ref_coords_field[node_idx, :] = interface_coords[i, :]

            # Compute distance to rotation axis for each node
            distance_to_axis = np.zeros(n_mesh_nodes, dtype=np.float64)
            for i, node_idx in enumerate(interface_node_indices):
                if node_idx < n_mesh_nodes:
                    rel_pos = interface_coords[i, :] - self._coord_transforms.center
                    axis = self._coord_transforms.axis
                    parallel = np.dot(rel_pos, axis) * axis
                    perp = rel_pos - parallel
                    distance_to_axis[node_idx] = np.linalg.norm(perp)

            initial_fields = {
                "NodalMass": nodal_mass_field,
                "ElementMass": element_mass_field,
                "F_Gravity_Global": self._expand_interface_forces_to_full(
                    F_gravity_global.reshape(-1, 3)
                ),
                "F_Centrifugal_Initial": self._expand_interface_forces_to_full(
                    F_centrifugal_initial
                ),
                "RefCoords": ref_coords_field,
                "DistanceToAxis": distance_to_axis,
                "RotationAxis": np.tile(self._coord_transforms.axis, (n_mesh_nodes, 1)).astype(
                    np.float64
                ),
                "RotationCenter": np.tile(self._coord_transforms.center, (n_mesh_nodes, 1)).astype(
                    np.float64
                ),
            }

            self._handle_checkpoint(
                t=0.0,
                time_step=0,
                dt=self.dt,
                theta=self._theta,
                omega=omega_initial,
                u_red=u.array.copy(),
                v_red=v.array.copy(),
                a_red=a.array.copy(),
                u_full=bc_manager.expand_solution(u).array.copy(),
                v_full=bc_manager.expand_state_vector(v).array.copy(),
                a_full=bc_manager.expand_state_vector(a).array.copy(),
                extra_fields=initial_fields,
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
        self.v = bc_manager.expand_state_vector(v)
        self.a = bc_manager.expand_state_vector(a)

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

        custom_meshes = {}
        initial_data = None
        if getattr(self, "_send_omega_to_precice", True):
            # Prepare GlobalSolidMesh with a single vertex at origin (0,0,0)
            # This matches the OpenFOAM adapter which creates a single vertex at origin
            # for global data exchange (like AngularVelocity)
            # Use origin (0,0,0) for global mesh - this is what OpenFOAM adapter expects
            origin_vertex = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
            custom_meshes["GlobalSolidMesh"] = origin_vertex

            # Publish the omega that will be used in the first time window
            # (t -> t + dt), so fluid does not start from a stale zero value.
            initial_omega, _ = self._omega_provider.get_omega(t + self.dt)
            initial_data = {"AngularVelocity": initial_omega}
            _logger.info(f"Initial angular velocity for preCICE: {initial_omega:.4f} rad/s")

        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
            custom_mesh_coords=custom_meshes if custom_meshes else None,
            initial_data_values=initial_data,
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
    ) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec, float, int, bool]:
        """Setup initial conditions and restore from checkpoint if available.

        Returns:
            Tuple of (u, v, a, t, time_step, starting_from_zero).
        """
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

        # Note: Initial state checkpoint is written in solve() after computing
        # all debugging fields (gravity, masses, etc.)
        return u, v, a, t, time_step, starting_from_zero

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
        # Omega must remain constant inside each implicit-coupling time window.
        # Recompute only once at the window start, then hold for all sub-iterations.
        omega, alpha = self._omega_provider.get_omega(t + self.dt)
        theta_target = self._theta + omega * self.dt
        omega_initialized_for_window = False

        while self.precice_participant.is_coupling_ongoing:
            step += 1

            # Handle preCICE checkpointing
            if self.precice_participant.requires_writing_checkpoint:
                _logger.debug("Step %d: Writing checkpoint at t = %.6f s", step, t)
                ckpt_data = (u, v, a, t, self._theta)
                if isinstance(self._omega_provider, (ComputedOmega, RampedComputedOmega)):
                    ckpt_data += (self._omega_provider.get_state(),)
                self.precice_participant.store_checkpoint(ckpt_data)

            # Compute omega only once per time window and hold it fixed for
            # all implicit sub-iterations in that window.
            if not omega_initialized_for_window:
                t_target = t + self.dt
                omega, alpha = self._omega_provider.get_omega(t_target)
                theta_target = self._theta + omega * self.dt

                # Update K_G if omega changed significantly (window-level update only)
                K_eff, K_G_red = self._update_geometric_stiffness_if_needed(
                    omega, K_red, M_red, K_eff, C_red, K_G_red, bc_manager, coeffs
                )
                self._prev_omega = omega
                omega_initialized_for_window = True

            if not self.precice_participant.requires_reading_checkpoint:
                self._v_guess_next = None

            # Use the omega computed at time window start (constant during sub-iterations)
            t_target = t + self.dt

            # [DUAL FRAME FSI - REVERTED]
            # User correctly pointed out that OpenFOAM adapter uses a STATIC mesh interface.
            # Rotating the CSM mesh in PreCICE would cause mesh drift vs OpenFOAM's static interface.
            # We rely on the Static-to-Static mapping and send Global Displacements (R·u).


            # Read and process forces
            data_global, applied_force, applied_max_nodal, n_nodes = self._read_and_reduce_forces()
            raw_force = applied_force.copy()
            raw_max_nodal = applied_max_nodal

            # --- Force sanity: detect CFD divergence before it contaminates the solve ---
            force_total_mag = float(np.linalg.norm(applied_force))
            if self._max_force_seen > 0.0 and force_total_mag > self._force_jump_factor * self._max_force_seen:
                raise RuntimeError(
                    f"Diverged force from fluid solver detected at t={t_target:.6f} s: "
                    f"|F| = {force_total_mag:.3e} N is "
                    f"{force_total_mag / self._max_force_seen:.1f}× the running max "
                    f"({self._max_force_seen:.3e} N). "
                    f"CFD solver likely diverged (Courant blow-up or PIMPLE non-convergence). "
                    "Aborting to prevent FSI hang."
                )
            if self._force_max_magnitude is not None and force_total_mag > self._force_max_magnitude:
                raise RuntimeError(
                    f"Force exceeds configured limit at t={t_target:.6f} s: "
                    f"|F| = {force_total_mag:.3e} N > {self._force_max_magnitude:.3e} N. Aborting."
                )
            # Update running maximum (only after ramp so it reflects real physics)
            self._max_force_seen = max(self._max_force_seen, force_total_mag)
            # -------------------------------------------------------------------------

            # Apply force ramp if configured
            ramp_factor = 1.0
            if self._force_ramp_time > 1e-12:
                ramp_check = min(t_target / self._force_ramp_time, 1.0)
                if ramp_check < 1.0:
                    ramp_factor = ramp_check
                    data_global *= ramp_factor
                    applied_force *= ramp_factor
                    applied_max_nodal *= ramp_factor

            mesh_dim = self.precice_participant.mesh_dimensions

            # Transform forces from global (inertial) frame to rotating frame
            # OpenFOAM computes forces in the global frame (where the structure rotates)
            # The structural solver works in the co-rotating frame (structure appears static)
            # Therefore: F_local = R^T · F_global
            data_local_2d = self._coord_transforms.transform_force_to_rotating(
                data_global.flatten(), theta_target
            ).reshape(-1, mesh_dim)

            # Export debug interface data (forces in rotating frame)
            # We pass data_local_2d as both raw and applied because clipping/ramping
            # is not currently implemented in the rotor solver loop, but rotation is the key transformation.
            self._export_interface_debug_data(step, t_target, None, data_local_2d)

            local_force_mag = np.linalg.norm(np.sum(data_local_2d, axis=0))

            # Compute inertial forces
            v_for_inertial = self._v_guess_next if self._v_guess_next is not None else v
            v_full = bc_manager.expand_solution(v_for_inertial)
            interface_velocities = self._get_interface_nodal_velocities(v_full, interface_dofs)

            # Only include Euler force when enabled AND there is angular acceleration (alpha != 0)
            include_euler = self._include_euler and abs(alpha) > 1e-12

            F_inertial, inertial_diags = self._inertial_calculator.compute_all_inertial_forces(
                nodal_coords=interface_coords,
                nodal_velocities=interface_velocities,
                nodal_masses=interface_masses,
                omega=omega,
                alpha=alpha,
                include_centrifugal=self._include_centrifugal,
                include_coriolis=self._include_coriolis,
                include_euler=include_euler,
            )

            # Transform gravity to rotating frame
            F_gravity_local = self._transform_gravity_to_local(
                F_gravity_global, theta_target, data_local_2d.shape
            )

            # Combine forces
            data_combined = data_local_2d + F_inertial + F_gravity_local

            # Compute interface displacements for torque calculations
            u_full_disp = bc_manager.expand_solution(u)
            interface_disps = self._get_interface_displacements(u_full_disp, interface_dofs)

            # Compute driving torque for dynamic omega (rotor angular momentum balance)
            # Only EXTERNAL forces contribute to rotor acceleration:
            #   - CFD forces (aerodynamic torque)
            #   - Gravity forces (if there's mass imbalance)
            # Inertial forces (centrifugal, Coriolis, Euler) are fictitious forces
            # from the rotating reference frame and don't accelerate the rotor.
            if isinstance(self._omega_provider, (ComputedOmega, RampedComputedOmega)):
                force_for_rotor_dynamics = data_local_2d + F_gravity_local
                torque_driving_vec, _ = self._compute_rotor_torque(
                    interface_coords, interface_disps, force_for_rotor_dynamics
                )
                driving_torque_mag = np.dot(torque_driving_vec, self._coord_transforms.axis)
            else:
                driving_torque_mag = 0.0

            # Compute torque and performance metrics (includes all forces for reporting)
            torque_global, torque_power = self._compute_rotor_torque(
                interface_coords, interface_disps, data_combined
            )

            # Update rotor radius with deformation for accurate performance coefficients
            current_radius = self._compute_rotor_radius(interface_coords, interface_disps)

            thrust = np.dot(applied_force, self._coord_transforms.axis)
            power_watts = torque_power * omega
            ct, cp, tsr = self._compute_performance_coefficients(
                thrust, power_watts, omega, radius=current_radius
            )

            # Log time step info
            self._log_time_step_header(time_step, step, t_target, np.degrees(theta_target))
            self._log_force_statistics(
                raw_force,
                raw_max_nodal,
                applied_force,
                applied_max_nodal,
                n_nodes,
                local_force_mag,
                omega,
                inertial_diags,
                F_gravity_local if self._include_gravity else None,
                torque_global,
                torque_power,
                ramp_factor=ramp_factor if ramp_factor < 1.0 else None,
                force_ramp_time=self._force_ramp_time,
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

            # Check for divergence to prevent hanging the coupling
            if not np.isfinite(max_disp_iter) or max_disp_iter > 1e10:
                _logger.error(
                    "Solver diverged! max|u| = %s. Aborting to prevent FSI hang.", max_disp_iter
                )
                raise RuntimeError(f"Solid solver divergence detected: max|u| = {max_disp_iter}")

            # Update velocity guess for next sub-iteration
            self._update_velocity_guess(u, u_new, v, a, beta, gamma)

            # Write displacement to preCICE
            self._write_displacement_to_precice(u_new, interface_dofs, theta_target, bc_manager)

            # Write angular velocity to preCICE before every advance() so preCICE
            # does not see a zero buffer in sub-iterations.
            self._write_angular_velocity_to_precice(omega)

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

                ckpt_data = self.precice_participant.retrieve_checkpoint()
                u_old, v_old, a_old, t_old, theta_old = ckpt_data[0:5]

                if (
                    isinstance(self._omega_provider, (ComputedOmega, RampedComputedOmega))
                    and len(ckpt_data) > 5
                ):
                    self._omega_provider.set_state(ckpt_data[5])

                # Copy checkpoint data into existing vectors instead of destroying them
                # This is necessary because retrieve_checkpoint() returns the same
                # vector references each time, not fresh copies
                u.array[:] = u_old.array
                v.array[:] = v_old.array
                a.array[:] = a_old.array
                t = t_old
                self._theta = theta_old
            else:
                time_step += 1
                self._theta += omega * self.dt
                omega_initialized_for_window = False

                # Update Omega state after a converged time window.
                rotor_dynamics_info = (None, None, None, None)
                omega_next_for_coupling = omega
                if isinstance(self._omega_provider, (ComputedOmega, RampedComputedOmega)):
                    self._omega_provider.update(driving_torque_mag, self.dt)
                    if isinstance(self._omega_provider, RampedComputedOmega):
                        # Use next window target time to report/publish the omega that
                        # will be used at the start of the next window.
                        # get_state() returns internal state during ramp, not ramped values.
                        next_window_t = t + 2.0 * self.dt
                        new_omega, new_alpha = self._omega_provider.get_omega(next_window_t)
                        omega_next_for_coupling = new_omega
                        phase_info = "ramp" if self._omega_provider.is_in_ramp_phase else "dynamic"
                        rotor_dynamics_info = (driving_torque_mag, new_alpha, new_omega, phase_info)
                    else:
                        new_omega, new_alpha = self._omega_provider.get_state()
                        omega_next_for_coupling = new_omega
                        rotor_dynamics_info = (driving_torque_mag, new_alpha, new_omega, None)
                elif isinstance(self._omega_provider, RampedOmega):
                    # RampedOmega: publish omega for next window start.
                    next_window_t = t + 2.0 * self.dt
                    new_omega, new_alpha = self._omega_provider.get_omega(next_window_t)
                    omega_next_for_coupling = new_omega
                    phase_info = "ramp" if new_alpha > 0 else "constant"
                    rotor_dynamics_info = (driving_torque_mag, new_alpha, new_omega, phase_info)
                elif isinstance(self._omega_provider, ConstantOmega):
                    # ConstantOmega: fixed omega, alpha=0
                    new_omega, new_alpha = self._omega_provider.get_omega(t)
                    omega_next_for_coupling = new_omega
                    rotor_dynamics_info = (driving_torque_mag, new_alpha, new_omega, "constant")

                v_new, a_new = self._newmark_update(u, u_new, v, a, coeffs, beta)

                u.destroy()
                v.destroy()
                a.destroy()
                u, v, a = u_new, v_new, a_new
                t += self.dt

                max_disp = u.norm(PETSc.NormType.INFINITY)
                max_vel = v.norm(PETSc.NormType.INFINITY)
                max_acc = a.norm(PETSc.NormType.INFINITY)

                self._log_time_window_converged(t, max_disp, max_vel, max_acc, *rotor_dynamics_info)

                # Write angular velocity for the NEXT time window start.
                self._write_angular_velocity_to_precice(omega_next_for_coupling)

                # Log converged performance to CSV
                self._log_rotor_performance(
                    t,
                    omega * 30.0 / np.pi,
                    np.degrees(self._theta),
                    torque_power,
                    torque_global,
                    thrust,
                    power_watts,
                    cp,
                    ct,
                    tsr,
                    deformed_radius=current_radius,
                )

                # Prepare force fields for checkpoint visualization
                # Also add nodal masses for debugging using proper node indices
                n_mesh_nodes = self.domain.mesh.node_count
                mass_field = np.zeros(n_mesh_nodes, dtype=np.float64)
                interface_node_indices = self.precice_participant.interface_node_indices
                for i, node_idx in enumerate(interface_node_indices):
                    if node_idx < n_mesh_nodes:
                        mass_field[node_idx] = interface_masses[i]

                force_fields = {
                    "Force CFD": self._expand_interface_forces_to_full(data_local_2d),
                    "Force Inertial": self._expand_interface_forces_to_full(F_inertial),
                    "Force Gravity": self._expand_interface_forces_to_full(F_gravity_local),
                    "Force Total": self._expand_interface_forces_to_full(data_combined),
                    "NodalMass": mass_field,
                }

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
                    v_full=bc_manager.expand_state_vector(v).array.copy(),
                    a_full=bc_manager.expand_state_vector(a).array.copy(),
                    extra_fields=force_fields,
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
        mesh_dim = self.precice_participant.mesh_dimensions

        if data_global.ndim == 1:
            data_global = data_global.reshape(-1, mesh_dim)

        # Sanity check: reject NaN/Inf immediately (fluid divergence produces these)
        if not np.all(np.isfinite(data_global)):
            n_bad = np.count_nonzero(~np.isfinite(data_global))
            raise RuntimeError(
                f"Diverged force data received from fluid solver via preCICE: "
                f"{n_bad} non-finite values (NaN/Inf) in force array. "
                "CFD solver likely diverged. Aborting to prevent FSI hang."
            )

        n_nodes = data_global.shape[0]
        raw_force_local = np.sum(data_global, axis=0)
        raw_force = raw_force_local
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
        torque_global = torque_local_sum
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
        F_new.setValues(interface_dofs, data_combined)
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
        """
        Transform displacement to global frame and write to preCICE.

        In the co-rotational formulation:
        - The structural solver works in a STATIC mesh but conceptually in a rotating frame
        - It computes elastic deformation u_local in this co-rotating frame
        - OpenFOAM rotates its mesh physically using omega from preCICE
        - OpenFOAM then applies the displacement from preCICE ON TOP of its rotation

        For a node at reference position x_ref:
        - OpenFOAM rotates to: R(θ) · x_ref
        - Then adds displacement: R(θ) · x_ref + u_precice
        - The correct final position should be: R(θ) · (x_ref + u_local)

        Therefore: u_precice = R(θ) · u_local

        This transforms the elastic deformation from the rotating frame to the
        inertial/global frame so OpenFOAM can apply it correctly.
        """
        u_full_new = bc_manager.expand_solution(u_new)

        # Get elastic displacement in rotating frame
        u_interface_local = u_full_new.array[interface_dofs].reshape(-1, 3)

        # Optionally transform displacement to global/inertial frame
        if self._transform_displacement:
            # Transform: u_global = R · u_local
            # This is necessary when OpenFOAM first rotates its mesh, then adds
            # the displacement. For the final position to be correct, the displacement
            # must be expressed in the global frame.
            u_interface_output = self._coord_transforms.transform_displacement_to_inertial(
                u_interface_local.flatten(), theta_target
            ).reshape(-1, 3)
        else:
            # No transformation - send displacement as-is in the rotating frame
            # Use this when OpenFOAM expects displacement in the co-rotating frame
            u_interface_output = u_interface_local

        _logger.debug("Writing displacement data to preCICE...")
        u_full_output = u_full_new.array.copy()

        # Log write statistics
        if self._is_primary_rank():
            u_local_max = np.max(np.abs(u_interface_local))
            u_output_max = np.max(np.abs(u_interface_output))
            transform_str = "R·u" if self._transform_displacement else "u (no transform)"
            
            # FSI DEBUG: Displacement
            u_local_norm_max = np.max(np.linalg.norm(u_interface_local, axis=1))
            u_output_norm_max = np.max(np.linalg.norm(u_interface_output, axis=1))
            
            print(f"  ┌─ DISPLACEMENT (Sending to preCICE)", flush=True)
            print(f"  │  Elastic Disp (Local) Max Norm:   {u_local_norm_max:.4e} m", flush=True)
            print(f"  │  Output Disp ({transform_str}) Max Norm: {u_output_norm_max:.4e} m", flush=True)
            print(f"  │  (Should be small elastic deforms, NOT rigid rotation magnitude)", flush=True)
            print(f"  └" + "─" * 40, flush=True)

        u_full_output[interface_dofs.flatten()] = u_interface_output.flatten()
        self.precice_participant.write_data(u_full_output)
        u_full_new.destroy()

    def _write_angular_velocity_to_precice(self, omega: float) -> None:
        """
        Write angular velocity to preCICE on GlobalSolidMesh.

        This writes a single scalar (omega) to the single vertex registered on
        GlobalSolidMesh (the rotation centre).  AngularVelocity lives on a
        *secondary* mesh and is therefore not present in write_data_names
        (which only lists data for the primary Solid-Mesh); we use
        _send_omega_to_precice as the sole gate.
        """
        if getattr(self, "_send_omega_to_precice", True) is False:
            return

        # AngularVelocity is always exchanged under this fixed name on GlobalSolidMesh
        omega_name = "AngularVelocity"

        # Create scalar field for the single GlobalSolidMesh vertex
        # The mesh was registered as a single point (rotation center)
        # So we write a single scalar value: omega

        # Ensure data is an array
        data_to_write = np.array([omega], dtype=np.float64)

        # 4. Write directly to interface
        if self._is_primary_rank():
            print(
                f"  → Writing preCICE Angular Velocity: {omega:.4f} rad/s ({omega_name})",
                flush=True,
            )

        try:
            self.precice_participant.write_interface_data(
                data_to_write, omega_name, mesh_name="GlobalSolidMesh"
            )
        except ValueError:
            # Fallback (unlikely if registered correctly above, but safety for existing flows)
            pass
