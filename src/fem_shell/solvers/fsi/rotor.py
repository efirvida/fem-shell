"""
Co-rotational FSI Solver for Rotating Structures.

This module implements a linear dynamic Fluid-Structure Interaction (FSI) solver
for rotating structures (wind turbine blades, helicopter rotors, propellers) using
a co-rotational formulation in a rotating reference frame coupled to a CFD solver
via the preCICE library.

Governing Equation (Rotating Reference Frame)
----------------------------------------------
The equation of motion solved at each time step is (cf. ANSYS MAPDL Theory
Reference, Eq. 14-57, §14.4.1):

    [M]{ü} + [C]{u̇} + ([K] + [K_G] + [K_SP]){u} = {F_aero} + {F_cf} + {F_cor} + {F_euler} + {F_g}

where:
    [M]    — Lumped mass matrix (diagonal, row-sum of consistent mass).
    [C]    — Rayleigh damping matrix: C = η_m·M + η_k·K.
    [K]    — Linear elastic stiffness matrix (MITC3 shell elements).
    [K_G]  — Geometric stiffness (stress stiffening) from centrifugal prestress.
             Assembled element-by-element: K_G = ∫ B_G^T · S̃ · B_G dA,
             where the in-plane stress S̃ is estimated from the centrifugal
             prestress σ_cf ≈ ρ·ω²·r·L_char.
    [K_SP] — Spin softening matrix (ANSYS Eq. 3-74 / 14-55):
             K_SP = -ω² · M · (I - n̂⊗n̂).
             Diagonal for lumped mass. Reduces effective stiffness in the plane
             perpendicular to the rotation axis. This captures the increase in
             centrifugal loading due to elastic displacement without requiring
             explicit force evaluation at deformed coordinates.

LHS vs RHS treatment of physical effects:
    - [K_G] on LHS:  Stress stiffening (INCREASES natural frequencies)
    - [K_SP] on LHS: Spin softening  (DECREASES natural frequencies in rotation plane)
    - {F_cf} on RHS: Centrifugal force at UNDEFORMED coordinates X₀.
                     The correction for deformed coords (ω×(ω×u)) is captured
                     implicitly by K_SP·u on the LHS. Evaluating F_cf at X₀+u
                     when K_SP is active would double-count the spin softening.
    - {F_cor} on RHS: Coriolis force = -2·m·(ω × v), explicit (lagged velocity).
    - {F_euler} on RHS: Euler force = -m·(α × r) at DEFORMED coordinates X₀+u,
                        evaluated only when angular acceleration α ≠ 0.
                        No LHS correction exists for Euler, so explicit evaluation
                        at deformed coords is appropriate.
    - {F_g} on RHS: Gravity force transformed to rotating frame via R^T(θ).

Time Integration
----------------
Newmark-β method with β = 0.25, γ = 0.5 (average acceleration, unconditionally
stable for linear systems). The effective stiffness formulation yields:

    K_eff = [K] + [K_G] + [K_SP] + a₀·[M] + a₁·[C]
    F_eff = {F} + [M]·(a₀·u + a₂·v + a₃·a) + [C]·(a₁·u + a₄·v + a₅·a)

where a₀..a₅ are Newmark coefficients derived from β, γ, and dt.

FSI Coupling Architecture
-------------------------
The solver uses preCICE with implicit coupling (IQN-ILS acceleration):
    1. At each time window, preCICE orchestrates sub-iterations between the
       fluid (OpenFOAM) and solid (this solver) participants.
    2. Within a time window, ω is held constant — omega is computed once at
       the window start and not re-evaluated during sub-iterations.
    3. K_G and K_SP are rebuilt only when |Δω| > threshold between windows.
    4. Convergence of the FSI fixed-point iteration is handled by preCICE's
       IQN-ILS quasi-Newton acceleration, not by internal structural iterations.

Co-rotational Frame
-------------------
The FEM mesh is STATIC in global coordinates. Elastic deformation u is computed
in the co-rotating frame (structure appears stationary). Data exchange with the
fluid solver follows:
    - Forces:       F_local = R^T(θ) · F_global   (global → rotating)
    - Displacement: u_global = R(θ) · u_local      (rotating → global)

Relationship between K_G and K_SP (ANSYS §3.4–3.5, Eq. 3-88)
--------------------------------------------------------------
K_G and K_SP model DIFFERENT physical effects and coexist:
    - K_G captures the geometric nonlinear stiffening from internal membrane
      stress induced by centrifugal loading (analogous to a taut string).
    - K_SP captures the variation of the external centrifugal FORCE with
      displacement (the force increases as the node moves outward).
    - In ANSYS notation: [K_total] = [K] + [S] + [S̃₂], where [S] is stress
      stiffening and [S̃₂] is spin softening (both functions of ω²).
    - For a rotating blade: K_G stiffens flapwise modes, K_SP softens
      in-plane (lead-lag) modes. Both are essential for correct Campbell diagrams.

Rotor Torque and Angular Velocity Dynamics
------------------------------------------
The solver supports both prescribed and dynamic angular velocity via
OmegaProvider subclasses. The dynamic case solves the rigid-body
rotational equation of motion for the rotor as a whole:

    I · dω/dt = τ_aero + τ_gravity + τ_shaft

where:
    I         — Total moment of inertia about the rotation axis [kg·m²],
                computed as I = Σᵢ mᵢ · r_⊥,ᵢ² (from lumped mass and nodal
                coordinates, or prescribed by user).
    τ_aero    — Aerodynamic driving torque from CFD forces, projected onto
                the rotation axis: τ_aero = n̂ · Σᵢ (rᵢ × F_cfd,ᵢ).
    τ_gravity — Gravitational torque (relevant for mass imbalance).
    τ_shaft   — External shaft torque [N·m] (user-specified, signed).
                Positive drives rotation, negative resists (e.g. generator).

The integration uses explicit Euler:

    α = (τ_driving + τ_shaft) / I
    ω^{n+1} = ω^n + α · Δt

CRITICAL: Only EXTERNAL forces (aerodynamic + gravity) contribute to τ_driving.
Inertial forces (centrifugal, Coriolis, Euler) are fictitious forces in the
rotating frame and do NOT accelerate the rotor.

The torque for structural analysis (logged to CSV, performance metrics) includes
ALL forces (inertial + external) and is computed at deformed coordinates:

    τ = Σᵢ (X₀,ᵢ + uᵢ − center) × F_combined,ᵢ

OmegaProvider Modes
-------------------
    ConstantOmega:       ω = const, α = 0 always.
    RampedOmega:         Linear ramp: ω(t) = ω_target · min(t/t_ramp, 1),
                         α = ω_target/t_ramp during ramp, 0 after.
    ComputedOmega:       Dynamic ω from torque balance (Euler integration).
    RampedComputedOmega: Two-phase — linear ramp then dynamic torque balance.
    TableOmega:          Prescribed ω(t) from tabulated time-series.
    FunctionOmega:       Prescribed ω(t) from user callable.

Angular velocity is exchanged with the fluid solver via preCICE on a
dedicated GlobalSolidMesh (single vertex at rotation center). OpenFOAM
reads this value to drive its dynamic mesh rotation.

Performance Coefficients
------------------------
At each converged time window, the solver computes (using **aerodynamic**
forces/power only, consistent with standard wind energy definitions):

    Thrust    = F_aero · n̂      (aerodynamic force projected on rotation axis)
    P_aero    = τ_aero · ω      (aerodynamic torque × angular velocity)
    Ct = Thrust / (½ · ρ · V∞² · π · R²)
    Cp = P_aero / (½ · ρ · V∞³ · π · R²)
    Cq = τ_aero / (½ · ρ · V∞² · π · R² · R)
    TSR = ω · R / V∞

Additionally, the solver reports a torque breakdown (aerodynamic, inertial,
gravitational, total), the net non-aerodynamic torque
τ_non-aero = τ_total - τ_aero, two power columns (aero and total), and a
structural efficiency based on the opposing non-aerodynamic torque.
Freestream parameters (ρ, V∞) are configured under the ``postprocess:`` YAML key.

The shaft torque sign convention is:
    - Positive: drives rotation (e.g. motor powering a propeller)
    - Negative: resists rotation (e.g. generator in a wind turbine)

where R is the deformed rotor radius (max perpendicular distance from
rotation axis, updated each step with elastic deformation).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.postprocess.stress_recovery import StressRecovery, StressType

from .base import NewmarkCoefficients
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

    Solves the rotating-frame equation of motion (ANSYS Eq. 14-57):

        [M]{ü} + [C]{u̇} + ([K] + [K_G] + [K_SP]){u} = {F}

    where the RHS includes aerodynamic forces (from preCICE), centrifugal force
    at undeformed coordinates, Coriolis force, Euler force at deformed coordinates,
    and gravity transformed to the rotating frame.

    The solver operates in a rotating reference frame, which keeps the stiffness
    matrix constant while inertial effects are handled through:
    - **Stress stiffening [K_G]**: Geometric stiffness from centrifugal prestress
      on the LHS (stiffens flapwise/out-of-plane modes).
    - **Spin softening [K_SP]**: Negative stiffness = -ω²·M·(I - n̂⊗n̂) on the
      LHS (softens in-plane modes in the rotation plane). ANSYS Eq. 3-74.
    - **Centrifugal force**: F_cf = m·ω²·r_⊥ evaluated at X₀ on the RHS.
      The displacement-dependent correction is captured implicitly by K_SP·u.
    - **Coriolis force**: F_cor = -2·m·(ω × v) explicit on the RHS using the
      best available velocity estimate (lagged or sub-iteration guess).
    - **Euler force**: F_euler = -m·(α × r) evaluated at deformed coordinates
      X₀ + u, only when angular acceleration α ≠ 0.
    - **Coordinate transforms**: Forces R^T(θ)·F_global; displacements R(θ)·u_local.

    Numerical Scheme Classification
    --------------------------------
    - Spin softening: IMPLICIT (K_SP on LHS, solved simultaneously)
    - Stress stiffening: IMPLICIT (K_G on LHS)
    - Centrifugal: IMPLICIT via K_SP + explicit base load at X₀
    - Coriolis: EXPLICIT (force on RHS, lagged velocity)
    - Euler: EXPLICIT (force at X₀+u, only when α ≠ 0)
    - FSI convergence: Handled by preCICE IQN-ILS (no internal iterations)

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
    solver.rotor.include_spin_softening : bool
        Include K_SP for spin softening (ANSYS Eq. 3-74). Default: True.
        When True, centrifugal force stays at X₀ (K_SP captures the correction).
        When False, no spin softening correction is applied.
    solver.rotor.include_centrifugal : bool
        Include centrifugal forces on RHS. Default: True
    solver.rotor.include_coriolis : bool
        Include Coriolis forces on RHS. Default: True
    solver.rotor.include_euler : bool
        Include Euler forces at deformed coords (only when α ≠ 0). Default: True
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
       very flexible structures unless small time steps are used. The ANSYS-consistent
       approach would place Coriolis as antisymmetric [G] matrix on the LHS, but
       this requires a non-symmetric solver (reserved for future).

    2. Small Strain Assumption:
       Assumes linear elasticity with stress stiffening only. Does not implement a
       full Geometrically Exact Beam or St. Venant-Kirchhoff model for large
       rotations relative to the local frame.

    3. Lumped Mass Approximation:
       Inertial forces and K_SP are calculated using a diagonal (lumped) mass
       matrix. This simplifies K_SP to a diagonal matrix (O(n) cost) but may
       approximate rotational inertia terms less accurately than a consistent
       mass formulation.

    4. K_SP Singularity Risk:
       At very high ω, K_SP can make K_eff singular if ω²·m > k_elastic for
       some DOF (spin-buckle). In practice K_G dominates and prevents this,
       but eigenvalue monitoring is recommended near critical speeds.

    5. Explicit Euler Integration for ω:
       Dynamic omega (ComputedOmega) uses forward Euler: ω^{n+1} = ω^n + α·dt.
       This is first-order and may accumulate error for large dt or rapidly
       varying torque. Higher-order integrators (RK4) are not yet implemented.

    6. Torque Balance — Fictitious Forces Excluded:
       The driving torque for ω dynamics excludes centrifugal, Coriolis, and
       Euler forces. These are artifacts of the rotating reference frame and
       do not produce net angular acceleration of the rotor assembly.
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
        """Initialize rotor-specific configuration from model properties.

        Sets up the angular velocity provider (OmegaProvider) and all rotor
        physics flags. The OmegaProvider hierarchy is:

        1. auto-inertia + ramp → RampedOmega initially, replaced by
           RampedComputedOmega in solve() after I is estimated from mesh.
        2. explicit I + ramp   → RampedComputedOmega (ramp then dynamic).
        3. explicit I, no ramp → ComputedOmega (dynamic from t=0).
        4. ramp only           → RampedOmega (prescribed linear ramp).
        5. constant            → ConstantOmega (fixed ω, α=0).

        Also initializes CoordinateTransforms (rotation R(θ)) and
        InertialForcesCalculator (centrifugal, Coriolis, Euler).
        """
        rotor_cfg = self.solver_params.get("rotor", {})

        # Angular velocity configuration
        omega_value = float(rotor_cfg.get("omega", 0.0))
        omega_ramp_time = float(rotor_cfg.get("omega_ramp_time", 0.0))

        # Check for dynamic omega (ComputedOmega)
        moment_of_inertia = rotor_cfg.get("moment_of_inertia")
        # Support both 'shaft_torque' (preferred) and deprecated 'resistive_torque'
        shaft_torque = rotor_cfg.get("shaft_torque", None)
        if shaft_torque is None:
            # Backward compat: negate resistive_torque to match new sign convention
            legacy = rotor_cfg.get("resistive_torque", 0.0)
            shaft_torque = -float(legacy) if float(legacy) != 0.0 else 0.0
        else:
            shaft_torque = float(shaft_torque)
        self._auto_inertia = False

        # Priority: auto-inertia > explicit inertia > ramp-only > constant
        if isinstance(moment_of_inertia, str) and moment_of_inertia.lower() == "auto":
            # Auto-compute inertia from mesh - will be resolved in solve()
            self._auto_inertia = True
            self._auto_inertia_params = {
                "target_omega": omega_value,
                "ramp_time": omega_ramp_time,
                "shaft_torque": shaft_torque,
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
                    shaft_torque=shaft_torque,
                )
            else:
                # No ramp: pure dynamic from start
                self._omega_provider = ComputedOmega(
                    moment_of_inertia=inertia_val,
                    initial_omega=omega_value,
                    shaft_torque=shaft_torque,
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
        self._include_spin_softening = rotor_cfg.get("include_spin_softening", True)
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

        # On restart, CFD may produce transient force spikes while it re-stabilizes.
        # During the grace period, forces that exceed the jump factor are clamped
        # to the last known maximum instead of aborting the simulation.
        self._restart_force_grace_windows: int = int(
            rotor_cfg.get("restart_force_grace_windows", 5)
        )
        self._restart_grace_remaining: int = 0

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

        # Aerodynamic performance parameters (for Cp, Cq, Ct, TSR — no effect on physics).
        # Preferred location: postprocess.fluid_density / postprocess.flow_velocity
        # Deprecated location: rotor.fluid_density / rotor.flow_velocity
        perf_cfg = self.solver_params.get("performance", {})
        self._fluid_density = float(
            perf_cfg.get("fluid_density")
            or rotor_cfg.get("fluid_density")
            or _DEFAULT_FLUID_DENSITY
        )
        self._flow_velocity = float(
            perf_cfg.get("flow_velocity")
            or rotor_cfg.get("flow_velocity")
            or _DEFAULT_FLOW_VELOCITY
        )
        if not perf_cfg and (
            rotor_cfg.get("fluid_density") is not None or rotor_cfg.get("flow_velocity") is not None
        ):
            _logger.warning(
                "fluid_density / flow_velocity under 'rotor:' is deprecated. "
                "Move them to 'postprocess:' in your simulation YAML."
            )
        self._rotor_radius: Optional[float] = rotor_cfg.get("radius")
        if self._rotor_radius is not None:
            self._rotor_radius = float(self._rotor_radius)

        # Reserved for future dynamic omega computation
        self._moment_of_inertia = rotor_cfg.get("moment_of_inertia", None)

    def _init_solver_config(self) -> None:
        """Initialize solver configuration parameters."""
        damping_cfg = self.solver_params.get("damping") or {}

        self._damping_enabled: bool = damping_cfg.get("enabled", True)
        # Auto mode: compute coefficients from modal analysis at assembly time
        self._damping_auto: bool = (
            self._damping_enabled
            and damping_cfg.get("eta_m") is None
            and damping_cfg.get("eta_k") is None
            and bool(damping_cfg)  # only auto when damping section is present
        )
        self._damping_cfg: dict = damping_cfg

        if not self._damping_enabled:
            self._eta_m = 0.0
            self._eta_k = 0.0
        elif self._damping_auto:
            # Placeholder — will be overwritten in _assemble_system_matrices
            self._eta_m = 0.0
            self._eta_k = 0.0
        else:
            # Manual mode: read from nested damping dict, fall back to flat keys
            self._eta_m = float(
                damping_cfg["eta_m"]
                if damping_cfg.get("eta_m") is not None
                else self.solver_params.get("eta_m", 0.0)
            )
            self._eta_k = float(
                damping_cfg["eta_k"]
                if damping_cfg.get("eta_k") is not None
                else self.solver_params.get("eta_k", 0.0)
            )

        # Solver type configuration
        self._solver_type = self.solver_params.get("solver_type", "auto")

    def _init_state_tracking(self) -> None:
        """Initialize state tracking variables."""
        # Current rotation angle (updated during time stepping)
        self._theta = 0.0

        # State tracking for dynamic updates
        self._prev_omega: Optional[float] = None
        self._v_guess_next: Optional[PETSc.Vec] = None

        # On restart, ramps must NOT be re-applied — the simulation
        # continues from the checkpoint state where ramps already completed.
        # Explicit is better than implicit (PEP 20).
        self._skip_ramps = False

    # =========================================================================
    # Checkpoint Peeking
    # =========================================================================

    def _read_restart_state(self, t_target: float) -> Optional[Tuple[float, float, float]]:
        """Look up (theta, omega, alpha) at *t_target* from ``rotor_restart_state.csv``.

        Finds the last row with ``time <= t_target`` and extrapolates forward
        using the second-order kinematics:

            theta(t) ≈ theta_0 + omega_0 · Δt + ½ · alpha_0 · Δt²

        Parameters
        ----------
        t_target : float
            Target time to recover kinematics for [s].

        Returns
        -------
        tuple of (theta_rad, omega, alpha) or None if file absent / unreadable.
        """
        import csv as _csv

        output_folder = self.solver_params.get("output_folder", "results")
        csv_path = os.path.join(output_folder, "rotor_restart_state.csv")

        if not os.path.exists(csv_path):
            return None

        try:
            best: Optional[Tuple[float, float, float, float]] = None  # (t, theta, omega, alpha)
            with open(csv_path, newline="") as fh:
                reader = _csv.reader(fh)
                header = next(reader)
                t_idx = header.index("Time [s]")
                th_idx = header.index("Theta [rad]")
                om_idx = header.index("Omega [rad/s]")
                al_idx = header.index("Alpha [rad/s2]")
                for row in reader:
                    try:
                        t_row = float(row[t_idx])
                    except (ValueError, IndexError):
                        continue
                    if t_row <= t_target + 1e-12:
                        if best is None or t_row > best[0]:
                            best = (
                                t_row,
                                float(row[th_idx]),
                                float(row[om_idx]),
                                float(row[al_idx]),
                            )

            if best is None:
                return None

            t0, theta0, omega0, alpha0 = best
            dt = t_target - t0
            theta_t = theta0 + omega0 * dt + 0.5 * alpha0 * dt * dt
            omega_t = omega0 + alpha0 * dt
            return theta_t, omega_t, alpha0
        except Exception as e:
            _logger.warning("Could not read rotor_restart_state.csv: %s", e)
            return None

    def _peek_checkpoint_theta(self) -> float:
        """Read the rotation angle at the fluid restart time.

        Strategy (in order of preference):

        1. Read ``theta_ckpt`` and ``t_ckpt`` from the latest checkpoint NPZ.
        2. Find ``t_fluid`` (the actual OpenFOAM restart time).
        3. If ``t_fluid == t_ckpt`` → use NPZ theta directly.
        4. If ``t_fluid > t_ckpt`` → look up exact/interpolated theta in
           ``rotor_restart_state.csv`` (written every converged window).
        5. If the restart state is missing → extrapolate with NPZ omega and warn.
        6. Guard: if the gap exceeds ``max_time_gap`` (default 5·dt) abort early.

        Returns
        -------
        float
            Rotation angle [rad] aligned to ``t_fluid``, or 0.0 on failure.
        """
        start_from = self.solver_params.get("start_from", "startTime")
        if start_from not in ("latestTime", "firstTime"):
            return 0.0

        if self._checkpoint_manager is None:
            return 0.0

        if start_from == "firstTime":
            info = self._checkpoint_manager.find_first()
        else:
            info = self._checkpoint_manager.find_latest()

        if info is None:
            return 0.0

        npz_path = os.path.join(info.path, "state.npz")
        if not os.path.exists(npz_path):
            return 0.0

        try:
            with np.load(npz_path) as data:
                t_ckpt = float(data["t"]) if "t" in data.files else 0.0
                if "theta" not in data.files:
                    if self._is_primary_rank():
                        print(
                            "  ↳ WARNING: theta not found in checkpoint NPZ, defaulting to 0.0 rad",
                            flush=True,
                        )
                    return 0.0
                theta_ckpt = float(data["theta"])
                omega_ckpt = float(data["omega"]) if "omega" in data.files else 0.0
                alpha_ckpt = float(data["alpha"]) if "alpha" in data.files else 0.0

            t_fluid = self._find_fluid_restart_time(t_ckpt)
            dt_gap = t_fluid - t_ckpt

            # ── Determine theta at t_fluid ──────────────────────────────────
            # dt_gap can be positive (fluid ahead) OR negative (fluid behind).
            # Both cases require angular correction.
            source = "npz"
            theta_rad = theta_ckpt
            abs_gap = abs(dt_gap)

            if abs_gap > 1e-12:
                # Try exact lookup from restart state history
                state = self._read_restart_state(t_fluid)
                if state is not None:
                    theta_rad, _, _ = state
                    source = "restart_state"
                else:
                    # Fallback: second-order extrapolation from NPZ.
                    # Works correctly for both signs of dt_gap:
                    #   dt_gap > 0 → extrapolate forward
                    #   dt_gap < 0 → interpolate backward
                    theta_rad = theta_ckpt + omega_ckpt * dt_gap + 0.5 * alpha_ckpt * dt_gap**2
                    source = "extrapolated"

                # ── Guard: abort if gap is large and no restart state ───────
                coupling_cfg = self.model_properties.get("coupling", {})
                max_time_gap = coupling_cfg.get(
                    "restart_max_time_gap",
                    None,
                )
                if max_time_gap is None:
                    dt_solid = self.solver_params.get("time_step", 1e-4)
                    write_interval = coupling_cfg.get("fluid_write_interval", dt_solid * 10)
                    max_time_gap = 2.0 * float(write_interval)

                if source == "extrapolated" and abs_gap > float(max_time_gap) + 1e-12:
                    direction = "ahead of" if dt_gap > 0 else "behind"
                    raise RuntimeError(
                        f"Restart temporal gap too large to recover safely:\n"
                        f"  t_solid  = {t_ckpt:.6f} s (solid checkpoint)\n"
                        f"  t_fluid  = {t_fluid:.6f} s (fluid latest time, {direction} solid)\n"
                        f"  |Δt_gap| = {abs_gap:.6f} s  (limit = {float(max_time_gap):.6f} s)\n"
                        f"  Δθ_est   = {np.degrees(omega_ckpt * abs_gap):.2f}°\n"
                        f"\n"
                        f"  rotor_restart_state.csv is absent or does not cover t_fluid.\n"
                        f"  The solid and fluid checkpoints are not temporally aligned.\n"
                        f"  Options:\n"
                        f"  1. Align fluid time to solid: keep only fluid times ≤ t={t_ckpt:.6f} s.\n"
                        f"  2. Align solid to fluid: re-run solid checkpoint at t={t_fluid:.6f} s.\n"
                        f"  3. Increase 'coupling.restart_max_time_gap' to suppress this check\n"
                        f"     (only if the angular offset is within the RBF support radius)."
                    )

            if self._is_primary_rank():
                print(
                    f"  ↳ Checkpoint θ: t_solid={t_ckpt:.6f} s, t_fluid={t_fluid:.6f} s"
                    f"  [gap={dt_gap:+.6f} s, src={source}]",
                    flush=True,
                )
                print(
                    f"  ↳ θ(t_fluid) = {np.degrees(theta_rad):.4f}° = {theta_rad:.4f} rad",
                    flush=True,
                )
                if source == "extrapolated":
                    print(
                        "  ↳ WARNING: rotor_restart_state.csv absent — using kinematic"
                        f" extrapolation (Δt={dt_gap:+.4f} s).",
                        flush=True,
                    )

            return theta_rad
        except Exception as e:
            _logger.warning("Could not peek checkpoint theta: %s", e)
            return 0.0

    def _peek_checkpoint_omega(self) -> Optional[float]:
        """Read the angular velocity from the latest checkpoint NPZ.

        On restart, the omega provider is freshly created and has no knowledge
        of the dynamic omega that evolved during the previous run.  This method
        reads the saved ``omega`` field from the checkpoint so it can be used
        as the initial value for preCICE and for restoring the provider state.

        Returns
        -------
        float or None
            Angular velocity [rad/s] at the checkpoint time, or None if
            no checkpoint / no restart / omega not found in NPZ.
        """
        start_from = self.solver_params.get("start_from", "startTime")
        if start_from not in ("latestTime", "firstTime"):
            return None

        if self._checkpoint_manager is None:
            return None

        if start_from == "firstTime":
            info = self._checkpoint_manager.find_first()
        else:
            info = self._checkpoint_manager.find_latest()

        if info is None:
            return None

        npz_path = os.path.join(info.path, "state.npz")
        if not os.path.exists(npz_path):
            return None

        try:
            with np.load(npz_path) as data:
                if "omega" in data.files:
                    omega_val = float(data["omega"])
                    if self._is_primary_rank():
                        print(f"  ↳ Checkpoint ω: {omega_val:.4f} rad/s", flush=True)
                    return omega_val
            return None
        except Exception as e:
            _logger.warning("Could not peek checkpoint omega: %s", e)
            return None

    def _read_angle_from_performance_csv(self, t_target: float) -> float:
        """Read the rotation angle [deg] from ``rotor_performance.csv``.

        Finds the entry whose time is closest to *t_target* without
        exceeding it (i.e. the last entry at or before *t_target*).

        Parameters
        ----------
        t_target : float
            Target time [s] (typically the fluid restart time).

        Returns
        -------
        float
            Rotation angle in **degrees**, or 0.0 if the CSV is not found
            or cannot be parsed.
        """
        import csv

        output_folder = self.solver_params.get("output_folder", "results")
        csv_path = os.path.join(output_folder, "rotor_performance.csv")

        if not os.path.exists(csv_path):
            _logger.warning("rotor_performance.csv not found at %s", csv_path)
            return 0.0

        best_time = -1.0
        best_angle = 0.0

        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                time_idx = header.index("Time [s]")
                angle_idx = header.index("Angle [deg]")
                for row in reader:
                    t_row = float(row[time_idx])
                    if t_row <= t_target + 1e-12 and t_row > best_time:
                        best_time = t_row
                        best_angle = float(row[angle_idx])
        except Exception as e:
            _logger.warning("Failed to read rotor_performance.csv: %s", e)
            return 0.0

        if best_time < 0.0:
            _logger.warning(
                "No entry found in rotor_performance.csv at or before t=%.6f",
                t_target,
            )
            return 0.0

        if self._is_primary_rank():
            _logger.info(
                "Read angle from rotor_performance.csv: t_target=%.6f s, t_csv=%.6f s, angle=%.4f°",
                t_target,
                best_time,
                best_angle,
            )
        return best_angle

    def _find_fluid_restart_time(self, t_solid: float) -> float:
        """Find the latest *valid* time directory in the fluid case on disk.

        This method trusts ONLY what exists on the filesystem — it does NOT
        use ``t_solid`` as a lower bound or fallback.  The fluid may be ahead
        of, behind, or exactly at the solid checkpoint time; all three cases
        are legitimate.

        Handles both serial (top-level time dirs) and parallel
        (``processorN/``) OpenFOAM layouts.  A time directory is considered
        **valid** only when:

        * **Parallel** — it exists in ALL ``processorN/`` directories AND
          contains at least one non-hidden regular file in ``processor0``.
        * **Serial** — it contains at least one non-hidden regular file.

        This rejects incomplete last-write directories produced by mid-crash
        I/O.

        Parameters
        ----------
        t_solid : float
            Solid checkpoint time [s].  Used only for diagnostic printing;
            does NOT influence which fluid time is selected.

        Returns
        -------
        float
            Latest valid fluid time found on disk, or ``t_solid`` only if
            the fluid case directory does not exist or contains no valid
            time directories at all (with a warning).
        """
        import re

        coupling_cfg = self.model_properties.get("coupling", {})
        fluid_case_dir = coupling_cfg.get("fluid_case_dir", "../fluid")

        if not os.path.isabs(fluid_case_dir):
            fluid_case_dir = os.path.normpath(os.path.join(os.getcwd(), fluid_case_dir))

        if not os.path.isdir(fluid_case_dir):
            if self._is_primary_rank():
                print(
                    f"  ↳ WARNING: fluid case dir not found ({fluid_case_dir}), "
                    f"assuming t_fluid = t_solid = {t_solid:.6f} s",
                    flush=True,
                )
            return t_solid

        time_pattern = re.compile(r"^\d+(?:\.\d+)?$")
        latest_time: Optional[float] = None

        # ── Detect decomposed vs serial ──────────────────────────────────────
        proc_dirs = sorted(
            d
            for d in os.listdir(fluid_case_dir)
            if re.match(r"^processor\d+$", d) and os.path.isdir(os.path.join(fluid_case_dir, d))
        )

        if proc_dirs:
            # ── Parallel case ────────────────────────────────────────────────
            time_sets = []
            for proc in proc_dirs:
                proc_path = os.path.join(fluid_case_dir, proc)
                try:
                    times = {
                        float(e)
                        for e in os.listdir(proc_path)
                        if time_pattern.match(e) and os.path.isdir(os.path.join(proc_path, e))
                    }
                    time_sets.append(times)
                except OSError:
                    pass

            if time_sets:
                common_times = time_sets[0]
                for ts in time_sets[1:]:
                    common_times = common_times & ts

                proc0_path = os.path.join(fluid_case_dir, proc_dirs[0])
                proc0_name_map: dict[float, str] = {}
                try:
                    for e in os.listdir(proc0_path):
                        if time_pattern.match(e) and os.path.isdir(os.path.join(proc0_path, e)):
                            proc0_name_map[float(e)] = e
                except OSError:
                    pass

                for t_val in sorted(common_times, reverse=True):
                    dir_name = proc0_name_map.get(t_val)
                    if dir_name is None:
                        continue
                    t_dir = os.path.join(proc0_path, dir_name)
                    try:
                        has_fields = any(
                            os.path.isfile(os.path.join(t_dir, f))
                            for f in os.listdir(t_dir)
                            if not f.startswith(".")
                        )
                    except OSError:
                        has_fields = False
                    if has_fields:
                        latest_time = t_val
                        break

        else:
            # ── Serial case ──────────────────────────────────────────────────
            try:
                for entry in os.listdir(fluid_case_dir):
                    if not time_pattern.match(entry):
                        continue
                    t_val = float(entry)
                    t_dir = os.path.join(fluid_case_dir, entry)
                    try:
                        has_fields = any(
                            os.path.isfile(os.path.join(t_dir, f))
                            for f in os.listdir(t_dir)
                            if not f.startswith(".")
                        )
                    except OSError:
                        has_fields = False
                    if has_fields and (latest_time is None or t_val > latest_time):
                        latest_time = t_val
            except OSError as e:
                _logger.debug("Could not scan fluid case dir %s: %s", fluid_case_dir, e)

        # ── Fallback when no valid fluid time was found ──────────────────────
        if latest_time is None:
            if self._is_primary_rank():
                print(
                    f"  ↳ WARNING: no valid fluid time directory found in {fluid_case_dir}, "
                    f"assuming t_fluid = t_solid = {t_solid:.6f} s",
                    flush=True,
                )
            return t_solid

        if self._is_primary_rank():
            dt_gap = latest_time - t_solid
            direction = "ahead" if dt_gap > 1e-12 else "behind" if dt_gap < -1e-12 else "aligned"
            print(
                f"  ↳ Fluid restart time: t_fluid={latest_time:.6f} s  "
                f"(t_solid={t_solid:.6f} s, {direction}, Δt={dt_gap:+.6f} s)",
                flush=True,
            )

        return latest_time

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
        """Configure direct LU solver - best for small/medium problems.

        Uses MUMPS if available, otherwise falls back to PETSc native LU.
        """
        self._solver.setType("preonly")
        pc.setType("lu")
        if self._has_mumps():
            pc.setFactorSolverType("mumps")
        else:
            pc.setFactorSolverType("petsc")
        self._solver.setTolerances(rtol=_DIRECT_SOLVER_RTOL, atol=_DIRECT_SOLVER_ATOL, max_it=1)

    def _configure_iterative_solver(self, pc: PETSc.PC, opts: PETSc.Options) -> None:
        """Configure iterative solver with GAMG preconditioner.

        Uses conservative aggregation settings suitable for anisotropic shell
        meshes (high aspect ratios, mixed edge lengths).
        """
        self._solver.setType("cg")
        pc.setType("gamg")
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 2
        opts["pc_gamg_threshold"] = 0.005
        opts["pc_gamg_square_graph"] = 1
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_ksp_max_it"] = 4
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "lu"
        self._solver.setTolerances(
            rtol=_ITERATIVE_SOLVER_RTOL,
            atol=_ITERATIVE_SOLVER_ATOL,
            max_it=_ITERATIVE_SOLVER_MAX_IT,
        )

    def _solve_linear_system(
        self, K_eff: PETSc.Mat, F_eff: PETSc.Vec
    ) -> tuple[PETSc.Vec, int, int]:
        """Solve K_eff * u = F_eff with automatic fallback on divergence.

        Fallback chain (iterative solver only):
          1. Primary CG+GAMG solve
          2. BCGS+GAMG with threshold=0 (handles indefinite PC)
          3. Direct LU (PETSc native, then MUMPS if available)

        Returns (u_new, ksp_iterations, ksp_converged_reason).
        """
        u_new = K_eff.createVecRight()
        self._solver.setOperators(K_eff)
        self._solver.solve(F_eff, u_new)

        ksp_its = self._solver.getIterationNumber()
        ksp_reason = self._solver.getConvergedReason()

        is_iterative = self._solver.getType() != "preonly"
        if ksp_reason >= 0 or not is_iterative:
            return u_new, ksp_its, ksp_reason

        # --- Fallback 1: BCGS+GAMG threshold=0 ---
        # CG requires SPD preconditioner; GAMG aggregation can produce
        # indefinite actions (reason=-8).  BCGS has no such requirement.
        _logger.warning(
            "KSP diverged (reason=%d). Retrying with BCGS + relaxed GAMG...",
            ksp_reason,
        )
        fb1 = PETSc.KSP().create(self.comm)
        fb1.setType("bcgs")
        fb1_pc = fb1.getPC()
        fb1_pc.setType("gamg")
        opts = PETSc.Options()
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 2
        opts["pc_gamg_threshold"] = 0.0
        opts["pc_gamg_square_graph"] = 1
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_ksp_max_it"] = 4
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "lu"
        fb1.setTolerances(
            rtol=_ITERATIVE_SOLVER_RTOL,
            atol=_ITERATIVE_SOLVER_ATOL,
            max_it=_ITERATIVE_SOLVER_MAX_IT,
        )
        fb1.setOperators(K_eff)
        fb1.setFromOptions()

        u_new.zeroEntries()
        fb1.solve(F_eff, u_new)
        ksp_its = fb1.getIterationNumber()
        ksp_reason = fb1.getConvergedReason()
        fb1.destroy()

        if ksp_reason >= 0:
            _logger.info(
                "BCGS+GAMG fallback converged (reason=%d, its=%d).",
                ksp_reason, ksp_its,
            )
            return u_new, ksp_its, ksp_reason

        # --- Fallback 2: Direct LU solver ---
        # Try without specifying MUMPS first (PETSc native LU always exists).
        # If the build has MUMPS, try that as a second attempt.
        _logger.warning(
            "BCGS+GAMG also diverged (reason=%d). Falling back to direct LU...",
            ksp_reason,
        )
        for solver_pkg in (None, "mumps"):
            if solver_pkg == "mumps" and not self._has_mumps():
                continue
            fb2 = PETSc.KSP().create(self.comm)
            fb2.setType("preonly")
            fb2_pc = fb2.getPC()
            fb2_pc.setType("lu")
            if solver_pkg is not None:
                fb2_pc.setFactorSolverType(solver_pkg)
            fb2.setTolerances(rtol=_DIRECT_SOLVER_RTOL, atol=_DIRECT_SOLVER_ATOL, max_it=1)
            fb2.setOperators(K_eff)
            fb2.setFromOptions()
            try:
                u_new.zeroEntries()
                fb2.solve(F_eff, u_new)
                ksp_its = fb2.getIterationNumber()
                ksp_reason = fb2.getConvergedReason()
                fb2.destroy()
                if ksp_reason >= 0:
                    pkg_label = solver_pkg or "petsc-native"
                    _logger.info("Direct LU (%s) converged (reason=%d).", pkg_label, ksp_reason)
                    return u_new, ksp_its, ksp_reason
            except PETSc.Error as exc:
                pkg_label = solver_pkg or "petsc-native"
                _logger.warning("Direct LU (%s) failed: %s", pkg_label, exc)
                fb2.destroy()

        _logger.error("ALL solver fallbacks exhausted. Solution may be invalid.")
        return u_new, 0, -1

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

        Parameters
        ----------
        K : PETSc.Mat
            Global elastic stiffness matrix.
        M : PETSc.Mat
            Global lumped mass matrix.

        Returns
        -------
        PETSc.Mat or None
            Damping matrix if either η_m or η_k is non-zero, otherwise None.
        """
        if self._eta_m == 0.0 and self._eta_k == 0.0:
            return None

        # copy() preserves values; duplicate() only copies sparsity (values = 0)
        C = K.copy()
        C.scale(self._eta_k)
        C.axpy(self._eta_m, M)
        return C

    def _compute_rayleigh_auto(self) -> Tuple[float, float]:
        """Compute Rayleigh damping coefficients α (η_k) and β (η_m) automatically.

        Runs a reduced modal analysis on copies of the assembled K and M matrices
        and applies the two-point method described in:

            SimScale Knowledge Base — "How to Compute the Coefficients for
            Rayleigh Damping?" (2022).

        Given damping ratios ζ_i and ζ_j at two natural frequencies ω_i and ω_j
        (from modal analysis), the system

            ζ = ½ · (α·ω + β/ω)

        is solved at both reference modes:

            α (η_k) = 2·(ζ_i·ω_i − ζ_j·ω_j) / (ω_i² − ω_j²)
            β (η_m) = 2·ω_i·ω_j·(ζ_j·ω_i − ζ_i·ω_j) / (ω_i² − ω_j²)

        For equal damping ratios ζ_i = ζ_j = ζ this simplifies to:

            α = 2·ζ / (ω_i + ω_j)
            β = 2·ζ·ω_i·ω_j / (ω_i + ω_j)

        Returns
        -------
        Tuple[float, float]
            (eta_k, eta_m) — stiffness-proportional and mass-proportional
            Rayleigh coefficients.

        Raises
        ------
        RuntimeError
            If the modal solve does not converge enough modes.
        ValueError
            If the two reference modes share the same natural frequency.
        """
        from slepc4py import SLEPc  # local import — only needed in auto mode

        cfg = self._damping_cfg
        zeta = float(cfg.get("zeta", 0.02))
        zeta_i = float(cfg["zeta_1"]) if cfg.get("zeta_1") is not None else zeta
        zeta_j = float(cfg["zeta_2"]) if cfg.get("zeta_2") is not None else zeta
        mode_i = int(cfg.get("mode_i", 1))
        mode_j = int(cfg.get("mode_j", 2))
        num_modes = int(cfg.get("num_modes", max(mode_j + 2, 6)))

        if mode_i == mode_j:
            raise ValueError("mode_i and mode_j must be different for Rayleigh auto-computation.")

        # copy() preserves both sparsity structure AND values (unlike duplicate())
        K_dup = self.K.copy()
        M_dup = self.M.copy()
        F_tmp = K_dup.createVecRight()
        F_tmp.set(0.0)

        bc_tmp = BoundaryConditionManager(K_dup, F_tmp, M_dup, self.domain.dofs_per_node)
        bc_tmp.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = bc_tmp.reduced_system

        # Generalized eigenvalue problem: K·φ = λ·M·φ  (λ = ω²)
        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(K_red, M_red)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)
        ksp_st = st.getKSP()
        ksp_st.setType("preonly")
        pc_st = ksp_st.getPC()
        pc_st.setType("lu")
        # Prefer MUMPS when available (better for large matrices); fall back to
        # the PETSc built-in LU which is always present.
        # We probe availability by asking PETSc to register the solver type —
        # if it can't, we stay with "petsc" (no-op means already set above).
        try:
            _probe = PETSc.Mat().createAIJ((2, 2), nnz=1, comm=PETSc.COMM_SELF)
            _probe.setValue(0, 0, 1.0)
            _probe.setValue(1, 1, 1.0)
            _probe.assemble()
            _ksp_probe = PETSc.KSP().create(PETSc.COMM_SELF)
            _ksp_probe.setType("preonly")
            _pc_probe = _ksp_probe.getPC()
            _pc_probe.setType("lu")
            _pc_probe.setFactorSolverType("mumps")
            _ksp_probe.setOperators(_probe)
            _ksp_probe.setUp()  # raises petsc4py.PETSc.Error if MUMPS missing
            _ksp_probe.destroy()
            _probe.destroy()
            pc_st.setFactorSolverType("mumps")
        except Exception:
            pc_st.setFactorSolverType("petsc")

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(0.0)
        eps.setDimensions(num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-8, max_it=500)
        try:
            eps.solve()
        except Exception as exc:
            K_red.destroy()
            if M_red is not None:
                M_red.destroy()
            F_red.destroy()
            K_dup.destroy()
            M_dup.destroy()
            F_tmp.destroy()
            raise RuntimeError(
                f"Rayleigh auto: SLEPc eigenvalue solve failed — {exc}. "
                "Check boundary conditions and that K/M are properly assembled."
            ) from exc

        nconv = eps.getConverged()

        # Extract and sort positive eigenvalues (λ = ω² > 0)
        raw_eigs = sorted(eps.getEigenvalue(i).real for i in range(nconv))
        positive_eigs = [lam for lam in raw_eigs if lam > 1e-8]

        # Cleanup temporary objects
        K_red.destroy()
        if M_red is not None:
            M_red.destroy()
        F_red.destroy()
        K_dup.destroy()
        M_dup.destroy()
        F_tmp.destroy()

        needed = max(mode_i, mode_j)
        if len(positive_eigs) < needed:
            raise RuntimeError(
                f"Rayleigh auto: modal solve found only {len(positive_eigs)} positive "
                f"eigenvalues but mode {needed} is required. "
                f"Increase num_modes (currently {num_modes}) or check boundary conditions."
            )

        omega_i = float(np.sqrt(positive_eigs[mode_i - 1]))  # rad/s
        omega_j = float(np.sqrt(positive_eigs[mode_j - 1]))  # rad/s

        denom = omega_i**2 - omega_j**2
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Rayleigh auto: modes {mode_i} and {mode_j} have the same "
                f"natural frequency (ω ≈ {omega_i:.4e} rad/s). "
                "Choose two distinct modes."
            )

        alpha = 2.0 * (zeta_i * omega_i - zeta_j * omega_j) / denom  # η_k
        beta = 2.0 * omega_i * omega_j * (zeta_j * omega_i - zeta_i * omega_j) / denom  # η_m

        if alpha < 0.0 or beta < 0.0:
            _logger.warning(
                "Rayleigh auto produced a negative coefficient: η_k=%.3e, η_m=%.3e. "
                "The damping ratio may be non-monotone. Verify your mode selection.",
                alpha,
                beta,
            )

        if self._is_primary_rank():
            f_i = omega_i / (2.0 * np.pi)
            f_j = omega_j / (2.0 * np.pi)
            print(
                f"  [Rayleigh auto] Mode {mode_i}: f={f_i:.3f} Hz "
                f"(ω={omega_i:.3f} rad/s), ζ={zeta_i:.4f}",
                flush=True,
            )
            print(
                f"  [Rayleigh auto] Mode {mode_j}: f={f_j:.3f} Hz "
                f"(ω={omega_j:.3f} rad/s), ζ={zeta_j:.4f}",
                flush=True,
            )
            print(
                f"  [Rayleigh auto] η_k (stiffness) = {alpha:.4e} s",
                flush=True,
            )
            print(
                f"  [Rayleigh auto] η_m (mass)      = {beta:.4e} 1/s",
                flush=True,
            )

        return alpha, beta  # (eta_k, eta_m)

    def _build_effective_stiffness(
        self,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        coeffs: NewmarkCoefficients,
        K_G_red: Optional[PETSc.Mat] = None,
        C_red: Optional[PETSc.Mat] = None,
        K_SP_red: Optional[PETSc.Mat] = None,
    ) -> PETSc.Mat:
        """Build the Newmark effective stiffness matrix.

        Assembles K_eff for the implicit Newmark-β scheme:

            K_eff = [K] + [K_G] + [K_SP] + a₀·[M] + a₁·[C]

        where a₀ = 1/(β·dt²) and a₁ = γ/(β·dt) are Newmark coefficients.
        This matrix is factorized once and reused for all sub-iterations
        within a preCICE time window. It is rebuilt only when ω changes
        significantly (triggering K_G and/or K_SP updates).

        Parameters
        ----------
        K_red : PETSc.Mat
            Reduced elastic stiffness matrix (free DOFs only).
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        coeffs : NewmarkCoefficients
            Precomputed Newmark integration coefficients.
        K_G_red : PETSc.Mat, optional
            Reduced geometric stiffness (stress stiffening). Adds positive
            stiffness from centrifugal prestress.
        C_red : PETSc.Mat, optional
            Reduced Rayleigh damping matrix.
        K_SP_red : PETSc.Mat, optional
            Reduced spin softening matrix. Adds NEGATIVE stiffness in the
            rotation plane: K_SP = -ω²·M·(I - n̂⊗n̂).

        Returns
        -------
        PETSc.Mat
            Effective stiffness matrix ready for KSP solve.
        """
        # copy() preserves values; duplicate() only copies sparsity (values = 0).
        # K must be explicitly present in K_eff for the correct Newmark formulation.
        K_eff = K_red.copy()
        K_eff.axpy(coeffs.a0, M_red)
        if K_G_red is not None:
            K_eff.axpy(1.0, K_G_red)
        if K_SP_red is not None:
            K_eff.axpy(1.0, K_SP_red)
        if C_red is not None:
            K_eff.axpy(coeffs.a1, C_red)
        return K_eff

    def _build_spin_softening_matrix(self, omega: float) -> PETSc.Mat:
        """Build spin softening matrix K_SP (full system, unreduced).

        Implements ANSYS Eq. 3-74 / 14-55 for lumped mass:
            K_SP = -ω² · M · (I - n⊗n)

        For each node, the diagonal entries are:
            K_SP[dof_j] = -ω² · m_node · (1 - axis[j]²)   for translational DOFs
            K_SP[dof_j] = 0                                  for rotational DOFs

        This is a negative-definite matrix that softens the structure in the
        plane perpendicular to the rotation axis (spin softening effect).

        Parameters
        ----------
        omega : float
            Current angular velocity magnitude [rad/s].

        Returns
        -------
        PETSc.Mat
            Diagonal spin softening matrix (full system size).
        """
        n_dofs = self.domain.dofs_count
        dofs_per_node = self.domain.dofs_per_node
        axis = self._coord_transforms.axis

        # Get lumped mass diagonal
        M_diag = self.M.getDiagonal()
        m_array = M_diag.getArray(readonly=True)

        # Build K_SP diagonal
        ksp_diag = PETSc.Vec().createMPI(n_dofs, comm=self.comm)
        ksp_array = ksp_diag.getArray()

        omega_sq = omega * omega
        n_nodes = n_dofs // dofs_per_node

        for i in range(n_nodes):
            base = i * dofs_per_node
            # Translational DOFs (first 3): K_SP = -ω²·m·(1 - n_j²)
            for j in range(3):
                dof = base + j
                if dof < n_dofs:
                    m_node = m_array[dof]
                    ksp_array[dof] = -omega_sq * m_node * (1.0 - axis[j] ** 2)
            # Rotational DOFs (3..dofs_per_node-1): leave as 0

        M_diag.destroy()

        # Create diagonal matrix
        K_SP = PETSc.Mat().createAIJ(size=(n_dofs, n_dofs), nnz=1, comm=self.comm)
        K_SP.setDiagonal(ksp_diag)
        K_SP.assemble()
        ksp_diag.destroy()

        return K_SP

    # =========================================================================
    # Interface Data Extraction
    # =========================================================================

    def _extract_nodal_masses(self, M_lumped: PETSc.Mat, interface_dofs: np.ndarray) -> np.ndarray:
        """
        Extract nodal masses from lumped mass matrix diagonal in serial.

        Parameters
        ----------
        M_lumped : PETSc.Mat
            Lumped mass matrix (diagonal).
        interface_dofs : np.ndarray
            DOF indices for interface nodes, shape (n_nodes, dofs_per_node)
            or (n_nodes * 3,).

        Returns
        -------
        np.ndarray
            Scalar mass per interface node, shape (n_nodes,).
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
        """Estimate total moment of inertia about the rotation axis.

        Computes the parallel-axis contribution of all mesh nodes:

            I = Σᵢ mᵢ · r_⊥,ᵢ²

        where mᵢ is the lumped mass (first translational DOF diagonal entry)
        and r_⊥,ᵢ is the perpendicular distance from the node to the rotation
        axis:

            r_⊥² = |r|² − (r · n̂)²

        This is a point-mass approximation; rotational DOF inertia (drilling,
        tilting) is not included. The result is used by ComputedOmega /
        RampedComputedOmega for the torque balance: I·α = τ_driving + τ_shaft.

        Returns
        -------
        float
            Total moment of inertia [kg·m²] about the rotation axis.
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
        """Get coordinates of interface nodes in rotating frame.

        Returns
        -------
        np.ndarray
            Reference coordinates X₀, shape (n_nodes, 3).
        """
        return self.precice_participant.interface_coordinates

    def _extract_interface_vector(
        self, full_vec: PETSc.Vec, interface_dofs: np.ndarray
    ) -> np.ndarray:
        """
        Extract 3D vectors for interface nodes from a full DOF vector.

        Parameters
        ----------
        full_vec : PETSc.Vec
            Full DOF vector (displacement, velocity, etc.).
        interface_dofs : np.ndarray
            DOF indices for interface nodes, shape (n_nodes, dofs_per_node)
            or (n_nodes * 3,).

        Returns
        -------
        np.ndarray
            Translational vector values for each interface node,
            shape (n_nodes, 3).
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
        """Extract velocity vectors for interface nodes.

        Parameters
        ----------
        v_full : PETSc.Vec
            Full velocity DOF vector.
        interface_dofs : np.ndarray
            DOF indices for interface nodes.

        Returns
        -------
        np.ndarray
            Velocity vectors, shape (n_nodes, 3).
        """
        return self._extract_interface_vector(v_full, interface_dofs)

    def _get_interface_displacements(
        self, u_full: PETSc.Vec, interface_dofs: np.ndarray
    ) -> np.ndarray:
        """Extract displacement vectors for interface nodes.

        Parameters
        ----------
        u_full : PETSc.Vec
            Full displacement DOF vector.
        interface_dofs : np.ndarray
            DOF indices for interface nodes.

        Returns
        -------
        np.ndarray
            Displacement vectors, shape (n_nodes, 3).
        """
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
        """Compute rotor radius as max perpendicular distance from rotation axis.

        Uses the deformed configuration (X₀ + u) when displacements are
        provided. The deformed radius is used for:
         - Performance coefficients: Ct, Cp, TSR depend on swept area A = πR².
         - Reporting: monitors blade elongation/contraction under load.

        The perpendicular distance is computed as:

            r_⊥ = |r − (r · n̂) · n̂|

        where r = X − center, and n̂ is the rotation axis unit vector.

        Parameters
        ----------
        interface_coords : np.ndarray, shape (n_nodes, 3)
            Undeformed reference coordinates X₀ of interface nodes.
        interface_disps : np.ndarray, optional, shape (n_nodes, 3)
            Elastic displacements u. When provided, uses X₀ + u.

        Returns
        -------
        float
            Maximum perpendicular distance from the rotation axis [m].
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

        Parameters
        ----------
        interface_coords : np.ndarray, shape (n_nodes, 3)
            Reference coordinates of interface nodes.

        Returns
        -------
        float
            Maximum perpendicular distance from the rotation axis [m].
        """
        return self._compute_rotor_radius(interface_coords, interface_disps=None)

    def _compute_gravity_forces(
        self, interface_masses: np.ndarray, n_nodes: int
    ) -> Tuple[np.ndarray, float]:
        """Compute gravity force vectors for interface nodes.

        Gravity is applied in the GLOBAL (inertial) frame as:

            F_g,i = mᵢ · g

        where g = [gx, gy, gz] is the user-specified gravity vector.
        These forces are later transformed to the rotating frame via
        R^T(θ) before assembly, so gravity direction rotates relative
        to the blade (as physically expected for a rotating structure).

        Parameters
        ----------
        interface_masses : np.ndarray, shape (n_nodes,)
            Scalar nodal masses at interface nodes [kg].
        n_nodes : int
            Number of interface nodes.

        Returns
        -------
        Tuple[np.ndarray, float]
            (F_gravity [n_nodes, 3] in global frame, total |F_g| [N]).
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
        power_aero: float,
        torque_aero: float,
        omega: float,
        radius: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute non-dimensional rotor performance coefficients.

        Standard wind turbine / propeller aerodynamic coefficients:

            Ct  = Thrust / (½ · ρ · V∞² · A)
            Cp  = P_aero / (½ · ρ · V∞³ · A)
            Cq  = Q_aero / (½ · ρ · V∞² · A · R)
            TSR = |ω| · R / V∞

        where A = π·R² is the rotor swept area, ρ is the fluid density,
        V∞ is the freestream velocity, and R is the rotor radius
        (deformed if provided, else reference).

        All coefficients use **aerodynamic** forces/power only (not total),
        consistent with standard wind energy definitions.

        Parameters
        ----------
        thrust : float
            Axial thrust force [N] (aerodynamic force projected on rotation axis).
        power_aero : float
            Aerodynamic power [W] = τ_aero · ω.
        torque_aero : float
            Aerodynamic torque [N·m] (CFD forces only, projected on rotation axis).
        omega : float
            Angular velocity [rad/s].
        radius : float, optional
            Rotor radius [m]. If None, uses stored reference radius.
            Pass the deformed radius for accurate instantaneous coefficients.

        Returns
        -------
        Tuple[float, float, float, float]
            (Ct, Cp, Cq, TSR).
        """
        # Use provided radius (deformed) or fall back to stored radius
        effective_radius = radius if radius is not None else self._rotor_radius
        if effective_radius is None:
            effective_radius = 1.0

        area = np.pi * effective_radius**2
        q_dynamic = 0.5 * self._fluid_density * self._flow_velocity**2

        denom_force = q_dynamic * area
        denom_power = q_dynamic * area * self._flow_velocity
        denom_torque = q_dynamic * area * effective_radius

        ct = thrust / denom_force if abs(denom_force) > _MIN_DENOMINATOR else 0.0
        cp = power_aero / denom_power if abs(denom_power) > _MIN_DENOMINATOR else 0.0
        cq = torque_aero / denom_torque if abs(denom_torque) > _MIN_DENOMINATOR else 0.0
        tsr = (
            (abs(omega) * effective_radius) / self._flow_velocity
            if abs(self._flow_velocity) > _MIN_DENOMINATOR
            else 0.0
        )

        return ct, cp, cq, tsr

    # =========================================================================
    # Logging and Output
    # =========================================================================

    def _is_primary_rank(self) -> bool:
        """Always True in serial."""
        return True

    def _print_header(self, title: str) -> None:
        """Print a formatted header section.

        Parameters
        ----------
        title : str
            Header text to display.
        """
        if self._is_primary_rank():
            print("\n" + "═" * 70, flush=True)
            print(f"  {title}", flush=True)
            print("═" * 70, flush=True)

    def _print_separator(self) -> None:
        """Print a section separator."""
        if self._is_primary_rank():
            print("═" * 70, flush=True)

    def _print_phase(self, phase: int, total: int, message: str) -> None:
        """Print a phase progress message.

        Parameters
        ----------
        phase : int
            Current phase number.
        total : int
            Total number of phases.
        message : str
            Progress description.
        """
        if self._is_primary_rank():
            print(f"  [{phase}/{total}] {message}", flush=True)

    def _print_info(self, message: str) -> None:
        """Print an info message.

        Parameters
        ----------
        message : str
            Informational text to display.
        """
        if self._is_primary_rank():
            print(f"  [Info] {message}", flush=True)

    def _write_restart_state(
        self,
        t: float,
        theta: float,
        omega: float,
        alpha: float,
    ) -> None:
        """Append one row to ``rotor_restart_state.csv``.

        This file is the authoritative kinematic history for restart: it records
        the cumulative angle, angular velocity, and angular acceleration at every
        *converged* time window so that, on restart, ``_peek_checkpoint_theta``
        can recover the exact rotor pose at the fluid's ``latestTime``, which may
        differ from the solid checkpoint time.

        Unlike ``rotor_performance.csv`` (a reporting artefact), this file is
        deliberately minimal and always written regardless of ``postprocess``
        configuration.

        Parameters
        ----------
        t     : float  Current time after the converged window [s].
        theta : float  Cumulative rotation angle [rad].
        omega : float  Angular velocity [rad/s].
        alpha : float  Angular acceleration [rad/s²].
        """
        if not self._is_primary_rank():
            return

        import csv as _csv

        output_folder = self.solver_params.get("output_folder", "results")
        csv_path = os.path.join(output_folder, "rotor_restart_state.csv")
        os.makedirs(output_folder, exist_ok=True)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            writer = _csv.writer(fh)
            if write_header:
                writer.writerow(["Time [s]", "Theta [rad]", "Omega [rad/s]", "Alpha [rad/s2]"])
            writer.writerow([f"{t:.9f}", f"{theta:.9f}", f"{omega:.9f}", f"{alpha:.9f}"])

    def _log_rotor_performance(
        self,
        t: float,
        omega_rpm: float,
        omega_rad: float,
        alpha: float,
        angle_deg: float,
        thrust: float,
        torque_aero: float,
        torque_non_aero: float,
        torque_inertial: float,
        torque_gravity: float,
        torque_total: float,
        power_aero: float,
        power_total: float,
        structural_efficiency: float,
        cp: float,
        cq: float,
        ct: float,
        tsr: float,
        torque_aero_global: np.ndarray,
        torque_total_global: np.ndarray,
        max_displacement: float,
        deformed_radius: Optional[float] = None,
    ) -> None:
        """Write rotor performance metrics to CSV log (rank 0 only).

        Columns are grouped conceptually:

        1. Time & kinematics: time, angle, speed, omega, alpha
        2. Axial force: aerodynamic thrust
        3. Torque breakdown (scalar projections on rotation axis):
              aerodynamic, non-aerodynamic, inertial, gravitational, total
        4. Power: aerodynamic (extracted from wind), total (net after
              structural losses), structural efficiency (clamped [0, 1])
        5. Performance coefficients (aero-based): Cp, Cq, Ct, TSR
        6. Torque vector components in the **global (inertial) frame**:
           aerodynamic and total (X, Y, Z)
        7. Structural response: max displacement, deformed radius

        Parameters
        ----------
        t : float
            Current time [s].
        omega_rpm : float
            Angular velocity [RPM].
        omega_rad : float
            Angular velocity [rad/s].
        alpha : float
            Angular acceleration [rad/s²].
        angle_deg : float
            Cumulative rotation angle [°].
        thrust : float
            Aerodynamic thrust (axial CFD force) [N].
        torque_aero : float
            Aerodynamic torque (CFD forces only) on rotation axis [N·m].
        torque_non_aero : float
            Net non-aerodynamic torque = τ_total - τ_aero [N·m].
        torque_inertial : float
            Inertial torque (centrifugal + Coriolis + Euler) on axis [N·m].
        torque_gravity : float
            Gravitational torque on rotation axis [N·m].
        torque_total : float
            Total torque on rotation axis [N·m].
        power_aero : float
            Aerodynamic power = τ_aero × ω [W].
        power_total : float
            Total power = τ_total × ω [W].
        structural_efficiency : float
            Structural efficiency in [0, 1], based on opposing non-aerodynamic torque [-].
        cp : float
            Power coefficient (aero) [-].
        cq : float
            Torque coefficient (aero) [-].
        ct : float
            Thrust coefficient [-].
        tsr : float
            Tip speed ratio [-].
        torque_aero_global : np.ndarray, shape (3,)
            Aerodynamic torque vector in global frame [N·m].
        torque_total_global : np.ndarray, shape (3,)
            Total torque vector in global frame [N·m].
        max_displacement : float
            Maximum nodal displacement magnitude [m].
        deformed_radius : float, optional
            Current deformed rotor radius [m].
        """
        if not self._is_primary_rank():
            return

        log_path = os.path.join(self.solver_params["output_folder"], "rotor_performance.csv")
        file_exists = os.path.exists(log_path)

        try:
            with open(log_path, "a") as f:
                if not file_exists:
                    header = (
                        "Time [s],Angle [deg],Speed [RPM],Omega [rad/s],Alpha [rad/s2],"
                        "Aero Thrust [N],"
                        "Aero Torque [Nm],Inertial Torque [Nm],"
                        "Gravity Torque [Nm],Total Torque [Nm],"
                        "Aero Power [W],Total Power [W],Structural Efficiency,"
                        "Cp,Cq,Ct,TSR,"
                        "Aero Torque X [Nm],Aero Torque Y [Nm],Aero Torque Z [Nm],"
                        "Total Torque X [Nm],Total Torque Y [Nm],Total Torque Z [Nm],"
                        "Max Displacement [m],Deformed Radius [m]\n"
                    )
                    f.write(header)

                radius_str = f"{deformed_radius:.6f}" if deformed_radius is not None else ""
                line = (
                    f"{t:.6f},{angle_deg:.4f},{omega_rpm:.4f},"
                    f"{omega_rad:.4f},{alpha:.6e},"
                    f"{thrust:.6e},"
                    f"{torque_aero:.6e},{torque_inertial:.6e},"
                    f"{torque_gravity:.6e},{torque_total:.6e},"
                    f"{power_aero:.6e},{power_total:.6e},{structural_efficiency:.6f},"
                    f"{cp:.6f},{cq:.6f},{ct:.6f},{tsr:.6f},"
                    f"{torque_aero_global[0]:.6e},{torque_aero_global[1]:.6e},"
                    f"{torque_aero_global[2]:.6e},"
                    f"{torque_total_global[0]:.6e},{torque_total_global[1]:.6e},"
                    f"{torque_total_global[2]:.6e},"
                    f"{max_displacement:.6e},{radius_str}\n"
                )
                f.write(line)
        except Exception as e:
            _logger.warning("Failed to write rotor log: %s", e)

    def _log_configuration_summary(
        self,
        beta: float,
        gamma: float,
        omega_initial: float,
        K_G_assembled: bool,
        K_SP_assembled: bool,
        C_enabled: bool,
        gravity_force_mag: float,
    ) -> None:
        """Log the solver configuration summary.

        Parameters
        ----------
        beta : float
            Newmark β parameter.
        gamma : float
            Newmark γ parameter.
        omega_initial : float
            Initial angular velocity [rad/s].
        K_G_assembled : bool
            Whether geometric stiffness matrix was assembled at startup.
        K_SP_assembled : bool
            Whether spin softening matrix was assembled at startup.
        C_enabled : bool
            Whether Rayleigh damping is active.
        gravity_force_mag : float
            Magnitude of total gravity force [N].
        """
        if not self._is_primary_rank():
            return

        self._print_separator()
        print(f"  dt = {self.dt:.6f} s  │  Newmark β={beta:.2f}, γ={gamma:.2f}", flush=True)
        print(
            f"  ω = {omega_initial:.4f} rad/s  │  axis = {self._coord_transforms.axis}",
            flush=True,
        )
        print(
            f"  K_G: {'enabled' if self._include_geometric_stiffness else 'disabled'}  │  "
            f"K_SP: {'enabled' if self._include_spin_softening else 'disabled'}  │  "
            f"Damping: {'enabled' if C_enabled else 'disabled'}  │  "
            f"Solver: {self._solver_type}",
            flush=True,
        )
        if self._include_geometric_stiffness or self._include_spin_softening:
            print(
                f"  Startup assembly: K_G={'on' if K_G_assembled else 'off'}  │  "
                f"K_SP={'on' if K_SP_assembled else 'off'}  │  "
                f"ω(0) = {omega_initial:.4f} rad/s (zero-valued if ω=0)",
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
        """Log time step header information.

        Parameters
        ----------
        time_step : int
            Current time window index.
        step : int
            Global iteration counter.
        t_target : float
            Target time for this window [s].
        theta_deg : float
            Estimated rotation angle [°].
        """
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
        alpha: float,
        inertial_diags: Dict[str, Any],
        F_gravity_local: Optional[np.ndarray],
        torque_total_vec: np.ndarray,
        torque_aero: float,
        torque_non_aero: float,
        torque_inertial: float,
        torque_gravity: float,
        torque_total: float,
        ct: float = 0.0,
        cp: float = 0.0,
        cq: float = 0.0,
        tsr: float = 0.0,
        ramp_factor: Optional[float] = None,
        force_ramp_time: Optional[float] = None,
    ) -> None:
        """Log force statistics for current time step.

        Parameters
        ----------
        raw_force : np.ndarray
            Raw total force from preCICE [N], shape (3,).
        raw_max_nodal : float
            Maximum nodal force magnitude before ramping [N].
        applied_force : np.ndarray
            Force after ramp scaling [N], shape (3,).
        applied_max_nodal : float
            Maximum nodal force magnitude after ramping [N].
        n_nodes : int
            Number of interface nodes.
        local_force_mag : float
            Total force magnitude in rotating frame [N].
        omega : float
            Current angular velocity [rad/s].
        alpha : float
            Current angular acceleration [rad/s²].
        inertial_diags : Dict[str, Any]
            Diagnostic dict from InertialForcesCalculator with per-force
            magnitudes (centrifugal, coriolis, euler, total_inertial).
        F_gravity_local : np.ndarray or None
            Gravity forces in rotating frame, shape (n_nodes, 3).
        torque_total_vec : np.ndarray
            Total torque vector in rotating frame [N·m], shape (3,).
        torque_aero : float
            Aerodynamic torque projected on rotation axis [N·m].
        torque_non_aero : float
            Net non-aerodynamic torque (τ_total - τ_aero) on axis [N·m].
        torque_inertial : float
            Inertial torque projected on rotation axis [N·m].
        torque_gravity : float
            Gravitational torque projected on rotation axis [N·m].
        torque_total : float
            Total torque projected on rotation axis [N·m].
        ramp_factor : float, optional
            Current force ramp factor in [0, 1].
        force_ramp_time : float, optional
            Ramp duration [s].
        """
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
        # Log omega ramp status if using a ramped provider
        if isinstance(self._omega_provider, RampedOmega):
            target = self._omega_provider._target_omega
            ramp_t = self._omega_provider._ramp_time
            pct = min(omega / target, 1.0) * 100.0 if target > 0 else 100.0
            if pct < 100.0:
                print(
                    f"  ├─ ► ω RAMP ACTIVE: {pct:5.1f}%  │  "
                    f"ω={omega:.4f} / {target:.4f} rad/s  │  "
                    f"α={alpha:.4f} rad/s²  │  Target: {ramp_t:.4f}s",
                    flush=True,
                )
            else:
                print(
                    f"  ├─ ✓ ω RAMP COMPLETE  │  ω={omega:.4f} rad/s (target reached)",
                    flush=True,
                )
            print("  │", flush=True)
        elif isinstance(self._omega_provider, RampedComputedOmega):
            target = self._omega_provider._target_omega
            ramp_t = self._omega_provider._ramp_time
            if not self._omega_provider._ramp_completed:
                pct = min(omega / target, 1.0) * 100.0 if target > 0 else 100.0
                print(
                    f"  ├─ ► ω RAMP ACTIVE: {pct:5.1f}%  │  "
                    f"ω={omega:.4f} / {target:.4f} rad/s  │  "
                    f"α={alpha:.4f} rad/s²  │  Target: {ramp_t:.4f}s",
                    flush=True,
                )
            else:
                print(
                    f"  ├─ ✓ ω RAMP COMPLETE (dynamic)  │  "
                    f"ω={omega:.4f} rad/s  │  α={alpha:.4f} rad/s²",
                    flush=True,
                )
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
            f"  │  Vector: [{torque_total_vec[0]:.4e}, "
            f"{torque_total_vec[1]:.4e}, {torque_total_vec[2]:.4e}] N·m",
            flush=True,
        )
        print(f"  │  Aero:       {torque_aero:+.4e} N·m", flush=True)
        print(f"  │  Non-aero:   {torque_non_aero:+.4e} N·m", flush=True)
        print(f"  │  Inertial:   {torque_inertial:+.4e} N·m", flush=True)
        print(f"  │  Gravity:    {torque_gravity:+.4e} N·m", flush=True)
        print(f"  │  Total:      {torque_total:+.4e} N·m  (on axis)", flush=True)
        print("  │", flush=True)
        print("  ├─ PERFORMANCE COEFFICIENTS", flush=True)
        print(
            f"  │  Ct = {ct:.4f}  │  Cp = {cp:.4f}  │  Cq = {cq:.4f}  │  TSR = {tsr:.4f}",
            flush=True,
        )
        print("  └" + "─" * 67, flush=True)

    def _log_solver_response(
        self, ksp_its: int, ksp_reason: int, max_disp: float, iter_wall_time: float = 0.0
    ) -> None:
        """Log linear solver response.

        Parameters
        ----------
        ksp_its : int
            Number of KSP iterations performed.
        ksp_reason : int
            PETSc convergence reason code (positive = converged).
        max_disp : float
            Maximum displacement magnitude [m].
        iter_wall_time : float
            Wall-clock time for this iteration [s].
        """
        if not self._is_primary_rank():
            return

        print("  ┌─ SOLVER RESPONSE", flush=True)
        print(f"  │  KSP iterations: {ksp_its}  (reason: {ksp_reason})", flush=True)
        print(f"  │  max|u_new| = {max_disp:.4e} m", flush=True)
        print(
            f"  │  ⏱ solve time: {iter_wall_time:.3f} s (read forces + compute + KSP)", flush=True
        )
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
        window_wall_time: float = 0.0,
        window_iters: int = 0,
    ) -> None:
        """Log time window convergence.

        Parameters
        ----------
        t : float
            Current time after convergence [s].
        max_disp : float
            Maximum displacement magnitude [m].
        max_vel : float
            Maximum velocity magnitude [m/s].
        max_acc : float
            Maximum acceleration magnitude [m/s²].
        driving_torque : float, optional
            Driving torque for this window [N·m].
        new_alpha : float, optional
            Updated angular acceleration [rad/s²].
        new_omega : float, optional
            Updated angular velocity for next window [rad/s].
        phase_info : str, optional
            OmegaProvider phase description (e.g. "ramp", "dynamic").
        window_wall_time : float
            Wall-clock time for this time window [s].
        window_iters : int
            Number of sub-iterations in this time window.
        """
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
        if window_iters > 0:
            avg_iter = window_wall_time / window_iters
            print(
                f"  │  ⏱ window: {window_wall_time:.3f} s  │  "
                f"{window_iters} iters  │  avg: {avg_iter:.3f} s/iter",
                flush=True,
            )
        print(
            f"  └─ Advanced to t = {t:.6f} s  │  θ = {np.degrees(self._theta):.1f}°",
            flush=True,
        )

    def _log_finalization(
        self,
        t: float,
        time_step: int,
        step: int,
        total_iter_time: float = 0.0,
        total_window_time: float = 0.0,
        total_windows: int = 0,
        total_iters: int = 0,
    ) -> None:
        """Log simulation finalization.

        Parameters
        ----------
        t : float
            Final simulation time [s].
        time_step : int
            Total number of time windows completed.
        step : int
            Total number of sub-iterations across all time windows.
        total_iter_time : float
            Cumulative wall time of all iterations [s].
        total_window_time : float
            Cumulative wall time of all time windows [s].
        total_windows : int
            Number of completed time windows.
        total_iters : int
            Number of completed iterations.
        """
        if not self._is_primary_rank():
            return

        self._print_header("FSI SIMULATION COMPLETED (CO-ROTATIONAL)")
        print(f"  Final time:        {t:.6f} s", flush=True)
        print(f"  Final θ:           {np.degrees(self._theta):.1f}°", flush=True)
        print(f"  Time steps:        {time_step}", flush=True)
        print(f"  Total iterations:  {step}", flush=True)
        if time_step > 0:
            print(f"  Avg iters/step:    {step / time_step:.2f}", flush=True)
        if total_iters > 0:
            avg_iter_time = total_iter_time / total_iters
            print(f"  ⏱ Total iter time:  {total_iter_time:.1f} s", flush=True)
            print(f"  ⏱ Avg iter time:   {avg_iter_time:.3f} s", flush=True)
        if total_windows > 0:
            avg_window_time = total_window_time / total_windows
            print(f"  ⏱ Total window time: {total_window_time:.1f} s", flush=True)
            print(f"  ⏱ Avg window time:  {avg_window_time:.3f} s", flush=True)
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
        """Compute the Newmark effective force vector.

        Assembles the RHS of the Newmark implicit system:

            F_eff = {F_new} + [M]·(a₀·u + a₂·v + a₃·a) + [C]·(a₁·u + a₄·v + a₅·a)

        where {F_new} contains all external and inertial forces at the new
        time step (aerodynamic + centrifugal + Coriolis + Euler + gravity),
        and the [M] and [C] terms account for the Newmark time integration
        history from the current state (u, v, a).

        Parameters
        ----------
        F_new_red : PETSc.Vec
            Reduced external + inertial force vector at time t+dt.
        u, v, a : PETSc.Vec
            Current displacement, velocity, acceleration (reduced).
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        C_red : PETSc.Mat, optional
            Reduced damping matrix.
        coeffs : NewmarkCoefficients
            Precomputed integration coefficients.

        Returns
        -------
        PETSc.Vec
            Effective force vector for the Newmark solve.
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
        """Perform Newmark-β state update after a converged time window.

        Computes new acceleration and velocity from the displacement increment:

            a_new = (Δu - dt·v - (0.5 - β)·dt²·a) / (β·dt²)
            v_new = v + dt·(1 - γ)·a + dt·γ·a_new

        With β = 0.25, γ = 0.5 this is the average acceleration method
        (unconditionally stable, second-order accurate, no numerical damping).

        Parameters
        ----------
        u : PETSc.Vec
            Displacement at current time step (reduced).
        u_new : PETSc.Vec
            Displacement at new time step (reduced).
        v : PETSc.Vec
            Velocity at current time step (reduced).
        a : PETSc.Vec
            Acceleration at current time step (reduced).
        coeffs : NewmarkCoefficients
            Precomputed integration coefficients.
        beta : float
            Newmark β parameter.

        Returns
        -------
        Tuple[PETSc.Vec, PETSc.Vec]
            (v_new, a_new) velocity and acceleration at t+dt.
        """
        delta_u = u_new - u
        a_new = (delta_u - self.dt * v - (0.5 - beta) * self.dt**2 * a) / (beta * self.dt**2)
        v_new = v + coeffs.a6 * a + coeffs.a7 * a_new

        return v_new, a_new

    # =========================================================================
    # Main Solve Method
    # =========================================================================

    def solve(self) -> Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """Perform co-rotational dynamic FSI analysis.

        Orchestrates the full simulation pipeline:

        1. **Matrix assembly** (once):
           - K (elastic), M (lumped mass), C (Rayleigh damping)
           - K_G (geometric stiffness from centrifugal prestress)
           - K_SP (spin softening, diagonal: -ω²·M·(I - n̂⊗n̂))

        2. **Effective stiffness** (once, rebuilt when ω changes):
           K_eff = K + K_G + K_SP + a₀·M + a₁·C

        3. **preCICE time loop** (implicit coupling with sub-iterations):
           a. Get ω, α from OmegaProvider (constant within time window)
           b. Read aerodynamic forces from fluid → transform to rotating frame
           c. Compute inertial forces:
              - Centrifugal at X₀ (spin softening via K_SP handles X₀+u)
              - Coriolis at current velocity estimate
              - Euler at X₀+u (only when α ≠ 0)
           d. Solve: K_eff · u_new = F_eff
           e. Transform u_local → u_global and write to preCICE
           f. preCICE sub-iteration or advance to next window

        Returns
        -------
        Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec]
            Final (displacement, velocity, acceleration) vectors.
        """
        omega_initial, _ = self._omega_provider.get_omega(0.0)

        self._print_header("FSI DYNAMIC ANALYSIS - CO-ROTATIONAL ROTOR SOLVER")

        # Phase 1: Matrix Assembly
        matrices, bc_manager, C, K_G = self._assemble_system_matrices(omega_initial)

        # Auto-compute inertia if requested
        if getattr(self, "_auto_inertia", False):
            estimated_inertia = self._compute_estimated_inertia()
            if self._is_primary_rank():
                print(
                    f"  ↳ Auto-computed Moment of Inertia: {estimated_inertia:.4e} kg·m²",
                    flush=True,
                )

            # Re-initialize provider with computed inertia
            ramp_time = self._auto_inertia_params.get("ramp_time", 0.0)
            target_omega = self._auto_inertia_params["target_omega"]
            shaft_torque = self._auto_inertia_params["shaft_torque"]

            if ramp_time > 0.0:
                # Use combined ramp + computed provider
                self._omega_provider = RampedComputedOmega(
                    target_omega=target_omega,
                    ramp_time=ramp_time,
                    moment_of_inertia=estimated_inertia,
                    shaft_torque=shaft_torque,
                )
                if self._is_primary_rank():
                    print(
                        f"  ↳ Omega mode: Ramp ({ramp_time:.3f} s) → Dynamic (I={estimated_inertia:.4e} kg·m²)",
                        flush=True,
                    )
            else:
                # Pure dynamic mode from start
                self._omega_provider = ComputedOmega(
                    moment_of_inertia=estimated_inertia,
                    initial_omega=target_omega,
                    shaft_torque=shaft_torque,
                )
                if self._is_primary_rank():
                    print(f"  ↳ Omega mode: Dynamic (I={estimated_inertia:.4e} kg·m²)", flush=True)

        # Phase 2: preCICE Initialization
        t, step, time_step = self._initialize_precice(bc_manager)

        # Phase 3: Reduce matrices and setup solver
        K_red, F_red, M_red = bc_manager.reduced_system
        self.free_dofs = bc_manager.free_dofs

        # Tell GAMG to aggregate at the node level (dofs_per_node DOFs per block).
        # This must be set on the PRECONDITIONER matrix (K_red), not on K_eff.
        # K_red is used as the fixed preconditioner for the entire run — its
        # sparsity/values never change, so GAMG setup runs exactly once.
        K_red.setBlockSize(self.domain.dofs_per_node)

        C_red = bc_manager.reduce_matrix(C) if C is not None else None
        K_G_red = bc_manager.reduce_matrix(K_G) if K_G is not None else None

        # Build spin softening matrix (ANSYS Eq. 3-74 / 14-55)
        K_SP_red = None
        if self._include_spin_softening:
            K_SP = self._build_spin_softening_matrix(omega_initial)
            K_SP_red = bc_manager.reduce_matrix(K_SP)

        # Phase 4: Newmark coefficients and effective stiffness
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        coeffs = NewmarkCoefficients.from_newmark_params(beta, gamma, self.dt)

        K_eff = self._build_effective_stiffness(K_red, M_red, coeffs, K_G_red, C_red, K_SP_red)

        if not self._prepared:
            self._setup_solver()
            self._prepared = True

        # Use K_red (pure elastic stiffness, constant) as the GAMG preconditioner
        # matrix.  K_eff changes every window when ω changes (K_G/K_SP rebuilt),
        # but K_red never changes.  PETSc only re-runs PCSetUp when the
        # preconditioner matrix state changes → GAMG hierarchy is built once and
        # reused throughout the simulation, eliminating the ~90 s spike per window.
        self._solver.setOperators(K_eff, K_red)
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

        # Write initial state checkpoint with the same VTU fields used in
        # subsequent checkpoints, so visualization data is consistent.
        if (
            self._checkpoint_manager is not None
            and self.solver_params.get("write_initial_state", True)
            and starting_from_zero
            and t == 0
        ):
            # Keep field names consistent across all checkpoints.
            zeros_interface = np.zeros_like(F_gravity_global)
            F_total_initial = F_centrifugal_initial + F_gravity_global

            initial_fields = {
                "F_AERO": self._expand_interface_forces_to_full(zeros_interface),
                "F_INERT": self._expand_interface_forces_to_full(F_centrifugal_initial),
                "F_GRAV": self._expand_interface_forces_to_full(F_gravity_global.reshape(-1, 3)),
                "F_TOTAL": self._expand_interface_forces_to_full(F_total_initial),
            }

            # Stress / strain fields (TOP, MID, BOT for shells; full 3-D for solids)
            u_full_init = bc_manager.expand_solution(u).array.copy()
            initial_fields.update(self._compute_stress_fields(u_full_init))

            self._handle_checkpoint(
                t=0.0,
                time_step=0,
                dt=self.dt,
                theta=self._theta,
                omega=omega_initial,
                u_red=u.array.copy(),
                v_red=v.array.copy(),
                a_red=a.array.copy(),
                u_full=u_full_init,
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
            K_SP_red is not None,
            C_red is not None,
            gravity_force_mag,
        )

        # Time stepping loop
        (
            t,
            time_step,
            step,
            u,
            v,
            a,
            _total_iter_time,
            _total_window_time,
            _total_windows,
            _total_iters,
        ) = self._time_stepping_loop(
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
            K_SP_red,
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
        self._log_finalization(
            t,
            time_step,
            step,
            total_iter_time=_total_iter_time,
            total_window_time=_total_window_time,
            total_windows=_total_windows,
            total_iters=_total_iters,
        )

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
        """Assemble stiffness, mass, damping, and geometric stiffness matrices.

        Parameters
        ----------
        omega_initial : float
            Initial angular velocity [rad/s], used to compute the centrifugal
            prestress for K_G.

        Returns
        -------
        Tuple[Tuple[PETSc.Mat, PETSc.Mat], BoundaryConditionManager, Optional[PETSc.Mat], Optional[PETSc.Mat]]
            ((K, M), bc_manager, C, K_G) where C and K_G may be None.
        """
        self._print_phase(1, 7, "Assembling stiffness matrix...")
        self.K = self.domain.assemble_stiffness_matrix()

        self._print_phase(2, 7, "Assembling mass matrix (lumped)...")
        M_consistent = self.domain.assemble_mass_matrix()
        self.M = self.lump_mass_matrix(M_consistent)

        # Geometric stiffness
        K_G = None
        if self._include_geometric_stiffness:
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
        if not self._damping_enabled:
            self._print_phase(4, 7, "Rayleigh damping: disabled (enabled=false)")
        elif self._damping_auto:
            self._print_phase(4, 7, "Rayleigh damping: auto-computing from modal analysis...")
            self._eta_k, self._eta_m = self._compute_rayleigh_auto()
            self._print_phase(
                4, 7, f"Creating Rayleigh damping (η_m={self._eta_m:.4e}, η_k={self._eta_k:.4e})..."
            )
            C = self._create_damping_matrix(self.K, self.M)
        elif self._eta_m != 0.0 or self._eta_k != 0.0:
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
        """Initialize preCICE coupling and return timing state.

        On restart, the OpenFOAM fluid mesh is read from the saved time
        directory (``<time>/polyMesh/points``), which contains the **rotated**
        mesh points.  The solid mesh, however, is loaded from
        ``deformed_mesh.h5`` in the co-rotating frame (unrotated).  To make
        the preCICE RBF mapping work, the solid interface coordinates must
        be rotated by the checkpoint angle θ₀ so that both meshes are in the
        same (lab) frame.

        Parameters
        ----------
        bc_manager : BoundaryConditionManager
            Boundary condition manager with fixed/free DOF partitioning.

        Returns
        -------
        Tuple[float, int, int]
            (t, step, time_step) initial time, iteration counter, and
            time window index.
        """
        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.solver_params.get("time_step", 1.0)))
        bootstrap_dt = float(self.solver_params.get("time_step", 1.0))

        self._print_phase(6, 7, "Initializing preCICE coupling...")

        # ── Peek at checkpoint θ so we can rotate the Solid-Mesh vertices ──
        # This must happen BEFORE preCICE initialize(), because vertex
        # coordinates are registered during that call and cannot be changed
        # afterwards.
        restart_theta = self._peek_checkpoint_theta()
        precice_coord_transform = None
        if abs(restart_theta) > 1e-12:
            R = self._coord_transforms.rotation_matrix(restart_theta)
            center = self._coord_transforms.center

            def precice_coord_transform(coords: np.ndarray) -> np.ndarray:
                """Rotate interface coords from co-rotating frame to lab frame."""
                shifted = coords - center
                rotated = shifted @ R.T  # (n,3) @ (3,3)^T
                return rotated + center

            if self._is_primary_rank():
                print(
                    f"  ↳ Rotating Solid-Mesh vertices by θ₀ = {np.degrees(restart_theta):.1f}°"
                    f" to match OpenFOAM rotated mesh",
                    flush=True,
                )

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
            # preCICE owns the authoritative dt, but before initialize() returns
            # we only have the configured startup step available.
            #
            # On restart, the omega provider is freshly created and get_omega(0+dt)
            # returns a ramped value near zero instead of the actual omega at the
            # checkpoint time.  Use the checkpoint omega directly if available.
            restart_omega = self._peek_checkpoint_omega()
            if restart_omega is not None:
                initial_omega = restart_omega
            else:
                initial_omega, _ = self._omega_provider.get_omega(t + bootstrap_dt)
            initial_data = {"AngularVelocity": initial_omega}
            if self._is_primary_rank():
                print(f"  ↳ Initial ω for preCICE: {initial_omega:.4f} rad/s", flush=True)

        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
            custom_mesh_coords=custom_meshes if custom_meshes else None,
            initial_data_values=initial_data,
            precice_coord_transform=precice_coord_transform,
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

        Parameters
        ----------
        K_red : PETSc.Mat
            Reduced elastic stiffness matrix.
        F_red : PETSc.Vec
            Reduced force vector (used for initial acceleration).
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        bc_manager : BoundaryConditionManager
            Boundary condition manager.
        t : float
            Initial time [s].
        time_step : int
            Initial time window index.
        omega_initial : float
            Initial angular velocity [rad/s] (for θ estimation on restart).

        Returns
        -------
        Tuple[PETSc.Vec, PETSc.Vec, PETSc.Vec, float, int, bool]
            (u, v, a, t, time_step, starting_from_zero).
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

            # Restore omega provider state from checkpoint so that ramps are
            # not re-applied and the dynamic omega evolution is preserved.
            ckpt_omega = checkpoint_state.get("omega", None)
            if ckpt_omega is not None:
                ckpt_omega = float(ckpt_omega)
                if isinstance(self._omega_provider, RampedComputedOmega):
                    self._omega_provider.set_state((ckpt_omega, 0.0, True, t))
                    if self._is_primary_rank():
                        print(
                            f"  ↳ Omega provider restored: ω = {ckpt_omega:.4f} rad/s"
                            f" (ramp completed)",
                            flush=True,
                        )
                elif isinstance(self._omega_provider, ComputedOmega):
                    self._omega_provider.set_state((ckpt_omega, 0.0))
                    if self._is_primary_rank():
                        print(
                            f"  ↳ Omega provider restored: ω = {ckpt_omega:.4f} rad/s",
                            flush=True,
                        )
                elif isinstance(self._omega_provider, RampedOmega):
                    # RampedOmega uses absolute time, so no state to restore,
                    # but if t > ramp_time the ramp is already complete.
                    if self._is_primary_rank():
                        print(
                            f"  ↳ Omega provider (RampedOmega): checkpoint ω = {ckpt_omega:.4f} rad/s",
                            flush=True,
                        )

            # Explicitly skip ramps on restart — simulation continues from
            # checkpoint state where ramps already completed.
            self._skip_ramps = True
            if self._is_primary_rank():
                print("  ↳ Ramps bypassed (restart from checkpoint)", flush=True)

            # Restore running maximum force for sanity checks.
            ckpt_max_force = checkpoint_state.get("max_force_seen", None)
            if ckpt_max_force is not None:
                self._max_force_seen = float(ckpt_max_force)
                if self._is_primary_rank():
                    print(
                        f"  ↳ Force baseline restored: |F|_max = {self._max_force_seen:.3e} N",
                        flush=True,
                    )

            # Activate grace period: clamp (not abort) force spikes while
            # the CFD solver re-stabilizes after restart.
            self._restart_grace_remaining = self._restart_force_grace_windows
            if self._is_primary_rank() and self._restart_grace_remaining > 0:
                print(
                    f"  ↳ Force grace period: {self._restart_grace_remaining} windows"
                    f" (clamp instead of abort)",
                    flush=True,
                )

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
        K_SP_red: Optional[PETSc.Mat],
        bc_manager: BoundaryConditionManager,
        coeffs: NewmarkCoefficients,
        beta: float,
        gamma: float,
        interface_coords: np.ndarray,
        interface_dofs: np.ndarray,
        interface_masses: np.ndarray,
        F_gravity_global: np.ndarray,
    ) -> Tuple[float, int, int, PETSc.Vec, PETSc.Vec, PETSc.Vec]:
        """Main preCICE-driven time stepping loop.

        Implements the implicit FSI coupling protocol:

        For each time window [t, t+dt]:
          1. ω is computed once and held fixed for all sub-iterations.
          2. K_G and K_SP are rebuilt if |Δω| > threshold.
          3. Forces from fluid solver are read, transformed R^T(θ)·F_global,
             and combined with inertial + gravity forces.
          4. The Newmark system K_eff·u_new = F_eff is solved.
          5. Displacement is transformed R(θ)·u_local and written to preCICE.
          6. preCICE decides: sub-iterate (restore checkpoint) or converge.

        Force evaluation strategy (consistent with ANSYS §14.4):
          - Centrifugal F_cf(X₀) on RHS + K_SP·u on LHS = total centrifugal
            effect at deformed position, handled implicitly.
          - Coriolis F_cor(v_guess) on RHS, explicit with best velocity estimate.
          - Euler F_euler(X₀+u) on RHS, only when α ≠ 0. No LHS correction
            exists for Euler, so explicit at deformed coords is appropriate.
          - Gravity F_g transformed to rotating frame via R^T(θ).

        Parameters
        ----------
        u, v, a : PETSc.Vec
            Initial displacement, velocity, acceleration (reduced).
        t : float
            Start time [s].
        time_step : int
            Starting time window index.
        step : int
            Starting global iteration counter.
        K_red : PETSc.Mat
            Reduced elastic stiffness matrix.
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        K_eff : PETSc.Mat
            Current effective stiffness matrix.
        C_red : PETSc.Mat, optional
            Reduced damping matrix.
        K_G_red : PETSc.Mat, optional
            Reduced geometric stiffness matrix.
        K_SP_red : PETSc.Mat, optional
            Reduced spin softening matrix.
        bc_manager : BoundaryConditionManager
            Boundary condition manager.
        coeffs : NewmarkCoefficients
            Precomputed integration coefficients.
        beta : float
            Newmark β parameter.
        gamma : float
            Newmark γ parameter.
        interface_coords : np.ndarray, shape (n_nodes, 3)
            Reference coordinates of interface nodes.
        interface_dofs : np.ndarray
            DOF indices for interface nodes.
        interface_masses : np.ndarray, shape (n_nodes,)
            Scalar nodal masses at interface nodes [kg].
        F_gravity_global : np.ndarray, shape (n_nodes, 3)
            Gravity forces in inertial frame [N].

        Returns
        -------
        Tuple[float, int, int, PETSc.Vec, PETSc.Vec, PETSc.Vec]
            (final_t, final_time_step, final_step, u, v, a).
        """
        # Omega must remain constant inside each implicit-coupling time window.
        # Recompute only once at the window start, then hold for all sub-iterations.
        omega, alpha = self._omega_provider.get_omega(t + self.dt)
        theta_target = self._theta + omega * self.dt
        omega_initialized_for_window = False
        window_iter_count = 0
        window_wall_start = time.perf_counter()
        total_iter_time = 0.0
        total_window_time = 0.0
        total_windows_completed = 0
        total_iters_completed = 0

        while self.precice_participant.is_coupling_ongoing:
            step += 1
            iter_wall_start = time.perf_counter()
            window_iter_count += 1

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

                # Update K_G and K_SP if omega changed significantly (window-level update only)
                K_eff, K_G_red, K_SP_red = self._update_geometric_stiffness_if_needed(
                    omega, K_red, M_red, K_eff, C_red, K_G_red, K_SP_red, bc_manager, coeffs
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
            force_jump_exceeded = (
                self._max_force_seen > 0.0
                and force_total_mag > self._force_jump_factor * self._max_force_seen
            )

            if force_jump_exceeded:
                if self._restart_grace_remaining > 0:
                    # Grace period: clamp forces to last known maximum
                    clamp_scale = self._max_force_seen / force_total_mag
                    data_global *= clamp_scale
                    applied_force *= clamp_scale
                    applied_max_nodal *= clamp_scale
                    if self._is_primary_rank():
                        print(
                            f"  ⚠ Force clamped (grace {self._restart_grace_remaining}): "
                            f"|F| = {force_total_mag:.3e} → {self._max_force_seen:.3e} N",
                            flush=True,
                        )
                else:
                    raise RuntimeError(
                        f"Diverged force from fluid solver detected at t={t_target:.6f} s: "
                        f"|F| = {force_total_mag:.3e} N is "
                        f"{force_total_mag / self._max_force_seen:.1f}× the running max "
                        f"({self._max_force_seen:.3e} N). "
                        f"CFD solver likely diverged (Courant blow-up or PIMPLE non-convergence). "
                        "Aborting to prevent FSI hang."
                    )
            if (
                self._force_max_magnitude is not None
                and force_total_mag > self._force_max_magnitude
            ):
                if self._restart_grace_remaining > 0:
                    clamp_scale = self._force_max_magnitude / force_total_mag
                    data_global *= clamp_scale
                    applied_force *= clamp_scale
                    applied_max_nodal *= clamp_scale
                    if self._is_primary_rank():
                        print(
                            f"  ⚠ Force clamped to limit (grace {self._restart_grace_remaining}): "
                            f"|F| = {force_total_mag:.3e} → {self._force_max_magnitude:.3e} N",
                            flush=True,
                        )
                else:
                    raise RuntimeError(
                        f"Force exceeds configured limit at t={t_target:.6f} s: "
                        f"|F| = {force_total_mag:.3e} N > {self._force_max_magnitude:.3e} N. Aborting."
                    )
            # Update running maximum only when forces reflect real physics:
            # - not during force ramp (CFD is transitioning, forces are not representative)
            # - not when a jump/clamp occurred (spike would inflate the baseline)
            # -------------------------------------------------------------------------

            # Apply force ramp if configured (skip on restart)
            ramp_factor = 1.0
            if self._force_ramp_time > 1e-12 and not self._skip_ramps:
                ramp_check = min(t_target / self._force_ramp_time, 1.0)
                if ramp_check < 1.0:
                    ramp_factor = ramp_check
                    data_global *= ramp_factor
                    applied_force *= ramp_factor
                    applied_max_nodal *= ramp_factor

            if not force_jump_exceeded and ramp_factor >= 1.0:
                self._max_force_seen = max(self._max_force_seen, force_total_mag)

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

            # Extract interface displacements (needed for Euler at deformed coords and torque)
            u_full_disp = bc_manager.expand_solution(u)
            interface_disps = self._get_interface_displacements(u_full_disp, interface_dofs)

            # Only include Euler force when enabled AND there is angular acceleration (alpha != 0)
            include_euler = self._include_euler and abs(alpha) > 1e-12

            # Centrifugal and Coriolis always evaluated at X₀ (initial coords).
            # When K_SP is active, the spin softening correction ω×(ω×u) is handled
            # implicitly via K_SP on the LHS — evaluating centrifugal at X₀+u would
            # double-count. Euler force is evaluated at X₀+u since it depends on α
            # (angular acceleration) which has no LHS matrix correction.
            F_inertial, inertial_diags = self._inertial_calculator.compute_all_inertial_forces(
                nodal_coords=interface_coords,
                nodal_velocities=interface_velocities,
                nodal_masses=interface_masses,
                omega=omega,
                alpha=alpha,
                include_centrifugal=self._include_centrifugal,
                include_coriolis=self._include_coriolis,
                include_euler=False,  # Euler handled separately at deformed coords
            )

            if include_euler:
                euler_coords = interface_coords + interface_disps
                F_euler = self._inertial_calculator.compute_euler_force(
                    nodal_coords=euler_coords,
                    nodal_masses=interface_masses,
                    alpha=alpha,
                )
                F_inertial = F_inertial + F_euler
                inertial_diags["euler_max"] = float(np.max(np.linalg.norm(F_euler, axis=1)))

            # Transform gravity to rotating frame
            F_gravity_local = self._transform_gravity_to_local(
                F_gravity_global, theta_target, data_local_2d.shape
            )

            # Combine forces
            data_combined = data_local_2d + F_inertial + F_gravity_local

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

            # --- Torque breakdown per force component ---
            # Total torque (all forces: aero + inertial + gravity)
            torque_total_local, torque_total_scalar = self._compute_rotor_torque(
                interface_coords, interface_disps, data_combined
            )
            # Aerodynamic torque (CFD surface forces only)
            torque_aero_local, torque_aero_scalar = self._compute_rotor_torque(
                interface_coords, interface_disps, data_local_2d
            )
            # Inertial torque (centrifugal + Coriolis + Euler)
            torque_inertial_local, torque_inertial_scalar = self._compute_rotor_torque(
                interface_coords, interface_disps, F_inertial
            )
            # Gravitational torque
            torque_gravity_local, torque_gravity_scalar = self._compute_rotor_torque(
                interface_coords, interface_disps, F_gravity_local
            )

            # Transform torque vectors to global (inertial) frame: τ_global = R(θ) · τ_local
            R_theta = self._coord_transforms.rotation_matrix(theta_target)
            torque_aero_global = R_theta @ torque_aero_local
            torque_total_global = R_theta @ torque_total_local

            # Update rotor radius with deformation for accurate performance coefficients
            current_radius = self._compute_rotor_radius(interface_coords, interface_disps)

            # Power: aero (extracted from wind) vs total (net after structural losses)
            thrust = np.dot(applied_force, self._coord_transforms.axis)
            power_aero = torque_aero_scalar * omega
            power_total = torque_total_scalar * omega

            # Signed non-aero contribution: inertial + gravity (+ any other non-CFD terms).
            torque_non_aero_scalar = torque_total_scalar - torque_aero_scalar

            # Efficiency is the fraction of aero torque opposed by non-aero effects.
            # This avoids >100% values when non-aero terms assist rotor motion.
            if abs(torque_aero_scalar) > _MIN_DENOMINATOR:
                aero_sign = np.sign(torque_aero_scalar)
                opposing_non_aero_torque = -torque_non_aero_scalar * aero_sign
                structural_efficiency = np.clip(
                    opposing_non_aero_torque / abs(torque_aero_scalar),
                    0.0,
                    1.0,
                )
            else:
                structural_efficiency = 0.0

            ct, cp, cq, tsr = self._compute_performance_coefficients(
                thrust, power_aero, torque_aero_scalar, omega, radius=current_radius
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
                alpha,
                inertial_diags,
                F_gravity_local if self._include_gravity else None,
                torque_total_local,
                torque_aero_scalar,
                torque_non_aero_scalar,
                torque_inertial_scalar,
                torque_gravity_scalar,
                torque_total_scalar,
                ct=ct,
                cp=cp,
                cq=cq,
                tsr=tsr,
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
            solve_wall_time = time.perf_counter() - iter_wall_start
            self._log_solver_response(ksp_its, ksp_reason, max_disp_iter, solve_wall_time)

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

                # Measure full iteration time (includes preCICE advance + wait)
                iter_wall_time = time.perf_counter() - iter_wall_start
                total_iter_time += iter_wall_time
                total_iters_completed += 1
                if self._is_primary_rank():
                    print(
                        f"  ⏱ iteration total: {iter_wall_time:.3f} s (sub-iter, restored checkpoint)",
                        flush=True,
                    )
            else:
                # Measure full iteration time (includes preCICE advance + wait)
                iter_wall_time = time.perf_counter() - iter_wall_start
                total_iter_time += iter_wall_time
                total_iters_completed += 1

                window_wall_time = time.perf_counter() - window_wall_start
                total_window_time += window_wall_time
                total_windows_completed += 1
                time_step += 1
                self._theta += omega * self.dt
                omega_initialized_for_window = False

                # Decrement restart grace period after each converged window
                if self._restart_grace_remaining > 0:
                    self._restart_grace_remaining -= 1

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

                self._log_time_window_converged(
                    t,
                    max_disp,
                    max_vel,
                    max_acc,
                    *rotor_dynamics_info,
                    window_wall_time=window_wall_time,
                    window_iters=window_iter_count,
                )

                # Reset window timer for next window
                window_iter_count = 0
                window_wall_start = time.perf_counter()

                # Write angular velocity for the NEXT time window start.
                self._write_angular_velocity_to_precice(omega_next_for_coupling)

                # Log converged performance to CSV
                self._log_rotor_performance(
                    t,
                    omega * 30.0 / np.pi,
                    omega,
                    alpha,
                    np.degrees(self._theta),
                    thrust,
                    torque_aero_scalar,
                    torque_non_aero_scalar,
                    torque_inertial_scalar,
                    torque_gravity_scalar,
                    torque_total_scalar,
                    power_aero,
                    power_total,
                    structural_efficiency,
                    cp,
                    cq,
                    ct,
                    tsr,
                    torque_aero_global,
                    torque_total_global,
                    max_disp,
                    deformed_radius=current_radius,
                )

                # Write lightweight kinematic state for restart alignment
                self._write_restart_state(t, self._theta, omega, alpha)

                # Prepare consistent force fields for checkpoint visualization.
                force_fields = {
                    "F_AERO": self._expand_interface_forces_to_full(data_local_2d),
                    "F_INERT": self._expand_interface_forces_to_full(F_inertial),
                    "F_GRAV": self._expand_interface_forces_to_full(F_gravity_local),
                    "F_TOTAL": self._expand_interface_forces_to_full(data_combined),
                }

                # Stress / strain fields (TOP, MID, BOT for shells; full 3-D for solids)
                u_full_ckpt = bc_manager.expand_solution(u).array.copy()
                force_fields.update(self._compute_stress_fields(u_full_ckpt))

                self._handle_checkpoint(
                    t=t,
                    time_step=time_step,
                    dt=self.dt,
                    theta=self._theta,
                    omega=omega,
                    max_force_seen=self._max_force_seen,
                    u_red=u.array.copy(),
                    v_red=v.array.copy(),
                    a_red=a.array.copy(),
                    u_full=u_full_ckpt,
                    v_full=bc_manager.expand_state_vector(v).array.copy(),
                    a_full=bc_manager.expand_state_vector(a).array.copy(),
                    extra_fields=force_fields,
                )

                # Debug: write interface node positions at this checkpoint
                self._write_interface_debug(
                    t,
                    self._theta,
                    u_full_ckpt,
                    interface_coords,
                    interface_dofs,
                )

        return (
            t,
            time_step,
            step,
            u,
            v,
            a,
            total_iter_time,
            total_window_time,
            total_windows_completed,
            total_iters_completed,
        )

    # =========================================================================
    # Interface Debugging
    # =========================================================================

    def _write_interface_debug(
        self,
        t: float,
        theta: float,
        u_full: np.ndarray,
        interface_coords: np.ndarray,
        interface_dofs: np.ndarray,
    ) -> None:
        """Write interface node coordinates at checkpoint time for debugging.

        Saves three sets of coordinates per interface node:
        1. **Reference (co-rotating)**: X₀ from the FEM mesh
        2. **Deformed (co-rotating)**: X₀ + u_local (elastic only)
        3. **Deformed (lab frame)**: R(θ) · (X₀ + u_local)

        This allows diagnosing restart mismatches between the solid mesh
        (co-rotating) and the OpenFOAM mesh (lab frame).

        Parameters
        ----------
        t : float
            Current physical time [s].
        theta : float
            Current cumulative rotation angle [rad].
        u_full : np.ndarray
            Full displacement vector (all DOFs).
        interface_coords : np.ndarray
            Interface node reference coordinates, shape (n, 3).
        interface_dofs : np.ndarray
            Interface DOF indices.
        """
        if self._checkpoint_manager is None:
            return
        if not self._checkpoint_manager.should_write(t):
            return
        if not self._is_primary_rank():
            return

        try:
            # Extract interface displacements
            n_nodes = interface_coords.shape[0]
            u_interface = np.zeros((n_nodes, 3), dtype=np.float64)
            for i in range(n_nodes):
                for j in range(3):
                    dof = (
                        interface_dofs[i, j]
                        if interface_dofs.ndim == 2
                        else interface_dofs[i * 3 + j]
                    )
                    if dof < len(u_full):
                        u_interface[i, j] = u_full[dof]

            # Deformed positions in co-rotating frame
            deformed_local = interface_coords + u_interface

            # Deformed positions in lab frame
            R = self._coord_transforms.rotation_matrix(theta)
            center = self._coord_transforms.center
            shifted = deformed_local - center
            deformed_lab = (shifted @ R.T) + center

            # Write to checkpoint directory
            time_str = f"{t:.6f}"
            output_dir = os.path.join(self.solver_params.get("output_folder", "results"), time_str)
            os.makedirs(output_dir, exist_ok=True)

            node_ids = self.precice_participant.interface_node_ids
            debug_path = os.path.join(output_dir, "interface_debug.csv")
            with open(debug_path, "w") as f:
                f.write(f"# t={t:.6f} theta={theta:.6f} theta_deg={np.degrees(theta):.2f}\n")
                f.write(
                    "node_id,"
                    "ref_x,ref_y,ref_z,"
                    "def_local_x,def_local_y,def_local_z,"
                    "def_lab_x,def_lab_y,def_lab_z\n"
                )
                for i in range(n_nodes):
                    f.write(
                        f"{node_ids[i]},"
                        f"{interface_coords[i, 0]:.8e},{interface_coords[i, 1]:.8e},{interface_coords[i, 2]:.8e},"
                        f"{deformed_local[i, 0]:.8e},{deformed_local[i, 1]:.8e},{deformed_local[i, 2]:.8e},"
                        f"{deformed_lab[i, 0]:.8e},{deformed_lab[i, 1]:.8e},{deformed_lab[i, 2]:.8e}\n"
                    )
        except Exception as e:
            _logger.debug("Could not write interface debug: %s", e)

    # =========================================================================
    # Stress / Strain Post-Processing for Checkpoint Export
    # =========================================================================

    def _compute_stress_fields(
        self,
        u_full: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute stress fields and return a flat dict for VTU export.

        For **shell** elements the stresses are evaluated at three
        through-thickness locations (TOP, MID, BOT) and the dictionary
        keys are prefixed accordingly (e.g. ``TOP_von_mises``).

        For **solid** elements the Gauss-to-Node extrapolated stresses
        are returned without prefix.

        This method is called *only* when a checkpoint is being written
        and therefore does **not** impact the Newmark time-integration
        performance.
        """
        sr = StressRecovery(self.domain, u_full)

        # Detect whether we have shell elements
        has_shell = any(sr._is_shell(e) for e in self.domain._element_map.values())
        has_solid = any(sr._is_solid(e) for e in self.domain._element_map.values())

        out: Dict[str, np.ndarray] = {}

        if has_shell and not has_solid:
            # Pure shell mesh → export all three layers
            out.update(
                sr.compute_nodal_stresses_all_layers_dict(
                    stress_type=StressType.TOTAL,
                )
            )
            out.update(sr.compute_nodal_strains_all_layers_dict())
        elif has_solid and not has_shell:
            # Pure solid mesh → single set of results (no layer prefix)
            result = sr.compute_nodal_stresses()
            out.update(result.to_dict())
            out.update({f"strain_{k}": v for k, v in sr.compute_nodal_strains().to_dict().items()})
        else:
            # Mixed mesh → export shell layers + solid (with prefix)
            out.update(
                sr.compute_nodal_stresses_all_layers_dict(
                    stress_type=StressType.TOTAL,
                )
            )
            out.update(sr.compute_nodal_strains_all_layers_dict())

        return out

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
        K_SP_red: Optional[PETSc.Mat],
        bc_manager: BoundaryConditionManager,
        coeffs: NewmarkCoefficients,
    ) -> Tuple[PETSc.Mat, Optional[PETSc.Mat], Optional[PETSc.Mat]]:
        """Rebuild K_G and K_SP when ω changes, then re-form K_eff.

        Called once per preCICE time window (not per sub-iteration). Both K_G
        and K_SP scale with ω² but through different mechanisms:
          - K_G: element-level assembly from centrifugal prestress σ_cf ∝ ω².
          - K_SP: diagonal matrix K_SP = -ω²·M·(I - n̂⊗n̂), trivial to rebuild.

        The threshold |Δω| > 1e-4 rad/s avoids unnecessary refactorizations
        when ω is essentially constant.

        Parameters
        ----------
        omega : float
            New angular velocity [rad/s].
        K_red, M_red : PETSc.Mat
            Reduced elastic stiffness and mass matrices (constant).
        K_eff : PETSc.Mat
            Current effective stiffness (returned unchanged if no rebuild).
        C_red : PETSc.Mat, optional
            Reduced damping matrix.
        K_G_red : PETSc.Mat, optional
            Current reduced geometric stiffness.
        K_SP_red : PETSc.Mat, optional
            Current reduced spin softening matrix.
        bc_manager : BoundaryConditionManager
            For reducing new full-system matrices to free DOFs.
        coeffs : NewmarkCoefficients
            For rebuilding K_eff.

        Returns
        -------
        Tuple[PETSc.Mat, Optional[PETSc.Mat], Optional[PETSc.Mat]]
            Updated (K_eff, K_G_red, K_SP_red).
        """
        if self._prev_omega is not None and abs(omega - self._prev_omega) > _OMEGA_CHANGE_THRESHOLD:
            rebuild = False
            reassembled = []

            if self._include_geometric_stiffness:
                try:
                    K_G_new = self.domain.assemble_geometric_stiffness(
                        omega=omega,
                        rotation_axis=self._coord_transforms.axis,
                        rotation_center=self._coord_transforms.center,
                    )
                    K_G_red = bc_manager.reduce_matrix(K_G_new)
                    rebuild = True
                    reassembled.append("K_G")
                except Exception as e:
                    _logger.warning("Failed to update K_G: %s", e)

            if self._include_spin_softening:
                K_SP_new = self._build_spin_softening_matrix(omega)
                K_SP_red = bc_manager.reduce_matrix(K_SP_new)
                rebuild = True
                reassembled.append("K_SP")

            if rebuild:
                K_eff = self._build_effective_stiffness(
                    K_red, M_red, coeffs, K_G_red, C_red, K_SP_red
                )
                # K_red is the fixed preconditioner matrix (never changes).
                # Passing it explicitly prevents GAMG from re-running PCSetUp.
                self._solver.setOperators(K_eff, K_red)
                if self._is_primary_rank():
                    matrices_str = " + ".join(reassembled)
                    print(
                        f"  ┌─ ⟳ REASSEMBLED: {matrices_str}  │  "
                        f"ω: {self._prev_omega:.4f} → {omega:.4f} rad/s  │  "
                        f"Δω = {abs(omega - self._prev_omega):.4f} rad/s",
                        flush=True,
                    )
                    print(
                        "  └─ K_eff rebuilt and solver operators updated",
                        flush=True,
                    )

        return K_eff, K_G_red, K_SP_red

    def _read_and_reduce_forces(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Read forces from preCICE and compute global reductions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, int]
            (data_global, raw_force, raw_max_nodal, n_nodes) where
            data_global has shape (n_nodes, dim), raw_force is the summed
            force vector, raw_max_nodal the max nodal magnitude, and
            n_nodes the number of interface nodes.
        """
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
        """Transform gravity forces from global (inertial) to rotating frame.

        Applies F_local = R^T(θ) · F_global to each nodal gravity vector.
        Since gravity is constant in the inertial frame but the blade rotates,
        the gravity direction in the co-rotating frame changes with θ. This
        produces the cyclic gravitational loading (1P frequency) experienced
        by each blade as it sweeps through 360°.

        Parameters
        ----------
        F_gravity_global : np.ndarray
            Gravity forces in inertial frame, shape (n_nodes, 3).
        theta : float
            Current rotation angle [rad].
        shape : Tuple[int, int]
            Output array shape, typically (n_nodes, 3).

        Returns
        -------
        np.ndarray
            Gravity forces in the co-rotating frame, shape ``shape``.
        """
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
        """Compute rotor torque about the rotation center.

        Evaluates the total moment from distributed nodal forces at the
        DEFORMED configuration:

            τ = Σᵢ (X₀,ᵢ + uᵢ − center) × Fᵢ

        The torque vector is projected onto the rotation axis to obtain the
        scalar driving/resisting torque:

            τ_power = τ · n̂

        This method is called TWICE per step with different force sets:
         1. With EXTERNAL forces only (aero + gravity) → driving torque for
            the ω dynamics equation I·α = τ_driving − τ_gen.
         2. With ALL forces (aero + inertial + gravity) → total torque for
            performance reporting and CSV logging.

        Parameters
        ----------
        interface_coords : np.ndarray, shape (n_nodes, 3)
            Undeformed reference coordinates X₀ of interface nodes.
        interface_disps : np.ndarray, shape (n_nodes, 3)
            Elastic displacement u at interface nodes.
        data_combined : np.ndarray, shape (n_nodes, 3)
            Nodal force vectors to compute torque from.

        Returns
        -------
        Tuple[np.ndarray, float]
            (torque_vector [3], torque_power_scalar) where torque_power is
            the projection onto the rotation axis [N·m].
        """
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
        """Solve K_eff · u_new = F_eff for a single time step / sub-iteration.

        Assembles the full force vector by scattering the combined interface
        forces (aerodynamic + inertial + gravity) into the global DOF vector,
        reduces to free DOFs, computes F_eff via Newmark history terms, and
        solves the linear system using the pre-factorized K_eff.

        Parameters
        ----------
        u : PETSc.Vec
            Current displacement (reduced).
        v : PETSc.Vec
            Current velocity (reduced).
        a : PETSc.Vec
            Current acceleration (reduced).
        data_combined : np.ndarray
            Combined interface forces (aero + inertial + gravity), flat array
            aligned with ``interface_dofs``.
        interface_dofs : np.ndarray
            DOF indices for interface nodes.
        K_eff : PETSc.Mat
            Effective stiffness matrix (pre-factorized).
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        C_red : PETSc.Mat, optional
            Reduced damping matrix.
        bc_manager : BoundaryConditionManager
            For vector reduction.
        coeffs : NewmarkCoefficients
            Precomputed integration coefficients.

        Returns
        -------
        Tuple[PETSc.Vec, int, int]
            (u_new, ksp_iterations, ksp_converged_reason).
        """
        # Assemble force vector
        F_new = self.F.duplicate()
        self.F.copy(F_new)
        F_new.setValues(interface_dofs, data_combined)
        F_new_red = bc_manager.reduce_vector(F_new)

        # Compute effective force
        F_eff = self._compute_effective_force(F_new_red, u, v, a, M_red, C_red, coeffs)

        # Solve
        _logger.debug("Solving linear system...")
        u_new, ksp_its, ksp_reason = self._solve_linear_system(K_eff, F_eff)

        # Cleanup
        F_new.destroy()
        F_new_red.destroy()
        F_eff.destroy()

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
        """Compute velocity estimate for the next preCICE sub-iteration.

        Uses the current displacement solution u_new to estimate v_new via
        Newmark relations. This velocity guess is used to evaluate the
        explicit Coriolis force F_cor = -2·m·(ω × v) in the next sub-iteration,
        improving convergence of the implicit FSI loop.

        Parameters
        ----------
        u : PETSc.Vec
            Displacement at current time step (reduced).
        u_new : PETSc.Vec
            Displacement solution from latest solve (reduced).
        v : PETSc.Vec
            Velocity at current time step (reduced).
        a : PETSc.Vec
            Acceleration at current time step (reduced).
        beta : float
            Newmark β parameter.
        gamma : float
            Newmark γ parameter.
        """
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

        Parameters
        ----------
        u_new : PETSc.Vec
            Displacement solution (reduced DOFs).
        interface_dofs : np.ndarray
            DOF indices for interface nodes.
        theta_target : float
            Target rotation angle [rad] for R(θ) transformation.
        bc_manager : BoundaryConditionManager
            For expanding reduced solution to full DOF vector.
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

            print("  ┌─ DISPLACEMENT (Sending to preCICE)", flush=True)
            print(f"  │  Elastic Disp (Local) Max Norm:   {u_local_norm_max:.4e} m", flush=True)
            print(
                f"  │  Output Disp ({transform_str}) Max Norm: {u_output_norm_max:.4e} m",
                flush=True,
            )
            print(
                "  │  (Should be small elastic deforms, NOT rigid rotation magnitude)", flush=True
            )
            print("  └" + "─" * 40, flush=True)

        u_full_output[interface_dofs.flatten()] = u_interface_output.flatten()
        self.precice_participant.write_data(u_full_output)
        u_full_new.destroy()

    def _write_angular_velocity_to_precice(self, omega: float) -> None:
        """Publish angular velocity to the fluid solver via preCICE.

        Writes ω [rad/s] as a scalar to the GlobalSolidMesh — a dedicated
        single-vertex mesh registered at the rotation center. The OpenFOAM
        adapter reads this value to set the angular velocity of its dynamic
        mesh (solidBodyMotionFunction), ensuring the CFD mesh rotates at
        the same ω computed by the structural solver.

        The value is written:
         - Before every preCICE advance() (including sub-iterations), so the
           fluid participant always sees a valid ω.
         - After a converged time window, with ω for the NEXT window start,
           so that the fluid solver can begin the next window with the
           correct mesh rotation velocity.

        For dynamic omega (ComputedOmega), this closes the feedback loop:
           τ_aero → α = (τ - τ_gen)/I → ω^{n+1} → preCICE → OpenFOAM mesh.

        Parameters
        ----------
        omega : float
            Angular velocity to publish [rad/s].
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
