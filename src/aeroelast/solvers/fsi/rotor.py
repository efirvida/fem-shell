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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from petsc4py import PETSc

from aeroelast.core.bc import BoundaryConditionManager
from aeroelast.core.mesh import MeshModel
from aeroelast.elements import ElementFamily
from aeroelast.postprocess.stress_recovery import StressRecovery, StressType

from .corotational import (
    ComputedOmega,
    ConstantOmega,
    CoordinateTransforms,
    InertialForcesCalculator,
    RampedComputedOmega,
    RampedOmega,
)
from .linear_dynamic import LinearDynamicFSISolver


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


# Logging
_logger = logging.getLogger(__name__)


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
        self._omega_mesh_name: str = rotor_cfg.get("omega_mesh_name", "GlobalSolidMesh")
        self._omega_write_data_name: str = rotor_cfg.get("omega_write_data", "AngularVelocity")

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

    def _init_solver_config(self) -> None:
        """Initialize solver configuration parameters."""
        super()._init_solver_config()

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

    def _init_state_tracking(self) -> None:
        """Initialize state tracking variables."""
        # Current rotation angle (updated during time stepping)
        self._theta = 0.0

        # State tracking for dynamic updates

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
        matrices, bc_manager, K_G = self._assemble_system_matrices(omega_initial)

        # Auto-compute inertia if requested
        if getattr(self, "_auto_inertia", False):
            import _aeroelast  # noqa: PLC0415

            diag_vec = self.M.getDiagonal()
            mass_diag = diag_vec.getArray(readonly=True).copy()
            diag_vec.destroy()
            node_coords = np.array([n.coords for n in self.domain.nodes], dtype=np.float64).ravel()
            estimated_inertia = _aeroelast.compute_estimated_inertia(
                node_coords,
                mass_diag,
                self.domain.dofs_per_node,
                list(self._coord_transforms.center),
                list(self._coord_transforms.axis),
            )
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

        # Phase 2: Interface extraction
        coupling_boundaries = self.model_properties["solver"]["coupling_boundaries"]
        mesh = self.domain.mesh
        node_sets = [mesh.node_sets[name] for name in coupling_boundaries]
        nodes = {node.id: node.coords for _set in node_sets for node in _set.nodes.values()}
        sorted_node_ids = sorted(nodes.keys())
        self._interface_node_ids = np.array(sorted_node_ids, dtype=np.int64)
        _iface_coords = np.array([nodes[nid] for nid in sorted_node_ids])
        if self.domain.spatial_dim == 2 and _iface_coords.shape[1] > 2:
            _iface_coords = _iface_coords[:, :2]
        self._interface_coords = _iface_coords
        raw_dofs = np.array([self.domain._node_dofs_map[nid] for nid in sorted_node_ids])
        if raw_dofs.ndim == 2 and raw_dofs.shape[1] > 3:
            self._interface_dofs = raw_dofs[:, :3].astype(int)
        else:
            self._interface_dofs = raw_dofs.astype(int)

        if self._rotor_radius is None:
            import _aeroelast  # noqa: PLC0415

            self._rotor_radius = _aeroelast.compute_rotor_radius(
                self._interface_coords.ravel().astype(np.float64),
                list(self._coord_transforms.center),
                list(self._coord_transforms.axis),
            )
            self._print_info(f"Auto-detected rotor radius: {self._rotor_radius:.4f} m")

        return self._solve_via_rust(
            bc_manager=bc_manager,
            interface_coords_flat=self._interface_coords.ravel().astype(np.float64),
            interface_dofs_global_flat=self._interface_dofs.ravel().astype(np.uint64),
            K_G=K_G,
        )

    # =========================================================================
    # Solve Sub-Methods
    # =========================================================================

    def _assemble_system_matrices(
        self, omega_initial: float
    ) -> Tuple[
        Tuple[PETSc.Mat, PETSc.Mat],
        BoundaryConditionManager,
        Optional[PETSc.Mat],
    ]:
        """Assemble stiffness, mass, and geometric stiffness matrices.

        Parameters
        ----------
        omega_initial : float
        Initial angular velocity [rad/s], used to compute the centrifugal
        prestress for K_G.

        Returns
        -------
        Tuple[Tuple[PETSc.Mat, PETSc.Mat], BoundaryConditionManager, Optional[PETSc.Mat]]
        ((K, M), bc_manager, K_G) where K_G may be None.
        """
        import _aeroelast  # noqa: PLC0415

        self._print_phase(1, 6, "Assembling stiffness matrix...")
        self.K = self.domain.assemble_stiffness_matrix()

        self._print_phase(2, 6, "Assembling mass matrix (lumped in Rust)...")
        self.M = self.domain.assemble_mass_matrix_lumped()

        # Geometric stiffness
        K_G = None
        if self._include_geometric_stiffness:
            self._print_phase(3, 6, "Assembling geometric stiffness (centrifugal)...")
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
            self._print_phase(3, 6, "Geometric stiffness: skipped")

        # Force vector and boundary conditions
        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        self._print_phase(4, 6, "Applying boundary conditions...")
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)

        if self._is_primary_rank():
            print(
                f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, "
                f"Free: {len(bc_manager.free_dofs)} DOFs",
                flush=True,
            )

        # Rayleigh damping (coefficients only — Rust builds C internally)
        if not self._damping_enabled:
            self._print_phase(5, 6, "Rayleigh damping: disabled (enabled=false)")
        elif self._damping_auto:
            self._print_phase(
                5, 6, "Rayleigh damping: auto-computing via Rust SLEPc modal analysis..."
            )
            import numpy as np  # noqa: PLC0415

            cfg = self._damping_cfg
            zeta = float(cfg.get("zeta", 0.02))
            zeta_i = float(cfg["zeta_1"]) if cfg.get("zeta_1") is not None else zeta
            zeta_j = float(cfg["zeta_2"]) if cfg.get("zeta_2") is not None else zeta
            mode_i = int(cfg.get("mode_i", 1))
            mode_j = int(cfg.get("mode_j", 2))
            num_modes = int(cfg.get("num_modes", max(mode_j + 2, 6)))

            k_rows, k_cols, k_vals = self._petsc_to_coo(self.K)
            m_rows, m_cols, m_vals = self._petsc_to_coo(self.M)
            free_dofs = bc_manager.free_dofs.astype(np.int32)

            self._eta_k, self._eta_m = _aeroelast.compute_rayleigh_auto(
                k_rows,
                k_cols,
                k_vals,
                m_rows,
                m_cols,
                m_vals,
                free_dofs,
                num_modes,
                mode_i,
                mode_j,
                zeta_i,
                zeta_j,
            )

            if self._eta_k < 0.0 or self._eta_m < 0.0:
                _logger.warning(
                    "Rayleigh auto produced a negative coefficient: η_k=%.3e, η_m=%.3e. "
                    "The damping ratio may be non-monotone. Verify your mode selection.",
                    self._eta_k,
                    self._eta_m,
                )

            self._print_phase(
                5, 6, f"Rayleigh auto: η_k={self._eta_k:.4e} s  η_m={self._eta_m:.4e} 1/s"
            )
        elif self._eta_m != 0.0 or self._eta_k != 0.0:
            self._print_phase(
                5, 6, f"Rayleigh damping: η_m={self._eta_m:.4e}  η_k={self._eta_k:.4e}"
            )
        else:
            self._print_phase(5, 6, "Rayleigh damping: disabled")

        self._print_phase(6, 6, "Matrix assembly complete.")
        return (self.K, self.M), bc_manager, K_G

    # =========================================================================
    # Rust FSI fast-path helpers
    # =========================================================================

    def _map_omega_provider(
        self,
    ) -> tuple:
        """Map the Python OmegaProvider instance to Rust-compatible params.

        Returns
        -------
        tuple
        ``(mode, omega, omega_target, t_ramp, moment_of_inertia, shaft_torque)``
        where optional values are ``None`` when not applicable.
        """
        p = self._omega_provider
        if isinstance(p, ConstantOmega):
            return "constant", float(p._omega), None, None, None, None
        if isinstance(p, RampedOmega):
            return "ramped", 0.0, float(p._target_omega), float(p._ramp_time), None, None
        if isinstance(p, ComputedOmega):
            return (
                "computed",
                float(p._omega),
                None,
                None,
                float(p._I),
                float(p._tau_shaft),
            )
        if isinstance(p, RampedComputedOmega):
            return (
                "ramped_computed",
                0.0,
                float(p._target_omega),
                float(p._ramp_time),
                float(p._I),
                float(p._tau_shaft),
            )
        # Fallback for TableOmega / FunctionOmega: sample current omega at t=0
        omega_val, _ = p.get_omega(0.0)
        return "constant", float(omega_val), None, None, None, None

    def _solve_via_rust(
        self,
        bc_manager,
        interface_coords_flat,
        interface_dofs_global_flat,
        K_G=None,
    ):
        """Run the co-rotational rotor FSI loop via the Rust binding.

        Delegates to ``_aeroelast.run_rotor_fsi_solver``, which handles the
        full Newmark time integration and preCICE coupling in compiled code.

        Parameters
        ----------
        bc_manager : BoundaryConditionManager
        Provides free/fixed DOF partitioning.
        interface_coords_flat : np.ndarray
        Flat (n_iface_nodes * 3,) array of interface node coordinates.
        interface_dofs_global_flat : np.ndarray
        Flat array of global DOF indices for the interface nodes.
        """
        import _aeroelast  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rust_asm = getattr(self.domain, "_rust", None)

        k_rows, k_cols, k_vals = self._petsc_to_coo(self.K)
        m_rows, m_cols, m_vals = self._petsc_to_coo(self.M)

        # Extract K_G COO (full DOF space) so Rust can apply it to the
        # reduced free-DOF system at startup.
        if K_G is not None and self._include_geometric_stiffness:
            kg0_rows, kg0_cols, kg0_vals = self._petsc_to_coo(K_G)
            kg0_rows = kg0_rows.astype(np.int64)
            kg0_cols = kg0_cols.astype(np.int64)
            kg0_vals = kg0_vals.astype(np.float64)
        else:
            kg0_rows = kg0_cols = kg0_vals = None

        free_dofs = bc_manager.free_dofs.astype(np.int32)
        self.free_dofs = bc_manager.free_dofs
        n_full_dofs: int = int(self.K.getSize()[0])

        # ── All-node coordinates (flat, co-rotating frame) ─────────────────
        all_node_coords = np.array([n.coords for n in self.domain.nodes], dtype=np.float64).ravel()

        # ── Per-node scalar lumped mass (first translational DOF per node) ──
        diag_vec = self.M.getDiagonal()
        diag_arr = diag_vec.getArray(readonly=True)
        dofs = self.domain.dofs_per_node
        n_nodes = len(self.domain.nodes)
        all_node_masses = np.array(
            [
                float(diag_arr[i * dofs]) if i * dofs < len(diag_arr) else 0.0
                for i in range(n_nodes)
            ],
            dtype=np.float64,
        )
        diag_vec.destroy()

        # ── OmegaProvider mapping ───────────────────────────────────────────
        omega_mode, omega_val, omega_target, t_ramp, moi, shaft_tau = self._map_omega_provider()

        # ── Checkpoint restore ──────────────────────────────────────────────
        checkpoint_state = self._try_restore_checkpoint()
        u0 = v0 = a0 = None
        t0 = float(self.solver_params.get("start_time", 0.0))
        theta0 = float(getattr(self, "_theta", 0.0))

        if checkpoint_state is not None:
            stored_dofs = len(checkpoint_state.get("u_red", []))
            expected_dofs = int(free_dofs.shape[0])
            if stored_dofs and stored_dofs != expected_dofs:
                if self._is_primary_rank():
                    print(
                        f"  ⚠️ Checkpoint DOF mismatch ({stored_dofs} vs {expected_dofs}). "
                        "Checkpoint ignored.",
                        flush=True,
                    )
                checkpoint_state = None

        if checkpoint_state is not None and "u_red" in checkpoint_state:
            u0 = checkpoint_state["u_red"].astype(np.float64)
            v0 = checkpoint_state["v_red"].astype(np.float64)
            a0 = checkpoint_state["a_red"].astype(np.float64)
            t0 = float(checkpoint_state["t"])
            theta0 = float(checkpoint_state.get("theta", 0.0))
            if self._is_primary_rank():
                print(f"  ✓ Restored from checkpoint at t = {t0:.6f} s", flush=True)

        # ── coupling config ─────────────────────────────────────────────────
        cfg = self._coupling_cfg
        mesh_name = cfg["coupling_mesh"]
        write_data = (
            cfg["write_data"] if isinstance(cfg["write_data"], str) else cfg["write_data"][0]
        )
        read_data = cfg["read_data"] if isinstance(cfg["read_data"], str) else cfg["read_data"][0]

        beta = float(self.solver_params["beta"])
        gamma = float(self.solver_params["gamma"])
        dt_hint = float(self.solver_params.get("dt", 0.01))

        if self._is_primary_rank():
            print(
                f"  → Delegating rotor FSI loop to Rust: "
                f"omega_mode={omega_mode}  ω₀={omega_val:.4f} rad/s  "
                f"η_k={self._eta_k:.3e}  η_m={self._eta_m:.3e}  "
                f"β={beta}  γ={gamma}",
                flush=True,
            )

        # ── Per-step callback ───────────────────────────────────────────────
        n_total: int = n_full_dofs
        _free_dofs_arr: np.ndarray = self.free_dofs
        _fixed_dof_vals: dict = dict(bc_manager.fixed_dofs)

        def _step_cb(
            t,
            time_step,
            dt,
            u_red,
            v_red,
            a_red,
            force_mag,
            forces_iface,
            omega,
            alpha,
            theta,
            rotor_perf_tuple,
        ):
            tau_aero, ct, cp, cq, tsr = rotor_perf_tuple

            u_full = np.zeros(n_total, dtype=np.float64)
            u_full[_free_dofs_arr] = u_red
            for dof, val in _fixed_dof_vals.items():
                u_full[dof] = val

            v_full = np.zeros(n_total, dtype=np.float64)
            v_full[_free_dofs_arr] = v_red

            a_full = np.zeros(n_total, dtype=np.float64)
            a_full[_free_dofs_arr] = a_red

            stress_fields = self._compute_stress_fields(u_full)

            iface_dofs_flat = interface_dofs_global_flat
            force_fields = {}
            if forces_iface is not None and len(forces_iface) > 0:
                f_raw_full = np.zeros(n_total, dtype=np.float64)
                for local_idx in range(min(len(forces_iface), len(iface_dofs_flat))):
                    gdof = int(iface_dofs_flat[local_idx])
                    if gdof < n_total:
                        f_raw_full[gdof] += forces_iface[local_idx]
                force_fields["F_AERO_RAW"] = f_raw_full
                force_fields["F_AERO"] = f_raw_full
                force_fields["F_TOTAL"] = f_raw_full.copy()
            force_fields.update(stress_fields)
            force_fields["OMEGA"] = omega
            force_fields["ALPHA"] = alpha
            force_fields["THETA"] = theta
            force_fields["TAU_AERO"] = tau_aero
            force_fields["CT"] = ct
            force_fields["CP"] = cp
            force_fields["CQ"] = cq
            force_fields["TSR"] = tsr

            self._log_structural_report(
                t=t,
                time_step=time_step,
                u_full=u_full,
                v_full=v_full,
                a_full=a_full,
                stress_fields=force_fields,
                applied_force_mag=force_mag,
            )
            self._log_probe_data(
                t=t,
                time_step=time_step,
                u_full=u_full,
                v_full=v_full,
                stress_fields=force_fields,
            )
            self._handle_checkpoint(
                t=t,
                time_step=time_step,
                dt=dt,
                u_red=u_red,
                v_red=v_red,
                a_red=a_red,
                u_full=u_full,
                v_full=v_full,
                a_full=a_full,
                extra_fields=force_fields,
                theta=theta,
                omega=omega,
            )

        # ── omega preCICE coupling params (optional) ────────────────────────
        if self._send_omega_to_precice:
            _omega_mesh = self._omega_mesh_name
            _omega_data = self._omega_write_data_name
            _omega_coord = list(self._coord_transforms.center)
            _logger.info(
                "[RotorFSI] omega→preCICE ENABLED: mesh=%s  data=%s  vertex=%s",
                _omega_mesh,
                _omega_data,
                _omega_coord,
            )
        else:
            _omega_mesh = None
            _omega_data = None
            _omega_coord = None
            _logger.info("[RotorFSI] omega→preCICE DISABLED (send_omega_to_precice=False)")

        u_final_red, v_final_red, a_final_red, times = _aeroelast.run_rotor_fsi_solver(
            rust_asm,
            n_full_dofs,
            self._kg_update_interval,
            list(self._coord_transforms.axis),
            list(self._coord_transforms.center),
            all_node_coords,
            all_node_masses,
            omega_mode,
            omega_val,
            omega_target,
            t_ramp,
            moi,
            shaft_tau,
            list(self._gravity),
            self._include_centrifugal,
            self._include_coriolis,
            self._include_euler,
            self._include_geometric_stiffness,
            self._include_spin_softening,
            float(self.solver_params.get("ksp_omega_threshold", 1e-4)),
            self.domain.dofs_per_node,
            self._fluid_density,
            self._flow_velocity,
            float(self._rotor_radius),
            k_rows,
            k_cols,
            k_vals,
            m_rows,
            m_cols,
            m_vals,
            free_dofs,
            self._eta_k,
            self._eta_m,
            beta,
            gamma,
            dt_hint,
            interface_coords_flat,
            interface_dofs_global_flat,
            self.domain.spatial_dim,
            cfg["participant"],
            cfg["config_file"],
            mesh_name,
            write_data,
            read_data,
            float(self._force_ramp_time),
            getattr(self, "_force_max_magnitude", None),
            # omega preCICE coupling (optional — None disables it)
            _omega_mesh,
            _omega_data,
            _omega_coord,
            u0,
            v0,
            a0,
            t0,
            theta0,
            kg0_rows,
            kg0_cols,
            kg0_vals,
            step_callback=_step_cb,
        )

        n_steps = len(times)
        if n_steps > 0 and self._is_primary_rank():
            print(
                f"  ✓ Rotor FSI loop complete: "
                f"{n_steps} converged steps, t_final={times[-1]:.4f} s",
                flush=True,
            )
        elif self._is_primary_rank():
            print("  ⚠️ Rotor FSI loop returned 0 converged steps.", flush=True)

        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        if n_steps > 0:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            u_final_full[_free_dofs_arr] = np.asarray(u_final_red, dtype=np.float64)
            for dof, val in _fixed_dof_vals.items():
                u_final_full[dof] = val

            v_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full[_free_dofs_arr] = np.asarray(v_final_red, dtype=np.float64)

            a_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full[_free_dofs_arr] = np.asarray(a_final_red, dtype=np.float64)
        else:
            u_final_full = np.zeros(n_total, dtype=np.float64)
            v_final_full = np.zeros(n_total, dtype=np.float64)
            a_final_full = np.zeros(n_total, dtype=np.float64)

        self.u = u_final_full
        self.v = v_final_full
        self.a = a_final_full
        return self.u, self.v, self.a

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
        has_shell = self.domain.element_family == ElementFamily.SHELL
        has_solid = self.domain.element_family == ElementFamily.SOLID

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
