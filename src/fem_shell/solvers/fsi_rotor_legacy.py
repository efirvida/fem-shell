"""
FSI Rotor Solver for Wind Turbine Applications.
================================================

This module implements a Fluid-Structure Interaction (FSI) solver for wind turbine
rotors, coupling structural dynamics with aerodynamic loads via the preCICE library.

Theoretical Background
----------------------

**Governing Equations**

The rotor dynamics are governed by two coupled systems:

1. **Structural Dynamics** (Newmark-β integration):

   M·ü + C·u̇ + K·u = F_aero

   where:
   - M: Mass matrix [kg]
   - C: Damping matrix [N·s/m]
   - K: Stiffness matrix [N/m]
   - u: Displacement vector [m]
   - F_aero: Aerodynamic forces from CFD [N]

2. **Rotational Dynamics** (1-DOF rigid body rotation):

   I·α = τ_aero - τ_load - c·ω

   where:
   - I: Total rotor moment of inertia [kg·m²]
   - α: Angular acceleration [rad/s²]
   - τ_aero: Aerodynamic torque around rotation axis [N·m]
   - τ_load: Resistive load torque (generator, gearbox) [N·m]
   - c: Rotational damping coefficient [N·m·s/rad]
   - ω: Angular velocity [rad/s]

**Torque Calculation**

Aerodynamic torque is computed from nodal forces:

   τ = Σᵢ (rᵢ × Fᵢ) · n̂_axis

where:
- rᵢ: Position vector from rotor center to node i [m]
- Fᵢ: Force vector at node i [N]
- n̂_axis: Unit vector along rotation axis

**Load Torque Models**

The resistive load torque τ_load models the generator/gearbox reaction:

- **none**: τ_load = 0 (free rotation)
- **constant**: τ_load = τ₀ (constant braking torque)
- **linear**: τ_load = k·ω (viscous-like, simple generator model)
- **quadratic**: τ_load = k·ω² (optimal TSR tracking, region 2 control)
- **rated_power**: τ_load = P_rated/ω (constant power extraction, region 3)

Wind Turbine Operating Regions:
- Region 1: Below cut-in, τ_load ≈ 0
- Region 2: Optimal TSR tracking, τ_load ∝ ω² (quadratic mode)
- Region 3: Rated power, τ_load = P_rated/ω (rated_power mode)

**Moment of Inertia Calculation**

Three modes for specifying rotor inertia:

1. **"total"**: Direct input of total rotor inertia I_total
2. **"hub_plus_blades"**: I_total = I_hub + I_blades
   - I_hub: User-specified hub inertia
   - I_blades: Computed from mesh using lumped mass matrix
3. **"fraction"**: I_total = I_blades × (1 + f_hub)
   - f_hub: Hub contribution as fraction of blade inertia

Blade inertia is computed as:
   I_blades = Σᵢ mᵢ · r_⊥ᵢ²

where mᵢ is the lumped mass at node i and r_⊥ᵢ is its perpendicular
distance from the rotation axis.

**Aerodynamic Coefficients**

Standard wind turbine performance coefficients:

- Tip Speed Ratio (TSR): λ = ω·R / V∞
- Power Coefficient: Cp = P / (½·ρ·A·V∞³)
- Torque Coefficient: Cq = τ / (½·ρ·A·R·V∞²)
- Thrust Coefficient: Ct = T / (½·ρ·A·V∞²)

where:
- R: Rotor radius [m]
- V∞: Freestream wind speed [m/s]
- ρ: Air density [kg/m³]
- A: Rotor swept area = π·R² [m²]
- P: Mechanical power = τ·ω [W]
- T: Thrust force (axial component) [N]

**Inertial Forces (Centrifugal and Gravitational)**

Since the structural mesh does not physically rotate, inertial forces must
be applied as equivalent nodal loads. Two effects are modeled:

1. **Centrifugal Force** (body force due to rotation):

   F_c,i = mᵢ · ω² · r_⊥,i · ê_r

   where:
   - mᵢ: Lumped mass at node i [kg]
   - ω: Angular velocity [rad/s]
   - r_⊥,i: Perpendicular distance from rotation axis [m]
   - ê_r: Unit vector pointing radially outward from axis

   This force acts radially outward and creates tensile stresses along
   the blade span (centrifugal stiffening effect).

2. **Gravitational Force** (weight in rotating frame):

   F_g,i = mᵢ · R_axis(θ) · g

   where:
   - g: Gravity vector in global frame [m/s²]
   - R_axis(θ): Rotation matrix around rotor axis by angle θ
   - θ: Accumulated rotation angle [rad]

   Since the mesh is fixed but represents a rotating blade, we rotate
   the gravity vector inversely to simulate the blade's position in its
   rotation cycle. This creates cyclic bending loads (edgewise fatigue).

   For axis 'x' with θ rotation:
       g_rotated = Rx(-θ) · g

   The negative angle accounts for the "virtual" rotation of the blade.

**Time Integration**

Angular state is integrated using explicit Euler with trapezoidal
position update for improved accuracy:

   α(t+dt) = (τ_aero - τ_load - c·ω(t)) / I
   ω(t+dt) = ω(t) + α·dt
   θ(t+dt) = θ(t) + ½·(ω(t) + ω(t+dt))·dt

**FSI Coupling**

The solver uses implicit coupling with preCICE:
1. Read forces from CFD
2. Compute structural response (multiple sub-iterations)
3. Update angular state
4. Write displacements to CFD
5. Repeat until convergence

Checkpoint/restore mechanism ensures convergence in implicit schemes:
- Checkpoint saves: CheckpointState(u, v, a, t, ω, θ, α)
- Restore resets all state variables on non-convergence

Maintenance Notes
-----------------

**Code Organization**

Key methods (in call order during simulation):
1. solve() - Main FSI loop
2. _prepare_solver() - Assemble matrices, compute inertia
3. _compute_aerodynamic_torque() - Sum moments from nodal forces
4. _update_angular_state() - Integrate rotational dynamics
6. _compute_aerodynamic_coefficients() - Calculate Cp, Cq, Ct, TSR
7. _write_rotor_log() - Log converged timestep data

**State Variables**

Angular state (saved/restored in checkpoint):
- _omega: Current angular velocity [rad/s]
- _theta: Accumulated rotation angle [rad]
- _alpha: Angular acceleration [rad/s²]

Computed once at startup:
- _I_rotor: Total moment of inertia [kg·m²]
- _rotor_radius: Max perpendicular distance from axis [m]
- _rotor_area: Swept area = π·R² [m²]

**Adding New Load Torque Models**

To add a new load torque model:
1. Add option to load_torque_mode validation in __init__
2. Add case in GeneratorController.compute_torque() method
3. Update docstrings and log header in _init_rotor_log()
4. Update console output in solve() if needed

**Testing Considerations**

- Use omega_mode="constant" to validate against fixed-RPM CFD simulations
- Compare "dynamic" vs "constant" to quantify FSI effects on rotor speed
- Verify energy balance: P_aero = P_load + P_damping + d(½Iω²)/dt

Author: fem-shell contributors
License: MIT
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
from petsc4py import PETSc

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.solvers.fsi import Adapter
from fem_shell.solvers.linear import LinearDynamicSolver

# =============================================================================
# Configuration Classes (Phase 4: Structural Refactoring)
# =============================================================================


class LoadTorqueMode(Enum):
    """Available load torque modes for generator control."""

    NONE = "none"
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    RATED_POWER = "rated_power"


class OmegaMode(Enum):
    """Angular velocity control modes."""

    DYNAMIC = "dynamic"
    CONSTANT = "constant"


class InertiaMode(Enum):
    """Moment of inertia specification modes."""

    TOTAL = "total"
    HUB_PLUS_BLADES = "hub_plus_blades"
    FRACTION = "fraction"


@dataclass
class RotorConfig:
    """
    Configuration dataclass for rotor dynamics parameters.

    This class validates all rotor parameters at initialization time,
    providing early error detection and clear error messages.

    Parameters
    ----------
    rotor_center : List[float]
        Center of rotation in global coordinates [x, y, z] [m].
    rotor_axis : str
        Rotation axis: 'x', 'y', or 'z'.
    omega_mode : str
        Angular velocity mode: 'dynamic' or 'constant'.
    initial_omega : float
        Initial angular velocity [rad/s].
    rotational_damping : float
        Mechanical damping coefficient [N·m·s/rad].
    inertia_mode : str
        Inertia specification mode.
    rotor_inertia : Optional[float]
        Total rotor inertia [kg·m²] (for inertia_mode='total').
    hub_inertia : float
        Hub inertia [kg·m²].
    hub_fraction : float
        Hub inertia as fraction of blade inertia.
    load_torque_mode : str
        Generator load torque mode.
    load_torque_value : float
        Constant torque value [N·m].
    load_torque_coeff : float
        Torque coefficient for linear/quadratic modes.
    rated_power : float
        Rated electrical power [W].
    omega_min_rated_power : float
        Minimum omega for rated_power mode regularization [rad/s].
    enable_centrifugal : bool
        Enable centrifugal forces.
    enable_gravity : bool
        Enable gravitational forces in rotating frame.
    enable_coriolis : bool
        Enable Coriolis forces (velocity-dependent).
    enable_euler : bool
        Enable Euler forces (angular acceleration-dependent).
    gravity_vector : List[float]
        Gravity acceleration vector [m/s²].
    air_density : float
        Air density [kg/m³].
    wind_speed : Optional[float]
        Freestream wind speed [m/s].

    Raises
    ------
    ValueError
        If any parameter has an invalid value.
    """

    # Geometry
    rotor_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotor_axis: str = "x"

    # Angular velocity
    omega_mode: str = "dynamic"
    initial_omega: float = 0.0
    rotational_damping: float = 0.0

    # Inertia
    inertia_mode: str = "total"
    rotor_inertia: Optional[float] = None
    hub_inertia: float = 0.0
    hub_fraction: float = 0.1

    # Load torque
    load_torque_mode: str = "none"
    load_torque_value: float = 0.0
    load_torque_coeff: float = 0.0
    rated_power: float = 0.0
    omega_min_rated_power: float = 0.1  # Configurable minimum omega for rated_power
    torque_max_rated_power: float = 1e6  # Maximum torque limit for rated_power mode [N·m]

    # Inertial forces
    enable_centrifugal: bool = False
    enable_gravity: bool = False
    enable_coriolis: bool = False  # Phase 3: Coriolis forces
    enable_euler: bool = False  # Euler forces (angular acceleration-dependent)
    coriolis_use_rotating_frame: bool = True  # Transform velocities to rotating frame
    gravity_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    # Aerodynamic
    air_density: float = 1.225
    wind_speed: Optional[float] = None

    # Logging
    rotor_log_file: Optional[str] = None
    rotor_log_separator: str = ","

    # Solver tolerances and parameters
    solver_rtol_large: float = 1e-5  # Relative tolerance for large problems (GAMG)
    solver_atol_large: float = 1e-8  # Absolute tolerance for large problems
    solver_rtol_small: float = 1e-8  # Relative tolerance for small problems (ILU)
    solver_atol_small: float = 1e-12  # Absolute tolerance for small problems
    solver_rtol_mass: float = 1e-12  # Mass matrix solver relative tolerance
    solver_shift_amount: float = 1e-10  # Positive definite shift amount
    solver_max_it_large: int = 1000  # Max iterations for large problems
    solver_max_it_small: int = 500  # Max iterations for small problems
    solver_dofs_threshold: int = 10000  # DOFs threshold for GAMG vs ILU

    # Dynamic stiffness update (geometric stiffening from centrifugal prestress)
    enable_dynamic_stiffness: bool = False  # Enable K_eff recalculation during solve
    stiffness_update_interval: int = 1  # Update K_eff every N time steps (if enabled)
    stiffness_update_mode: str = "interval"  # "interval" or "adaptive"
    omega_change_threshold: float = (
        0.01  # Relative change in omega to trigger update (for adaptive mode)
    )

    # Structural damping (Rayleigh damping: C = alpha*M + beta*K)
    enable_structural_damping: bool = False  # Enable Rayleigh damping in Newmark integration
    rayleigh_alpha: float = 0.0  # Mass-proportional damping coefficient [1/s]
    rayleigh_beta: float = 0.0  # Stiffness-proportional damping coefficient [s]
    # Alternative: specify damping ratios at two frequencies
    use_damping_ratios: bool = False  # If True, compute alpha/beta from damping ratios
    damping_ratio_1: float = 0.01  # Damping ratio at frequency 1 (1% default)
    damping_ratio_2: float = 0.01  # Damping ratio at frequency 2 (1% default)
    damping_freq_1: float = 1.0  # First frequency [Hz] for damping ratio specification
    damping_freq_2: float = 10.0  # Second frequency [Hz] for damping ratio specification

    def __post_init__(self):
        """Validate all configuration parameters."""
        self._validate()

    def _validate(self):
        """Perform comprehensive validation of all parameters."""
        logger = logging.getLogger(__name__)

        # Validate rotor_axis
        valid_axes = ("x", "y", "z")
        if self.rotor_axis.lower() not in valid_axes:
            raise ValueError(
                f"Invalid rotor_axis: '{self.rotor_axis}'. Must be one of {valid_axes}."
            )
        self.rotor_axis = self.rotor_axis.lower()

        # Validate omega_mode
        try:
            OmegaMode(self.omega_mode.lower())
            self.omega_mode = self.omega_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in OmegaMode]
            raise ValueError(
                f"Invalid omega_mode: '{self.omega_mode}'. Must be one of {valid_modes}."
            )

        # Validate inertia_mode
        try:
            InertiaMode(self.inertia_mode.lower())
            self.inertia_mode = self.inertia_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in InertiaMode]
            raise ValueError(
                f"Invalid inertia_mode: '{self.inertia_mode}'. Must be one of {valid_modes}."
            )

        # Validate load_torque_mode
        try:
            LoadTorqueMode(self.load_torque_mode.lower())
            self.load_torque_mode = self.load_torque_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in LoadTorqueMode]
            raise ValueError(
                f"Invalid load_torque_mode: '{self.load_torque_mode}'. Must be one of {valid_modes}."
            )

        # Validate numerical parameters
        if self.initial_omega < 0:
            raise ValueError(f"initial_omega must be non-negative, got {self.initial_omega}")

        if self.rotational_damping < 0:
            raise ValueError(
                f"rotational_damping must be non-negative, got {self.rotational_damping}"
            )

        if self.inertia_mode == "total":
            if self.rotor_inertia is None:
                raise ValueError("rotor_inertia must be specified when inertia_mode='total'")
            if self.rotor_inertia <= 0:
                raise ValueError(f"rotor_inertia must be positive, got {self.rotor_inertia}")

        if self.hub_inertia < 0:
            raise ValueError(f"hub_inertia must be non-negative, got {self.hub_inertia}")

        if self.hub_fraction < 0:
            raise ValueError(f"hub_fraction must be non-negative, got {self.hub_fraction}")

        if self.load_torque_mode == "constant" and self.load_torque_value < 0:
            raise ValueError(
                f"load_torque_value must be non-negative, got {self.load_torque_value}"
            )

        if self.load_torque_mode in ("linear", "quadratic") and self.load_torque_coeff < 0:
            raise ValueError(
                f"load_torque_coeff must be non-negative, got {self.load_torque_coeff}"
            )

        if self.load_torque_mode == "rated_power":
            if self.rated_power <= 0:
                raise ValueError(
                    f"rated_power must be positive when mode='rated_power', got {self.rated_power}"
                )
            if self.omega_min_rated_power <= 0:
                raise ValueError(
                    f"omega_min_rated_power must be positive, got {self.omega_min_rated_power}"
                )

        if self.torque_max_rated_power <= 0:
            raise ValueError(
                f"torque_max_rated_power must be positive, got {self.torque_max_rated_power}"
            )
            raise ValueError(f"air_density must be positive, got {self.air_density}")

        # Validate solver tolerances
        if self.solver_rtol_large <= 0:
            raise ValueError(f"solver_rtol_large must be positive, got {self.solver_rtol_large}")
        if self.solver_atol_large <= 0:
            raise ValueError(f"solver_atol_large must be positive, got {self.solver_atol_large}")
        if self.solver_rtol_small <= 0:
            raise ValueError(f"solver_rtol_small must be positive, got {self.solver_rtol_small}")
        if self.solver_atol_small <= 0:
            raise ValueError(f"solver_atol_small must be positive, got {self.solver_atol_small}")
        if self.solver_rtol_mass <= 0:
            raise ValueError(f"solver_rtol_mass must be positive, got {self.solver_rtol_mass}")
        if self.solver_shift_amount <= 0:
            raise ValueError(
                f"solver_shift_amount must be positive, got {self.solver_shift_amount}"
            )
        if self.solver_max_it_large <= 0:
            raise ValueError(
                f"solver_max_it_large must be positive, got {self.solver_max_it_large}"
            )
        if self.solver_max_it_small <= 0:
            raise ValueError(
                f"solver_max_it_small must be positive, got {self.solver_max_it_small}"
            )
        if self.solver_dofs_threshold <= 0:
            raise ValueError(
                f"solver_dofs_threshold must be positive, got {self.solver_dofs_threshold}"
            )

        if self.wind_speed is not None and self.wind_speed < 0:
            raise ValueError(f"wind_speed must be non-negative, got {self.wind_speed}")

        # Validate stiffness update mode
        valid_stiffness_modes = ("interval", "adaptive")
        if self.stiffness_update_mode.lower() not in valid_stiffness_modes:
            raise ValueError(
                f"Invalid stiffness_update_mode: '{self.stiffness_update_mode}'. "
                f"Must be one of {valid_stiffness_modes}."
            )
        self.stiffness_update_mode = self.stiffness_update_mode.lower()

        if self.omega_change_threshold <= 0 or self.omega_change_threshold > 1:
            raise ValueError(
                f"omega_change_threshold must be in (0, 1], got {self.omega_change_threshold}"
            )

        # Validate Rayleigh damping parameters
        if self.enable_structural_damping:
            if self.use_damping_ratios:
                if self.damping_ratio_1 < 0 or self.damping_ratio_1 > 1:
                    raise ValueError(
                        f"damping_ratio_1 must be in [0, 1], got {self.damping_ratio_1}"
                    )
                if self.damping_ratio_2 < 0 or self.damping_ratio_2 > 1:
                    raise ValueError(
                        f"damping_ratio_2 must be in [0, 1], got {self.damping_ratio_2}"
                    )
                if self.damping_freq_1 <= 0:
                    raise ValueError(f"damping_freq_1 must be positive, got {self.damping_freq_1}")
                if self.damping_freq_2 <= 0:
                    raise ValueError(f"damping_freq_2 must be positive, got {self.damping_freq_2}")
                if abs(self.damping_freq_1 - self.damping_freq_2) < 1e-6:
                    raise ValueError("damping_freq_1 and damping_freq_2 must be different")
            else:
                if self.rayleigh_alpha < 0:
                    raise ValueError(
                        f"rayleigh_alpha must be non-negative, got {self.rayleigh_alpha}"
                    )
                if self.rayleigh_beta < 0:
                    raise ValueError(
                        f"rayleigh_beta must be non-negative, got {self.rayleigh_beta}"
                    )

        # Validate vector dimensions
        if len(self.rotor_center) != 3:
            raise ValueError(f"rotor_center must have 3 components, got {len(self.rotor_center)}")

        if len(self.gravity_vector) != 3:
            raise ValueError(
                f"gravity_vector must have 3 components, got {len(self.gravity_vector)}"
            )

        # Warnings for physics configurations
        if self.enable_centrifugal:
            if self.enable_dynamic_stiffness:
                logging.getLogger(__name__).info(
                    "Centrifugal forces and geometric stiffening enabled. "
                    "Natural frequencies will increase with rotor speed (Campbell diagram effect)."
                )
            else:
                logging.getLogger(__name__).warning(
                    "Centrifugal forces enabled but geometric stiffening disabled. "
                    "Natural frequencies will NOT increase with rotor speed. "
                    "Set enable_dynamic_stiffness=True for complete physics."
                )

        if self.enable_coriolis:
            logging.getLogger(__name__).info(
                "Coriolis forces enabled. These velocity-dependent forces will be computed "
                "at each timestep based on nodal velocities."
            )

        if self.enable_euler:
            logging.getLogger(__name__).info(
                "Euler forces enabled. These angular acceleration-dependent forces will be "
                "computed at each timestep when omega_mode='dynamic'. "
                "F_Euler = -m·(α × r) where α = dω/dt."
            )

    @classmethod
    def from_solver_params(cls, solver_params: Dict) -> "RotorConfig":
        """Create RotorConfig from solver parameters dictionary."""
        return cls(
            rotor_center=solver_params.get("rotor_center", [0.0, 0.0, 0.0]),
            rotor_axis=solver_params.get("rotor_axis", "x"),
            omega_mode=solver_params.get("omega_mode", "dynamic"),
            initial_omega=solver_params.get("initial_omega", 0.0),
            rotational_damping=solver_params.get("rotational_damping", 0.0),
            inertia_mode=solver_params.get("inertia_mode", "total"),
            rotor_inertia=solver_params.get("rotor_inertia", None),
            hub_inertia=solver_params.get("hub_inertia", 0.0),
            hub_fraction=solver_params.get("hub_fraction", 0.1),
            load_torque_mode=solver_params.get("load_torque_mode", "none"),
            load_torque_value=solver_params.get("load_torque_value", 0.0),
            load_torque_coeff=solver_params.get("load_torque_coeff", 0.0),
            rated_power=solver_params.get("rated_power", 0.0),
            omega_min_rated_power=solver_params.get("omega_min_rated_power", 0.1),
            enable_centrifugal=solver_params.get("enable_centrifugal", False),
            enable_gravity=solver_params.get("enable_gravity", False),
            enable_coriolis=solver_params.get("enable_coriolis", False),
            enable_euler=solver_params.get("enable_euler", False),
            coriolis_use_rotating_frame=solver_params.get("coriolis_use_rotating_frame", True),
            gravity_vector=solver_params.get("gravity_vector", [0.0, 0.0, -9.81]),
            air_density=solver_params.get("air_density", 1.225),
            wind_speed=solver_params.get("wind_speed", None),
            rotor_log_file=solver_params.get("rotor_log_file", None),
            rotor_log_separator=solver_params.get("rotor_log_separator", ","),
            enable_dynamic_stiffness=solver_params.get("enable_dynamic_stiffness", False),
            stiffness_update_interval=solver_params.get("stiffness_update_interval", 1),
            stiffness_update_mode=solver_params.get("stiffness_update_mode", "interval"),
            omega_change_threshold=solver_params.get("omega_change_threshold", 0.01),
            enable_structural_damping=solver_params.get("enable_structural_damping", False),
            rayleigh_alpha=solver_params.get("rayleigh_alpha", 0.0),
            rayleigh_beta=solver_params.get("rayleigh_beta", 0.0),
            use_damping_ratios=solver_params.get("use_damping_ratios", False),
            damping_ratio_1=solver_params.get("damping_ratio_1", 0.01),
            damping_ratio_2=solver_params.get("damping_ratio_2", 0.01),
            damping_freq_1=solver_params.get("damping_freq_1", 1.0),
            damping_freq_2=solver_params.get("damping_freq_2", 10.0),
        )


class GeneratorController:
    """
    Generator/gearbox load torque controller.

    This class encapsulates the load torque calculation logic, making it
    easier to extend with new control strategies and test in isolation.

    Parameters
    ----------
    mode : LoadTorqueMode
        The control mode for load torque calculation.
    constant_torque : float
        Constant torque value [N·m] for CONSTANT mode.
    torque_coeff : float
        Torque coefficient for LINEAR and QUADRATIC modes.
    rated_power : float
        Rated electrical power [W] for RATED_POWER mode.
    omega_min : float
        Minimum omega for regularization in RATED_POWER mode [rad/s].
    tau_max : float
        Maximum torque limit for RATED_POWER mode [N·m].

    Examples
    --------
    >>> controller = GeneratorController(
    ...     mode=LoadTorqueMode.QUADRATIC,
    ...     torque_coeff=2.5e4
    ... )
    >>> tau_load = controller.compute_torque(omega=1.2)  # Returns k*omega^2
    """

    def __init__(
        self,
        mode: LoadTorqueMode,
        constant_torque: float = 0.0,
        torque_coeff: float = 0.0,
        rated_power: float = 0.0,
        omega_min: float = 0.1,  # Use class constant FSIRotorSolver._OMEGA_MIN_DEFAULT
        tau_max: float = 1e6,  # Maximum torque limit for rated_power mode [N·m]
    ):
        self.mode = mode
        self.constant_torque = constant_torque
        self.torque_coeff = torque_coeff
        self.rated_power = rated_power
        self.omega_min = omega_min
        self.tau_max = tau_max

    def compute_torque(self, omega: float) -> float:
        """
        Compute load torque based on current angular velocity.

        Parameters
        ----------
        omega : float
            Current angular velocity [rad/s].

        Returns
        -------
        tau_load : float
            Load torque [N·m]. For most modes, returns non-negative magnitude.
            For QUADRATIC mode, can be negative to oppose rotation direction.
        """
        if self.mode == LoadTorqueMode.NONE:
            return 0.0

        elif self.mode == LoadTorqueMode.CONSTANT:
            return self.constant_torque

        elif self.mode == LoadTorqueMode.LINEAR:
            # τ_load = k·|ω|
            return self.torque_coeff * abs(omega)

        elif self.mode == LoadTorqueMode.QUADRATIC:
            # τ_load = k·ω² with proper sign handling (torque opposes motion)
            # For physical realism: sign(ω) * k * ω² = k * ω * |ω|
            return self.torque_coeff * omega * abs(omega)

        elif self.mode == LoadTorqueMode.RATED_POWER:
            # τ_load = P_rated / ω with smooth regularization and maximum limit
            # Using sqrt(ω² + ω_min²) for smooth transition near zero
            omega_regularized = np.sqrt(omega**2 + self.omega_min**2)
            tau_calculated = self.rated_power / omega_regularized
            # Apply maximum torque limit for physical realism
            return min(tau_calculated, self.tau_max)

        return 0.0

    @classmethod
    def from_config(cls, config: RotorConfig) -> "GeneratorController":
        """Create GeneratorController from RotorConfig."""
        return cls(
            mode=LoadTorqueMode(config.load_torque_mode),
            constant_torque=config.load_torque_value,
            torque_coeff=config.load_torque_coeff,
            rated_power=config.rated_power,
            omega_min=config.omega_min_rated_power,
            tau_max=config.torque_max_rated_power,
        )


@dataclass
class CheckpointState:
    """State storage for implicit coupling checkpoints."""

    u: PETSc.Vec
    v: PETSc.Vec
    a: PETSc.Vec
    t: float
    omega: float
    theta: float
    alpha: float

    def to_tuple(self) -> Tuple:
        """Convert state to tuple for storage.

        Note: We pass references. The Adapter's SolverState class handles
        creating deep copies of PETSc vectors for safe storage.
        """
        return (self.u, self.v, self.a, self.t, self.omega, self.theta, self.alpha)

    @classmethod
    def from_tuple(cls, data: Tuple) -> "CheckpointState":
        """Restore state from tuple."""
        u, v, a, t, omega, theta, alpha = data
        return cls(u, v, a, t, omega, theta, alpha)


class RotorLogger:
    """
    Handles logging of rotor dynamics data to a CSV file.

    Encapsulates all file I/O and formatting logic for the rotor log.
    """

    def __init__(
        self,
        config: RotorConfig,
        rotor_radius: float,
        rotor_area: float,
        rotor_inertia: float,
        solver_params: Dict,
    ):
        self.config = config
        self.log_file = config.rotor_log_file
        self.separator = config.rotor_log_separator
        self.handle: Optional[TextIO] = None

        # Store properties for header
        self.rotor_radius = rotor_radius
        self.rotor_area = rotor_area
        self.rotor_inertia = rotor_inertia
        self.solver_params = solver_params

    def initialize(self):
        """Initialize the log file with header information."""
        if self.log_file is None:
            return

        # Ensure directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.handle = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            logging.getLogger(__name__).warning("Could not open rotor log file: %s", e)
            self.handle = None
            return

        self._write_header()

    def _write_header(self):
        """Write the detailed header to the log file."""
        sep = self.separator
        h = self.handle

        h.write("# FSI Rotor Dynamics Log\n")
        h.write(f"# Generated: {datetime.now().isoformat()}\n")
        h.write("#\n")
        h.write("# === ROTOR CONFIGURATION ===\n")
        h.write(f"# Rotor center [m]: {self.config.rotor_center}\n")
        h.write(f"# Rotation axis: {self.config.rotor_axis.upper()}\n")
        h.write(f"# Omega mode: {self.config.omega_mode}\n")
        h.write(f"# Inertia mode: {self.config.inertia_mode}\n")
        h.write(f"# Total inertia [kg·m²]: {self.rotor_inertia:.6e}\n")

        if self.config.inertia_mode == "hub_plus_blades":
            h.write(f"# Hub inertia [kg·m²]: {self.config.hub_inertia:.6e}\n")
        elif self.config.inertia_mode == "fraction":
            h.write(f"# Hub fraction: {self.config.hub_fraction:.4f}\n")

        h.write(f"# Initial omega [rad/s]: {self.config.initial_omega:.6f}\n")
        h.write(f"# Rotational damping [N·m·s/rad]: {self.config.rotational_damping:.6e}\n")
        h.write("#\n")
        h.write("# === LOAD TORQUE (Generator/Gearbox) ===\n")
        h.write(f"# Load torque mode: {self.config.load_torque_mode}\n")
        # ... (Add specific load torque details if needed) ...

        h.write("#\n")
        h.write("# === INERTIAL FORCES ===\n")
        h.write(f"# Centrifugal: {self.config.enable_centrifugal}\n")
        h.write(f"# Gravity: {self.config.enable_gravity}\n")
        h.write(f"# Coriolis: {self.config.enable_coriolis}\n")
        if self.config.enable_gravity:
            h.write(f"# Gravity vector: {self.config.gravity_vector}\n")

        h.write("#\n")
        h.write("# === ROTOR GEOMETRY ===\n")
        h.write(f"# Rotor radius [m]: {self.rotor_radius:.6f}\n")
        h.write(f"# Rotor area [m²]: {self.rotor_area:.6f}\n")
        h.write("#\n")
        h.write("# === AERODYNAMIC PARAMETERS ===\n")
        h.write(f"# Air density [kg/m³]: {self.config.air_density:.6f}\n")
        if self.config.wind_speed is not None:
            h.write(f"# Wind speed [m/s]: {self.config.wind_speed:.6f}\n")
        else:
            h.write("# Wind speed: Not specified\n")

        h.write("#\n")
        h.write("# === SIMULATION PARAMETERS ===\n")
        # h.write(f"# Time step [s]: {self.dt:.6e}\n") # dt not available here easily without passing it
        h.write(f"# Newmark beta: {self.solver_params.get('beta', 0.25):.4f}\n")
        h.write(f"# Newmark gamma: {self.solver_params.get('gamma', 0.5):.4f}\n")
        h.write("#\n")

        # Columns
        columns = [
            "time",
            "torque_aero",
            "torque_inertial",
            "torque_net",
            "thrust",
            "power",
            "omega_rad_s",
            "alpha_rad_s2",
            "theta_rad",
            "omega_RPM",
            "theta_deg",
            "power_kW",
        ]

        if self.config.wind_speed is not None:
            columns.extend(["TSR", "Cp", "Cq", "Ct"])

        h.write(sep.join(columns) + "\n")
        h.flush()

    def log_timestep(
        self,
        t: float,
        torque_aero: float,
        torque_inertial: float,
        torque_net: float,
        thrust: float,
        power: float,
        omega: float,
        alpha: float,
        theta: float,
        tsr: float = 0.0,
        cp: float = 0.0,
        cq: float = 0.0,
        ct: float = 0.0,
    ):
        """Write a single timestep entry."""
        if self.handle is None:
            return

        sep = self.separator
        omega_rpm = omega * 60.0 / (2.0 * np.pi)
        theta_deg = np.degrees(theta)
        power_kW = power / 1000.0

        values = [
            f"{t:.6e}",
            f"{torque_aero:.6e}",
            f"{torque_inertial:.6e}",
            f"{torque_net:.6e}",
            f"{thrust:.6e}",
            f"{power:.6e}",
            f"{omega:.6e}",
            f"{alpha:.6e}",
            f"{theta:.6e}",
            f"{omega_rpm:.6e}",
            f"{theta_deg:.6e}",
            f"{power_kW:.6e}",
        ]

        if self.config.wind_speed is not None:
            values.extend([
                f"{tsr:.6e}",
                f"{cp:.6e}",
                f"{cq:.6e}",
                f"{ct:.6e}",
            ])

        self.handle.write(sep.join(values) + "\n")
        self.handle.flush()

    def log_initial_state(
        self,
        dt: float,
        beta: float,
        gamma: float,
        rotor_center: List[float],
        rotor_axis: str,
        air_density: float,
        wind_speed: Optional[float],
        rotational_damping: float,
    ):
        """Log initial simulation state parameters."""
        if self.handle is None:
            return

        # Write initial state as comments in the log file
        h = self.handle
        h.write("# === SIMULATION INITIAL STATE ===\n")
        h.write(f"# Time step [s]: {dt:.6e}\n")
        h.write(f"# Newmark beta: {beta:.6f}\n")
        h.write(f"# Newmark gamma: {gamma:.6f}\n")
        h.write(f"# Rotor center [m]: {rotor_center}\n")
        h.write(f"# Rotor axis: {rotor_axis.upper()}\n")
        h.write(f"# Air density [kg/m³]: {air_density:.6f}\n")
        if wind_speed is not None:
            h.write(f"# Wind speed [m/s]: {wind_speed:.6f}\n")
        else:
            h.write("# Wind speed: Not specified\n")
        h.write(f"# Rotational damping [N·m·s/rad]: {rotational_damping:.6e}\n")
        h.write("# =====================================\n")
        h.flush()

    def write_to_file(self, data: str):
        """Write arbitrary data to the log file.

        Parameters
        ----------
        data : str
            Data to write to the log file.
        """
        if self.handle is not None:
            self.handle.write(data)
            self.handle.flush()

    def close(self):
        """Close the log file."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class FSIRotorSolver(LinearDynamicSolver):
    """
    FSI solver for wind turbine rotor blades with rotational dynamics.

    This solver couples structural finite element analysis with aerodynamic
    loads from CFD, while simultaneously solving rotor rotational dynamics.

    Physical Model
    --------------
    The solver integrates two coupled dynamic systems:

    1. **Structural Response**: Shell/beam deformation under aerodynamic loads
       using Newmark-β time integration.

    2. **Rotational Dynamics**: Single-DOF rotation governed by:

       I·α = τ_aero - τ_load - c·ω

       This equation balances:
       - Aerodynamic driving torque (τ_aero)
       - Generator/gearbox resistive torque (τ_load)
       - Mechanical damping losses (c·ω)

    Reference Frames
    ----------------
    This solver operates in a **Co-Rotating (Local Blade) Reference Frame**.
    The structural mesh is fixed in the blade coordinate system, and all
    internal computations (stiffness, mass, inertial forces) are performed
    in this rotating frame.

    **FSI Interface Coordinate Conventions**:

    - **Forces from CFD**: Expected in **Global Inertial Frame**.
      Must be transformed to Local Frame via: F_local = R(-θ) · F_global

    - **Displacements to CFD**: Computed in **Local Rotating Frame**.
      Must be transformed to Global Frame via: u_global = R(θ) · u_local

    .. warning::

       The coordinate transformations for FSI data exchange are **NOT YET
       IMPLEMENTED**. Forces are currently applied directly without rotation,
       and displacements are sent without transformation. This will cause
       incorrect results when θ ≠ 0.

    Usage Modes
    -----------
    **omega_mode="dynamic"** (default):
        Angular velocity evolves according to the torque balance equation.
        Use this for full FSI coupling where rotor speed responds to
        aerodynamic and electrical loads.

    **omega_mode="constant"**:
        Angular velocity remains fixed at initial_omega.
        Use this for comparison studies or when coupling with CFD
        that prescribes rotor speed.

    Configuration Examples
    ----------------------
    Example 1: Free rotation (no generator load)::

        solver_config = {
            "omega_mode": "dynamic",
            "inertia_mode": "total",
            "rotor_inertia": 1.5e7,  # kg·m²
            "initial_omega": 1.57,   # rad/s (~15 RPM)
            "load_torque_mode": "none",
        }

    Example 2: Region 2 control (optimal TSR tracking)::

        solver_config = {
            "omega_mode": "dynamic",
            "inertia_mode": "hub_plus_blades",
            "hub_inertia": 5e6,      # kg·m²
            "initial_omega": 1.2,    # rad/s
            "load_torque_mode": "quadratic",
            "load_torque_coeff": 2.5e4,  # N·m·s²/rad² (k_opt = ½·ρ·π·R⁵·Cp_max/λ_opt³)
        }

    Example 3: Region 3 control (rated power)::

        solver_config = {
            "omega_mode": "dynamic",
            "rotor_inertia": 1.5e7,
            "initial_omega": 1.26,   # rated speed
            "load_torque_mode": "rated_power",
            "rated_power": 5e6,      # 5 MW
        }

    Example 4: Fixed speed for CFD comparison::

        solver_config = {
            "omega_mode": "constant",
            "initial_omega": 1.26,   # fixed at rated
        }

    Parameters
    ----------
    mesh : MeshModel
        The structural mesh model.
    fem_model_properties : Dict
        Model properties including solver configuration with the following
        rotor-specific options:

    Rotor Geometry
    ~~~~~~~~~~~~~~
    rotor_center : list[float], default=[0, 0, 0]
        Center of rotation in global coordinates [x, y, z] [m].
    rotor_axis : str, default='x'
        Rotation axis: 'x', 'y', or 'z'.

    Angular Velocity Control
    ~~~~~~~~~~~~~~~~~~~~~~~~
    omega_mode : str, default='dynamic'
        - "dynamic": ω evolves from torque balance (FSI coupling)
        - "constant": ω stays fixed at initial_omega
    initial_omega : float, default=0.0
        Initial angular velocity [rad/s].
    rotational_damping : float, default=0.0
        Mechanical damping coefficient [N·m·s/rad].
        Only used when omega_mode="dynamic".

    Moment of Inertia
    ~~~~~~~~~~~~~~~~~
    inertia_mode : str, default='total'
        - "total": Use rotor_inertia directly
        - "hub_plus_blades": I = hub_inertia + I_blades (computed from mesh)
        - "fraction": I = I_blades × (1 + hub_fraction)
    rotor_inertia : float
        Total rotor inertia [kg·m²]. Required if inertia_mode="total".
    hub_inertia : float
        Hub inertia [kg·m²]. Required if inertia_mode="hub_plus_blades".
    hub_fraction : float, default=0.1
        Hub inertia as fraction of blade inertia. Used if inertia_mode="fraction".

    Generator/Gearbox Load
    ~~~~~~~~~~~~~~~~~~~~~~
    load_torque_mode : str, default='none'
        - "none": No load (free rotation)
        - "constant": τ_load = load_torque_value
        - "linear": τ_load = k·ω (generator torque ∝ speed)
        - "quadratic": τ_load = k·ω² (optimal TSR control)
        - "rated_power": τ_load = P/ω (constant power extraction)
    load_torque_value : float
        Constant torque [N·m]. For load_torque_mode="constant".
    load_torque_coeff : float
        Torque coefficient k. Units: [N·m·s/rad] for linear, [N·m·s²/rad²] for quadratic.
    rated_power : float
        Rated electrical power [W]. For load_torque_mode="rated_power".

    Inertial Forces
    ~~~~~~~~~~~~~~~
    enable_centrifugal : bool, default=False
        Enable centrifugal force calculation (F = m·ω²·r).
        Creates radial tensile loads that increase with rotor speed.
    enable_gravity : bool, default=False
        Enable gravitational force in rotating frame.
        Creates cyclic edgewise bending loads.
    gravity_vector : list[float], default=[0, 0, -9.81]
        Gravity acceleration vector in global coordinates [m/s²].
        Typical values:
        - [0, 0, -9.81]: Z-up coordinate system
        - [0, -9.81, 0]: Y-up coordinate system

    Aerodynamic Coefficients
    ~~~~~~~~~~~~~~~~~~~~~~~~
    air_density : float, default=1.225
        Air density [kg/m³].
    wind_speed : float, optional
        Freestream velocity [m/s]. Required for computing Cp, Cq, Ct, TSR.
        If None, only raw torque/omega are logged.

    Logging
    ~~~~~~~
    rotor_log_file : str, optional
        Path to CSV log file. If None, no log is written.
    rotor_log_separator : str, default=','
        Column separator for log file.

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
    Adapter : preCICE coupling adapter.

    Notes
    -----
    The angular state (ω, θ, α) participates in the preCICE implicit
    coupling loop and is saved/restored with checkpoints to ensure
    convergence of the coupled FSI system.

    **Formulation Assumptions**:

    - **Linear Elasticity**: Material behavior is linear elastic.
      Geometric nonlinearity is partially included via stress stiffening (K_G)
      but large displacement/rotation formulations are not implemented.

    - **Small Deformations**: The strain-displacement relation assumes small
      strains. For large blade deflections (>10-15% of radius), accuracy
      may degrade.

    - **Lumped Mass**: The mass matrix is lumped (diagonal) for efficiency.
      This is standard practice for explicit/semi-implicit time integration.
    """

    # =========================================================================
    # Class Constants
    # =========================================================================

    #: Mapping from axis name to array index for vectorized operations
    AXIS_MAP: Dict[str, int] = {"x": 0, "y": 1, "z": 2}

    #: Small number for numerical stability checks
    _EPS: float = 1e-10

    #: Generator controller defaults
    _OMEGA_MIN_DEFAULT: float = 0.1  # Default minimum angular velocity [rad/s]

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        """
        Initialize the FSI rotor solver.

        Sets up all configuration parameters and initializes state variables.
        Actual matrix assembly and inertia computation occur in solve().

        Parameters
        ----------
        mesh : MeshModel
            Structural mesh for the rotor blades.
        fem_model_properties : Dict
            Complete model properties including 'solver' configuration.

        Raises
        ------
        ValueError
            If omega_mode or load_torque_mode has an invalid value.
        """
        super().__init__(mesh, fem_model_properties)
        adapter_cfg = fem_model_properties["solver"]["adapter_cfg"]
        base_path = fem_model_properties.get("base_path")
        self.precice_participant = Adapter(adapter_cfg, base_path=base_path)
        self._prepared = False

        # Initialize logger
        self._logger = logging.getLogger(__name__)

        # =====================================================================
        # Configuration Validation (Phase 1: Input Validation)
        # =====================================================================
        # Use RotorConfig for comprehensive parameter validation
        self._config = RotorConfig.from_solver_params(self.solver_params)

        # =====================================================================
        # Rotor Geometry Configuration
        # =====================================================================
        self._rotor_center = np.array(self._config.rotor_center)
        self._rotor_axis = self._config.rotor_axis

        # Pre-compute axis indices (Phase 1: Refactor axis logic)
        self._axis_idx, self._perp_indices = self._get_axis_indices()

        # =====================================================================
        # Angular Velocity Configuration
        # =====================================================================
        self._omega_mode = self._config.omega_mode
        self._inertia_mode = self._config.inertia_mode
        self._rotor_inertia_input = self._config.rotor_inertia
        self._hub_inertia = self._config.hub_inertia
        self._hub_fraction = self._config.hub_fraction
        self._rotational_damping = self._config.rotational_damping

        # =====================================================================
        # Generator Controller (Phase 4: Extract class)
        # =====================================================================
        self._generator = GeneratorController.from_config(self._config)

        # Keep legacy attributes for compatibility
        self._load_torque_mode = self._config.load_torque_mode
        self._load_torque_value = self._config.load_torque_value
        self._load_torque_coeff = self._config.load_torque_coeff
        self._rated_power = self._config.rated_power

        # =====================================================================
        # Inertial Forces Configuration
        # =====================================================================
        self._enable_centrifugal = self._config.enable_centrifugal
        self._enable_gravity = self._config.enable_gravity
        self._enable_coriolis = self._config.enable_coriolis  # Phase 3: Coriolis
        self._enable_euler = self._config.enable_euler  # Euler forces
        self._coriolis_use_rotating_frame = self._config.coriolis_use_rotating_frame
        self._gravity_vector = np.array(self._config.gravity_vector)

        # =====================================================================
        # Angular State Variables
        # =====================================================================
        self._omega = self._config.initial_omega
        self._theta = 0.0
        self._alpha = 0.0
        self._I_rotor = None

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

        # Current aerodynamic state
        self._thrust = 0.0
        self._power = 0.0
        self._tsr = 0.0
        self._Cp = 0.0
        self._Cq = 0.0
        self._Ct = 0.0

        # Rotor dynamics logging
        self._rotor_log_file: Optional[str] = self._config.rotor_log_file
        self._rotor_log_separator: str = self._config.rotor_log_separator
        self._rotor_log_handle: Optional[TextIO] = None

        # Cached nodal masses (computed once when mass matrix is available)
        self._cached_nodal_masses: Optional[np.ndarray] = None

        # Validate Newmark-β parameters for numerical stability
        self._validate_newmark_stability()

    # =========================================================================
    # Axis Helper Methods (Phase 1: Refactor duplicate logic)
    # =========================================================================

    def _get_axis_indices(self) -> Tuple[int, List[int]]:
        """
        Get axis index and perpendicular indices based on rotor_axis.

        Returns
        -------
        axis_idx : int
            Index of the rotation axis (0, 1, or 2 for x, y, z).
        perp_indices : List[int]
            Indices of the two perpendicular axes.
        """
        axis_idx = self.AXIS_MAP.get(self._rotor_axis, 0)
        perp_indices = [i for i in range(3) if i != axis_idx]
        return axis_idx, perp_indices

    def _validate_newmark_stability(self) -> None:
        """
        Validate Newmark-β parameters for numerical stability.

        Issues warnings if the parameter values may lead to numerical instability.

        Stability Conditions (based on MOOSE framework documentation):
        - For unconditional stability: γ ≥ 0.5 and β ≥ (γ + 0.5)²/4
        - For γ = 0.5: method is second-order and unconditionally stable for 0.5 ≤ γ ≤ 2β

        Standard stable combinations:
        - β = 0.25, γ = 0.5: Constant average acceleration (no numerical damping)
        - β = 1/6, γ = 0.5: Linear acceleration (small numerical damping)

        Warnings
        --------
        UserWarning
            If parameters may cause numerical instability.
        """
        import warnings

        # Get Newmark-β parameters with defaults
        beta = self.solver_params.get("beta", 0.25)
        gamma = self.solver_params.get("gamma", 0.5)

        self._logger.debug(f"Validating Newmark-β parameters: β={beta}, γ={gamma}")

        # Check basic stability condition: γ ≥ 0.5
        if gamma < 0.5:
            warnings.warn(
                f"Newmark-β parameter γ={gamma} < 0.5 may cause numerical instability. "
                f"For unconditional stability, use γ ≥ 0.5. "
                f"Consider using γ=0.5 (standard value).",
                UserWarning,
                stacklevel=2,
            )
            self._logger.warning(f"Newmark γ={gamma} < 0.5: potential numerical instability")

        # Check unconditional stability condition: β ≥ (γ + 0.5)²/4
        beta_min = (gamma + 0.5) ** 2 / 4
        if beta < beta_min:
            warnings.warn(
                f"Newmark-β parameter β={beta} < {beta_min:.4f} may cause numerical instability. "
                f"For unconditional stability with γ={gamma}, use β ≥ {beta_min:.4f}. "
                f"Consider using standard values: β=0.25, γ=0.5 (constant average acceleration) "
                f"or β={1 / 6:.4f}, γ=0.5 (linear acceleration).",
                UserWarning,
                stacklevel=2,
            )
            self._logger.warning(
                f"Newmark β={beta} < {beta_min:.4f}: potential numerical instability"
            )

        # Check for common stable combinations and provide recommendations
        if abs(beta - 0.25) < 1e-10 and abs(gamma - 0.5) < 1e-10:
            self._logger.info(
                "Using Newmark constant average acceleration (β=0.25, γ=0.5): "
                "unconditionally stable, second-order, no numerical damping"
            )
        elif abs(beta - 1 / 6) < 1e-10 and abs(gamma - 0.5) < 1e-10:
            self._logger.info(
                "Using Newmark linear acceleration (β=1/6, γ=0.5): "
                "unconditionally stable, second-order, small numerical damping"
            )
        else:
            # Additional check for gamma = 0.5 special case
            if abs(gamma - 0.5) < 1e-10:
                if beta < 0.25:  # For γ=0.5, need 0.5 ≤ γ ≤ 2β, so 2β ≥ 0.5, so β ≥ 0.25
                    warnings.warn(
                        f"For γ=0.5, unconditional stability requires β ≥ 0.25, but β={beta}. "
                        f"Consider using standard β=0.25 for constant average acceleration.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    self._logger.info(
                        f"Using custom Newmark parameters (β={beta}, γ={gamma}): "
                        f"should be unconditionally stable for γ=0.5"
                    )
            else:
                self._logger.info(
                    f"Using custom Newmark parameters (β={beta}, γ={gamma}): "
                    f"stability conditions satisfied"
                )

    def _ensure_3d_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Ensure coordinate array is 3D by padding with zeros if needed.

        Parameters
        ----------
        coords : np.ndarray
            Coordinate array of shape (n_points, n_dims) where n_dims can be 2 or 3.

        Returns
        -------
        coords_3d : np.ndarray
            Coordinate array of shape (n_points, 3) with zero-padding if needed.
        """
        if coords.shape[1] < 3:
            return np.hstack([
                coords,
                np.zeros((coords.shape[0], 3 - coords.shape[1])),
            ])
        return coords

    def _compute_position_from_center(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute position vectors from rotor center.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates, shape (n_nodes, 3) [m].

        Returns
        -------
        r : np.ndarray
            Position vectors from rotor center, shape (n_nodes, 3) [m].
        """
        return coords - self._rotor_center

    def _compute_perpendicular_distances(self, position_vectors: np.ndarray) -> np.ndarray:
        """
        Compute perpendicular distances from rotation axis.

        Parameters
        ----------
        position_vectors : np.ndarray
            Position vectors from rotor center, shape (n_nodes, 3) [m].

        Returns
        -------
        r_perp : np.ndarray
            Perpendicular distances from rotation axis, shape (n_nodes,) [m].
        """
        return np.sqrt(np.sum(position_vectors[:, self._perp_indices] ** 2, axis=1))

    # =========================================================================
    # Torque and Force Calculations
    # =========================================================================

    def _compute_torque(self, coords: np.ndarray, forces: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute torque around rotor axis from nodal forces.

        The torque is computed as the sum of moment contributions from all
        nodes in the mesh:

            τ = Σᵢ (rᵢ × Fᵢ)

        where rᵢ is the position vector from the rotor center to node i,
        and Fᵢ is the force at that node.

        The axial component (around the rotation axis) drives rotor acceleration,
        while perpendicular components create bending moments on the structure.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in global frame, shape (n_nodes, 3) [m].
            Should be deformed coordinates (initial + displacement) for accuracy.
        forces : np.ndarray
            Force vectors at nodes, shape (n_nodes, 3) [N].

        Returns
        -------
        torque_axial : float
            Torque component around the rotor axis [N·m].
            Positive value indicates torque in direction of rotation.
        torque_vector : np.ndarray
            Full torque vector [Tx, Ty, Tz] [N·m].
        """
        # Validate inputs
        if coords.shape[0] != forces.shape[0]:
            raise ValueError(
                f"coords and forces must have same number of nodes: "
                f"{coords.shape[0]} vs {forces.shape[0]}"
            )

        # Check for NaN/Inf in forces
        if not np.all(np.isfinite(forces)):
            self._logger.warning("Non-finite values detected in forces, replacing with zeros")
            forces = np.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)

        # Position vectors from rotor center
        r = self._compute_position_from_center(coords)

        # Torque at each node: τ = r × F
        torque_per_node = np.cross(r, forces)

        # Total torque vector
        torque_vector = np.sum(torque_per_node, axis=0)

        # Extract axial component using pre-computed index
        torque_axial = torque_vector[self._axis_idx]

        return torque_axial, torque_vector

    # =========================================================================
    # Inertial Forces (Centrifugal and Gravitational)
    # =========================================================================

    def _compute_centrifugal_forces(self, coords: np.ndarray, nodal_mass: np.ndarray) -> np.ndarray:
        """
        Compute centrifugal forces at mesh nodes due to rotor rotation.

        The centrifugal force on each node is:

            F_c,i = mᵢ · ω² · r_⊥,i · ê_r

        where:
        - mᵢ: Lumped mass at node i [kg]
        - ω: Current angular velocity [rad/s]
        - r_⊥,i: Perpendicular distance from rotation axis [m]
        - ê_r: Unit radial vector (outward from axis)

        Physical Interpretation
        -----------------------
        Centrifugal forces create tensile stresses along the blade span,
        which provide a "centrifugal stiffening" effect that increases
        blade natural frequencies at higher rotor speeds. This effect is
        important for:
        - Resonance avoidance (Campbell diagram analysis)
        - Buckling resistance in the spar caps
        - Reducing flapwise deflections

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in global frame, shape (n_nodes, 3) [m].
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].

        Returns
        -------
        F_centrifugal : np.ndarray
            Centrifugal force vectors at nodes, shape (n_nodes, 3) [N].
            Forces point radially outward from the rotation axis.

        Notes
        -----
        - Forces are zero when ω = 0 (no rotation)
        - Forces scale with ω² (quadratic increase with speed)
        - At rated speed, centrifugal loads can be comparable to aerodynamic loads
        """
        n_nodes = coords.shape[0]
        F_centrifugal = np.zeros((n_nodes, 3))

        if abs(self._omega) < self._EPS:
            # No rotation, no centrifugal force
            return F_centrifugal

        # Position vectors from rotor center
        r = self._compute_position_from_center(coords)

        # Use pre-computed axis indices
        # Perpendicular distance from axis (r_perp)
        r_perp_vec = np.zeros_like(r)
        r_perp_vec[:, self._perp_indices] = r[:, self._perp_indices]
        r_perp_mag = np.linalg.norm(r_perp_vec, axis=1)

        # Unit radial vectors (avoid division by zero)
        r_perp_mag_safe = np.where(r_perp_mag > self._EPS, r_perp_mag, 1.0)
        e_r = r_perp_vec / r_perp_mag_safe[:, np.newaxis]

        # Centrifugal force: F = m · ω² · r_perp · ê_r
        omega_squared = self._omega**2
        F_centrifugal = (nodal_mass * omega_squared * r_perp_mag)[:, np.newaxis] * e_r

        return F_centrifugal

    def _compute_gravity_forces(self, nodal_mass: np.ndarray) -> np.ndarray:
        """
        Compute gravitational forces in the rotating blade reference frame.

        Since the structural mesh is fixed (non-rotating), we simulate the
        effect of gravity on a rotating blade by rotating the gravity vector
        inversely by the accumulated rotation angle θ:

            F_g,i = mᵢ · R_axis(-θ) · g

        where:
        - mᵢ: Lumped mass at node i [kg]
        - R_axis(-θ): Rotation matrix around rotor axis by -θ
        - g: Gravity vector in global frame [m/s²]
        - θ: Accumulated rotation angle [rad]

        Physical Interpretation
        -----------------------
        As the blade rotates through its cycle:
        - At θ = 0° (blade pointing up): Gravity causes compression in leading edge
        - At θ = 90° (blade horizontal): Gravity causes flapwise bending
        - At θ = 180° (blade pointing down): Gravity causes tension in leading edge
        - At θ = 270° (blade horizontal): Gravity causes opposite flapwise bending

        This creates cyclic edgewise bending at 1P frequency (once per revolution),
        which is a major contributor to blade fatigue loads.

        Parameters
        ----------
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].

        Returns
        -------
        F_gravity : np.ndarray
            Gravity force vectors at nodes, shape (n_nodes, 3) [N].
            Includes effect of virtual blade rotation.

        Notes
        -----
        - Gravity direction rotates with -θ to simulate blade rotation
        - Forces oscillate at 1P frequency relative to azimuth angle
        - Maximum edgewise loads occur at 90° and 270° azimuth
        """
        # Rotate gravity vector by -θ around the rotor axis
        # This simulates the blade "seeing" gravity from different angles
        g_rotated = self._rotate_vector_around_axis(
            self._gravity_vector,
            self._rotor_axis,
            -self._theta,  # Negative because blade rotates forward, gravity "rotates" backward
        )

        # Apply gravity force to each node: F = m · g_rotated
        F_gravity = nodal_mass[:, np.newaxis] * g_rotated

        return F_gravity

    def _rotate_vector_around_axis(self, vector: np.ndarray, axis: str, angle: float) -> np.ndarray:
        """
        Rotate a 3D vector or array of vectors around a coordinate axis.

        Uses Rodrigues' rotation formula for rotation around principal axes.

        Parameters
        ----------
        vector : np.ndarray
            Vector(s) to rotate, shape (3,) or (N, 3).
        axis : str
            Rotation axis: 'x', 'y', or 'z'.
        angle : float
            Rotation angle [rad]. Positive follows right-hand rule.

        Returns
        -------
        rotated : np.ndarray
            Rotated vector(s), shape (3,) or (N, 3).

        Notes
        -----
        Rotation matrices for principal axes:

        Rx(θ) = [[1,    0,       0    ],
                 [0,  cos(θ), -sin(θ)],
                 [0,  sin(θ),  cos(θ)]]

        Ry(θ) = [[ cos(θ), 0, sin(θ)],
                 [   0,    1,   0    ],
                 [-sin(θ), 0, cos(θ)]]

        Rz(θ) = [[cos(θ), -sin(θ), 0],
                 [sin(θ),  cos(θ), 0],
                 [  0,       0,    1]]
        """
        R = self._get_rotation_matrix(axis, angle)
        # Handle (N, 3) arrays by using dot product with transpose
        # For (3,) vector v: v @ R.T is equivalent to R @ v
        # For (N, 3) vectors V: V @ R.T rotates each row vector correctly
        return vector @ R.T

    def _get_rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """
        Get 3x3 rotation matrix for rotation around a principal axis.

        Parameters
        ----------
        axis : str
            Rotation axis: 'x', 'y', or 'z'.
        angle : float
            Rotation angle [rad]. Positive follows right-hand rule.

        Returns
        -------
        R : np.ndarray
            3x3 rotation matrix.
        """
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            return np.eye(3)

    def _transform_vectors_to_rotating_frame(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transform vectors from inertial frame to rotating frame.

        Applies R(-θ) to transform from lab frame to blade-attached frame:

            v_rot = R(-θ) · v_inertial

        Parameters
        ----------
        vectors : np.ndarray
            Vectors in inertial frame, shape (n_vectors, 3).

        Returns
        -------
        vectors_rot : np.ndarray
            Vectors in rotating frame, shape (n_vectors, 3).

        Notes
        -----
        The rotating frame co-rotates with the blade at angle θ.
        Using -θ because we transform FROM inertial TO rotating.
        """
        R = self._get_rotation_matrix(self._rotor_axis, -self._theta)
        return vectors @ R.T  # Equivalent to (R @ v.T).T for each row

    def _transform_vectors_to_inertial_frame(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transform vectors from rotating frame to inertial frame.

        Applies R(θ) to transform from blade-attached frame to lab frame:

            v_inertial = R(θ) · v_rot

        Parameters
        ----------
        vectors : np.ndarray
            Vectors in rotating frame, shape (n_vectors, 3).

        Returns
        -------
        vectors_inertial : np.ndarray
            Vectors in inertial frame, shape (n_vectors, 3).

        Notes
        -----
        This is the inverse transformation of _transform_vectors_to_rotating_frame.

        **Primary Use Case**: Transform structural displacements computed in the
        Local Rotating Frame to the Global Inertial Frame before sending to CFD.
        This ensures kinematic compatibility at the FSI interface.
        """
        R = self._get_rotation_matrix(self._rotor_axis, self._theta)
        return vectors @ R.T  # Equivalent to (R @ v.T).T for each row

    def _get_nodal_masses(self) -> np.ndarray:
        """
        Extract nodal masses from the lumped mass matrix.

        Returns
        -------
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].

        Notes
        -----
        Extracts translational mass from the first 3 DOFs of each node
        (assumes consistent mass distribution across x, y, z translations).
        """
        M_diag = self.M.getDiagonal()
        mass_array = M_diag.array

        dofs_per_node = self.domain.dofs_per_node
        n_nodes = len(mass_array) // dofs_per_node

        # Extract translational mass (average of first 3 DOFs per node)
        nodal_mass = np.zeros(n_nodes)
        for i in range(n_nodes):
            start_idx = i * dofs_per_node
            nodal_mass[i] = np.mean(mass_array[start_idx : start_idx + 3])

        return nodal_mass

    def _compute_inertial_forces(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute total inertial forces (centrifugal + gravitational) at nodes.

        This method combines centrifugal and gravitational effects based on
        the current configuration flags. The total force vector is:

            F_inertial = F_centrifugal + F_gravity

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in global frame, shape (n_nodes, 3) [m].

        Returns
        -------
        F_inertial : np.ndarray
            Total inertial force vectors at nodes, shape (n_nodes, 3) [N].

        Notes
        -----
        - Returns zeros if both centrifugal and gravity are disabled
        - Nodal masses are extracted from the lumped mass matrix
        - Forces are applied in global coordinates
        """
        n_nodes = coords.shape[0]
        F_inertial = np.zeros((n_nodes, 3))

        if (
            not self._enable_centrifugal
            and not self._enable_gravity
            and not self._enable_coriolis
            and not self._enable_euler
        ):
            return F_inertial

        # Get nodal masses (use cached if available)
        if self._cached_nodal_masses is not None:
            nodal_mass = self._cached_nodal_masses
        else:
            nodal_mass = self._get_nodal_masses()
            self._cached_nodal_masses = nodal_mass

        # Add centrifugal forces if enabled
        if self._enable_centrifugal:
            F_inertial += self._compute_centrifugal_forces(coords, nodal_mass)

        # Add gravity forces if enabled
        if self._enable_gravity:
            F_inertial += self._compute_gravity_forces(nodal_mass)

        return F_inertial

    def _compute_coriolis_forces(
        self, coords: np.ndarray, nodal_mass: np.ndarray, nodal_velocities: np.ndarray
    ) -> np.ndarray:
        """
        Compute Coriolis forces at mesh nodes due to rotation.

        The Coriolis force on each node is:

            F_cor,i = -2 · mᵢ · (ω⃗ × v⃗_rel,i)

        where:
        - mᵢ: Lumped mass at node i [kg]
        - ω⃗: Angular velocity vector [rad/s]
        - v⃗_rel,i: Velocity of node i relative to rotating frame [m/s]

        Physical Model (Rotating Reference Frame)
        -----------------------------------------
        This solver operates in a **Rotating Reference Frame** where the mesh
        coordinates are fixed (do not physically rotate). In this formulation:

        1. The nodal velocities computed by Newmark integration ARE the relative
           velocities (v_rel) - they represent the rate of deformation in the
           co-rotating frame.

        2. NO coordinate transformation is needed because we are already working
           in the rotating frame.

        3. The Coriolis force couples flapwise and edgewise motion, introducing
           apparent damping (or anti-damping) effects.

        Important: Previous versions incorrectly applied R(-θ) to velocities,
        which introduced spurious forces equivalent to a "double centrifugal"
        effect. This has been corrected.

        Physical Interpretation
        -----------------------
        Coriolis forces arise from the interaction between nodal velocities
        and the rotating reference frame. They:
        - Act perpendicular to both velocity and rotation axis
        - Can cause coupling between flapwise and edgewise modes
        - Introduce apparent damping (or anti-damping) effects
        - Are important for very long, flexible blades at high tip speeds

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates, shape (n_nodes, 3). Used only for array sizing.
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        nodal_velocities : np.ndarray
            Nodal velocity vectors (deformation rate in rotating frame),
            shape (n_nodes, 3) [m/s].

        Returns
        -------
        F_coriolis : np.ndarray
            Coriolis force vectors at nodes, shape (n_nodes, 3) [N].

        Notes
        -----
        - Forces are zero when ω = 0 (no rotation)
        - Forces are proportional to ω (linear in rotation speed)
        - Forces are proportional to nodal velocity (velocity-dependent)
        - This is a non-conservative force that can affect system stability
        """
        n_nodes = coords.shape[0]
        F_coriolis = np.zeros((n_nodes, 3))

        if abs(self._omega) < self._EPS:
            return F_coriolis

        # Construct angular velocity vector along rotation axis
        omega_vec = np.zeros(3)
        omega_vec[self._axis_idx] = self._omega

        # In the rotating frame formulation, nodal_velocities ARE v_rel
        # (the deformation velocities relative to the co-rotating frame).
        # No transformation is needed - this was the source of the previous bug.
        v_rel = nodal_velocities

        # F_cor = -2 * m * (ω × v_rel) for each node
        omega_cross_v = np.cross(omega_vec, v_rel)
        F_coriolis = -2.0 * nodal_mass[:, np.newaxis] * omega_cross_v

        return F_coriolis

    def _compute_euler_forces(self, coords: np.ndarray, nodal_mass: np.ndarray) -> np.ndarray:
        """
        Compute Euler forces at mesh nodes due to angular acceleration.

        The Euler force on each node is:

            F_euler,i = -mᵢ · (α⃗ × r⃗_rot,i)

        where:
        - mᵢ: Lumped mass at node i [kg]
        - α⃗: Angular acceleration vector [rad/s²] (α = dω/dt)
        - r⃗_rot,i: Position vector of node i in rotating frame [m]

        Frame Transformation
        --------------------
        When `coriolis_use_rotating_frame=True` (default), positions are properly
        transformed to the rotating frame before computing Euler forces:

            r_rot = R(-θ) · r_inertial

        The resulting forces are then transformed back to the inertial frame
        for application to the structural solver:

            F_inertial = R(θ) · F_rot

        This provides the physically correct Euler force for rotating structures.

        Physical Interpretation
        -----------------------
        Euler forces (also called azimuthal or tangential inertial forces) arise
        when the angular velocity of the rotating frame changes over time. They:
        - Act tangentially to the rotation (perpendicular to both α and r)
        - Appear only when angular acceleration is non-zero
        - Are distinct from Coriolis forces (velocity-dependent) and
          centrifugal forces (position-dependent, ω²)

        For wind turbines, Euler forces become significant during:
        - Start-up and shutdown transients
        - Rapid power regulation changes
        - Grid fault ride-through events
        - Variable-speed operation with changing rotor speed

        Reference
        ---------
        https://en.wikipedia.org/wiki/Euler_force

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in inertial frame, shape (n_nodes, 3) [m].
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].

        Returns
        -------
        F_euler : np.ndarray
            Euler force vectors at nodes in inertial frame, shape (n_nodes, 3) [N].

        Notes
        -----
        - Only computed when omega_mode='dynamic' and α ≠ 0
        - Uses angular acceleration stored in self._alpha
        - The minus sign comes from the fictitious force in the rotating frame
        """
        n_nodes = coords.shape[0]
        F_euler = np.zeros((n_nodes, 3))

        # Angular acceleration vector (α = dω/dt)
        alpha = getattr(self, "_alpha", 0.0)
        if abs(alpha) < self._EPS:
            return F_euler

        # Build angular acceleration vector along rotor axis
        alpha_vec = np.zeros(3)
        alpha_vec[self._axis_idx] = alpha

        # Get positions in the appropriate frame
        if self._coriolis_use_rotating_frame:
            # Transform positions to rotating frame: r_rot = R(-θ) · r_inertial
            r_rot = self._transform_vectors_to_rotating_frame(coords)
        else:
            # Use inertial positions directly (approximation)
            r_rot = coords

        # F_euler = -m * (α × r_rot) for each node in rotating frame
        alpha_cross_r = np.cross(alpha_vec, r_rot)
        F_euler_rot = -nodal_mass[:, np.newaxis] * alpha_cross_r

        if self._coriolis_use_rotating_frame:
            # Transform forces back to inertial frame: F_inertial = R(θ) · F_rot
            F_euler = self._transform_vectors_to_inertial_frame(F_euler_rot)
        else:
            F_euler = F_euler_rot

        return F_euler

    # =========================================================================
    # Moment of Inertia Calculations
    # =========================================================================

    def _compute_blades_inertia(self) -> float:
        """
        Compute moment of inertia of blades about the rotor axis.

        Uses the lumped mass matrix to extract nodal masses, then computes
        the moment of inertia using the parallel axis theorem:

            I_blades = Sum_i m_i * r_perp_i^2

        where:
        - m_i is the lumped mass at node i [kg]
        - r_perp_i is the perpendicular distance from the rotation axis [m]

        The perpendicular distance is computed as:
            r_perp = sqrt(r_j^2 + r_k^2)

        where j and k are the indices perpendicular to the rotation axis.

        Implementation Notes
        --------------------
        - Mass is extracted from the first 3 DOFs (translational) of each node
        - Rotational DOFs are ignored as they contribute negligibly to inertia
        - This calculation assumes the mesh represents only the blades,
          not the hub or nacelle

        Approximation: Point Mass Model
        --------------------------------
        This implementation treats each node as a **point mass**, ignoring
        the rotational inertia of individual shell elements about their
        own centroids.

        The complete moment of inertia would be:

            I_total = Sum_i (m_i * r_perp_i^2 + I_element_i)

        where I_element is the element's rotational inertia about its centroid,
        projected onto the rotor axis. For a rectangular shell element:

            I_element = (rho * h * A) / 12 * (a^2 + b^2)

        **Error Estimation:**
        For typical wind turbine blade meshes:
        - Element size: a ~ b ~ 0.1 m
        - Distance to axis: r ~ 30-60 m

        Relative error:
            I_element / I_translational ~ (a^2 + b^2) / (12 * r^2)
                                        ~ 0.02 / (12 * 900)
                                        ~ 2e-6 (0.0002%)

        This approximation is excellent for wind turbine applications where
        r >> element_size. It would only become significant for elements
        very close to the rotation axis or with very large element sizes.

        Returns
        -------
        I_blades : float
            Moment of inertia of blades about rotor axis [kg*m^2].

        See Also
        --------
        _get_total_inertia : Combines blade inertia with hub contribution.
        """
        # Use existing method to get nodal masses (eliminates code duplication)
        nodal_mass = self._get_nodal_masses()

        # Get node coordinates
        coords = self.mesh_obj.coords_array

        # Position vectors from rotor center
        r = self._compute_position_from_center(coords)

        # Perpendicular distance squared from rotation axis (use pre-computed indices)
        r_perp_squared = np.sum(r[:, self._perp_indices] ** 2, axis=1)

        # I = Σ m_i × r_⊥²
        I_blades = np.sum(nodal_mass * r_perp_squared)

        return I_blades

    def _get_total_inertia(self) -> float:
        """
        Compute total rotor moment of inertia based on configuration mode.

        The total moment of inertia includes contributions from:
        - Rotor blades (can be computed from mesh)
        - Hub and pitch system
        - Low-speed shaft (rotor side of gearbox)
        - Gearbox (referred to low-speed side)

        Three configuration modes are available:

        **"total"** (default):
            User provides the complete rotor inertia directly.
            Use when you have the total inertia from manufacturer data
            or detailed CAD models.

        **"hub_plus_blades"**:
            I_total = I_hub + I_blades
            - I_hub: User-specified (includes hub, pitch, shaft)
            - I_blades: Computed from mesh using lumped mass
            Use when you have hub inertia data but want to compute
            blade inertia from the structural model.

        **"fraction"**:
            I_total = I_blades × (1 + f_hub)
            - I_blades: Computed from mesh
            - f_hub: Hub contribution as fraction of blade inertia
            Use for quick estimates when detailed hub data is unavailable.
            Typical values: f_hub = 0.05-0.15 for utility-scale turbines.

        Returns
        -------
        I_total : float
            Total moment of inertia about rotor axis [kg·m²].

        Raises
        ------
        ValueError
            If required parameters are missing for the selected mode,
            or if an unknown mode is specified.

        Examples
        --------
        For a 5 MW turbine with known total inertia::

            config = {"inertia_mode": "total", "rotor_inertia": 3.5e7}

        For detailed modeling with known hub inertia::

            config = {"inertia_mode": "hub_plus_blades", "hub_inertia": 1.2e7}

        For quick estimate assuming hub is 10% of blade inertia::

            config = {"inertia_mode": "fraction", "hub_fraction": 0.1}
        """
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
            I_hub_estimated = I_blades * self._hub_fraction
            self._logger.info(f"        Blade inertia:      {I_blades:.4e} kg·m²")
            self._logger.info(f"        Hub fraction:       {self._hub_fraction:.2%}")
            self._logger.info(f"        Hub inertia (est):  {I_hub_estimated:.4e} kg·m²")
            self._logger.info(f"        Total inertia:      {I_total:.4e} kg·m²")
            return I_total

        else:
            raise ValueError(
                f"Unknown inertia_mode: {self._inertia_mode}. "
                "Use 'total', 'hub_plus_blades', or 'fraction'."
            )

    # =========================================================================
    # Angular State Integration
    # =========================================================================

    def _update_angular_state(self, torque_axial: float, dt: float) -> Tuple[float, float, float]:
        """
        Integrate the rotational equation of motion for one timestep.

        This method updates the angular velocity (omega), acceleration (alpha), and
        position (theta) based on the torque balance equation:

            I * alpha = tau_aero - tau_load - c * omega

        where:
        - I: Total rotor moment of inertia [kg*m^2]
        - alpha: Angular acceleration [rad/s^2]
        - tau_aero: Aerodynamic driving torque [N*m]
        - tau_load: Resistive load torque (generator) [N*m]
        - c: Rotational damping coefficient [N*m*s/rad]
        - omega: Angular velocity [rad/s]

        Integration Scheme
        ------------------
        Uses Newmark-beta method consistent with structural integration, providing
        O(dt^2) accuracy for all variables (theta, omega, alpha):

        Predictor step:
            omega_pred = omega_n + dt * (1 - gamma) * alpha_n

        Solve for acceleration (with implicit damping):
            alpha_{n+1} = (tau_net - c * omega_pred) / (I + c * dt * gamma)

        Corrector step:
            omega_{n+1} = omega_pred + dt * gamma * alpha_{n+1}
            theta_{n+1} = theta_n + dt * omega_n + dt^2 * [(0.5-beta) * alpha_n + beta * alpha_{n+1}]

        This scheme:
        - Uses the same beta, gamma parameters as structural integration
        - Treats damping implicitly for unconditional stability
        - Provides consistent O(dt^2) accuracy across FSI coupling

        Operating Modes
        ---------------
        **omega_mode="dynamic"** (default):
            Full integration of the equation of motion. Angular velocity
            responds to the torque imbalance between aerodynamic and load torques.

        **omega_mode="constant"**:
            Angular velocity is held fixed at initial_omega.
            Only theta accumulates (alpha is set to 0).
            Use for comparison with fixed-RPM CFD simulations.

        Parameters
        ----------
        torque_axial : float
            Aerodynamic torque component around rotor axis [N*m].
            Positive value accelerates rotor in direction of rotation.
        dt : float
            Time step size [s].

        Returns
        -------
        omega : float
            Updated angular velocity [rad/s].
        alpha : float
            Angular acceleration at end of timestep [rad/s^2].
        theta : float
            Updated cumulative rotation angle [rad].

        Notes
        -----
        - This method modifies instance state variables (_omega, _alpha, _theta)
        - State is saved/restored by preCICE checkpoint mechanism
        - For implicit coupling, this is called multiple times per timestep
          until convergence
        - Uses same Newmark-beta parameters as structural solver for consistency

        See Also
        --------
        GeneratorController.compute_torque : Computes tau_load based on selected mode.
        """
        if self._omega_mode == "constant":
            # Constant mode: ω fixed, α=0, only θ accumulates for position tracking
            self._alpha = 0.0
            self._theta += self._omega * dt
            return self._omega, self._alpha, self._theta

        # Dynamic mode: integrate rotational equation of motion
        # =====================================================================
        # Newmark-β Integration for Rotational Dynamics
        # =====================================================================
        # The equation of motion is:
        #   I·α = τ_net - c·ω
        #
        # where τ_net = τ_aero - τ_load
        #
        # Using Newmark-β method (same as structural integration) for consistency:
        #
        # Predictor (explicit):
        #   θ_pred = θ_n + dt·ω_n + dt²·(0.5 - β)·α_n
        #   ω_pred = ω_n + dt·(1 - γ)·α_n
        #
        # Corrector (with implicit damping for stability):
        #   I·α_{n+1} = τ_net - c·ω_{n+1}
        #   ω_{n+1} = ω_pred + dt·γ·α_{n+1}
        #
        # Substituting ω_{n+1} into equation of motion and solving for α_{n+1}:
        #   I·α_{n+1} = τ_net - c·(ω_pred + dt·γ·α_{n+1})
        #   (I + c·dt·γ)·α_{n+1} = τ_net - c·ω_pred
        #   α_{n+1} = (τ_net - c·ω_pred) / (I + c·dt·γ)
        #
        # This provides O(dt²) accuracy consistent with structural integration.
        # =====================================================================

        # Get Newmark-β parameters (same as structural)
        beta = self.solver_params.get("beta", 0.25)
        gamma = self.solver_params.get("gamma", 0.5)

        # Step 1: Compute load torque using GeneratorController
        tau_load = self._generator.compute_torque(self._omega)

        # Step 2: Compute net driving torque
        tau_net = torque_axial - tau_load

        # Step 3: Newmark-β integration
        I = self._I_rotor
        c = self._rotational_damping

        # Store old values
        theta_old = self._theta
        omega_old = self._omega
        alpha_old = self._alpha

        # Predictor step
        omega_pred = omega_old + dt * (1.0 - gamma) * alpha_old

        # Solve for new acceleration (implicit damping)
        # (I + c·dt·γ)·α_new = τ_net - c·ω_pred
        alpha_new = (tau_net - c * omega_pred) / (I + c * dt * gamma)

        # Corrector step for velocity
        omega_new = omega_pred + dt * gamma * alpha_new

        # Update position with Newmark-β formula
        # θ_{n+1} = θ_n + dt·ω_n + dt²·[(0.5-β)·α_n + β·α_{n+1}]
        theta_new = (
            theta_old + dt * omega_old + dt**2 * ((0.5 - beta) * alpha_old + beta * alpha_new)
        )

        # Step 4: Update state
        self._alpha = alpha_new
        self._omega = omega_new
        self._theta = theta_new

        return self._omega, self._alpha, self._theta

    def _compute_rotor_radius(self) -> float:
        """
        Compute rotor radius from mesh as maximum perpendicular distance from axis.

        Returns
        -------
        R : float
            Rotor radius [m].
        """
        coords = self.mesh_obj.coords_array
        r = self._compute_position_from_center(coords)

        # Use pre-computed perpendicular indices
        # Perpendicular distance from rotation axis
        r_perp = self._compute_perpendicular_distances(r)

        radius = np.max(r_perp)

        # Validate radius is positive
        if radius <= 0:
            raise ValueError(
                f"Computed rotor radius is non-positive: {radius}. "
                "Check mesh coordinates and rotor_center configuration."
            )

        return radius

    def _compute_aerodynamic_coefficients(self, torque: float, forces_3d: np.ndarray) -> None:
        """
        Compute aerodynamic coefficients for rotor characteristic curves.

        Computes thrust, power, TSR, Cp, Cq, Ct from current state.
        Results are stored in instance variables.

        Parameters
        ----------
        torque : float
            Aerodynamic torque around rotor axis [N·m].
        forces_3d : np.ndarray
            Force vectors at interface nodes, shape (n_nodes, 3).
        """
        # Thrust: sum of forces along rotor axis (positive = pushing rotor downstream)
        self._thrust = np.sum(forces_3d[:, self._axis_idx])

        # Power: P = τ × ω
        self._power = torque * self._omega

        # Only compute dimensionless coefficients if wind speed is provided
        if self._wind_speed is not None and self._wind_speed > 0:
            V = self._wind_speed
            R = self._rotor_radius
            A = self._rotor_area
            rho = self._air_density

            # Dynamic pressure
            q = 0.5 * rho * V**2

            # Tip Speed Ratio: λ = ωR / V
            self._tsr = self._omega * R / V

            # Power coefficient: Cp = P / (0.5 × ρ × A × V³)
            P_available = 0.5 * rho * A * V**3
            self._Cp = self._power / P_available if P_available > 0 else 0.0

            # Torque coefficient: Cq = Q / (0.5 × ρ × A × V² × R)
            self._Cq = torque / (q * A * R) if q > 0 else 0.0

            # Thrust coefficient: Ct = T / (0.5 × ρ × A × V²)
            self._Ct = self._thrust / (q * A) if q > 0 else 0.0
        else:
            self._tsr = 0.0
            self._Cp = 0.0
            self._Cq = 0.0
            self._Ct = 0.0

    def _precompute_interface_mapping(self, bc_manager: BoundaryConditionManager) -> None:
        """
        Pre-compute mapping from interface DOFs to reduced system DOFs.

        This optimization avoids expanding the full solution vector at every
        timestep just to extract interface velocities.
        """
        interface_dofs = self.precice_participant.interface_dofs

        # Create a map from global DOF index to reduced vector index
        # bc_manager.free_dofs is a dict {global_idx: reduced_idx} or similar structure
        # If it's a list, we need to build the map.
        # Looking at typical implementations, free_dofs is often a dict or we can infer it.
        # Let's assume we can get the mapping.

        # If bc_manager.free_dofs is a dict {global_idx: reduced_idx}:
        if isinstance(bc_manager.free_dofs, dict):
            global_to_reduced = bc_manager.free_dofs
        else:
            # Fallback if it's just a list of indices
            global_to_reduced = {dof: i for i, dof in enumerate(bc_manager.free_dofs)}

        self._interface_to_reduced_indices = []
        for global_dof in interface_dofs:
            # If the DOF is fixed (not in free_dofs), we map it to -1
            self._interface_to_reduced_indices.append(global_to_reduced.get(global_dof, -1))

        self._interface_to_reduced_indices = np.array(self._interface_to_reduced_indices, dtype=int)

    def _extract_interface_velocities(self, v_reduced: PETSc.Vec) -> np.ndarray:
        """
        Extract interface velocities directly from reduced velocity vector.

        Parameters
        ----------
        v_reduced : PETSc.Vec
            Reduced velocity vector from the solver.

        Returns
        -------
        velocities : np.ndarray
            Velocity vectors at interface nodes, shape (n_interface_nodes, 3).
        """
        # Get array from PETSc vector
        v_array = v_reduced.array

        # Initialize result array
        n_dofs = len(self._interface_to_reduced_indices)
        interface_vals = np.zeros(n_dofs)

        # Mask for free DOFs (indices >= 0)
        mask = self._interface_to_reduced_indices >= 0
        valid_indices = self._interface_to_reduced_indices[mask]

        # Fill values for free DOFs
        # Note: v_array is local to the process if MPI is used, but here we assume
        # we are gathering or working on a sequential part for the interface.
        # If v_reduced is MPI, getting .array might return local part.
        # However, preCICE adapter usually handles gathering if needed.
        # For safety in this refactoring step, we assume standard behavior.
        interface_vals[mask] = v_array[valid_indices]

        return interface_vals.reshape(-1, 3)

    def _compute_effective_stiffness(
        self,
        K_red: "PETSc.Mat",
        M_red: "PETSc.Mat",
        a0: float,
        K_geometric: Optional["PETSc.Mat"] = None,
    ) -> "PETSc.Mat":
        """
        Compute effective stiffness matrix for Newmark-beta integration.

        The effective stiffness matrix is:

            K_eff = K + a0 * M + K_g

        where:
        - K: Material stiffness matrix
        - a0 = 1 / (beta * dt^2): Newmark coefficient
        - M: Mass matrix
        - K_g: Geometric stiffness matrix (optional, for centrifugal stiffening)

        This method supports dynamic updates when enable_dynamic_stiffness=True,
        which is required for:
        - Geometric stiffening (centrifugal stress effects on frequencies)
        - Nonlinear material behavior
        - Contact problems

        Parameters
        ----------
        K_red : PETSc.Mat
            Reduced material stiffness matrix.
        M_red : PETSc.Mat
            Reduced mass matrix.
        a0 : float
            Newmark-beta coefficient (1 / (beta * dt^2)).
        K_geometric : PETSc.Mat, optional
            Geometric stiffness matrix for stress stiffening effects.
            If None, only K + a0*M is computed.

        Returns
        -------
        K_eff : PETSc.Mat
            Effective stiffness matrix ready for solver.

        Notes
        -----
        When enable_dynamic_stiffness=False (default), this is called once
        at the start of solve(). When enabled, it's called every
        stiffness_update_interval time steps.

        **Geometric Stiffening** (centrifugal prestress) is implemented via
        `_assemble_geometric_stiffness()`, which computes K_G based on
        centrifugal membrane stresses (σ ∝ ρ·ω²·r). The K_geometric parameter
        accepts this pre-assembled matrix.

        For Rayleigh damping (C = alpha*M + beta*K):
            K_eff = K + a0*M + a1*C = (1 + a1*beta)*K + (a0 + a1*alpha)*M
        where a1 = gamma / (beta * dt)
        """
        # Get Rayleigh damping coefficients
        alpha, beta_damp = self._get_rayleigh_coefficients()
        enable_damping = self._config.enable_structural_damping and (alpha > 0 or beta_damp > 0)

        # Start with copy of reduced stiffness
        K_eff = K_red.duplicate()
        K_eff.copy(K_red)

        if enable_damping:
            # With Rayleigh damping: K_eff = (1 + a1*beta)*K + (a0 + a1*alpha)*M
            # a1 = gamma / (beta_newmark * dt)
            gamma = self.solver_params.get("gamma", 0.5)
            beta_newmark = self.solver_params.get("beta", 0.25)
            a1 = gamma / (beta_newmark * self.dt)

            coef_K = 1.0 + a1 * beta_damp
            coef_M = a0 + a1 * alpha

            K_eff.scale(coef_K)
            K_eff.axpy(coef_M, M_red)

            self._logger.debug(
                f"Rayleigh damping active: alpha={alpha:.4e}, beta={beta_damp:.4e}, "
                f"coef_K={coef_K:.6f}, coef_M={coef_M:.4e}"
            )
        else:
            # No damping: K_eff = K + a0*M
            K_eff.axpy(a0, M_red)

        # Add geometric stiffness if provided (for centrifugal stiffening)
        if K_geometric is not None:
            K_eff.axpy(1.0, K_geometric)

        return K_eff

    def _get_rayleigh_coefficients(self) -> Tuple[float, float]:
        """
        Get Rayleigh damping coefficients (alpha, beta).

        If use_damping_ratios is True, computes alpha and beta from the
        specified damping ratios at two frequencies:

            ζ = alpha/(2*ω) + beta*ω/2

        Solving for two frequencies gives:
            alpha = 2*ω1*ω2*(ζ2*ω1 - ζ1*ω2) / (ω1² - ω2²)
            beta = 2*(ζ1*ω1 - ζ2*ω2) / (ω1² - ω2²)

        Returns
        -------
        alpha : float
            Mass-proportional damping coefficient [1/s].
        beta : float
            Stiffness-proportional damping coefficient [s].
        """
        if not self._config.enable_structural_damping:
            return 0.0, 0.0

        if not self._config.use_damping_ratios:
            return self._config.rayleigh_alpha, self._config.rayleigh_beta

        # Compute from damping ratios at two frequencies
        omega_1 = 2 * np.pi * self._config.damping_freq_1
        omega_2 = 2 * np.pi * self._config.damping_freq_2
        zeta_1 = self._config.damping_ratio_1
        zeta_2 = self._config.damping_ratio_2

        denom = omega_1**2 - omega_2**2

        alpha = 2 * omega_1 * omega_2 * (zeta_2 * omega_1 - zeta_1 * omega_2) / denom
        beta = 2 * (zeta_1 * omega_1 - zeta_2 * omega_2) / denom

        # Ensure non-negative coefficients (can happen with certain ratio combinations)
        alpha = max(0.0, alpha)
        beta = max(0.0, beta)

        return alpha, beta

    def _update_solver_operators(self, K_eff: "PETSc.Mat") -> None:
        """
        Update solver with new effective stiffness matrix.

        This method handles the solver operator update when K_eff changes
        during dynamic stiffness updates.

        Parameters
        ----------
        K_eff : PETSc.Mat
            New effective stiffness matrix.

        Notes
        -----
        For iterative solvers (CG, GMRES), updating operators may require
        rebuilding preconditioners. This method ensures proper handling.
        """
        self._solver.setOperators(K_eff)
        # Force preconditioner rebuild for iterative solvers
        self._solver.setConvergenceHistory()

    def _assemble_geometric_stiffness(self, bc_manager: "BoundaryConditionManager") -> "PETSc.Mat":
        """
        Assemble geometric stiffness matrix from centrifugal prestress.

        Computes the geometric stiffness matrix K_G that accounts for the
        stress stiffening effect due to centrifugal forces in a rotating
        structure. This matrix modifies the effective frequencies and mode
        shapes of the rotor blades.

        The geometric stiffness is computed from the centrifugal prestress:

            σ_c = ρ * ω² * r

        where:
        - ρ: Material density
        - ω: Angular velocity (rad/s)
        - r: Distance from rotation axis

        The assembled K_G is then reduced by eliminating constrained DOFs.

        Parameters
        ----------
        bc_manager : BoundaryConditionManager
            Boundary condition manager for DOF reduction.

        Returns
        -------
        K_G_red : PETSc.Mat
            Reduced geometric stiffness matrix.

        Notes
        -----
        This method uses the MeshAssembler's `assemble_geometric_stiffness()`
        method with centrifugal prestress parameters.

        The geometric stiffening effect:
        - Increases natural frequencies (stiffening)
        - For in-plane tension: adds positive stiffness
        - For compression: can lead to buckling (negative eigenvalues)

        References
        ----------
        Ko, Yongsu, Lee, Phill-Seung, & Bathe, Klaus-Jürgen. (2017).
        "A new MITC4+ shell element."
        Computers & Structures, 182, 404-418.
        """
        # Get rotation parameters from rotor configuration
        # Convert axis string ('x', 'y', 'z') to unit vector
        axis_vectors = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        rotation_axis = axis_vectors.get(self._rotor_axis, np.array([1.0, 0.0, 0.0]))
        rotation_center = np.array(self._rotor_center, dtype=np.float64)
        omega = self._omega  # Current angular velocity

        # Assemble full geometric stiffness matrix using centrifugal prestress
        K_G_full = self.domain.assemble_geometric_stiffness(
            omega=omega,
            rotation_axis=rotation_axis,
            rotation_center=rotation_center,
        )

        # Apply boundary conditions (reduce to free DOFs)
        K_G_red = bc_manager.reduce_matrix(K_G_full)

        self._logger.debug(
            f"Assembled geometric stiffness: ω={omega:.4f} rad/s, "
            f"axis={rotation_axis}, center={rotation_center}"
        )

        return K_G_red

    def _setup_solver(self) -> None:
        """Configure PETSc linear solver using configurable tolerances."""
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
        """
        Convert mass matrix to lumped (diagonal) form with validation.

        Uses row-sum lumping by default, with validation for positive masses.
        If negative masses are detected with higher-order elements,
        falls back to HRZ (Hinton-Rock-Zienkiewicz) lumping.

        Parameters
        ----------
        M : PETSc.Mat
            Consistent mass matrix to lump.

        Returns
        -------
        M_lumped : PETSc.Mat
            Lumped (diagonal) mass matrix with validated positive entries.

        Notes
        -----
        Row-sum lumping can produce negative masses with higher-order elements.
        HRZ lumping scales diagonal terms to preserve total mass while ensuring
        positivity: M_ii = (total_mass / n_nodes) if M_ii < 0.

        References
        ----------
        - Hinton, E., Rock, T., and Zienkiewicz, O. C. "A note on mass lumping
          and related processes in the finite element method."
          Earthquake Engineering & Structural Dynamics, 1976.
        """
        import warnings

        # Step 1: Standard row-sum lumping
        diag = PETSc.Vec().createMPI(M.getSize()[0], comm=M.getComm())
        M.getRowSum(diag)

        # Step 2: Validate masses are positive
        diag_array = diag.getArray()
        negative_count = np.sum(diag_array < 0)
        zero_count = np.sum(np.abs(diag_array) < self._EPS)
        total_nodes = len(diag_array)

        self._logger.debug(
            f"Mass lumping validation: {total_nodes} nodes, "
            f"{negative_count} negative masses, {zero_count} near-zero masses"
        )

        # Step 3: Handle negative masses if found
        if negative_count > 0:
            warnings.warn(
                f"Row-sum lumping produced {negative_count} negative masses "
                f"out of {total_nodes} nodes ({100 * negative_count / total_nodes:.1f}%). "
                f"This can occur with higher-order elements. Applying HRZ lumping correction.",
                UserWarning,
                stacklevel=2,
            )
            self._logger.warning(
                f"Negative masses detected: {negative_count}/{total_nodes} nodes. "
                f"Applying HRZ lumping correction."
            )

            # Apply HRZ lumping correction
            total_mass = np.sum(diag_array)
            uniform_mass = total_mass / total_nodes

            # Replace negative/zero masses with uniform distribution
            negative_mask = diag_array <= self._EPS
            diag_array[negative_mask] = uniform_mass

            # Renormalize to preserve total mass
            corrected_total = np.sum(diag_array)
            if corrected_total > self._EPS:
                diag_array *= total_mass / corrected_total

            # Update PETSc vector
            diag.setArray(diag_array)

            self._logger.info(
                f"HRZ lumping applied: corrected {negative_count} negative masses. "
                f"Total mass preserved: {total_mass:.6e} kg"
            )
        else:
            self._logger.debug(
                f"Row-sum lumping successful: all masses positive. "
                f"Total mass: {np.sum(diag_array):.6e} kg"
            )

        # Step 4: Final validation
        final_diag = diag.getArray()
        min_mass = np.min(final_diag)
        max_mass = np.max(final_diag)

        if min_mass <= 0:
            raise RuntimeError(
                f"Critical error: lumped mass matrix contains non-positive entries. "
                f"Min mass: {min_mass:.6e} kg. This indicates severe mesh issues "
                f"or numerical problems."
            )

        # Log mass statistics
        self._logger.debug(
            f"Lumped mass statistics: min={min_mass:.6e} kg, "
            f"max={max_mass:.6e} kg, ratio={max_mass / min_mass:.2e}"
        )

        # Step 5: Create lumped matrix
        M_lumped = PETSc.Mat().createAIJ(size=M.getSize(), comm=M.getComm())
        M_lumped.setDiagonal(diag)
        M_lumped.assemble()

        # Clean up
        diag.destroy()

        return M_lumped

    def _initialize_simulation(self) -> BoundaryConditionManager:
        """Initialize simulation: assembly, inertia, geometry, BCs, preCICE."""
        self._logger.info("\n" + "═" * 70)
        self._logger.info("  FSI ROTOR SOLVER - STRUCTURAL DYNAMICS")
        self._logger.info("═" * 70)

        # Assembly phase
        self._logger.info("  [1/5] Assembling stiffness matrix...")
        self.K = self.domain.assemble_stiffness_matrix()

        self._logger.info("  [2/5] Assembling mass matrix (lumped)...")
        self.M = self.domain.assemble_mass_matrix()
        self.M = self._lump_mass_matrix(self.M)

        # Compute rotor moment of inertia
        self._logger.info(f"  [2b/5] Computing rotor inertia (mode: {self._inertia_mode})...")
        self._logger.info(f"        Omega mode: {self._omega_mode}")
        if self._load_torque_mode != "none":
            load_info = f"{self._load_torque_mode}"
            if self._load_torque_mode == "constant":
                load_info += f" ({self._load_torque_value:.2e} N·m)"
            elif self._load_torque_mode == "rated_power":
                load_info += f" ({self._rated_power:.2e} W)"
            self._logger.info(f"        Load torque: {load_info}")

        if self._enable_centrifugal or self._enable_gravity or self._enable_coriolis:
            inertial_info = []
            if self._enable_centrifugal:
                inertial_info.append("centrifugal")
            if self._enable_gravity:
                inertial_info.append(f"gravity {self._gravity_vector.tolist()}")
            if self._enable_coriolis:
                inertial_info.append("coriolis")
            self._logger.info(f"        Inertial forces: {', '.join(inertial_info)}")

        self._I_rotor = self._get_total_inertia()

        # Compute rotor geometry from mesh
        self._logger.info("  [2c/5] Computing rotor geometry from mesh...")
        self._rotor_radius = self._compute_rotor_radius()
        self._rotor_area = np.pi * self._rotor_radius**2
        self._logger.info(f"        Rotor radius: {self._rotor_radius:.4f} m")
        self._logger.info(f"        Rotor area:   {self._rotor_area:.4f} m²")

        # Initialize rotor logger
        if self._rotor_log_file:
            self._logger.info(f"  [2d/5] Initializing rotor log: {self._rotor_log_file}")

        self.logger = RotorLogger(
            self._config, self._rotor_radius, self._rotor_area, self._I_rotor, self.solver_params
        )
        self.logger.initialize()

        # Initialize force vector
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

        # Pre-compute interface mapping for optimization
        self._precompute_interface_mapping(bc_manager)

        # Initialize preCICE
        self._logger.info("  [4/5] Initializing preCICE coupling...")
        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )

        self.dt = self.precice_participant.dt

        # Setup solver
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
            f"  Final ω:          {self._omega:.4f} rad/s ({self._omega * 60 / (2 * np.pi):.2f} RPM)"
        )
        self._logger.info(
            f"  Final θ:          {np.degrees(self._theta):.2f}° ({self._theta / (2 * np.pi):.2f} rev)"
        )
        self._logger.info(f"  Final α:          {self._alpha:.4e} rad/s²")
        if self._rotor_log_file:
            self._logger.info(f"  Log file:         {self._rotor_log_file}")
        self._logger.info("═" * 70 + "\n")

        # Close rotor dynamics log
        self.logger.close()

    def solve(self):
        """
        Perform FSI dynamic analysis.

        Main coupling loop:

        1. Read forces from CFD (in Global Inertial Frame)
        2. Transform forces to Local Rotating Frame: F_local = R(-θ)·F_global
        3. Compute inertial forces (centrifugal, Coriolis, gravity) in Local Frame
        4. Sum all forces and assemble global force vector
        5. Solve Newmark-β time integration for structural response
        6. Transform displacements to Global Frame: u_global = R(θ)·u_local
        7. Write displacements to CFD
        8. Update rotational dynamics (ω, θ, α)
        9. Advance preCICE coupling

        Coordinate Frame Transformations
        ---------------------------------
        The structural solver operates in the Local Rotating Frame (blade-attached),
        while the CFD solver operates in the Global Inertial Frame (lab frame).
        Proper coordinate transformations are applied:

        - **Forces**: R(-θ) transforms from Global to Local (inverse rotation)
        - **Displacements**: R(θ) transforms from Local to Global (forward rotation)

        where θ is the current rotor angular position.

        Dynamic Stiffness Update
        ------------------------
        When enable_dynamic_stiffness=True, the effective stiffness matrix K_eff
        is recomputed every stiffness_update_interval time steps. This is required
        for geometric stiffening effects where centrifugal stresses modify the
        structural stiffness.

        The update occurs at the beginning of each time step (before solving)
        to ensure consistency between forces and stiffness.
        """
        # Initialize simulation (Assembly, BCs, preCICE, Logging)
        bc_manager = self._initialize_simulation()

        # Reduced system matrices
        K_red, F_red, M_red = bc_manager.reduced_system

        # Store for potential dynamic updates
        self._K_red = K_red
        self._M_red = M_red

        # Newmark-β coefficients
        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]

        a0 = 1.0 / (beta * self.dt**2)
        a2 = 1.0 / (beta * self.dt)
        a3 = 1.0 / (2 * beta) - 1.0

        # Store coefficients for dynamic updates
        self._a0 = a0

        # Compute initial effective stiffness matrix using new method
        K_eff = self._compute_effective_stiffness(K_red, M_red, a0)
        self._K_eff = K_eff  # Store for potential updates
        self._update_solver_operators(K_eff)

        # Dynamic stiffness configuration
        enable_dynamic_stiffness = self._config.enable_dynamic_stiffness
        stiffness_update_interval = self._config.stiffness_update_interval
        stiffness_update_mode = self._config.stiffness_update_mode
        omega_change_threshold = self._config.omega_change_threshold
        omega_last_update = None  # Track omega at last K_G update (for adaptive mode)

        if enable_dynamic_stiffness:
            mode_info = (
                f"interval={stiffness_update_interval} steps"
                if stiffness_update_mode == "interval"
                else f"adaptive (Δω threshold={omega_change_threshold:.1%})"
            )
            self._logger.info(f"Dynamic stiffness update enabled ({mode_info})")

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

        # Time stepping variables
        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.dt))

        # Get interface coordinates for torque calculation
        interface_coords = self.precice_participant.interface_coordinates
        # Ensure 3D coordinates
        interface_coords = self._ensure_3d_coordinates(interface_coords)

        # Log initial state
        self.logger.log_initial_state(
            self.dt,
            beta,
            gamma,
            self._rotor_center,
            self._rotor_axis,
            self._air_density,
            self._wind_speed,
            self._rotational_damping,
        )

        # Main coupling loop
        while self.precice_participant.is_coupling_ongoing:
            step += 1

            # Checkpoint for implicit coupling
            if self.precice_participant.requires_writing_checkpoint:
                self._logger.debug(f"Step {step}: Writing checkpoint at t = {t:.6f} s")
                # Store structural state AND angular state for implicit coupling
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

            # Dynamic stiffness update for geometric (centrifugal) stiffening
            # Two modes: "interval" (every N steps) or "adaptive" (when omega changes)
            if enable_dynamic_stiffness:
                update_stiffness = False

                if stiffness_update_mode == "interval":
                    # Fixed interval mode: update every N time steps
                    update_stiffness = time_step % stiffness_update_interval == 0
                elif stiffness_update_mode == "adaptive":
                    # Adaptive mode: update when omega changes more than threshold
                    if omega_last_update is None:
                        update_stiffness = True  # First update
                    else:
                        # Relative change in omega
                        if abs(omega_last_update) > 1e-10:
                            omega_change = abs(self._omega - omega_last_update) / abs(
                                omega_last_update
                            )
                        else:
                            omega_change = abs(self._omega - omega_last_update)
                        update_stiffness = omega_change >= omega_change_threshold

                if update_stiffness:
                    # Compute geometric stiffness from centrifugal prestress
                    K_geometric = self._assemble_geometric_stiffness(bc_manager)

                    # Recompute effective stiffness: K_eff = K + a0*M + K_G
                    K_eff = self._compute_effective_stiffness(
                        self._K_red, self._M_red, self._a0, K_geometric
                    )
                    self._K_eff = K_eff
                    self._update_solver_operators(K_eff)
                    omega_last_update = self._omega
                    self._logger.debug(
                        f"Step {step}: Updated K_eff with geometric stiffness (ω = {self._omega:.4f} rad/s)"
                    )

            # Read forces from CFD (Aerodynamic Forces)
            # Forces from CFD are in Global Inertial Frame.
            # Transform to Local Rotating Frame: forces_local = R(-θ) · forces_global
            data = self.precice_participant.read_data()
            interface_dofs = self.precice_participant.interface_dofs

            # Reshape forces to (n_nodes, 3)
            if data.ndim == 1:
                forces_global = data.reshape(-1, 3)
            else:
                forces_global = self._ensure_3d_coordinates(data)

            # Transform forces from Global Inertial Frame to Local Rotating Frame
            forces_aero = self._transform_vectors_to_rotating_frame(forces_global)

            # Update Geometry for Torque Calculation
            # Extract current displacements at interface nodes
            u_full = bc_manager.expand_solution(u).array

            # Extract u at interface nodes using precomputed mapping if available, else direct
            # (For now we use direct extraction as in original, but we could optimize)
            u_interface_flat = u_full[interface_dofs]
            u_interface = u_interface_flat.reshape(-1, 3)

            # Deformed coordinates: r_current = r_initial + u
            current_coords = interface_coords + u_interface

            # Compute Inertial Forces (Centrifugal + Gravity)
            forces_inertial = np.zeros_like(forces_aero)
            if self._enable_centrifugal or self._enable_gravity:
                forces_inertial = self._compute_inertial_forces(current_coords)

            # Compute Coriolis Forces
            forces_coriolis = np.zeros_like(forces_aero)
            if self._enable_coriolis and step > 1:
                # Get nodal masses
                if self._cached_nodal_masses is None:
                    self._cached_nodal_masses = self._get_nodal_masses()

                # Extract interface velocities
                interface_velocities = self._extract_interface_velocities(v)

                F_coriolis = self._compute_coriolis_forces(
                    current_coords,
                    self._cached_nodal_masses[: len(interface_coords)],
                    interface_velocities,
                )
                forces_coriolis = F_coriolis

            # Compute Euler Forces (angular acceleration dependent)
            forces_euler = np.zeros_like(forces_aero)
            if self._enable_euler and step > 1:
                # Get nodal masses
                if self._cached_nodal_masses is None:
                    self._cached_nodal_masses = self._get_nodal_masses()

                F_euler = self._compute_euler_forces(
                    current_coords,
                    self._cached_nodal_masses[: len(interface_coords)],
                )
                forces_euler = F_euler

            # Total forces for FEM
            forces_total = forces_aero + forces_inertial + forces_coriolis + forces_euler

            # Force statistics (Total)
            force_x = np.sum(forces_total[:, 0])
            force_y = np.sum(forces_total[:, 1])
            force_z = np.sum(forces_total[:, 2])
            force_mag = np.sqrt(force_x**2 + force_y**2 + force_z**2)

            # Torque Calculation (Separated)
            # 1. Aerodynamic Torque (Driving) - Uses deformed geometry
            torque_aero, _ = self._compute_torque(current_coords, forces_aero)

            # 2. Inertial Torque (Parasitic/Driving) - Uses deformed geometry
            # Includes centrifugal, gravity, Coriolis, and Euler forces
            torque_inertial, _ = self._compute_torque(
                current_coords, forces_inertial + forces_coriolis + forces_euler
            )

            # 3. Net Torque for Dynamics
            torque_net = torque_aero + torque_inertial

            # Update angular state
            self._update_angular_state(torque_net, self.dt)

            # Compute aerodynamic coefficients (thrust, power, Cp, Cq, Ct, TSR)
            self._compute_aerodynamic_coefficients(torque_aero, forces_aero)

            t_target = t + self.dt

            # Log timestep info
            self.logger.log_timestep(
                t=t_target,
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

            # Assemble global force vector
            self.F.set(0.0)
            self.domain.apply_nodal_forces(self.F, interface_dofs, forces_total.flatten())

            # Apply boundary conditions to force vector
            bc_manager.apply_neumann(self.F)
            F_red_new = bc_manager.reduce_vector(self.F)

            # Effective load calculation
            # Without damping: R_eff = F + M * (a0*u + a2*v + a3*a)
            # With Rayleigh damping (C = alpha*M + beta*K):
            #   R_eff = F + M * (a0*u + a2*v + a3*a) + C * (a1*u + a4*v + a5*a)
            # where a4 = gamma/beta - 1, a5 = dt*(gamma/(2*beta) - 1)

            predictor = u.duplicate()
            predictor.copy(u)
            predictor.scale(a0)
            predictor.axpy(a2, v)
            predictor.axpy(a3, a)

            R_eff = F_red_new.duplicate()
            M_red.mult(predictor, R_eff)
            R_eff.axpy(1.0, F_red_new)

            # Add Rayleigh damping contribution if enabled
            alpha_damp, beta_damp = self._get_rayleigh_coefficients()
            if self._config.enable_structural_damping and (alpha_damp > 0 or beta_damp > 0):
                # Newmark coefficients for damping
                a1 = gamma / (beta * self.dt)
                a4 = gamma / beta - 1.0
                a5 = self.dt * (gamma / (2.0 * beta) - 1.0)

                # Damping predictor: a1*u + a4*v + a5*a
                damping_predictor = u.duplicate()
                damping_predictor.copy(u)
                damping_predictor.scale(a1)
                damping_predictor.axpy(a4, v)
                damping_predictor.axpy(a5, a)

                # C = alpha*M + beta*K, so C*predictor = alpha*M*predictor + beta*K*predictor
                temp_vec = damping_predictor.duplicate()

                # alpha*M contribution
                if alpha_damp > 0:
                    M_red.mult(damping_predictor, temp_vec)
                    R_eff.axpy(alpha_damp, temp_vec)

                # beta*K contribution
                if beta_damp > 0:
                    self._K_red.mult(damping_predictor, temp_vec)
                    R_eff.axpy(beta_damp, temp_vec)

                temp_vec.destroy()
                damping_predictor.destroy()

            # Solve for displacement
            self._solver.solve(R_eff, u)

            # Update velocity and acceleration
            # a_new = a0 * (u_new - u_old) - a2 * v_old - a3 * a_old
            # v_new = v_old + dt * ((1-gamma)*a_old + gamma*a_new)

            # Calculate new acceleration
            a_new = u.duplicate()
            a_new.copy(u)
            a_new.scale(a0)
            a_new.axpy(-1.0, predictor)  # predictor contains a0*u_old + a2*v_old + a3*a_old

            # Calculate new velocity
            v_new = v.duplicate()
            v_new.copy(v)

            # v_new = v_old + dt * (1-gamma) * a_old
            v_new.axpy(self.dt * (1.0 - gamma), a)

            # v_new += dt * gamma * a_new
            v_new.axpy(self.dt * gamma, a_new)

            # Update state vectors
            v.copy(v_new)
            a.copy(a_new)

            # Write displacements to CFD
            # Displacements are computed in Local Rotating Frame.
            # Transform to Global Inertial Frame: u_global = R(θ) · u_local
            u_full = bc_manager.expand_solution(u).array
            u_interface_flat = u_full[interface_dofs]
            u_interface_local = u_interface_flat.reshape(-1, 3)

            # Transform displacements from Local Rotating Frame to Global Inertial Frame
            u_interface_global = self._transform_vectors_to_inertial_frame(u_interface_local)

            # Write data to preCICE (Global Inertial Frame)
            self.precice_participant.write_data(u_interface_global.flatten())

            # Advance coupling
            self.precice_participant.advance(self.dt)

            # Check if we need to read a checkpoint (implicit coupling loop)
            if self.precice_participant.requires_reading_checkpoint:
                self._logger.debug(f"Step {step}: Reading checkpoint")
                # Restore structural state AND angular state
                checkpoint_data = self.precice_participant.read_checkpoint()
                checkpoint_state = CheckpointState.from_tuple(checkpoint_data)

                # CRITICAL: Copy values FROM checkpoint TO working vectors.
                # Do NOT assign (u = cp.u) because 'u' is modified in place during solve,
                # which would corrupt the stored checkpoint if we held a reference to it.
                checkpoint_state.u.copy(u)
                checkpoint_state.v.copy(v)
                checkpoint_state.a.copy(a)

                # Restore scalars
                t = checkpoint_state.t
                self._omega = checkpoint_state.omega
                self._theta = checkpoint_state.theta
                self._alpha = checkpoint_state.alpha
            else:
                # Time step accepted
                t += self.dt
                time_step += 1

                # Log to file
                self.logger.write_to_file(
                    t,
                    self._omega,
                    self._theta,
                    self._alpha,
                    torque_aero,
                    torque_inertial,
                    torque_net,
                    force_x,
                    force_y,
                    force_z,
                )

        self._finalize_simulation(t, step, time_step)
