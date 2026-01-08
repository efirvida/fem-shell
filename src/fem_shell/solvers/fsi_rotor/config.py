"""
Configuration classes and enums for FSI Rotor Solver.

This module contains all configuration-related classes:
- Enums for operating modes (LoadTorqueMode, OmegaMode, InertiaMode)
- RotorConfig dataclass with comprehensive parameter validation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LoadTorqueMode(Enum):
    """
    Available load torque modes for generator control.

    Attributes
    ----------
    NONE : str
        No load torque (free rotation).
    CONSTANT : str
        Constant braking torque τ_load = τ₀.
    LINEAR : str
        Linear with speed τ_load = k·ω.
    QUADRATIC : str
        Quadratic (optimal TSR tracking) τ_load = k·ω².
    RATED_POWER : str
        Constant power extraction τ_load = P_rated/ω.
    """

    NONE = "none"
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    RATED_POWER = "rated_power"


class OmegaMode(Enum):
    """
    Angular velocity control modes.

    Attributes
    ----------
    DYNAMIC : str
        Angular velocity evolves from torque balance (FSI coupling).
    CONSTANT : str
        Angular velocity stays fixed at initial_omega.
    """

    DYNAMIC = "dynamic"
    CONSTANT = "constant"


class InertiaMode(Enum):
    """
    Moment of inertia specification modes.

    Attributes
    ----------
    TOTAL : str
        Use rotor_inertia directly.
    HUB_PLUS_BLADES : str
        I = hub_inertia + I_blades (computed from mesh).
    FRACTION : str
        I = I_blades × (1 + hub_fraction).
    """

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

    # =========================================================================
    # Geometry
    # =========================================================================
    rotor_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotor_axis: str = "x"

    # =========================================================================
    # Angular velocity
    # =========================================================================
    omega_mode: str = "dynamic"
    initial_omega: float = 0.0
    rotational_damping: float = 0.0

    # =========================================================================
    # Inertia
    # =========================================================================
    inertia_mode: str = "total"
    rotor_inertia: Optional[float] = None
    hub_inertia: float = 0.0
    hub_fraction: float = 0.1

    # =========================================================================
    # Load torque
    # =========================================================================
    load_torque_mode: str = "none"
    load_torque_value: float = 0.0
    load_torque_coeff: float = 0.0
    rated_power: float = 0.0
    omega_min_rated_power: float = 0.1
    torque_max_rated_power: float = 1e6

    # =========================================================================
    # Inertial forces
    # =========================================================================
    enable_centrifugal: bool = False
    enable_gravity: bool = False
    enable_coriolis: bool = False
    enable_euler: bool = False
    coriolis_use_rotating_frame: bool = True
    gravity_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    # =========================================================================
    # Aerodynamic
    # =========================================================================
    air_density: float = 1.225
    wind_speed: Optional[float] = None

    # =========================================================================
    # Logging
    # =========================================================================
    rotor_log_file: Optional[str] = None
    rotor_log_separator: str = ","

    # =========================================================================
    # Solver tolerances and parameters
    # =========================================================================
    solver_rtol_large: float = 1e-5
    solver_atol_large: float = 1e-8
    solver_rtol_small: float = 1e-8
    solver_atol_small: float = 1e-12
    solver_rtol_mass: float = 1e-12
    solver_shift_amount: float = 1e-10
    solver_max_it_large: int = 1000
    solver_max_it_small: int = 500
    solver_dofs_threshold: int = 10000

    # =========================================================================
    # Dynamic stiffness update
    # =========================================================================
    enable_dynamic_stiffness: bool = False
    stiffness_update_interval: int = 1
    stiffness_update_mode: str = "interval"
    omega_change_threshold: float = 0.01

    # =========================================================================
    # Structural damping (Rayleigh)
    # =========================================================================
    enable_structural_damping: bool = False
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 0.0
    use_damping_ratios: bool = False
    damping_ratio_1: float = 0.01
    damping_ratio_2: float = 0.01
    damping_freq_1: float = 1.0
    damping_freq_2: float = 10.0

    def __post_init__(self):
        """Validate all configuration parameters."""
        self._validate()

    def _validate(self):
        """Perform comprehensive validation of all parameters."""
        self._validate_axis()
        self._validate_modes()
        self._validate_numerical_params()
        self._validate_load_torque()
        self._validate_damping()
        self._validate_vectors()
        self._log_physics_warnings()

    def _validate_axis(self):
        """Validate rotor axis parameter."""
        valid_axes = ("x", "y", "z")
        if self.rotor_axis.lower() not in valid_axes:
            raise ValueError(
                f"Invalid rotor_axis: '{self.rotor_axis}'. Must be one of {valid_axes}."
            )
        self.rotor_axis = self.rotor_axis.lower()

    def _validate_modes(self):
        """Validate operation mode parameters."""
        # Omega mode
        try:
            OmegaMode(self.omega_mode.lower())
            self.omega_mode = self.omega_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in OmegaMode]
            raise ValueError(
                f"Invalid omega_mode: '{self.omega_mode}'. Must be one of {valid_modes}."
            )

        # Inertia mode
        try:
            InertiaMode(self.inertia_mode.lower())
            self.inertia_mode = self.inertia_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in InertiaMode]
            raise ValueError(
                f"Invalid inertia_mode: '{self.inertia_mode}'. Must be one of {valid_modes}."
            )

        # Load torque mode
        try:
            LoadTorqueMode(self.load_torque_mode.lower())
            self.load_torque_mode = self.load_torque_mode.lower()
        except ValueError:
            valid_modes = [m.value for m in LoadTorqueMode]
            raise ValueError(
                f"Invalid load_torque_mode: '{self.load_torque_mode}'. Must be one of {valid_modes}."
            )

        # Stiffness update mode
        valid_stiffness_modes = ("interval", "adaptive")
        if self.stiffness_update_mode.lower() not in valid_stiffness_modes:
            raise ValueError(
                f"Invalid stiffness_update_mode: '{self.stiffness_update_mode}'. "
                f"Must be one of {valid_stiffness_modes}."
            )
        self.stiffness_update_mode = self.stiffness_update_mode.lower()

    def _validate_numerical_params(self):
        """Validate numerical parameters."""
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

        if self.omega_change_threshold <= 0 or self.omega_change_threshold > 1:
            raise ValueError(
                f"omega_change_threshold must be in (0, 1], got {self.omega_change_threshold}"
            )

        # Solver tolerances
        positive_params = [
            ("solver_rtol_large", self.solver_rtol_large),
            ("solver_atol_large", self.solver_atol_large),
            ("solver_rtol_small", self.solver_rtol_small),
            ("solver_atol_small", self.solver_atol_small),
            ("solver_rtol_mass", self.solver_rtol_mass),
            ("solver_shift_amount", self.solver_shift_amount),
            ("solver_max_it_large", self.solver_max_it_large),
            ("solver_max_it_small", self.solver_max_it_small),
            ("solver_dofs_threshold", self.solver_dofs_threshold),
        ]
        for name, value in positive_params:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def _validate_load_torque(self):
        """Validate load torque parameters."""
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

        if self.wind_speed is not None and self.wind_speed < 0:
            raise ValueError(f"wind_speed must be non-negative, got {self.wind_speed}")

    def _validate_damping(self):
        """Validate Rayleigh damping parameters."""
        if not self.enable_structural_damping:
            return

        if self.use_damping_ratios:
            if self.damping_ratio_1 < 0 or self.damping_ratio_1 > 1:
                raise ValueError(f"damping_ratio_1 must be in [0, 1], got {self.damping_ratio_1}")
            if self.damping_ratio_2 < 0 or self.damping_ratio_2 > 1:
                raise ValueError(f"damping_ratio_2 must be in [0, 1], got {self.damping_ratio_2}")
            if self.damping_freq_1 <= 0:
                raise ValueError(f"damping_freq_1 must be positive, got {self.damping_freq_1}")
            if self.damping_freq_2 <= 0:
                raise ValueError(f"damping_freq_2 must be positive, got {self.damping_freq_2}")
            if abs(self.damping_freq_1 - self.damping_freq_2) < 1e-6:
                raise ValueError("damping_freq_1 and damping_freq_2 must be different")
        else:
            if self.rayleigh_alpha < 0:
                raise ValueError(f"rayleigh_alpha must be non-negative, got {self.rayleigh_alpha}")
            if self.rayleigh_beta < 0:
                raise ValueError(f"rayleigh_beta must be non-negative, got {self.rayleigh_beta}")

    def _validate_vectors(self):
        """Validate vector dimensions."""
        if len(self.rotor_center) != 3:
            raise ValueError(f"rotor_center must have 3 components, got {len(self.rotor_center)}")

        if len(self.gravity_vector) != 3:
            raise ValueError(
                f"gravity_vector must have 3 components, got {len(self.gravity_vector)}"
            )

    def _log_physics_warnings(self):
        """Log warnings for physics configurations."""
        if self.enable_centrifugal:
            if self.enable_dynamic_stiffness:
                logger.info(
                    "Centrifugal forces and geometric stiffening enabled. "
                    "Natural frequencies will increase with rotor speed (Campbell diagram effect)."
                )
            else:
                logger.warning(
                    "Centrifugal forces enabled but geometric stiffening disabled. "
                    "Natural frequencies will NOT increase with rotor speed. "
                    "Set enable_dynamic_stiffness=True for complete physics."
                )

        if self.enable_coriolis:
            logger.info(
                "Coriolis forces enabled. These velocity-dependent forces will be computed "
                "at each timestep based on nodal velocities."
            )

        if self.enable_euler:
            logger.info(
                "Euler forces enabled. These angular acceleration-dependent forces will be "
                "computed at each timestep when omega_mode='dynamic'. "
                "F_Euler = -m·(α × r) where α = dω/dt."
            )

    @classmethod
    def from_solver_params(cls, solver_params: Dict) -> "RotorConfig":
        """
        Create RotorConfig from solver parameters dictionary.

        Parameters
        ----------
        solver_params : Dict
            Dictionary containing solver configuration parameters.

        Returns
        -------
        RotorConfig
            Validated configuration object.
        """
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
