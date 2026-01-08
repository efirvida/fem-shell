"""
Generator/gearbox load torque controller.

This module encapsulates the load torque calculation logic for wind turbine
generator control strategies.
"""

import numpy as np

from .config import LoadTorqueMode, RotorConfig


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

    Attributes
    ----------
    mode : LoadTorqueMode
        Current operating mode.
    constant_torque : float
        Constant torque value [N·m].
    torque_coeff : float
        Torque coefficient for speed-dependent modes.
    rated_power : float
        Rated power for constant-power mode [W].
    omega_min : float
        Minimum angular velocity for regularization [rad/s].
    tau_max : float
        Maximum allowable torque [N·m].

    Examples
    --------
    Create a controller for optimal TSR tracking (Region 2)::

        >>> controller = GeneratorController(
        ...     mode=LoadTorqueMode.QUADRATIC,
        ...     torque_coeff=2.5e4
        ... )
        >>> tau_load = controller.compute_torque(omega=1.2)  # Returns k*omega^2

    Create a controller for rated power operation (Region 3)::

        >>> controller = GeneratorController(
        ...     mode=LoadTorqueMode.RATED_POWER,
        ...     rated_power=5e6,  # 5 MW
        ...     omega_min=0.5,
        ...     tau_max=5e6,
        ... )
        >>> tau_load = controller.compute_torque(omega=1.26)
    """

    #: Default minimum angular velocity for regularization [rad/s]
    OMEGA_MIN_DEFAULT: float = 0.1

    def __init__(
        self,
        mode: LoadTorqueMode,
        constant_torque: float = 0.0,
        torque_coeff: float = 0.0,
        rated_power: float = 0.0,
        omega_min: float = 0.1,
        tau_max: float = 1e6,
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

        The torque model depends on the operating mode:

        - **NONE**: τ_load = 0
        - **CONSTANT**: τ_load = τ₀
        - **LINEAR**: τ_load = k·|ω|
        - **QUADRATIC**: τ_load = k·ω·|ω| (preserves sign)
        - **RATED_POWER**: τ_load = min(P/√(ω² + ω_min²), τ_max)

        Parameters
        ----------
        omega : float
            Current angular velocity [rad/s].

        Returns
        -------
        tau_load : float
            Load torque [N·m]. For most modes, returns non-negative magnitude.
            For QUADRATIC mode, can be negative to oppose rotation direction.

        Notes
        -----
        The QUADRATIC mode uses τ = k·ω·|ω| instead of k·ω² to preserve the
        sign of omega, ensuring the torque always opposes the direction of
        rotation (physically correct generator behavior).

        The RATED_POWER mode uses smooth regularization √(ω² + ω_min²) instead
        of max(|ω|, ω_min) to avoid discontinuities in the torque derivative,
        which improves numerical stability in coupled simulations.
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
        """
        Create GeneratorController from RotorConfig.

        Parameters
        ----------
        config : RotorConfig
            Rotor configuration object.

        Returns
        -------
        GeneratorController
            Configured generator controller.
        """
        return cls(
            mode=LoadTorqueMode(config.load_torque_mode),
            constant_torque=config.load_torque_value,
            torque_coeff=config.load_torque_coeff,
            rated_power=config.rated_power,
            omega_min=config.omega_min_rated_power,
            tau_max=config.torque_max_rated_power,
        )

    def __repr__(self) -> str:
        return (
            f"GeneratorController(mode={self.mode.value}, "
            f"constant_torque={self.constant_torque:.2e}, "
            f"torque_coeff={self.torque_coeff:.2e}, "
            f"rated_power={self.rated_power:.2e})"
        )
