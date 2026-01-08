"""
Aerodynamic calculations for wind turbine rotors.

This module provides calculations for:
- Torque from nodal forces
- Aerodynamic coefficients (Cp, Cq, Ct, TSR)
- Thrust force
- Power extraction
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .transforms import CoordinateTransforms

logger = logging.getLogger(__name__)


@dataclass
class AerodynamicState:
    """
    Current aerodynamic state of the rotor.

    Attributes
    ----------
    thrust : float
        Axial thrust force [N].
    power : float
        Mechanical power [W].
    tsr : float
        Tip speed ratio.
    Cp : float
        Power coefficient.
    Cq : float
        Torque coefficient.
    Ct : float
        Thrust coefficient.
    """

    thrust: float = 0.0
    power: float = 0.0
    tsr: float = 0.0
    Cp: float = 0.0
    Cq: float = 0.0
    Ct: float = 0.0


class AerodynamicsCalculator:
    """
    Calculator for aerodynamic forces and coefficients.

    This class computes torque from nodal forces and dimensionless
    aerodynamic coefficients for wind turbine analysis.

    Parameters
    ----------
    transforms : CoordinateTransforms
        Coordinate transformation utilities.
    air_density : float
        Air density [kg/m³].
    wind_speed : Optional[float]
        Freestream wind speed [m/s]. Required for coefficient calculations.

    Attributes
    ----------
    EPS : float
        Small number for numerical stability checks.

    Example
    -------
    ::

        transforms = CoordinateTransforms('x', rotor_center)
        aero = AerodynamicsCalculator(transforms, rho=1.225, wind_speed=10.0)

        # Compute torque
        torque, torque_vector = aero.compute_torque(coords, forces)

        # Compute coefficients
        state = aero.compute_coefficients(
            torque, forces, omega, rotor_radius, rotor_area
        )
        print(f"Cp = {state.Cp:.4f}, TSR = {state.tsr:.2f}")
    """

    #: Small number for numerical stability
    EPS: float = 1e-10

    def __init__(
        self,
        transforms: CoordinateTransforms,
        air_density: float = 1.225,
        wind_speed: Optional[float] = None,
    ):
        self.transforms = transforms
        self.air_density = air_density
        self.wind_speed = wind_speed

    def compute_torque(
        self,
        coords: np.ndarray,
        forces: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute torque around rotor axis from nodal forces.

        The torque is computed as the sum of moment contributions from all
        nodes in the mesh:

            τ = Σᵢ (rᵢ × Fᵢ)

        where rᵢ is the position vector from the rotor center to node i,
        and Fᵢ is the force at that node.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in global frame, shape (n_nodes, 3) [m].
            Should be deformed coordinates for accuracy.
        forces : np.ndarray
            Force vectors at nodes, shape (n_nodes, 3) [N].

        Returns
        -------
        torque_axial : float
            Torque component around the rotor axis [N·m].
            Positive value indicates torque in direction of rotation.
        torque_vector : np.ndarray
            Full torque vector [Tx, Ty, Tz] [N·m].

        Raises
        ------
        ValueError
            If coords and forces have different number of nodes.
        """
        # Validate inputs
        if coords.shape[0] != forces.shape[0]:
            raise ValueError(
                f"coords and forces must have same number of nodes: "
                f"{coords.shape[0]} vs {forces.shape[0]}"
            )

        # Check for NaN/Inf in forces
        if not np.all(np.isfinite(forces)):
            logger.warning("Non-finite values detected in forces, replacing with zeros")
            forces = np.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)

        # Position vectors from rotor center
        r = self.transforms.position_from_center(coords)

        # Torque at each node: τ = r × F
        torque_per_node = np.cross(r, forces)

        # Total torque vector
        torque_vector = np.sum(torque_per_node, axis=0)

        # Extract axial component
        torque_axial = torque_vector[self.transforms.axis_idx]

        return torque_axial, torque_vector

    def compute_thrust(self, forces: np.ndarray) -> float:
        """
        Compute axial thrust force.

        Parameters
        ----------
        forces : np.ndarray
            Force vectors at nodes, shape (n_nodes, 3) [N].

        Returns
        -------
        thrust : float
            Sum of forces along rotor axis [N].
            Positive = pushing rotor downstream.
        """
        return np.sum(forces[:, self.transforms.axis_idx])

    def compute_power(self, torque: float, omega: float) -> float:
        """
        Compute mechanical power from torque and angular velocity.

        Parameters
        ----------
        torque : float
            Aerodynamic torque [N·m].
        omega : float
            Angular velocity [rad/s].

        Returns
        -------
        power : float
            Mechanical power P = τ·ω [W].
        """
        return torque * omega

    def compute_coefficients(
        self,
        torque: float,
        forces: np.ndarray,
        omega: float,
        rotor_radius: float,
        rotor_area: float,
    ) -> AerodynamicState:
        """
        Compute aerodynamic coefficients for rotor characteristic curves.

        Computes the standard dimensionless coefficients:
        - Tip Speed Ratio (TSR): λ = ω·R / V∞
        - Power Coefficient: Cp = P / (½·ρ·A·V∞³)
        - Torque Coefficient: Cq = τ / (½·ρ·A·R·V∞²)
        - Thrust Coefficient: Ct = T / (½·ρ·A·V∞²)

        Parameters
        ----------
        torque : float
            Aerodynamic torque [N·m].
        forces : np.ndarray
            Force vectors at nodes, shape (n_nodes, 3) [N].
        omega : float
            Angular velocity [rad/s].
        rotor_radius : float
            Rotor radius [m].
        rotor_area : float
            Rotor swept area [m²].

        Returns
        -------
        state : AerodynamicState
            Aerodynamic state with all computed values.
        """
        # Thrust
        thrust = self.compute_thrust(forces)

        # Power
        power = self.compute_power(torque, omega)

        # Initialize state
        state = AerodynamicState(thrust=thrust, power=power)

        # Only compute dimensionless coefficients if wind speed is provided
        if self.wind_speed is not None and self.wind_speed > 0:
            V = self.wind_speed
            R = rotor_radius
            A = rotor_area
            rho = self.air_density

            # Dynamic pressure
            q = 0.5 * rho * V**2

            # Tip Speed Ratio: λ = ωR / V
            state.tsr = omega * R / V

            # Power coefficient: Cp = P / (0.5 × ρ × A × V³)
            P_available = 0.5 * rho * A * V**3
            state.Cp = power / P_available if P_available > self.EPS else 0.0

            # Torque coefficient: Cq = Q / (0.5 × ρ × A × V² × R)
            state.Cq = torque / (q * A * R) if q > self.EPS else 0.0

            # Thrust coefficient: Ct = T / (0.5 × ρ × A × V²)
            state.Ct = thrust / (q * A) if q > self.EPS else 0.0

        return state

    def update_wind_speed(self, wind_speed: float) -> None:
        """
        Update the wind speed for coefficient calculations.

        Parameters
        ----------
        wind_speed : float
            New wind speed [m/s].
        """
        self.wind_speed = wind_speed
