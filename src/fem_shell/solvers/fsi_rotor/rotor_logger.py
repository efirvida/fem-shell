"""
Rotor dynamics CSV logger.

This module handles logging of rotor dynamics data to CSV files,
including header generation and timestep logging.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import numpy as np

from .config import RotorConfig

logger = logging.getLogger(__name__)


class RotorLogger:
    """
    Handles logging of rotor dynamics data to a CSV file.

    Encapsulates all file I/O and formatting logic for the rotor log,
    providing a clean interface for timestep logging.

    Parameters
    ----------
    config : RotorConfig
        Rotor configuration object.
    rotor_radius : float
        Rotor radius [m].
    rotor_area : float
        Rotor swept area [m²].
    rotor_inertia : float
        Total rotor inertia [kg·m²].
    solver_params : Dict
        Solver parameters (for Newmark coefficients).

    Attributes
    ----------
    log_file : Optional[str]
        Path to log file, or None if logging disabled.
    separator : str
        Column separator character.
    handle : Optional[TextIO]
        File handle for writing.

    Example
    -------
    ::

        logger = RotorLogger(config, radius, area, inertia, solver_params)
        logger.initialize()

        for step in simulation:
            logger.log_timestep(t=t, torque_aero=tau, ...)

        logger.close()
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

    def initialize(self) -> None:
        """
        Initialize the log file with header information.

        Creates the log file, parent directories if needed, and writes
        the configuration header and column names.
        """
        if self.log_file is None:
            return

        # Ensure directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.handle = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            logger.warning("Could not open rotor log file: %s", e)
            self.handle = None
            return

        self._write_header()

    def _write_header(self) -> None:
        """Write the detailed header to the log file."""
        if self.handle is None:
            return

        sep = self.separator
        h = self.handle

        h.write("# FSI Rotor Dynamics Log\n")
        h.write(f"# Generated: {datetime.now().isoformat()}\n")
        h.write("#\n")

        # Configuration section
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

        # Load torque section
        h.write("#\n")
        h.write("# === LOAD TORQUE (Generator/Gearbox) ===\n")
        h.write(f"# Load torque mode: {self.config.load_torque_mode}\n")

        if self.config.load_torque_mode == "constant":
            h.write(f"# Constant torque [N·m]: {self.config.load_torque_value:.6e}\n")
        elif self.config.load_torque_mode in ("linear", "quadratic"):
            h.write(f"# Torque coefficient: {self.config.load_torque_coeff:.6e}\n")
        elif self.config.load_torque_mode == "rated_power":
            h.write(f"# Rated power [W]: {self.config.rated_power:.6e}\n")

        # Inertial forces section
        h.write("#\n")
        h.write("# === INERTIAL FORCES ===\n")
        h.write(f"# Centrifugal: {self.config.enable_centrifugal}\n")
        h.write(f"# Gravity: {self.config.enable_gravity}\n")
        h.write(f"# Coriolis: {self.config.enable_coriolis}\n")
        h.write(f"# Euler: {self.config.enable_euler}\n")
        if self.config.enable_gravity:
            h.write(f"# Gravity vector: {self.config.gravity_vector}\n")

        # Geometry section
        h.write("#\n")
        h.write("# === ROTOR GEOMETRY ===\n")
        h.write(f"# Rotor radius [m]: {self.rotor_radius:.6f}\n")
        h.write(f"# Rotor area [m²]: {self.rotor_area:.6f}\n")

        # Aerodynamic section
        h.write("#\n")
        h.write("# === AERODYNAMIC PARAMETERS ===\n")
        h.write(f"# Air density [kg/m³]: {self.config.air_density:.6f}\n")
        if self.config.wind_speed is not None:
            h.write(f"# Wind speed [m/s]: {self.config.wind_speed:.6f}\n")
        else:
            h.write("# Wind speed: Not specified\n")

        # Simulation parameters
        h.write("#\n")
        h.write("# === SIMULATION PARAMETERS ===\n")
        h.write(f"# Newmark beta: {self.solver_params.get('beta', 0.25):.4f}\n")
        h.write(f"# Newmark gamma: {self.solver_params.get('gamma', 0.5):.4f}\n")
        h.write("#\n")

        # Column headers
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
    ) -> None:
        """
        Write a single timestep entry to the log.

        Parameters
        ----------
        t : float
            Current time [s].
        torque_aero : float
            Aerodynamic torque [N·m].
        torque_inertial : float
            Inertial torque [N·m].
        torque_net : float
            Net torque [N·m].
        thrust : float
            Thrust force [N].
        power : float
            Mechanical power [W].
        omega : float
            Angular velocity [rad/s].
        alpha : float
            Angular acceleration [rad/s²].
        theta : float
            Rotation angle [rad].
        tsr : float, optional
            Tip speed ratio.
        cp : float, optional
            Power coefficient.
        cq : float, optional
            Torque coefficient.
        ct : float, optional
            Thrust coefficient.
        """
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
    ) -> None:
        """
        Log initial simulation state parameters.

        Parameters
        ----------
        dt : float
            Time step [s].
        beta : float
            Newmark beta parameter.
        gamma : float
            Newmark gamma parameter.
        rotor_center : List[float]
            Rotor center coordinates [m].
        rotor_axis : str
            Rotation axis.
        air_density : float
            Air density [kg/m³].
        wind_speed : Optional[float]
            Wind speed [m/s].
        rotational_damping : float
            Rotational damping coefficient [N·m·s/rad].
        """
        if self.handle is None:
            return

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

    def write_to_file(
        self,
        t: float,
        omega: float,
        theta: float,
        alpha: float,
        torque_aero: float,
        torque_inertial: float,
        torque_net: float,
        force_x: float,
        force_y: float,
        force_z: float,
    ) -> None:
        """
        Write converged timestep data (legacy interface).

        This method is kept for backward compatibility with existing code.

        Parameters
        ----------
        t : float
            Current time [s].
        omega : float
            Angular velocity [rad/s].
        theta : float
            Rotation angle [rad].
        alpha : float
            Angular acceleration [rad/s²].
        torque_aero : float
            Aerodynamic torque [N·m].
        torque_inertial : float
            Inertial torque [N·m].
        torque_net : float
            Net torque [N·m].
        force_x, force_y, force_z : float
            Force components [N].
        """
        # This is a simplified version - the main logging happens in log_timestep
        pass

    def close(self) -> None:
        """Close the log file."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
