"""
Inertial force calculations for rotating reference frames.

This module provides calculations for:
- Centrifugal forces (F = m·ω²·r)
- Gravitational forces in rotating frame
- Coriolis forces (F = -2m·ω×v)
- Euler forces (F = -m·α×r)
"""

import logging
from typing import Optional

import numpy as np

from .transforms import CoordinateTransforms

logger = logging.getLogger(__name__)


class InertialForcesCalculator:
    """
    Calculator for inertial forces in rotating reference frames.

    This class computes the fictitious forces that appear in a rotating
    reference frame: centrifugal, Coriolis, Euler, and gravitational forces.

    Parameters
    ----------
    transforms : CoordinateTransforms
        Coordinate transformation utilities.
    gravity_vector : np.ndarray
        Gravity acceleration vector in global frame [m/s²].

    Attributes
    ----------
    EPS : float
        Small number for numerical stability checks.

    Example
    -------
    ::

        transforms = CoordinateTransforms('x', rotor_center)
        calculator = InertialForcesCalculator(transforms, gravity_vector)

        # Compute all inertial forces
        forces = calculator.compute_total_forces(
            coords, nodal_masses, omega, theta,
            enable_centrifugal=True,
            enable_gravity=True,
        )
    """

    #: Small number for numerical stability
    EPS: float = 1e-10

    def __init__(
        self,
        transforms: CoordinateTransforms,
        gravity_vector: np.ndarray,
    ):
        self.transforms = transforms
        self.gravity_vector = np.asarray(gravity_vector, dtype=np.float64)

    def compute_centrifugal_forces(
        self,
        coords: np.ndarray,
        nodal_mass: np.ndarray,
        omega: float,
    ) -> np.ndarray:
        """
        Compute centrifugal forces at mesh nodes due to rotor rotation.

        The centrifugal force on each node is:

            F_c,i = mᵢ · ω² · r_⊥,i · ê_r

        where:
        - mᵢ: Lumped mass at node i [kg]
        - ω: Current angular velocity [rad/s]
        - r_⊥,i: Perpendicular distance from rotation axis [m]
        - ê_r: Unit radial vector (outward from axis)

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in global frame, shape (n_nodes, 3) [m].
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        omega : float
            Angular velocity [rad/s].

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

        if abs(omega) < self.EPS:
            return F_centrifugal

        # Position vectors from rotor center
        r = self.transforms.position_from_center(coords)

        # Perpendicular components only
        perp_indices = self.transforms.perp_indices
        r_perp_vec = np.zeros_like(r)
        r_perp_vec[:, perp_indices] = r[:, perp_indices]

        # Magnitude of perpendicular distance
        r_perp_mag = np.linalg.norm(r_perp_vec, axis=1)

        # Unit radial vectors (avoid division by zero)
        r_perp_mag_safe = np.where(r_perp_mag > self.EPS, r_perp_mag, 1.0)
        e_r = r_perp_vec / r_perp_mag_safe[:, np.newaxis]

        # Centrifugal force: F = m · ω² · r_perp · ê_r
        omega_squared = omega**2
        F_centrifugal = (nodal_mass * omega_squared * r_perp_mag)[:, np.newaxis] * e_r

        return F_centrifugal

    def compute_gravity_forces(
        self,
        nodal_mass: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Compute gravitational forces in the rotating blade reference frame.

        Since the structural mesh is fixed (non-rotating), we simulate the
        effect of gravity on a rotating blade by rotating the gravity vector
        inversely by the accumulated rotation angle θ:

            F_g,i = mᵢ · R_axis(-θ) · g

        Parameters
        ----------
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        theta : float
            Accumulated rotation angle [rad].

        Returns
        -------
        F_gravity : np.ndarray
            Gravity force vectors at nodes, shape (n_nodes, 3) [N].

        Notes
        -----
        As the blade rotates through its cycle:
        - At θ = 0° (blade pointing up): Gravity causes compression in leading edge
        - At θ = 90° (blade horizontal): Gravity causes flapwise bending
        - At θ = 180° (blade pointing down): Gravity causes tension in leading edge
        - At θ = 270° (blade horizontal): Gravity causes opposite flapwise bending

        This creates cyclic edgewise bending at 1P frequency (once per revolution),
        which is a major contributor to blade fatigue loads.
        """
        # Rotate gravity vector by -θ around the rotor axis
        g_rotated = self.transforms.rotate_vector(self.gravity_vector, -theta)

        # Apply gravity force to each node: F = m · g_rotated
        F_gravity = nodal_mass[:, np.newaxis] * g_rotated

        return F_gravity

    def compute_coriolis_forces(
        self,
        coords: np.ndarray,
        nodal_mass: np.ndarray,
        nodal_velocities: np.ndarray,
        omega: float,
    ) -> np.ndarray:
        """
        Compute Coriolis forces at mesh nodes due to rotation.

        The Coriolis force on each node is:

            F_cor,i = -2 · mᵢ · (ω⃗ × v⃗_rel,i)

        where:
        - mᵢ: Lumped mass at node i [kg]
        - ω⃗: Angular velocity vector [rad/s]
        - v⃗_rel,i: Velocity of node i relative to rotating frame [m/s]

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates, shape (n_nodes, 3). Used only for array sizing.
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        nodal_velocities : np.ndarray
            Nodal velocity vectors (deformation rate in rotating frame),
            shape (n_nodes, 3) [m/s].
        omega : float
            Angular velocity magnitude [rad/s].

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

        **Important**: In the rotating frame formulation, nodal_velocities ARE
        the relative velocities (deformation velocities). No transformation
        is needed.
        """
        n_nodes = coords.shape[0]
        F_coriolis = np.zeros((n_nodes, 3))

        if abs(omega) < self.EPS:
            return F_coriolis

        # Construct angular velocity vector along rotation axis
        omega_vec = np.zeros(3)
        omega_vec[self.transforms.axis_idx] = omega

        # In the rotating frame, nodal_velocities ARE v_rel
        v_rel = nodal_velocities

        # F_cor = -2 * m * (ω × v_rel)
        omega_cross_v = np.cross(omega_vec, v_rel)
        F_coriolis = -2.0 * nodal_mass[:, np.newaxis] * omega_cross_v

        return F_coriolis

    def compute_euler_forces(
        self,
        coords: np.ndarray,
        nodal_mass: np.ndarray,
        alpha: float,
        theta: float,
        use_rotating_frame: bool = True,
    ) -> np.ndarray:
        """
        Compute Euler forces at mesh nodes due to angular acceleration.

        The Euler force on each node is:

            F_euler,i = -mᵢ · (α⃗ × r⃗_rot,i)

        where:
        - mᵢ: Lumped mass at node i [kg]
        - α⃗: Angular acceleration vector [rad/s²] (α = dω/dt)
        - r⃗_rot,i: Position vector of node i in rotating frame [m]

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in inertial frame, shape (n_nodes, 3) [m].
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        alpha : float
            Angular acceleration [rad/s²].
        theta : float
            Current rotation angle [rad].
        use_rotating_frame : bool, optional
            If True, transform coordinates to rotating frame (default: True).

        Returns
        -------
        F_euler : np.ndarray
            Euler force vectors at nodes in inertial frame, shape (n_nodes, 3) [N].

        Notes
        -----
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
        """
        n_nodes = coords.shape[0]
        F_euler = np.zeros((n_nodes, 3))

        if abs(alpha) < self.EPS:
            return F_euler

        # Build angular acceleration vector along rotor axis
        alpha_vec = np.zeros(3)
        alpha_vec[self.transforms.axis_idx] = alpha

        # Get positions in the appropriate frame
        if use_rotating_frame:
            # Transform positions to rotating frame
            r_rot = self.transforms.to_rotating_frame(coords, theta)
        else:
            r_rot = coords

        # F_euler = -m * (α × r_rot)
        alpha_cross_r = np.cross(alpha_vec, r_rot)
        F_euler_rot = -nodal_mass[:, np.newaxis] * alpha_cross_r

        if use_rotating_frame:
            # Transform forces back to inertial frame
            F_euler = self.transforms.to_inertial_frame(F_euler_rot, theta)
        else:
            F_euler = F_euler_rot

        return F_euler

    def compute_total_forces(
        self,
        coords: np.ndarray,
        nodal_mass: np.ndarray,
        omega: float,
        theta: float,
        enable_centrifugal: bool = False,
        enable_gravity: bool = False,
        enable_coriolis: bool = False,
        enable_euler: bool = False,
        nodal_velocities: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        use_rotating_frame: bool = True,
    ) -> np.ndarray:
        """
        Compute total inertial forces from all enabled sources.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates, shape (n_nodes, 3) [m].
        nodal_mass : np.ndarray
            Lumped mass at each node, shape (n_nodes,) [kg].
        omega : float
            Angular velocity [rad/s].
        theta : float
            Rotation angle [rad].
        enable_centrifugal : bool, optional
            Enable centrifugal forces (default: False).
        enable_gravity : bool, optional
            Enable gravity forces (default: False).
        enable_coriolis : bool, optional
            Enable Coriolis forces (default: False).
        enable_euler : bool, optional
            Enable Euler forces (default: False).
        nodal_velocities : np.ndarray, optional
            Nodal velocities for Coriolis calculation, shape (n_nodes, 3) [m/s].
        alpha : float, optional
            Angular acceleration for Euler forces [rad/s²].
        use_rotating_frame : bool, optional
            Use rotating frame for Euler forces (default: True).

        Returns
        -------
        F_total : np.ndarray
            Total inertial forces, shape (n_nodes, 3) [N].
        """
        n_nodes = coords.shape[0]
        F_total = np.zeros((n_nodes, 3))

        if enable_centrifugal:
            F_total += self.compute_centrifugal_forces(coords, nodal_mass, omega)

        if enable_gravity:
            F_total += self.compute_gravity_forces(nodal_mass, theta)

        if enable_coriolis and nodal_velocities is not None:
            F_total += self.compute_coriolis_forces(coords, nodal_mass, nodal_velocities, omega)

        if enable_euler:
            F_total += self.compute_euler_forces(
                coords, nodal_mass, alpha, theta, use_rotating_frame
            )

        return F_total
