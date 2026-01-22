"""
Co-rotational framework for rotor FSI simulations.

This module provides utilities for solving FSI problems in a rotating reference frame,
which allows the stiffness matrix to remain constant while accounting for inertial
effects (centrifugal, Coriolis, and Euler forces).

Mathematical Formulation
------------------------
The position of a point in the inertial (global) frame is:

    x_global = R(θ) · (x_ref + u_local)

where:
- R(θ) is the rotation matrix for angle θ about the rotation axis
- x_ref is the reference position in the rotating frame
- u_local is the elastic displacement computed by FEM

Forces are transformed between frames:
- F_local = R^T · F_global  (for incoming aerodynamic forces)
- u_global = R · u_local    (for outgoing displacements)

Inertial forces in the rotating frame:
- Centrifugal: F_cf = m · ω × (ω × r)
- Coriolis: F_cor = 2m · ω × v
- Euler: F_euler = m · α × r  (where α = dω/dt)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CoordinateTransforms:
    """
    Coordinate transformation utilities for rotating reference frames.

    Uses Rodrigues' rotation formula to compute rotation matrices for
    arbitrary rotation axes.

    Parameters
    ----------
    rotation_axis : array-like, shape (3,)
        Unit vector defining the axis of rotation. Will be normalized.
    rotation_center : array-like, shape (3,), optional
        Point about which rotation occurs. Default is origin [0, 0, 0].

    Examples
    --------
    >>> transforms = CoordinateTransforms(rotation_axis=[0, 0, 1])
    >>> R = transforms.rotation_matrix(np.pi / 2)  # 90 degrees about Z
    >>> v_local = transforms.to_rotating(v_global, theta=np.pi/2)
    """

    def __init__(
        self,
        rotation_axis: Union[list, tuple, NDArray],
        rotation_center: Optional[Union[list, tuple, NDArray]] = None,
    ):
        axis = np.asarray(rotation_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("rotation_axis cannot be zero vector")
        self._axis = axis / norm

        if rotation_center is None:
            self._center = np.zeros(3, dtype=np.float64)
        else:
            self._center = np.asarray(rotation_center, dtype=np.float64)

        # Precompute skew-symmetric matrix for axis (used in Rodrigues formula)
        self._K = self._skew_symmetric(self._axis)
        self._K2 = self._K @ self._K

    @staticmethod
    def _skew_symmetric(v: NDArray) -> NDArray:
        """Compute skew-symmetric matrix from vector v such that K @ x = v × x."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)

    def rotation_matrix(self, theta: float) -> NDArray:
        """
        Compute 3x3 rotation matrix for angle theta using Rodrigues' formula.

        R = I + sin(θ)·K + (1 - cos(θ))·K²

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        R : ndarray, shape (3, 3)
            Rotation matrix.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.eye(3) + s * self._K + (1.0 - c) * self._K2

    def rotation_matrix_derivative(self, theta: float) -> NDArray:
        """
        Compute derivative of rotation matrix with respect to theta.

        dR/dθ = cos(θ)·K + sin(θ)·K²

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        dR : ndarray, shape (3, 3)
            Derivative of rotation matrix.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return c * self._K + s * self._K2

    def to_rotating(self, vec_global: NDArray, theta: float) -> NDArray:
        """
        Transform vector(s) from inertial (global) to rotating (local) frame.

        v_local = R^T · v_global

        Parameters
        ----------
        vec_global : ndarray, shape (3,) or (n, 3)
            Vector(s) in global coordinates.
        theta : float
            Current rotation angle in radians.

        Returns
        -------
        vec_local : ndarray
            Vector(s) in rotating frame, same shape as input.
        """
        R = self.rotation_matrix(theta)
        if vec_global.ndim == 1:
            return R.T @ vec_global
        else:
            # For array of vectors: (n, 3) @ (3, 3).T = (n, 3)
            return vec_global @ R

    def to_inertial(self, vec_local: NDArray, theta: float) -> NDArray:
        """
        Transform vector(s) from rotating (local) to inertial (global) frame.

        v_global = R · v_local

        Parameters
        ----------
        vec_local : ndarray, shape (3,) or (n, 3)
            Vector(s) in rotating coordinates.
        theta : float
            Current rotation angle in radians.

        Returns
        -------
        vec_global : ndarray
            Vector(s) in global frame, same shape as input.
        """
        R = self.rotation_matrix(theta)
        if vec_local.ndim == 1:
            return R @ vec_local
        else:
            return vec_local @ R.T

    def transform_force_to_rotating(self, force_global: NDArray, theta: float) -> NDArray:
        """
        Transform force data from CFD (global frame) to structural solver (rotating frame).

        F_local = R^T · F_global

        Parameters
        ----------
        force_global : ndarray, shape (n_nodes, 3) or (n_components,)
            Forces in global coordinates from preCICE.
        theta : float
            Current rotation angle in radians.

        Returns
        -------
        force_local : ndarray
            Forces in rotating frame, same shape as input.
        """
        R = self.rotation_matrix(theta)

        if force_global.ndim == 1:
            # Flat array: reshape to (n_nodes, 3), transform, flatten back
            n_nodes = len(force_global) // 3
            force_2d = force_global.reshape(n_nodes, 3)
            force_local_2d = force_2d @ R  # (n, 3) @ (3, 3) = R^T applied row-wise
            return force_local_2d.flatten()
        else:
            # Already (n_nodes, 3)
            return force_global @ R

    def transform_displacement_to_inertial(self, disp_local: NDArray, theta: float) -> NDArray:
        """
        Transform displacement from structural solver (rotating) to CFD (global frame).

        u_global = R · u_local

        Parameters
        ----------
        disp_local : ndarray, shape (n_dofs,) or (n_nodes, 3)
            Displacements in rotating coordinates.
        theta : float
            Current rotation angle in radians.

        Returns
        -------
        disp_global : ndarray
            Displacements in global frame, same shape as input.
        """
        R = self.rotation_matrix(theta)

        if disp_local.ndim == 1:
            n_nodes = len(disp_local) // 3
            disp_2d = disp_local.reshape(n_nodes, 3)
            disp_global_2d = disp_2d @ R.T
            return disp_global_2d.flatten()
        else:
            return disp_local @ R.T

    @property
    def axis(self) -> NDArray:
        """Return the normalized rotation axis."""
        return self._axis.copy()

    @property
    def center(self) -> NDArray:
        """Return the rotation center."""
        return self._center.copy()


class InertialForcesCalculator:
    """
    Calculate inertial (fictitious) forces in a rotating reference frame.

    In a non-inertial rotating frame, the equation of motion includes
    fictitious forces that account for the frame's acceleration:

    F_inertial = -m·(ω × (ω × r) + 2·ω × v + α × r)

    where:
    - ω × (ω × r): centrifugal acceleration (always radially outward)
    - 2·ω × v: Coriolis acceleration (perpendicular to velocity)
    - α × r: Euler acceleration (due to angular acceleration α = dω/dt)

    Parameters
    ----------
    rotation_axis : array-like, shape (3,)
        Unit vector defining the axis of rotation.
    rotation_center : array-like, shape (3,), optional
        Point about which rotation occurs. Default is origin.

    Notes
    -----
    The forces returned are in the ROTATING frame and should be added
    to the external forces before solving the structural equations.
    """

    def __init__(
        self,
        rotation_axis: Union[list, tuple, NDArray],
        rotation_center: Optional[Union[list, tuple, NDArray]] = None,
    ):
        axis = np.asarray(rotation_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("rotation_axis cannot be zero vector")
        self._axis = axis / norm

        if rotation_center is None:
            self._center = np.zeros(3, dtype=np.float64)
        else:
            self._center = np.asarray(rotation_center, dtype=np.float64)

    def compute_centrifugal_force(
        self,
        nodal_coords: NDArray,
        nodal_masses: NDArray,
        omega: float,
    ) -> NDArray:
        """
        Compute centrifugal force for each node.

        F_cf = m · ω² · r_perp  (radially outward from rotation axis)

        where r_perp is the perpendicular distance from the rotation axis.

        Parameters
        ----------
        nodal_coords : ndarray, shape (n_nodes, 3)
            Node coordinates in the rotating frame.
        nodal_masses : ndarray, shape (n_nodes,)
            Mass associated with each node.
        omega : float
            Angular velocity magnitude (rad/s).

        Returns
        -------
        F_centrifugal : ndarray, shape (n_nodes, 3)
            Centrifugal force vectors for each node.
        """
        if omega == 0.0:
            return np.zeros_like(nodal_coords)

        # Position vectors relative to rotation center
        r = nodal_coords - self._center

        # Perpendicular component: r_perp = r - (r · axis) * axis
        r_dot_axis = r @ self._axis
        r_perp = r - np.outer(r_dot_axis, self._axis)

        # Centrifugal force: F = m * omega^2 * r_perp (radially outward)
        omega_sq = omega * omega
        F_cf = nodal_masses[:, np.newaxis] * omega_sq * r_perp

        return F_cf

    def compute_coriolis_force(
        self,
        nodal_velocities: NDArray,
        nodal_masses: NDArray,
        omega: float,
    ) -> NDArray:
        """
        Compute Coriolis force for each node.

        F_cor = -2m · (ω × v)

        The negative sign is because we're computing the fictitious force
        that appears in the rotating frame equation of motion.

        Parameters
        ----------
        nodal_velocities : ndarray, shape (n_nodes, 3)
            Node velocities in the rotating frame.
        nodal_masses : ndarray, shape (n_nodes,)
            Mass associated with each node.
        omega : float
            Angular velocity magnitude (rad/s).

        Returns
        -------
        F_coriolis : ndarray, shape (n_nodes, 3)
            Coriolis force vectors for each node.
        """
        if omega == 0.0:
            return np.zeros_like(nodal_velocities)

        # Angular velocity vector
        omega_vec = omega * self._axis

        # Coriolis: F = -2m * (omega × v)
        # Cross product for each row
        omega_cross_v = np.cross(omega_vec, nodal_velocities)
        F_cor = -2.0 * nodal_masses[:, np.newaxis] * omega_cross_v

        return F_cor

    def compute_euler_force(
        self,
        nodal_coords: NDArray,
        nodal_masses: NDArray,
        alpha: float,
    ) -> NDArray:
        """
        Compute Euler force for each node (due to angular acceleration).

        F_euler = -m · (α × r)

        Parameters
        ----------
        nodal_coords : ndarray, shape (n_nodes, 3)
            Node coordinates in the rotating frame.
        nodal_masses : ndarray, shape (n_nodes,)
            Mass associated with each node.
        alpha : float
            Angular acceleration magnitude (rad/s²).

        Returns
        -------
        F_euler : ndarray, shape (n_nodes, 3)
            Euler force vectors for each node.
        """
        if alpha == 0.0:
            return np.zeros_like(nodal_coords)

        # Position vectors relative to rotation center
        r = nodal_coords - self._center

        # Angular acceleration vector
        alpha_vec = alpha * self._axis

        # Euler force: F = -m * (alpha × r)
        alpha_cross_r = np.cross(alpha_vec, r)
        F_euler = -nodal_masses[:, np.newaxis] * alpha_cross_r

        return F_euler

    def compute_all_inertial_forces(
        self,
        nodal_coords: NDArray,
        nodal_velocities: NDArray,
        nodal_masses: NDArray,
        omega: float,
        alpha: float = 0.0,
        include_centrifugal: bool = True,
        include_coriolis: bool = True,
        include_euler: bool = True,
    ) -> Tuple[NDArray, dict]:
        """
        Compute all inertial forces and return diagnostics.

        Parameters
        ----------
        nodal_coords : ndarray, shape (n_nodes, 3)
            Node coordinates in the rotating frame.
        nodal_velocities : ndarray, shape (n_nodes, 3)
            Node velocities in the rotating frame.
        nodal_masses : ndarray, shape (n_nodes,)
            Mass associated with each node.
        omega : float
            Angular velocity magnitude (rad/s).
        alpha : float, optional
            Angular acceleration magnitude (rad/s²). Default 0.
        include_centrifugal : bool, optional
            Whether to include centrifugal forces. Default True.
        include_coriolis : bool, optional
            Whether to include Coriolis forces. Default True.
        include_euler : bool, optional
            Whether to include Euler forces. Default True.

        Returns
        -------
        F_total : ndarray, shape (n_nodes, 3)
            Total inertial force per node.
        diagnostics : dict
            Dictionary with individual force magnitudes and statistics.
        """
        n_nodes = nodal_coords.shape[0]
        F_total = np.zeros((n_nodes, 3), dtype=np.float64)
        diagnostics = {}

        if include_centrifugal:
            F_cf = self.compute_centrifugal_force(nodal_coords, nodal_masses, omega)
            F_total += F_cf
            cf_mag = np.linalg.norm(F_cf, axis=1)
            diagnostics["centrifugal"] = {
                "total": float(np.linalg.norm(np.sum(F_cf, axis=0))),
                "max_nodal": float(np.max(cf_mag)),
                "mean_nodal": float(np.mean(cf_mag)),
            }

        if include_coriolis:
            F_cor = self.compute_coriolis_force(nodal_velocities, nodal_masses, omega)
            F_total += F_cor
            cor_mag = np.linalg.norm(F_cor, axis=1)
            diagnostics["coriolis"] = {
                "total": float(np.linalg.norm(np.sum(F_cor, axis=0))),
                "max_nodal": float(np.max(cor_mag)),
                "mean_nodal": float(np.mean(cor_mag)),
            }

        if include_euler and alpha != 0.0:
            F_euler = self.compute_euler_force(nodal_coords, nodal_masses, alpha)
            F_total += F_euler
            euler_mag = np.linalg.norm(F_euler, axis=1)
            diagnostics["euler"] = {
                "total": float(np.linalg.norm(np.sum(F_euler, axis=0))),
                "max_nodal": float(np.max(euler_mag)),
                "mean_nodal": float(np.mean(euler_mag)),
            }

        total_mag = np.linalg.norm(F_total, axis=1)
        diagnostics["total_inertial"] = {
            "total": float(np.linalg.norm(np.sum(F_total, axis=0))),
            "max_nodal": float(np.max(total_mag)),
            "mean_nodal": float(np.mean(total_mag)),
        }

        return F_total, diagnostics

    @property
    def axis(self) -> NDArray:
        """Return the rotation axis."""
        return self._axis.copy()

    @property
    def center(self) -> NDArray:
        """Return the rotation center."""
        return self._center.copy()


class OmegaProvider(ABC):
    """
    Abstract base class for angular velocity providers.

    This interface allows flexible specification of angular velocity:
    - Constant value
    - Time-varying function
    - Tabulated data with interpolation
    - Dynamically computed from torque balance (future)

    Subclasses must implement `get_omega(t)` which returns the current
    angular velocity and angular acceleration at time t.

    Future Extensions
    -----------------
    To add new omega modes, create a subclass:

    >>> class TableOmega(OmegaProvider):
    ...     def __init__(self, times, omegas):
    ...         self._times = np.array(times)
    ...         self._omegas = np.array(omegas)
    ...
    ...     def get_omega(self, t: float) -> Tuple[float, float]:
    ...         omega = np.interp(t, self._times, self._omegas)
    ...         # Compute alpha from finite difference
    ...         alpha = ...
    ...         return omega, alpha

    >>> class ComputedOmega(OmegaProvider):
    ...     '''Compute omega from torque balance: I * alpha = tau'''
    ...     def __init__(self, moment_of_inertia, initial_omega=0.0):
    ...         self._I = moment_of_inertia
    ...         self._omega = initial_omega
    ...
    ...     def update(self, torque: float, dt: float):
    ...         alpha = torque / self._I
    ...         self._omega += alpha * dt
    ...
    ...     def get_omega(self, t: float) -> Tuple[float, float]:
    ...         return self._omega, self._last_alpha
    """

    @abstractmethod
    def get_omega(self, t: float) -> Tuple[float, float]:
        """
        Get angular velocity and acceleration at time t.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.

        Returns
        -------
        omega : float
            Angular velocity in rad/s.
        alpha : float
            Angular acceleration in rad/s² (dω/dt).
        """
        pass

    @property
    @abstractmethod
    def initial_omega(self) -> float:
        """Return the initial angular velocity."""
        pass


class ConstantOmega(OmegaProvider):
    """
    Constant angular velocity provider.

    The angular velocity remains constant throughout the simulation,
    so angular acceleration is always zero.

    Parameters
    ----------
    omega : float
        Constant angular velocity in rad/s.

    Examples
    --------
    >>> provider = ConstantOmega(omega=10.0)  # 10 rad/s
    >>> omega, alpha = provider.get_omega(t=5.0)
    >>> print(omega, alpha)
    10.0 0.0
    """

    def __init__(self, omega: float):
        self._omega = float(omega)

    def get_omega(self, t: float) -> Tuple[float, float]:
        """Return constant omega and zero acceleration."""
        return self._omega, 0.0

    @property
    def initial_omega(self) -> float:
        """Return the constant omega value."""
        return self._omega

    def __repr__(self) -> str:
        return f"ConstantOmega(omega={self._omega})"


# =============================================================================
# Placeholder classes for future omega modes (documented for architecture)
# =============================================================================


class TableOmega(OmegaProvider):
    """
    Angular velocity from tabulated time-series data.

    Interpolates between provided time-omega pairs using linear interpolation.
    Angular acceleration is computed from finite differences.

    Parameters
    ----------
    times : array-like
        Time values in seconds (must be monotonically increasing).
    omegas : array-like
        Corresponding angular velocity values in rad/s.

    Notes
    -----
    This class is provided for future use. Implementation is complete
    but may need refinement based on actual use cases.
    """

    def __init__(self, times: Union[list, NDArray], omegas: Union[list, NDArray]):
        self._times = np.asarray(times, dtype=np.float64)
        self._omegas = np.asarray(omegas, dtype=np.float64)

        if len(self._times) != len(self._omegas):
            raise ValueError("times and omegas must have same length")
        if len(self._times) < 2:
            raise ValueError("Need at least 2 data points for interpolation")
        if not np.all(np.diff(self._times) > 0):
            raise ValueError("times must be monotonically increasing")

        # Precompute alpha (angular acceleration) using finite differences
        self._alphas = np.gradient(self._omegas, self._times)

    def get_omega(self, t: float) -> Tuple[float, float]:
        """Interpolate omega and alpha at time t."""
        omega = float(np.interp(t, self._times, self._omegas))
        alpha = float(np.interp(t, self._times, self._alphas))
        return omega, alpha

    @property
    def initial_omega(self) -> float:
        """Return omega at t=0."""
        return float(self._omegas[0])

    def __repr__(self) -> str:
        return f"TableOmega(t_range=[{self._times[0]}, {self._times[-1]}])"


class FunctionOmega(OmegaProvider):
    """
    Angular velocity from a user-defined function.

    Parameters
    ----------
    omega_func : callable
        Function that takes time t and returns omega.
        Signature: omega_func(t: float) -> float
    alpha_func : callable, optional
        Function that takes time t and returns alpha (dω/dt).
        If not provided, alpha is computed numerically.
    initial_omega : float, optional
        Initial omega value. If not provided, computed from omega_func(0).

    Examples
    --------
    >>> # Linear ramp-up
    >>> provider = FunctionOmega(
    ...     omega_func=lambda t: min(10.0, 2.0 * t),
    ...     alpha_func=lambda t: 2.0 if t < 5.0 else 0.0
    ... )
    """

    def __init__(
        self,
        omega_func: callable,
        alpha_func: Optional[callable] = None,
        initial_omega: Optional[float] = None,
    ):
        self._omega_func = omega_func
        self._alpha_func = alpha_func
        self._dt_numerical = 1e-6  # For numerical differentiation

        if initial_omega is not None:
            self._initial_omega = float(initial_omega)
        else:
            self._initial_omega = float(omega_func(0.0))

    def get_omega(self, t: float) -> Tuple[float, float]:
        """Evaluate omega and alpha at time t."""
        omega = float(self._omega_func(t))

        if self._alpha_func is not None:
            alpha = float(self._alpha_func(t))
        else:
            # Numerical differentiation
            omega_plus = self._omega_func(t + self._dt_numerical)
            alpha = (omega_plus - omega) / self._dt_numerical

        return omega, alpha

    @property
    def initial_omega(self) -> float:
        """Return omega at t=0."""
        return self._initial_omega

    def __repr__(self) -> str:
        return f"FunctionOmega(initial_omega={self._initial_omega})"
