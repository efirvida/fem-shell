"""
Coordinate frame transformations for rotating reference frames.

This module provides rotation matrices and transformation functions
for converting between inertial (lab) and rotating (blade) reference frames.
"""

from typing import Dict, List, Tuple

import numpy as np


class CoordinateTransforms:
    """
    Coordinate transformation utilities for rotating reference frames.

    This class provides methods for:
    - Building rotation matrices around principal axes
    - Transforming vectors between inertial and rotating frames
    - Computing axis indices and perpendicular distances

    Parameters
    ----------
    rotor_axis : str
        Rotation axis: 'x', 'y', or 'z'.
    rotor_center : np.ndarray
        Center of rotation [x, y, z] [m].

    Attributes
    ----------
    axis_idx : int
        Index of rotation axis (0, 1, or 2).
    perp_indices : List[int]
        Indices of perpendicular axes.

    Example
    -------
    ::

        transforms = CoordinateTransforms('x', np.array([0, 0, 0]))

        # Transform forces from inertial to rotating frame
        forces_rotating = transforms.to_rotating_frame(forces_inertial, theta)

        # Transform displacements from rotating to inertial frame
        u_inertial = transforms.to_inertial_frame(u_rotating, theta)
    """

    #: Mapping from axis name to array index
    AXIS_MAP: Dict[str, int] = {"x": 0, "y": 1, "z": 2}

    def __init__(self, rotor_axis: str, rotor_center: np.ndarray):
        self.rotor_axis = rotor_axis.lower()
        self.rotor_center = np.asarray(rotor_center, dtype=np.float64)

        # Pre-compute axis indices
        self.axis_idx = self.AXIS_MAP.get(self.rotor_axis, 0)
        self.perp_indices = [i for i in range(3) if i != self.axis_idx]

    def get_axis_indices(self) -> Tuple[int, List[int]]:
        """
        Get axis index and perpendicular indices.

        Returns
        -------
        axis_idx : int
            Index of the rotation axis (0, 1, or 2 for x, y, z).
        perp_indices : List[int]
            Indices of the two perpendicular axes.
        """
        return self.axis_idx, self.perp_indices

    def get_rotation_matrix(self, angle: float) -> np.ndarray:
        """
        Get 3x3 rotation matrix for rotation around the rotor axis.

        Parameters
        ----------
        angle : float
            Rotation angle [rad]. Positive follows right-hand rule.

        Returns
        -------
        R : np.ndarray
            3x3 rotation matrix.

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
        c = np.cos(angle)
        s = np.sin(angle)

        if self.rotor_axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif self.rotor_axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif self.rotor_axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            return np.eye(3)

    def rotate_vector(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate a 3D vector or array of vectors around the rotor axis.

        Uses the rotation matrix for the configured axis.

        Parameters
        ----------
        vector : np.ndarray
            Vector(s) to rotate, shape (3,) or (N, 3).
        angle : float
            Rotation angle [rad]. Positive follows right-hand rule.

        Returns
        -------
        rotated : np.ndarray
            Rotated vector(s), shape (3,) or (N, 3).
        """
        R = self.get_rotation_matrix(angle)
        # Handle (N, 3) arrays by using dot product with transpose
        # For (3,) vector v: v @ R.T is equivalent to R @ v
        # For (N, 3) vectors V: V @ R.T rotates each row vector correctly
        return vector @ R.T

    def to_rotating_frame(self, vectors: np.ndarray, theta: float) -> np.ndarray:
        """
        Transform vectors from inertial frame to rotating frame.

        Applies R(-θ) to transform from lab frame to blade-attached frame:

            v_rot = R(-θ) · v_inertial

        Parameters
        ----------
        vectors : np.ndarray
            Vectors in inertial frame, shape (n_vectors, 3).
        theta : float
            Current rotation angle [rad].

        Returns
        -------
        vectors_rot : np.ndarray
            Vectors in rotating frame, shape (n_vectors, 3).

        Notes
        -----
        The rotating frame co-rotates with the blade at angle θ.
        Using -θ because we transform FROM inertial TO rotating.
        """
        R = self.get_rotation_matrix(-theta)
        return vectors @ R.T

    def to_inertial_frame(self, vectors: np.ndarray, theta: float) -> np.ndarray:
        """
        Transform vectors from rotating frame to inertial frame.

        Applies R(θ) to transform from blade-attached frame to lab frame:

            v_inertial = R(θ) · v_rot

        Parameters
        ----------
        vectors : np.ndarray
            Vectors in rotating frame, shape (n_vectors, 3).
        theta : float
            Current rotation angle [rad].

        Returns
        -------
        vectors_inertial : np.ndarray
            Vectors in inertial frame, shape (n_vectors, 3).

        Notes
        -----
        This is the inverse transformation of `to_rotating_frame`.

        **Primary Use Case**: Transform structural displacements computed in the
        Local Rotating Frame to the Global Inertial Frame before sending to CFD.
        This ensures kinematic compatibility at the FSI interface.
        """
        R = self.get_rotation_matrix(theta)
        return vectors @ R.T

    def position_from_center(self, coords: np.ndarray) -> np.ndarray:
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
        return coords - self.rotor_center

    def perpendicular_distances(self, position_vectors: np.ndarray) -> np.ndarray:
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
        return np.sqrt(np.sum(position_vectors[:, self.perp_indices] ** 2, axis=1))

    @staticmethod
    def ensure_3d_coordinates(coords: np.ndarray) -> np.ndarray:
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
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3) if len(coords) % 3 == 0 else coords.reshape(-1, 2)

        if coords.shape[1] < 3:
            return np.hstack([
                coords,
                np.zeros((coords.shape[0], 3 - coords.shape[1])),
            ])
        return coords
