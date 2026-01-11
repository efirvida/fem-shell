"""
Geometric node selection utilities for MeshModel.
"""

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from fem_shell.core.mesh.model import MeshModel
    from fem_shell.core.mesh.entities import Node


def select_by_coordinate(
    mesh: "MeshModel",
    axis: Union[int, str],
    value: float,
    mode: str = "near",
    tol: float = 1e-6
) -> List[int]:
    """
    Select nodes based on a coordinate value.

    Parameters
    ----------
    mesh : MeshModel
        The mesh to select nodes from.
    axis : int or str
        Axis index (0, 1, 2) or name ('x', 'y', 'z').
    value : float
        The reference value for the coordinate.
    mode : str, optional
        Selection mode: 'near', 'greater', 'less'. Default is 'near'.
    tol : float, optional
        Tolerance for 'near' mode. Default is 1e-6.

    Returns
    -------
    List[int]
        List of node IDs satisfying the criteria.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map.get(axis.lower())
        if axis is None:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    coords = mesh.coords_array
    if mode == "near":
        mask = np.abs(coords[:, axis] - value) < tol
    elif mode == "greater":
        mask = coords[:, axis] > value
    elif mode == "less":
        mask = coords[:, axis] < value
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'near', 'greater', or 'less'.")

    indices = np.where(mask)[0]
    return [mesh.nodes[i].id for i in indices]


def select_by_box(
    mesh: "MeshModel",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
) -> List[int]:
    """Select nodes within a bounding box."""
    coords = mesh.coords_array
    mask = np.ones(coords.shape[0], dtype=bool)

    for i, r in enumerate([x_range, y_range, z_range]):
        if r is not None:
            mask &= (coords[:, i] >= r[0]) & (coords[:, i] <= r[1])

    indices = np.where(mask)[0]
    return [mesh.nodes[i].id for i in indices]


def select_by_distance(
    mesh: "MeshModel",
    point: Union[Iterable[float], np.ndarray],
    radius: float,
    mode: str = "inside"
) -> List[int]:
    """Select nodes based on distance to a point."""
    point = np.asarray(point)
    coords = mesh.coords_array
    dists = np.linalg.norm(coords - point, axis=1)

    if mode == "inside":
        mask = dists <= radius
    elif mode == "outside":
        mask = dists > radius
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'inside' or 'outside'.")

    indices = np.where(mask)[0]
    return [mesh.nodes[i].id for i in indices]


def select_by_direction(
    mesh: "MeshModel",
    point: Union[Iterable[float], np.ndarray],
    direction: Union[Iterable[float], np.ndarray],
    method: str = "axial",
    mode: str = "inside",
    distance: Optional[float] = None,
    range: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    angle: Optional[float] = None
) -> List[int]:
    """
    Select nodes based on a point and a direction vector.

    Parameters
    ----------
    mesh : MeshModel
        The mesh filter.
    point : array-like
        Origin point (P).
    direction : array-like
        Direction vector (V).
    method : str
        'axial' (prev. 'linear'): Selection based on distance along the vector (projection).
        'radial': Selection based on perpendicular distance from the axis line (tube).
    mode : str
        'inside': Inclusion based on thresholds (default).
        'outside': Exclusion based on thresholds.
    distance : float, optional
        Threshold value for the selected method.
    range : tuple of float, optional
        (min, max) interval for the selected method.
    radius : float, optional
        Alias for distance, often used in radial mode.
    angle : float, optional
        For 'radial' method, selects a cone with specified angle (radians).
    """
    coords = mesh.coords_array
    P = np.asarray(point)
    V = np.asarray(direction)
    norm_V = np.linalg.norm(V)
    if norm_V < 1e-12:
        raise ValueError("Direction vector cannot be zero.")
    V = V / norm_V

    # Relative coordinates from point P
    D = coords - P

    # Axial component: Projection of D onto V (distance along axis)
    dist_axial = np.dot(D, V)

    # Threshold preference
    threshold = distance if distance is not None else radius

    if method in ("axial", "linear"):
        vals = dist_axial
    elif method == "radial":
        # Radial component: Perpendicular distance from the axis line
        # Project Vector = dist_axial * V
        # Perpendicular Vector = D - Project Vector
        vals = np.linalg.norm(D - np.outer(dist_axial, V), axis=1)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'axial' or 'radial'.")

    # Determine selection mask
    if range is not None:
        mask = (vals >= range[0]) & (vals <= range[1])
    elif threshold is not None:
        mask = vals <= threshold
    elif angle is not None and method == "radial":
        # Conical selection (theta <= angle)
        # tan(theta) = radial / axial
        mask = (dist_axial > 0) & (vals / np.maximum(dist_axial, 1e-12) <= np.tan(angle))
    else:
        # Default behavior: half-space along the vector
        if method in ("axial", "linear"):
            mask = dist_axial > 0
        else:
            raise ValueError("No threshold, range, or angle provided for radial selection.")

    # Apply inside/outside mode
    if mode == "outside":
        mask = ~mask

    indices = np.where(mask)[0]
    return [mesh.nodes[i].id for i in indices]
