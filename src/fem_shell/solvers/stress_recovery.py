"""
Stress Recovery and Post-Processing Module for Shell Elements.

This module provides tools for computing stress fields, strain fields,
and derived quantities (von Mises, principal stresses, etc.) from
displacement solutions in finite element analysis.

Theory
------
For shell elements, stresses vary through the thickness. The stress state
at any point through the thickness is given by:

    σ(z) = σ_m + z · σ_b

where:
- σ_m: Membrane stress (constant through thickness)
- σ_b: Bending stress (varies linearly through thickness)
- z: Distance from mid-surface [-h/2, h/2]

The stresses are computed from strains using the constitutive relations:

    σ = C · ε

where C is the material constitutive matrix.

Stress Components
-----------------
For plane stress conditions (shell elements):
- σ_xx: Normal stress in x direction
- σ_yy: Normal stress in y direction
- σ_xy: In-plane shear stress
- σ_xz: Transverse shear stress
- σ_yz: Transverse shear stress

Derived Quantities
------------------
- Von Mises stress: σ_vm = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)
- Principal stresses: σ_1, σ_2 from eigenvalue decomposition
- Maximum shear stress: τ_max = (σ_1 - σ_2) / 2

References
----------
- Bathe, K.J. (2014). "Finite Element Procedures", 2nd Edition.
- Cook, R.D. et al. (2002). "Concepts and Applications of FEA", 4th Edition.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from fem_shell.core.assembler import MeshAssembler


class StressLocation(Enum):
    """Location through shell thickness for stress evaluation."""

    TOP = "top"  # z = +h/2 (top surface)
    MIDDLE = "middle"  # z = 0 (mid-surface)
    BOTTOM = "bottom"  # z = -h/2 (bottom surface)


class StressType(Enum):
    """Type of stress to compute."""

    MEMBRANE = "membrane"  # Constant through thickness
    BENDING = "bending"  # Linear variation through thickness
    TOTAL = "total"  # Membrane + Bending at specified location


@dataclass
class StressResult:
    """
    Container for stress computation results.

    Attributes
    ----------
    sigma_xx : np.ndarray
        Normal stress in x direction at each node/element
    sigma_yy : np.ndarray
        Normal stress in y direction
    sigma_xy : np.ndarray
        In-plane shear stress
    von_mises : np.ndarray
        Von Mises equivalent stress
    sigma_1 : np.ndarray
        First principal stress (maximum)
    sigma_2 : np.ndarray
        Second principal stress (minimum)
    tau_max : np.ndarray
        Maximum in-plane shear stress
    principal_angle : np.ndarray
        Angle of first principal direction from x-axis (radians)
    """

    sigma_xx: np.ndarray
    sigma_yy: np.ndarray
    sigma_xy: np.ndarray
    von_mises: np.ndarray
    sigma_1: np.ndarray
    sigma_2: np.ndarray
    tau_max: np.ndarray
    principal_angle: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for VTK output."""
        return {
            "sigma_xx": self.sigma_xx,
            "sigma_yy": self.sigma_yy,
            "sigma_xy": self.sigma_xy,
            "von_mises": self.von_mises,
            "sigma_1": self.sigma_1,
            "sigma_2": self.sigma_2,
            "tau_max": self.tau_max,
            "principal_angle": np.degrees(self.principal_angle),
        }


@dataclass
class StrainResult:
    """
    Container for strain computation results.

    Attributes
    ----------
    epsilon_xx : np.ndarray
        Normal strain in x direction
    epsilon_yy : np.ndarray
        Normal strain in y direction
    gamma_xy : np.ndarray
        In-plane shear strain (engineering strain = 2 * tensor strain)
    epsilon_1 : np.ndarray
        First principal strain (maximum)
    epsilon_2 : np.ndarray
        Second principal strain (minimum)
    gamma_max : np.ndarray
        Maximum shear strain
    """

    epsilon_xx: np.ndarray
    epsilon_yy: np.ndarray
    gamma_xy: np.ndarray
    epsilon_1: np.ndarray
    epsilon_2: np.ndarray
    gamma_max: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for VTK output."""
        return {
            "epsilon_xx": self.epsilon_xx,
            "epsilon_yy": self.epsilon_yy,
            "gamma_xy": self.gamma_xy,
            "epsilon_1": self.epsilon_1,
            "epsilon_2": self.epsilon_2,
            "gamma_max": self.gamma_max,
        }


class StressRecovery:
    """
    Stress recovery and post-processing for shell finite element analysis.

    This class computes stress and strain fields from displacement solutions,
    supporting both element-wise and nodal (smoothed) results.

    Parameters
    ----------
    domain : MeshAssembler
        The mesh assembler containing element information.
    u : np.ndarray
        Displacement solution vector (full system, not reduced).

    Examples
    --------
    >>> # After solving
    >>> stress_recovery = StressRecovery(solver.domain, solver.u.array)
    >>> stresses = stress_recovery.compute_nodal_stresses()
    >>> print(f"Max von Mises: {stresses.von_mises.max():.2e} Pa")
    """

    def __init__(self, domain: "MeshAssembler", u):
        """
        Initialize stress recovery.

        Parameters
        ----------
        domain : MeshAssembler
            The mesh assembler containing element information.
        u : PETSc.Vec or np.ndarray
            Displacement solution vector.
        """
        self.domain = domain
        # Handle both PETSc Vec and numpy array
        if hasattr(u, "array"):
            self.u = u.array.copy()
        else:
            self.u = np.asarray(u)
        self.dofs_per_node = domain.dofs_per_node
        self.n_nodes = len(domain.mesh.coords_array)
        self.n_elements = len(domain._element_map)
        # Get node ID to index mapping
        self._node_id_to_index = domain.mesh.node_id_to_index

    def compute_element_stresses(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        stress_type: StressType = StressType.TOTAL,
        gauss_point: Tuple[float, float] = (0.0, 0.0),
    ) -> StressResult:
        """
        Compute stresses at element centers (or specified Gauss point).

        Parameters
        ----------
        location : StressLocation
            Location through thickness for stress evaluation.
        stress_type : StressType
            Type of stress to compute (membrane, bending, or total).
        gauss_point : tuple
            Parametric coordinates (r, s) for stress evaluation.
            Default (0, 0) is element center.

        Returns
        -------
        StressResult
            Stress results at each element.
        """
        r, s = gauss_point
        n_elem = self.n_elements

        # Initialize arrays
        sigma_xx = np.zeros(n_elem)
        sigma_yy = np.zeros(n_elem)
        sigma_xy = np.zeros(n_elem)

        for elem_idx, element in self.domain._element_map.items():
            # Get element displacements
            u_elem = self._extract_element_displacements(element)

            # Compute stress at specified location
            sigma = self._compute_element_stress(element, u_elem, r, s, location, stress_type)

            sigma_xx[elem_idx] = sigma[0]
            sigma_yy[elem_idx] = sigma[1]
            sigma_xy[elem_idx] = sigma[2]

        # Compute derived quantities
        return self._compute_derived_stresses(sigma_xx, sigma_yy, sigma_xy)

    def compute_nodal_stresses(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        stress_type: StressType = StressType.TOTAL,
        smoothing: str = "average",
    ) -> StressResult:
        """
        Compute smoothed nodal stresses by extrapolating from Gauss points.

        This method computes stresses at each node by averaging contributions
        from all elements sharing that node (nodal averaging/smoothing).

        Parameters
        ----------
        location : StressLocation
            Location through thickness for stress evaluation.
        stress_type : StressType
            Type of stress to compute.
        smoothing : str
            Smoothing method: "average" (simple averaging) or
            "area_weighted" (weight by element area).

        Returns
        -------
        StressResult
            Smoothed stress results at each node.
        """
        n_nodes = self.n_nodes

        # Accumulators for nodal averaging
        sigma_xx_sum = np.zeros(n_nodes)
        sigma_yy_sum = np.zeros(n_nodes)
        sigma_xy_sum = np.zeros(n_nodes)
        weight_sum = np.zeros(n_nodes)

        # Gauss points for extrapolation to nodes (for 4-node elements)
        # These are the parametric coordinates of the element corners
        node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

        for _elem_idx, element in self.domain._element_map.items():
            # Get element displacements
            u_elem = self._extract_element_displacements(element)

            # Get element weight (for area-weighted averaging)
            if smoothing == "area_weighted":
                weight = self._compute_element_area(element)
            else:
                weight = 1.0

            # Extrapolate to each node of the element
            for local_node, (r, s) in enumerate(node_coords):
                # Compute stress at this corner
                sigma = self._compute_element_stress(element, u_elem, r, s, location, stress_type)

                # Get global node index from node ID
                node_id = element.node_ids[local_node]
                global_node = self._node_id_to_index[node_id]

                # Accumulate
                sigma_xx_sum[global_node] += weight * sigma[0]
                sigma_yy_sum[global_node] += weight * sigma[1]
                sigma_xy_sum[global_node] += weight * sigma[2]
                weight_sum[global_node] += weight

        # Average
        mask = weight_sum > 0
        sigma_xx = np.zeros(n_nodes)
        sigma_yy = np.zeros(n_nodes)
        sigma_xy = np.zeros(n_nodes)

        sigma_xx[mask] = sigma_xx_sum[mask] / weight_sum[mask]
        sigma_yy[mask] = sigma_yy_sum[mask] / weight_sum[mask]
        sigma_xy[mask] = sigma_xy_sum[mask] / weight_sum[mask]

        # Compute derived quantities
        return self._compute_derived_stresses(sigma_xx, sigma_yy, sigma_xy)

    def compute_element_strains(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        gauss_point: Tuple[float, float] = (0.0, 0.0),
    ) -> StrainResult:
        """
        Compute strains at element centers.

        Parameters
        ----------
        location : StressLocation
            Location through thickness for strain evaluation.
        gauss_point : tuple
            Parametric coordinates (r, s) for strain evaluation.

        Returns
        -------
        StrainResult
            Strain results at each element.
        """
        r, s = gauss_point
        n_elem = self.n_elements

        epsilon_xx = np.zeros(n_elem)
        epsilon_yy = np.zeros(n_elem)
        gamma_xy = np.zeros(n_elem)

        for elem_idx, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)
            epsilon = self._compute_element_strain(element, u_elem, r, s, location)

            epsilon_xx[elem_idx] = epsilon[0]
            epsilon_yy[elem_idx] = epsilon[1]
            gamma_xy[elem_idx] = epsilon[2]

        return self._compute_derived_strains(epsilon_xx, epsilon_yy, gamma_xy)

    def compute_nodal_strains(
        self,
        location: StressLocation = StressLocation.MIDDLE,
        smoothing: str = "average",
    ) -> StrainResult:
        """
        Compute smoothed nodal strains.

        Parameters
        ----------
        location : StressLocation
            Location through thickness for strain evaluation.
        smoothing : str
            Smoothing method.

        Returns
        -------
        StrainResult
            Smoothed strain results at each node.
        """
        n_nodes = self.n_nodes

        epsilon_xx_sum = np.zeros(n_nodes)
        epsilon_yy_sum = np.zeros(n_nodes)
        gamma_xy_sum = np.zeros(n_nodes)
        weight_sum = np.zeros(n_nodes)

        node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

        for _elem_idx, element in self.domain._element_map.items():
            u_elem = self._extract_element_displacements(element)

            if smoothing == "area_weighted":
                weight = self._compute_element_area(element)
            else:
                weight = 1.0

            for local_node, (r, s) in enumerate(node_coords):
                epsilon = self._compute_element_strain(element, u_elem, r, s, location)
                # Get global node index from node ID
                node_id = element.node_ids[local_node]
                global_node = self._node_id_to_index[node_id]

                epsilon_xx_sum[global_node] += weight * epsilon[0]
                epsilon_yy_sum[global_node] += weight * epsilon[1]
                gamma_xy_sum[global_node] += weight * epsilon[2]
                weight_sum[global_node] += weight

        mask = weight_sum > 0
        epsilon_xx = np.zeros(n_nodes)
        epsilon_yy = np.zeros(n_nodes)
        gamma_xy = np.zeros(n_nodes)

        epsilon_xx[mask] = epsilon_xx_sum[mask] / weight_sum[mask]
        epsilon_yy[mask] = epsilon_yy_sum[mask] / weight_sum[mask]
        gamma_xy[mask] = gamma_xy_sum[mask] / weight_sum[mask]

        return self._compute_derived_strains(epsilon_xx, epsilon_yy, gamma_xy)

    def _extract_element_displacements(self, element) -> np.ndarray:
        """Extract displacement vector for an element from global solution."""
        elem_dofs = []
        for node_id in element.node_ids:
            # Convert node ID to node index
            node_idx = self._node_id_to_index[node_id]
            start_dof = node_idx * self.dofs_per_node
            elem_dofs.extend(range(start_dof, start_dof + self.dofs_per_node))
        return self.u[elem_dofs]

    def _compute_element_stress(
        self,
        element,
        u_elem: np.ndarray,
        r: float,
        s: float,
        location: StressLocation,
        stress_type: StressType,
    ) -> np.ndarray:
        """
        Compute stress at a point within an element.

        Returns stress in Voigt notation: [σ_xx, σ_yy, σ_xy]
        """
        h = element.thickness

        # Get z coordinate for specified location
        if location == StressLocation.TOP:
            z = h / 2
        elif location == StressLocation.BOTTOM:
            z = -h / 2
        else:  # MIDDLE
            z = 0.0

        # Membrane contribution
        sigma_m = np.zeros(3)
        if stress_type in (StressType.MEMBRANE, StressType.TOTAL):
            # Extract membrane DOFs [u1, v1, u2, v2, u3, v3, u4, v4]
            membrane_dof_indices = np.array([0, 1, 6, 7, 12, 13, 18, 19])
            u_membrane = u_elem[membrane_dof_indices]

            # Compute strain and stress
            B_m = element.B_m(r, s)
            epsilon_m = B_m @ u_membrane
            C_m = element.Cm()
            sigma_m = C_m @ epsilon_m

        # Bending contribution
        sigma_b = np.zeros(3)
        if stress_type in (StressType.BENDING, StressType.TOTAL):
            # Extract bending DOFs [w, θx, θy] for each node
            # Element has 6 DOFs per node: [u, v, w, θx, θy, θz]
            # B_kappa expects 12 DOFs: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3, w4, θx4, θy4]
            bending_dof_indices = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
            u_bending = u_elem[bending_dof_indices]

            # Compute bending strain (curvature) and stress
            B_kappa = element.B_kappa(r, s)  # 3x12
            kappa = B_kappa @ u_bending  # Curvatures [κ_xx, κ_yy, κ_xy]

            # Bending strain at z: ε_b = z · κ
            epsilon_b = z * kappa

            # Bending stress using material matrix (not Cb which has h^3/12)
            C_m = element.Cm()  # Use membrane material matrix for stress
            sigma_b = C_m @ epsilon_b

        return sigma_m + sigma_b

    def _compute_element_strain(
        self,
        element,
        u_elem: np.ndarray,
        r: float,
        s: float,
        location: StressLocation,
    ) -> np.ndarray:
        """
        Compute strain at a point within an element.

        Returns strain in Voigt notation: [ε_xx, ε_yy, γ_xy]
        """
        h = element.thickness

        # Get z coordinate for specified location
        if location == StressLocation.TOP:
            z = h / 2
        elif location == StressLocation.BOTTOM:
            z = -h / 2
        else:
            z = 0.0

        # Membrane strain
        membrane_dof_indices = np.array([0, 1, 6, 7, 12, 13, 18, 19])
        u_membrane = u_elem[membrane_dof_indices]
        B_m = element.B_m(r, s)
        epsilon_m = B_m @ u_membrane

        # Bending strain (at z)
        # B_kappa expects 12 DOFs: [w, θx, θy] for each node
        # Element has 6 DOFs per node: [u, v, w, θx, θy, θz]
        bending_dof_indices = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
        u_bending = u_elem[bending_dof_indices]
        B_kappa = element.B_kappa(r, s)  # 3x12
        kappa = B_kappa @ u_bending
        epsilon_b = z * kappa

        return epsilon_m + epsilon_b

    def _compute_element_area(self, element) -> float:
        """Compute element area using Jacobian."""
        # Integrate det(J) over element (2x2 Gauss quadrature)
        gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        area = 0.0
        for r in gauss_pts:
            for s in gauss_pts:
                J_val, det_J = element.J(r, s)
                area += det_J
        return area

    def _compute_derived_stresses(
        self, sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: np.ndarray
    ) -> StressResult:
        """Compute von Mises, principal stresses, etc. from stress components."""
        # Von Mises stress (plane stress)
        von_mises = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

        # Principal stresses
        sigma_avg = (sigma_xx + sigma_yy) / 2
        sigma_diff = (sigma_xx - sigma_yy) / 2
        R = np.sqrt(sigma_diff**2 + sigma_xy**2)

        sigma_1 = sigma_avg + R  # Maximum principal stress
        sigma_2 = sigma_avg - R  # Minimum principal stress

        # Maximum shear stress
        tau_max = R

        # Principal angle (angle of σ_1 from x-axis)
        principal_angle = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)

        return StressResult(
            sigma_xx=sigma_xx,
            sigma_yy=sigma_yy,
            sigma_xy=sigma_xy,
            von_mises=von_mises,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            tau_max=tau_max,
            principal_angle=principal_angle,
        )

    def _compute_derived_strains(
        self, epsilon_xx: np.ndarray, epsilon_yy: np.ndarray, gamma_xy: np.ndarray
    ) -> StrainResult:
        """Compute principal strains, etc. from strain components."""
        # Principal strains
        epsilon_avg = (epsilon_xx + epsilon_yy) / 2
        epsilon_diff = (epsilon_xx - epsilon_yy) / 2
        R = np.sqrt(epsilon_diff**2 + (gamma_xy / 2) ** 2)

        epsilon_1 = epsilon_avg + R
        epsilon_2 = epsilon_avg - R
        gamma_max = 2 * R

        return StrainResult(
            epsilon_xx=epsilon_xx,
            epsilon_yy=epsilon_yy,
            gamma_xy=gamma_xy,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            gamma_max=gamma_max,
        )


def compute_von_mises(
    sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: np.ndarray
) -> np.ndarray:
    """
    Compute von Mises equivalent stress from plane stress components.

    The von Mises stress is a scalar measure of stress intensity that
    combines the stress components into a single value. It's commonly
    used as a yield criterion for ductile materials.

    For plane stress (σ_zz = σ_xz = σ_yz = 0):

        σ_vm = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

    Parameters
    ----------
    sigma_xx : np.ndarray
        Normal stress in x direction
    sigma_yy : np.ndarray
        Normal stress in y direction
    sigma_xy : np.ndarray
        Shear stress

    Returns
    -------
    np.ndarray
        Von Mises equivalent stress
    """
    return np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)


def compute_principal_stresses(
    sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal stresses and direction from plane stress components.

    Principal stresses are the maximum and minimum normal stresses that
    occur on planes where shear stress is zero.

    Parameters
    ----------
    sigma_xx : np.ndarray
        Normal stress in x direction
    sigma_yy : np.ndarray
        Normal stress in y direction
    sigma_xy : np.ndarray
        Shear stress

    Returns
    -------
    sigma_1 : np.ndarray
        First (maximum) principal stress
    sigma_2 : np.ndarray
        Second (minimum) principal stress
    theta_p : np.ndarray
        Principal angle (radians) - angle from x-axis to σ_1 direction
    """
    sigma_avg = (sigma_xx + sigma_yy) / 2
    sigma_diff = (sigma_xx - sigma_yy) / 2
    R = np.sqrt(sigma_diff**2 + sigma_xy**2)

    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    theta_p = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)

    return sigma_1, sigma_2, theta_p


def compute_max_shear_stress(
    sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: np.ndarray
) -> np.ndarray:
    """
    Compute maximum in-plane shear stress.

    The maximum shear stress occurs on planes oriented at 45° to the
    principal directions.

        τ_max = (σ_1 - σ_2) / 2

    Parameters
    ----------
    sigma_xx : np.ndarray
        Normal stress in x direction
    sigma_yy : np.ndarray
        Normal stress in y direction
    sigma_xy : np.ndarray
        Shear stress

    Returns
    -------
    np.ndarray
        Maximum in-plane shear stress
    """
    sigma_diff = (sigma_xx - sigma_yy) / 2
    return np.sqrt(sigma_diff**2 + sigma_xy**2)
