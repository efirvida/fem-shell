"""
Laminate Module for Composite Materials.

This module provides classes and functions for defining and analyzing
laminated composite materials using Classical Lamination Theory (CLT).

The main classes are:
- StrengthProperties: Material strength values for failure analysis
- Ply: Single lamina definition with material, thickness, and orientation
- Laminate: Multi-layer laminate with ABD stiffness matrix computation

References
----------
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from fem_shell.core.material import OrthotropicMaterial


@dataclass
class StrengthProperties:
    """
    Strength properties for composite lamina failure analysis.

    Parameters
    ----------
    Xt : float
        Tensile strength in fiber direction (1-direction) [Pa]
    Xc : float
        Compressive strength in fiber direction [Pa]
    Yt : float
        Tensile strength transverse to fibers (2-direction) [Pa]
    Yc : float
        Compressive strength transverse to fibers [Pa]
    S12 : float
        In-plane shear strength [Pa]
    S23 : float, optional
        Transverse shear strength (for Hashin criterion). 
        Default is S12/2 if not provided.

    Notes
    -----
    Typical values for Carbon/Epoxy T300/5208:
    - Xt = 1500 MPa, Xc = 1500 MPa
    - Yt = 40 MPa, Yc = 246 MPa
    - S12 = 68 MPa
    """

    Xt: float  # Tensile strength in fiber direction
    Xc: float  # Compressive strength in fiber direction
    Yt: float  # Tensile strength transverse to fibers
    Yc: float  # Compressive strength transverse to fibers
    S12: float  # In-plane shear strength
    S23: Optional[float] = None  # Transverse shear strength

    def __post_init__(self):
        if self.S23 is None:
            self.S23 = self.S12 / 2


@dataclass
class Ply:
    """
    Single ply/lamina definition for laminated composites.

    Parameters
    ----------
    material : OrthotropicMaterial
        Orthotropic material properties (E1, E2, G12, nu12, etc.)
    thickness : float
        Ply thickness [m]
    angle : float
        Fiber orientation angle [degrees] measured from x-axis
    strength : StrengthProperties, optional
        Strength properties for failure analysis

    Attributes
    ----------
    z_bottom : float
        Distance from laminate mid-plane to bottom of ply (computed)
    z_top : float
        Distance from laminate mid-plane to top of ply (computed)

    Notes
    -----
    The angle convention follows the standard:
    - 0° = fibers aligned with x-axis
    - 90° = fibers aligned with y-axis
    - Positive angle = counterclockwise rotation
    """

    material: OrthotropicMaterial
    thickness: float
    angle: float  # degrees
    strength: Optional[StrengthProperties] = None
    z_bottom: float = field(default=0.0, init=False)
    z_top: float = field(default=0.0, init=False)

    @property
    def angle_rad(self) -> float:
        """Fiber orientation angle in radians."""
        return np.radians(self.angle)


def compute_Q(material: OrthotropicMaterial) -> np.ndarray:
    """
    Compute reduced stiffness matrix Q in principal material coordinates.

    Parameters
    ----------
    material : OrthotropicMaterial
        Material with E1, E2, G12, nu12 properties

    Returns
    -------
    np.ndarray
        3x3 reduced stiffness matrix Q

    Notes
    -----
    The reduced stiffness matrix relates in-plane stresses to strains:
    [σ1, σ2, τ12]^T = Q @ [ε1, ε2, γ12]^T

    Components:
    Q11 = E1 / (1 - ν12·ν21)
    Q22 = E2 / (1 - ν12·ν21)
    Q12 = ν12·E2 / (1 - ν12·ν21)
    Q66 = G12
    """
    E1, E2, _ = material.E
    nu12, _, _ = material.nu
    G12, _, _ = material.G

    # Reciprocal Poisson's ratio from symmetry
    nu21 = nu12 * E2 / E1

    denom = 1 - nu12 * nu21

    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12

    return np.array([
        [Q11, Q12, 0],
        [Q12, Q22, 0],
        [0, 0, Q66]
    ])


def compute_Qbar(material: OrthotropicMaterial, theta_deg: float) -> np.ndarray:
    """
    Compute transformed reduced stiffness matrix Qbar for a rotated lamina.

    Parameters
    ----------
    material : OrthotropicMaterial
        Orthotropic material properties
    theta_deg : float
        Fiber orientation angle in degrees

    Returns
    -------
    np.ndarray
        3x3 transformed reduced stiffness matrix Qbar

    Notes
    -----
    The transformation accounts for the fiber orientation angle θ:
    
    Qbar_11 = Q11·c⁴ + 2(Q12 + 2Q66)·s²c² + Q22·s⁴
    Qbar_22 = Q11·s⁴ + 2(Q12 + 2Q66)·s²c² + Q22·c⁴
    Qbar_12 = (Q11 + Q22 - 4Q66)·s²c² + Q12·(s⁴ + c⁴)
    Qbar_16 = (Q11 - Q12 - 2Q66)·sc³ + (Q12 - Q22 + 2Q66)·s³c
    Qbar_26 = (Q11 - Q12 - 2Q66)·s³c + (Q12 - Q22 + 2Q66)·sc³
    Qbar_66 = (Q11 + Q22 - 2Q12 - 2Q66)·s²c² + Q66·(s⁴ + c⁴)

    where c = cos(θ), s = sin(θ)
    """
    Q = compute_Q(material)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2 = c**2, s**2
    c3, s3 = c**3, s**3
    c4, s4 = c**4, s**4

    Qbar = np.zeros((3, 3))

    Qbar[0, 0] = Q11 * c4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s4
    Qbar[1, 1] = Q11 * s4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c4
    Qbar[0, 1] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s4 + c4)
    Qbar[1, 0] = Qbar[0, 1]
    Qbar[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c3 + (Q12 - Q22 + 2 * Q66) * s3 * c
    Qbar[2, 0] = Qbar[0, 2]
    Qbar[1, 2] = (Q11 - Q12 - 2 * Q66) * s3 * c + (Q12 - Q22 + 2 * Q66) * s * c3
    Qbar[2, 1] = Qbar[1, 2]
    Qbar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s4 + c4)

    return Qbar


def compute_shear_Cbar(material: OrthotropicMaterial, theta_deg: float) -> np.ndarray:
    """
    Compute transformed transverse shear moduli for a rotated lamina.

    Parameters
    ----------
    material : OrthotropicMaterial
        Material with G13 and G23 transverse shear moduli
    theta_deg : float
        Fiber orientation angle in degrees

    Returns
    -------
    np.ndarray
        2x2 transformed shear stiffness matrix [C55, C45; C45, C44]

    Notes
    -----
    The transformation is:
    C44 = G23·cos²θ + G13·sin²θ
    C55 = G13·cos²θ + G23·sin²θ  
    C45 = (G13 - G23)·sinθ·cosθ
    """
    _, G23, G13 = material.G

    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2 = c**2, s**2

    C44 = G23 * c2 + G13 * s2
    C55 = G13 * c2 + G23 * s2
    C45 = (G13 - G23) * s * c

    return np.array([
        [C55, C45],
        [C45, C44]
    ])


@dataclass
class Laminate:
    """
    Multi-layer laminate composite definition with CLT stiffness matrices.

    Parameters
    ----------
    plies : List[Ply]
        List of Ply objects from bottom to top of laminate
    shear_correction_factor : float, optional
        Shear correction factor for transverse shear stiffness.
        Default is 0.75 (typical for cross-ply laminates).
        Recommended values:
        - Unidirectional: 0.78-0.85
        - Cross-ply: 0.75-0.80
        - Quasi-isotropic: 0.70-0.78
        - Sandwich with soft core: 0.15-0.40

    Attributes
    ----------
    A : np.ndarray
        3x3 extensional stiffness matrix [N/m]
    B : np.ndarray
        3x3 coupling stiffness matrix [N]
    D : np.ndarray
        3x3 bending stiffness matrix [N·m]
    Cs : np.ndarray
        2x2 transverse shear stiffness matrix [N/m]
    total_thickness : float
        Total laminate thickness [m]
    n_plies : int
        Number of plies

    Notes
    -----
    The ABD matrices relate stress resultants to mid-plane strains and curvatures:

    [N]   [A  B] [ε⁰]
    [M] = [B  D] [κ ]

    For symmetric laminates, B = 0 (no membrane-bending coupling).

    Examples
    --------
    >>> # Create a symmetric cross-ply laminate [0/90]s
    >>> material = OrthotropicMaterial(...)
    >>> plies = [
    ...     Ply(material, 0.125e-3, 0),
    ...     Ply(material, 0.125e-3, 90),
    ...     Ply(material, 0.125e-3, 90),
    ...     Ply(material, 0.125e-3, 0),
    ... ]
    >>> laminate = Laminate(plies)
    """

    plies: List[Ply]
    shear_correction_factor: float = 0.75

    # Computed attributes
    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    D: np.ndarray = field(init=False, repr=False)
    Cs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.plies) == 0:
            raise ValueError("Laminate must have at least one ply")

        self._compute_layer_positions()
        self._compute_ABD_matrices()
        self._compute_shear_stiffness()

    @property
    def total_thickness(self) -> float:
        """Total laminate thickness."""
        return sum(ply.thickness for ply in self.plies)

    @property
    def n_plies(self) -> int:
        """Number of plies in the laminate."""
        return len(self.plies)

    def _compute_layer_positions(self) -> None:
        """Compute z-coordinates for each ply relative to mid-plane."""
        h_total = self.total_thickness
        z = -h_total / 2  # Start from bottom

        for ply in self.plies:
            ply.z_bottom = z
            z += ply.thickness
            ply.z_top = z

    def _compute_ABD_matrices(self) -> None:
        """
        Compute ABD stiffness matrices by integrating through thickness.

        Uses Classical Lamination Theory formulas:
        A_ij = Σ Qbar_ij^(k) · (z_{k+1} - z_k)
        B_ij = (1/2) · Σ Qbar_ij^(k) · (z_{k+1}² - z_k²)
        D_ij = (1/3) · Σ Qbar_ij^(k) · (z_{k+1}³ - z_k³)
        """
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))

        for ply in self.plies:
            Qbar = compute_Qbar(ply.material, ply.angle)
            z_bot = ply.z_bottom
            z_top = ply.z_top

            self.A += Qbar * (z_top - z_bot)
            self.B += 0.5 * Qbar * (z_top**2 - z_bot**2)
            self.D += (1 / 3) * Qbar * (z_top**3 - z_bot**3)

    def _compute_shear_stiffness(self) -> None:
        """
        Compute transverse shear stiffness matrix.

        Integrates transformed shear moduli through thickness and
        applies shear correction factor.
        """
        A44, A45, A55 = 0.0, 0.0, 0.0

        for ply in self.plies:
            Cbar = compute_shear_Cbar(ply.material, ply.angle)
            t = ply.thickness

            A55 += Cbar[0, 0] * t
            A45 += Cbar[0, 1] * t
            A44 += Cbar[1, 1] * t

        k = self.shear_correction_factor
        self.Cs = k * np.array([
            [A55, A45],
            [A45, A44]
        ])

    @property
    def is_symmetric(self) -> bool:
        """
        Check if laminate is symmetric about mid-plane.

        A symmetric laminate has B = 0 (no membrane-bending coupling).
        """
        return np.allclose(self.B, 0, atol=1e-10 * np.max(np.abs(self.A)))

    @property
    def is_balanced(self) -> bool:
        """
        Check if laminate is balanced (no tension-shear coupling).

        A balanced laminate has A16 = A26 = 0.
        """
        tol = 1e-10 * np.max(np.abs(self.A))
        return np.abs(self.A[0, 2]) < tol and np.abs(self.A[1, 2]) < tol

    def get_ABD_matrix(self) -> np.ndarray:
        """
        Get the full 6x6 ABD stiffness matrix.

        Returns
        -------
        np.ndarray
            6x6 matrix [[A, B], [B, D]]
        """
        ABD = np.zeros((6, 6))
        ABD[:3, :3] = self.A
        ABD[:3, 3:] = self.B
        ABD[3:, :3] = self.B
        ABD[3:, 3:] = self.D
        return ABD

    def get_equivalent_properties(self) -> dict:
        """
        Compute equivalent engineering properties for the laminate.

        Returns
        -------
        dict
            Dictionary with Ex, Ey, Gxy, nuxy for membrane behavior
            and Ex_b, Ey_b for bending behavior.

        Notes
        -----
        For symmetric laminates only. Uses A matrix for membrane
        and D matrix for bending properties.
        """
        h = self.total_thickness

        # Membrane compliance
        a = np.linalg.inv(self.A)

        # Equivalent membrane properties
        Ex_m = 1 / (h * a[0, 0])
        Ey_m = 1 / (h * a[1, 1])
        Gxy_m = 1 / (h * a[2, 2])
        nuxy_m = -a[0, 1] / a[0, 0]

        # Bending compliance
        d = np.linalg.inv(self.D)

        # Equivalent bending properties (using 12/h^3 factor)
        Ex_b = 12 / (h**3 * d[0, 0])
        Ey_b = 12 / (h**3 * d[1, 1])

        return {
            'Ex_membrane': Ex_m,
            'Ey_membrane': Ey_m,
            'Gxy_membrane': Gxy_m,
            'nuxy_membrane': nuxy_m,
            'Ex_bending': Ex_b,
            'Ey_bending': Ey_b,
            'total_thickness': h,
        }

    def __repr__(self) -> str:
        angles = [f"{p.angle:.0f}" for p in self.plies]
        return f"Laminate([{'/'.join(angles)}], h={self.total_thickness*1000:.3f}mm)"


def create_symmetric_laminate(
    half_plies: List[Ply],
    shear_correction_factor: float = 0.75,
) -> Laminate:
    """
    Create a symmetric laminate from half the ply stack.

    Parameters
    ----------
    half_plies : List[Ply]
        Plies from bottom to mid-plane
    shear_correction_factor : float, optional
        Shear correction factor

    Returns
    -------
    Laminate
        Symmetric laminate with mirrored ply stack

    Examples
    --------
    >>> # Create [0/90]s laminate
    >>> half = [Ply(mat, t, 0), Ply(mat, t, 90)]
    >>> laminate = create_symmetric_laminate(half)
    >>> # Results in [0/90/90/0]
    """
    import copy

    # Mirror the plies
    mirrored = [copy.deepcopy(ply) for ply in reversed(half_plies)]
    full_plies = half_plies + mirrored

    return Laminate(full_plies, shear_correction_factor)


def create_laminate_from_angles(
    material: OrthotropicMaterial,
    ply_thickness: float,
    angles: List[float],
    strength: Optional[StrengthProperties] = None,
    shear_correction_factor: float = 0.75,
) -> Laminate:
    """
    Create a laminate from a list of ply angles.

    Parameters
    ----------
    material : OrthotropicMaterial
        Material for all plies
    ply_thickness : float
        Thickness of each ply [m]
    angles : List[float]
        List of fiber orientation angles [degrees]
    strength : StrengthProperties, optional
        Strength properties for all plies
    shear_correction_factor : float, optional
        Shear correction factor

    Returns
    -------
    Laminate
        Laminate with specified ply stack

    Examples
    --------
    >>> laminate = create_laminate_from_angles(
    ...     material, 0.125e-3, [0, 45, -45, 90, 90, -45, 45, 0]
    ... )
    """
    plies = [
        Ply(material, ply_thickness, angle, strength)
        for angle in angles
    ]
    return Laminate(plies, shear_correction_factor)


# Common layup patterns
def quasi_isotropic_layup(n_repeats: int = 1) -> List[float]:
    """
    Generate quasi-isotropic layup angles [0/±45/90]s.

    Parameters
    ----------
    n_repeats : int
        Number of repetitions of the basic pattern

    Returns
    -------
    List[float]
        List of angles for symmetric quasi-isotropic laminate
    """
    half = [0, 45, -45, 90] * n_repeats
    return half + half[::-1]


def cross_ply_layup(n_plies: int = 4) -> List[float]:
    """
    Generate cross-ply layup angles [0/90]ns.

    Parameters
    ----------
    n_plies : int
        Total number of plies (must be multiple of 4 for symmetric)

    Returns
    -------
    List[float]
        List of angles for symmetric cross-ply laminate
    """
    if n_plies % 4 != 0:
        raise ValueError("n_plies must be multiple of 4 for symmetric cross-ply")

    n_half = n_plies // 2
    half = [0 if i % 2 == 0 else 90 for i in range(n_half)]
    return half + half[::-1]


def angle_ply_layup(angle: float, n_plies: int = 4) -> List[float]:
    """
    Generate angle-ply layup [±θ]ns.

    Parameters
    ----------
    angle : float
        Ply angle in degrees
    n_plies : int
        Total number of plies (must be multiple of 4 for symmetric)

    Returns
    -------
    List[float]
        List of angles for symmetric angle-ply laminate
    """
    if n_plies % 4 != 0:
        raise ValueError("n_plies must be multiple of 4 for symmetric angle-ply")

    n_half = n_plies // 2
    half = [angle if i % 2 == 0 else -angle for i in range(n_half)]
    return half + half[::-1]
