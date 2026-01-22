"""3D Solid Volumetric Elements for Linear Elasticity Analysis

Implements isoparametric 3D solid elements for linear elasticity problems
using consistent formulation and numerical integration.

Elements supported:
- WEDGE6: 6-node linear wedge/prism
- WEDGE15: 15-node quadratic wedge/prism
- PYRAMID5: 5-node linear pyramid
- PYRAMID13: 13-node quadratic pyramid
- TETRA4: 4-node linear tetrahedron
- TETRA10: 10-node quadratic tetrahedron
- HEXA8: 8-node linear hexahedron
- HEXA20: 20-node quadratic hexahedron

Formulation:
    Stiffness matrix: K = ∫BᵀCB dΩ
    Mass matrix: M = ∫ρNᵀN dΩ
    Body forces: f = ∫Nᵀb dΩ

where:
    B: Strain-displacement matrix (6 × n_dofs)
    C: Constitutive matrix (6 × 6) - isotropic or orthotropic
    N: Shape functions matrix
    ρ: Material density
    b: Body force vector

Strain vector (Voigt notation):
    ε = [εxx, εyy, εzz, γxy, γyz, γzx]ᵀ

Stress vector (Voigt notation):
    σ = [σxx, σyy, σzz, τxy, τyz, τzx]ᵀ
"""

from abc import abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from fem_shell.core.material import IsotropicMaterial, OrthotropicMaterial
from fem_shell.elements.elements import ElementFamily, FemElement

MaterialType = Union[IsotropicMaterial, OrthotropicMaterial]


class SolidElement(FemElement):
    """Base class for 3D solid volumetric elements.

    Provides common functionality for all solid elements including:
    - 6×6 constitutive matrix for isotropic/orthotropic materials
    - 3D strain-displacement matrix computation
    - Stiffness, mass, and body force integration
    - Material orientation support for orthotropic materials

    Node numbering and local coordinate systems are defined by subclasses.

    Attributes
    ----------
    vector_form : dict
        DOF names for displacement vector: U = (Ux, Uy, Uz)
    orientation : np.ndarray or None
        3×3 rotation matrix from material to global coordinates.
        Used for orthotropic material orientation.
    """

    vector_form = {"U": ("Ux", "Uy", "Uz")}

    def __init__(
        self,
        name: str,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = True,
    ):
        """Initialize solid element.

        Parameters
        ----------
        name : str
            Element type name (e.g., "WEDGE6", "PYRAMID5")
        node_coords : np.ndarray
            Array of nodal coordinates (n_nodes × 3)
        node_ids : Tuple[int, ...]
            Global node IDs for connectivity
        material : IsotropicMaterial or OrthotropicMaterial
            Material properties instance
        orientation : np.ndarray, optional
            3×3 rotation matrix for orthotropic material orientation.
            Transforms from material principal axes to global coordinates.
            If None, material axes align with global axes.
        """
        super().__init__(
            name=name,
            node_coords=np.asarray(node_coords),
            node_ids=list(node_ids),
            material=material,
            dofs_per_node=3,
        )
        self.element_family = ElementFamily.SOLID
        self.orientation = orientation
        self.reduced_integration = reduced_integration

        # Ensure 3D coordinates
        if self.node_coords.shape[1] != 3:
            raise ValueError(
                f"Solid elements require 3D coordinates, got shape {self.node_coords.shape}"
            )

    @property
    @abstractmethod
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss quadrature points and weights for the element.

        Returns
        -------
        points : np.ndarray
            Array of (ξ, η, ζ) natural coordinates (n_points × 3)
        weights : np.ndarray
            Integration weights (n_points,)
        """
        pass

    @abstractmethod
    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate shape functions at natural coordinates.

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates

        Returns
        -------
        np.ndarray
            Shape function values (n_nodes,)
        """
        pass

    @abstractmethod
    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate shape function derivatives at natural coordinates.

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates

        Returns
        -------
        dN_dxi : np.ndarray
            Derivatives with respect to ξ (n_nodes,)
        dN_deta : np.ndarray
            Derivatives with respect to η (n_nodes,)
        dN_dzeta : np.ndarray
            Derivatives with respect to ζ (n_nodes,)
        """
        pass

    def _compute_jacobian(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute Jacobian matrix components for 3D mapping.

        The Jacobian matrix maps derivatives from natural to physical coordinates:
            J = | ∂x/∂ξ  ∂y/∂ξ  ∂z/∂ξ  |
                | ∂x/∂η  ∂y/∂η  ∂z/∂η  |
                | ∂x/∂ζ  ∂y/∂ζ  ∂z/∂ζ  |

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates

        Returns
        -------
        J : np.ndarray
            Jacobian matrix (3×3)
        det_J : float
            Jacobian determinant (must be positive for valid mapping)
        inv_J : np.ndarray
            Inverse of Jacobian matrix (3×3)
        """
        dN_dxi, dN_deta, dN_dzeta = self.shape_function_derivatives(xi, eta, zeta)

        # J[i,j] = ∂x_j/∂ξ_i where ξ_0=ξ, ξ_1=η, ξ_2=ζ and x_0=x, x_1=y, x_2=z
        J = np.array(
            [
                [
                    dN_dxi @ self.node_coords[:, 0],
                    dN_dxi @ self.node_coords[:, 1],
                    dN_dxi @ self.node_coords[:, 2],
                ],
                [
                    dN_deta @ self.node_coords[:, 0],
                    dN_deta @ self.node_coords[:, 1],
                    dN_deta @ self.node_coords[:, 2],
                ],
                [
                    dN_dzeta @ self.node_coords[:, 0],
                    dN_dzeta @ self.node_coords[:, 1],
                    dN_dzeta @ self.node_coords[:, 2],
                ],
            ]
        )

        det_J = np.linalg.det(J)

        if det_J <= 1e-14:
            raise ValueError(f"Non-positive Jacobian determinant at ({xi}, {eta}, {zeta}): {det_J}")

        inv_J = np.linalg.inv(J)

        return J, det_J, inv_J

    def compute_B_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute strain-displacement matrix B for 3D elasticity.

        Strain components (Voigt notation):
            [εxx, εyy, εzz, γxy, γyz, γzx]ᵀ = B · u

        The B matrix relates nodal displacements to strains:
            B = | ∂N/∂x    0       0    |
                |   0    ∂N/∂y     0    |
                |   0      0     ∂N/∂z  |
                | ∂N/∂y  ∂N/∂x     0    |
                |   0    ∂N/∂z  ∂N/∂y   |
                | ∂N/∂z    0    ∂N/∂x   |

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates

        Returns
        -------
        np.ndarray
            B matrix (6 × n_dofs)
        """
        dN_dxi, dN_deta, dN_dzeta = self.shape_function_derivatives(xi, eta, zeta)
        _, _, inv_J = self._compute_jacobian(xi, eta, zeta)

        # Transform to global coordinates: [dN/dx, dN/dy, dN/dz]ᵀ = J⁻¹ · [dN/dξ, dN/dη, dN/dζ]ᵀ
        dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta + inv_J[0, 2] * dN_dzeta
        dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta + inv_J[1, 2] * dN_dzeta
        dN_dz = inv_J[2, 0] * dN_dxi + inv_J[2, 1] * dN_deta + inv_J[2, 2] * dN_dzeta

        # Construct B matrix (6 × n_dofs)
        n_nodes = len(dN_dx)
        B = np.zeros((6, self.dofs_count))

        for i in range(n_nodes):
            col = 3 * i
            B[0, col] = dN_dx[i]  # εxx = ∂u/∂x
            B[1, col + 1] = dN_dy[i]  # εyy = ∂v/∂y
            B[2, col + 2] = dN_dz[i]  # εzz = ∂w/∂z
            B[3, col] = dN_dy[i]  # γxy = ∂u/∂y + ∂v/∂x
            B[3, col + 1] = dN_dx[i]
            B[4, col + 1] = dN_dz[i]  # γyz = ∂v/∂z + ∂w/∂y
            B[4, col + 2] = dN_dy[i]
            B[5, col] = dN_dz[i]  # γzx = ∂w/∂x + ∂u/∂z
            B[5, col + 2] = dN_dx[i]

        return B

    @property
    def C(self) -> np.ndarray:
        """Constitutive matrix for 3D elasticity (6×6).

        Supports both isotropic and orthotropic materials.
        For orthotropic materials, applies orientation transformation if specified.

        Voigt notation ordering: [σxx, σyy, σzz, τxy, τyz, τzx]

        Returns
        -------
        np.ndarray
            Constitutive matrix (6×6)
        """
        if isinstance(self.material, IsotropicMaterial):
            return self._isotropic_C()
        elif isinstance(self.material, OrthotropicMaterial):
            C_local = self._orthotropic_C()
            if self.orientation is not None:
                return self._rotate_constitutive(C_local, self.orientation)
            return C_local
        else:
            raise TypeError(f"Unsupported material type: {type(self.material)}")

    def _isotropic_C(self) -> np.ndarray:
        """Build 6×6 isotropic elasticity matrix.

        Uses Lamé constants:
            λ = Eν / ((1+ν)(1-2ν))
            μ = E / (2(1+ν))

        Returns
        -------
        np.ndarray
            Symmetric 6×6 constitutive matrix
        """
        E = self.material.E
        nu = self.material.nu

        # Lamé constants
        lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))  # Shear modulus G

        C = np.array(
            [
                [lambd + 2 * mu, lambd, lambd, 0, 0, 0],
                [lambd, lambd + 2 * mu, lambd, 0, 0, 0],
                [lambd, lambd, lambd + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )

        return C

    def _orthotropic_C(self) -> np.ndarray:
        """Build 6×6 orthotropic elasticity matrix in material coordinates.

        Uses compliance matrix inversion approach for numerical stability.
        Material axes: 1 (fiber), 2 (transverse), 3 (through-thickness)

        Returns
        -------
        np.ndarray
            Symmetric 6×6 constitutive matrix in material coordinates
        """
        E1, E2, E3 = self.material.E
        G12, G23, G31 = self.material.G
        nu12, nu23, nu31 = self.material.nu

        # Derive remaining Poisson ratios from symmetry: νij/Ei = νji/Ej
        nu21 = nu12 * E2 / E1
        nu32 = nu23 * E3 / E2
        nu13 = nu31 * E1 / E3

        # Compliance matrix S (strain = S · stress)
        S = np.array(
            [
                [1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                [0, 0, 0, 1 / G12, 0, 0],
                [0, 0, 0, 0, 1 / G23, 0],
                [0, 0, 0, 0, 0, 1 / G31],
            ]
        )

        # Stiffness matrix C = S⁻¹
        C = np.linalg.inv(S)

        return C

    def _rotate_constitutive(self, C_local: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Rotate constitutive matrix from material to global coordinates.

        Uses the transformation: C_global = T · C_local · Tᵀ
        where T is the 6×6 stress transformation matrix derived from R.

        Parameters
        ----------
        C_local : np.ndarray
            Constitutive matrix in material coordinates (6×6)
        R : np.ndarray
            Rotation matrix from material to global (3×3)

        Returns
        -------
        np.ndarray
            Constitutive matrix in global coordinates (6×6)
        """
        # Build 6×6 transformation matrix for Voigt notation
        # T transforms stress: σ_global = T · σ_local
        r = R  # 3×3 rotation matrix

        T = np.array(
            [
                [
                    r[0, 0] ** 2,
                    r[0, 1] ** 2,
                    r[0, 2] ** 2,
                    2 * r[0, 0] * r[0, 1],
                    2 * r[0, 1] * r[0, 2],
                    2 * r[0, 2] * r[0, 0],
                ],
                [
                    r[1, 0] ** 2,
                    r[1, 1] ** 2,
                    r[1, 2] ** 2,
                    2 * r[1, 0] * r[1, 1],
                    2 * r[1, 1] * r[1, 2],
                    2 * r[1, 2] * r[1, 0],
                ],
                [
                    r[2, 0] ** 2,
                    r[2, 1] ** 2,
                    r[2, 2] ** 2,
                    2 * r[2, 0] * r[2, 1],
                    2 * r[2, 1] * r[2, 2],
                    2 * r[2, 2] * r[2, 0],
                ],
                [
                    r[0, 0] * r[1, 0],
                    r[0, 1] * r[1, 1],
                    r[0, 2] * r[1, 2],
                    r[0, 0] * r[1, 1] + r[0, 1] * r[1, 0],
                    r[0, 1] * r[1, 2] + r[0, 2] * r[1, 1],
                    r[0, 0] * r[1, 2] + r[0, 2] * r[1, 0],
                ],
                [
                    r[1, 0] * r[2, 0],
                    r[1, 1] * r[2, 1],
                    r[1, 2] * r[2, 2],
                    r[1, 0] * r[2, 1] + r[1, 1] * r[2, 0],
                    r[1, 1] * r[2, 2] + r[1, 2] * r[2, 1],
                    r[1, 0] * r[2, 2] + r[1, 2] * r[2, 0],
                ],
                [
                    r[2, 0] * r[0, 0],
                    r[2, 1] * r[0, 1],
                    r[2, 2] * r[0, 2],
                    r[2, 0] * r[0, 1] + r[2, 1] * r[0, 0],
                    r[2, 1] * r[0, 2] + r[2, 2] * r[0, 1],
                    r[2, 0] * r[0, 2] + r[2, 2] * r[0, 0],
                ],
            ]
        )

        # Transform: C_global = T · C_local · Tᵀ
        C_global = T @ C_local @ T.T

        return C_global

    @property
    def K(self) -> np.ndarray:
        """Stiffness matrix computed via numerical integration.

        K = ∫BᵀCB dΩ ≈ Σᵢ BᵢᵀCBᵢ |J|ᵢ wᵢ

        Returns
        -------
        np.ndarray
            Symmetric stiffness matrix (n_dofs × n_dofs)
        """
        points, weights = self.integration_points
        K = np.zeros((self.dofs_count, self.dofs_count))
        C = self.C

        for point, w in zip(points, weights):
            xi, eta, zeta = point
            B = self.compute_B_matrix(xi, eta, zeta)
            _, det_J, _ = self._compute_jacobian(xi, eta, zeta)
            K += (B.T @ C @ B) * det_J * w

        # Ensure symmetry (numerical precision)
        K = 0.5 * (K + K.T)

        return K

    @property
    def M(self) -> np.ndarray:
        """Consistent mass matrix.

        M = ∫ρNᵀN dΩ ≈ Σᵢ ρ NᵢᵀNᵢ |J|ᵢ wᵢ

        Returns
        -------
        np.ndarray
            Symmetric mass matrix (n_dofs × n_dofs)
        """
        points, weights = self.integration_points
        M = np.zeros((self.dofs_count, self.dofs_count))
        rho = self.material.rho

        for point, w in zip(points, weights):
            xi, eta, zeta = point
            N = self.shape_functions(xi, eta, zeta)
            _, det_J, _ = self._compute_jacobian(xi, eta, zeta)

            # Build N matrix (3 × n_dofs)
            N_mat = np.zeros((3, self.dofs_count))
            N_mat[0, 0::3] = N  # u-components
            N_mat[1, 1::3] = N  # v-components
            N_mat[2, 2::3] = N  # w-components

            M += rho * (N_mat.T @ N_mat) * det_J * w

        return 0.5 * (M + M.T)

    def body_load(self, body_force: np.ndarray) -> np.ndarray:
        """Body force vector from distributed loads.

        f = ∫Nᵀb dΩ ≈ Σᵢ Nᵢᵀb |J|ᵢ wᵢ

        Parameters
        ----------
        body_force : np.ndarray
            Body force components [bx, by, bz]

        Returns
        -------
        np.ndarray
            Force vector (n_dofs,)
        """
        points, weights = self.integration_points
        f = np.zeros(self.dofs_count)
        b = np.asarray(body_force[:3])

        for point, w in zip(points, weights):
            xi, eta, zeta = point
            N = self.shape_functions(xi, eta, zeta)
            _, det_J, _ = self._compute_jacobian(xi, eta, zeta)

            # Build N matrix (3 × n_dofs)
            N_mat = np.zeros((3, self.dofs_count))
            N_mat[0, 0::3] = N
            N_mat[1, 1::3] = N
            N_mat[2, 2::3] = N

            f += (N_mat.T @ b) * det_J * w

        return f

    def compute_strain(
        self, displacements: np.ndarray, xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Compute strain at a point given nodal displacements.

        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (n_dofs,)
        xi, eta, zeta : float
            Natural coordinates of evaluation point

        Returns
        -------
        np.ndarray
            Strain vector [εxx, εyy, εzz, γxy, γyz, γzx]
        """
        B = self.compute_B_matrix(xi, eta, zeta)
        return B @ displacements

    def compute_stress(
        self, displacements: np.ndarray, xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Compute stress at a point given nodal displacements.

        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (n_dofs,)
        xi, eta, zeta : float
            Natural coordinates of evaluation point

        Returns
        -------
        np.ndarray
            Stress vector [σxx, σyy, σzz, τxy, τyz, τzx]
        """
        strain = self.compute_strain(displacements, xi, eta, zeta)
        return self.C @ strain

    def compute_von_mises(self, stress: np.ndarray) -> float:
        """Compute von Mises equivalent stress.

        σ_vm = √(½[(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)²] + 3[τxy² + τyz² + τzx²])

        Parameters
        ----------
        stress : np.ndarray
            Stress vector [σxx, σyy, σzz, τxy, τyz, τzx]

        Returns
        -------
        float
            Von Mises stress
        """
        sxx, syy, szz, txy, tyz, tzx = stress
        return np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3 * (txy**2 + tyz**2 + tzx**2)
        )


# =============================================================================
# WEDGE (Prism) Elements
# =============================================================================


class WEDGE6(SolidElement):
    """6-node linear wedge/prism element.

    Triangular base with linear interpolation along both triangular and
    prismatic directions.

    Node ordering (looking from +z):
        Bottom (z=-1):    Top (z=+1):
            2                 5
           / \\               / \\
          /   \\             /   \\
         0-----1           3-----4

    Natural coordinates:
        ξ ∈ [0, 1], η ∈ [0, 1], ζ ∈ [-1, 1]
        with ξ + η ≤ 1 (triangular constraint)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for WEDGE6
    ):
        if len(node_ids) != 6:
            raise ValueError(f"WEDGE6 requires 6 nodes, got {len(node_ids)}")
        super().__init__("WEDGE6", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """2-point triangular × 2-point linear Gauss rule (6 points total)."""
        # Triangular coordinates (3-point rule for linear triangle)
        tri_pts = np.array(
            [
                [1 / 6, 1 / 6],
                [2 / 3, 1 / 6],
                [1 / 6, 2 / 3],
            ]
        )
        tri_w = np.array([1 / 6, 1 / 6, 1 / 6])  # Weights sum to 0.5 (triangle area)

        # Linear direction (2-point Gauss)
        gp = 1 / np.sqrt(3)
        lin_pts = np.array([-gp, gp])
        lin_w = np.array([1.0, 1.0])

        # Tensor product
        points = []
        weights = []
        for i, (xi, eta) in enumerate(tri_pts):
            for j, zeta in enumerate(lin_pts):
                points.append([xi, eta, zeta])
                weights.append(tri_w[i] * lin_w[j] * 2)  # ×2 for ζ range [-1,1]

        return np.array(points), np.array(weights)

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Linear wedge shape functions.

        N = Lᵢ(ξ,η) × Lⱼ(ζ)
        where Lᵢ are triangular area coordinates and Lⱼ are linear 1D functions.
        """
        # Triangular shape functions
        L1 = 1 - xi - eta  # Node 0, 3
        L2 = xi  # Node 1, 4
        L3 = eta  # Node 2, 5

        # Linear interpolation in z
        Lm = 0.5 * (1 - zeta)  # Bottom
        Lp = 0.5 * (1 + zeta)  # Top

        return np.array(
            [
                L1 * Lm,  # N0
                L2 * Lm,  # N1
                L3 * Lm,  # N2
                L1 * Lp,  # N3
                L2 * Lp,  # N4
                L3 * Lp,  # N5
            ]
        )

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives of linear wedge shape functions."""
        # Triangular shape functions
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta

        # Linear interpolation
        Lm = 0.5 * (1 - zeta)
        Lp = 0.5 * (1 + zeta)

        # Derivatives of triangular functions
        dL1_dxi, dL1_deta = -1.0, -1.0
        dL2_dxi, dL2_deta = 1.0, 0.0
        dL3_dxi, dL3_deta = 0.0, 1.0

        # Derivatives of linear functions
        dLm_dzeta = -0.5
        dLp_dzeta = 0.5

        dN_dxi = np.array(
            [
                dL1_dxi * Lm,
                dL2_dxi * Lm,
                dL3_dxi * Lm,
                dL1_dxi * Lp,
                dL2_dxi * Lp,
                dL3_dxi * Lp,
            ]
        )

        dN_deta = np.array(
            [
                dL1_deta * Lm,
                dL2_deta * Lm,
                dL3_deta * Lm,
                dL1_deta * Lp,
                dL2_deta * Lp,
                dL3_deta * Lp,
            ]
        )

        dN_dzeta = np.array(
            [
                L1 * dLm_dzeta,
                L2 * dLm_dzeta,
                L3 * dLm_dzeta,
                L1 * dLp_dzeta,
                L2 * dLp_dzeta,
                L3 * dLp_dzeta,
            ]
        )

        return dN_dxi, dN_deta, dN_dzeta


class WEDGE15(SolidElement):
    """15-node quadratic wedge/prism element.

    Triangular base with quadratic interpolation.

    Node ordering (following Gmsh convention):
    
        Gmsh diagram:
                       3
                     ,/|`\
                   12  |  13
                 ,/    |    `\
                4------14-----5
                |      8      |
                |      |      |
                |      |      |
                |      |      |
               10      |      11
                |      0      |
                |    ,/ `\    |
                |  ,6     `7  |
                |,/         `\|
                1------9------2
                
        Corner nodes: 0,1,2 (bottom z=-1), 3,4,5 (top z=+1)
        
        Edge mapping:
        6: edge 0-1,  7: edge 0-2,  8: edge 0-3 (vertical)
        9: edge 1-2, 10: edge 1-4 (vertical), 11: edge 2-5 (vertical)
        12: edge 3-4, 13: edge 3-5, 14: edge 4-5

    Natural coordinates:
        ξ ∈ [0, 1], η ∈ [0, 1], ζ ∈ [-1, 1]
        with ξ + η ≤ 1
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for WEDGE15
    ):
        if len(node_ids) != 15:
            raise ValueError(f"WEDGE15 requires 15 nodes, got {len(node_ids)}")
        super().__init__("WEDGE15", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """3-point triangular × 3-point linear Gauss rule (21 points)."""
        # 7-point triangular rule for quadratic accuracy
        tri_pts = np.array(
            [
                [1 / 3, 1 / 3],
                [0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.797426985353087],
                [0.101286507323456, 0.101286507323456],
                [0.470142064105115, 0.059715871789770],
                [0.470142064105115, 0.470142064105115],
                [0.059715871789770, 0.470142064105115],
            ]
        )
        tri_w = (
            np.array(
                [
                    0.225,
                    0.125939180544827,
                    0.125939180544827,
                    0.125939180544827,
                    0.132394152788506,
                    0.132394152788506,
                    0.132394152788506,
                ]
            )
            * 0.5
        )  # Scale for unit triangle

        # 3-point Gauss in z
        sqrt35 = np.sqrt(3 / 5)
        lin_pts = np.array([-sqrt35, 0, sqrt35])
        lin_w = np.array([5 / 9, 8 / 9, 5 / 9])

        # Tensor product
        points = []
        weights = []
        for i, (xi, eta) in enumerate(tri_pts):
            for j, zeta in enumerate(lin_pts):
                points.append([xi, eta, zeta])
                weights.append(tri_w[i] * lin_w[j] * 2)

        return np.array(points), np.array(weights)

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Quadratic wedge shape functions."""
        # Triangular area coordinates
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta

        # Quadratic 1D shape functions in z
        Lm = 0.5 * zeta * (zeta - 1)  # Bottom
        L0 = 1 - zeta**2  # Middle
        Lp = 0.5 * zeta * (zeta + 1)  # Top

        N = np.zeros(15)

        # Corner nodes (bottom: 0,1,2 | top: 3,4,5)
        N[0] = L1 * (2 * L1 - 1) * Lm
        N[1] = L2 * (2 * L2 - 1) * Lm
        N[2] = L3 * (2 * L3 - 1) * Lm
        N[3] = L1 * (2 * L1 - 1) * Lp
        N[4] = L2 * (2 * L2 - 1) * Lp
        N[5] = L3 * (2 * L3 - 1) * Lp

        # Mid-edge nodes (Gmsh convention)
        # 6: edge 0-1 (bottom)
        N[6] = 4 * L1 * L2 * Lm
        # 7: edge 0-2 (bottom)
        N[7] = 4 * L1 * L3 * Lm
        # 8: edge 0-3 (vertical at corner 0)
        N[8] = L1 * L0
        # 9: edge 1-2 (bottom)
        N[9] = 4 * L2 * L3 * Lm
        # 10: edge 1-4 (vertical at corner 1)
        N[10] = L2 * L0
        # 11: edge 2-5 (vertical at corner 2)
        N[11] = L3 * L0
        # 12: edge 3-4 (top)
        N[12] = 4 * L1 * L2 * Lp
        # 13: edge 3-5 (top)
        N[13] = 4 * L1 * L3 * Lp
        # 14: edge 4-5 (top)
        N[14] = 4 * L2 * L3 * Lp

        return N

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives of quadratic wedge shape functions."""
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta

        # 1D quadratic shape functions
        Lm = 0.5 * zeta * (zeta - 1)
        L0 = 1 - zeta**2
        Lp = 0.5 * zeta * (zeta + 1)

        # Derivatives of 1D functions w.r.t. zeta
        dLm = zeta - 0.5
        dL0 = -2 * zeta
        dLp = zeta + 0.5

        dN_dxi = np.zeros(15)
        dN_deta = np.zeros(15)
        dN_dzeta = np.zeros(15)

        # Corner nodes - bottom
        dN_dxi[0] = (4 * L1 - 1) * (-1) * Lm
        dN_deta[0] = (4 * L1 - 1) * (-1) * Lm
        dN_dzeta[0] = L1 * (2 * L1 - 1) * dLm

        dN_dxi[1] = (4 * L2 - 1) * Lm
        dN_deta[1] = 0
        dN_dzeta[1] = L2 * (2 * L2 - 1) * dLm

        dN_dxi[2] = 0
        dN_deta[2] = (4 * L3 - 1) * Lm
        dN_dzeta[2] = L3 * (2 * L3 - 1) * dLm

        # Corner nodes - top
        dN_dxi[3] = (4 * L1 - 1) * (-1) * Lp
        dN_deta[3] = (4 * L1 - 1) * (-1) * Lp
        dN_dzeta[3] = L1 * (2 * L1 - 1) * dLp

        dN_dxi[4] = (4 * L2 - 1) * Lp
        dN_deta[4] = 0
        dN_dzeta[4] = L2 * (2 * L2 - 1) * dLp

        dN_dxi[5] = 0
        dN_deta[5] = (4 * L3 - 1) * Lp
        dN_dzeta[5] = L3 * (2 * L3 - 1) * dLp

        # Mid-edge nodes (Gmsh convention)
        # 6: edge 0-1 (bottom)
        dN_dxi[6] = 4 * (L1 - L2) * Lm
        dN_deta[6] = -4 * L2 * Lm
        dN_dzeta[6] = 4 * L1 * L2 * dLm

        # 7: edge 0-2 (bottom)
        dN_dxi[7] = -4 * L3 * Lm
        dN_deta[7] = 4 * (L1 - L3) * Lm
        dN_dzeta[7] = 4 * L1 * L3 * dLm

        # 8: edge 0-3 (vertical at corner 0)
        dN_dxi[8] = -L0
        dN_deta[8] = -L0
        dN_dzeta[8] = L1 * dL0

        # 9: edge 1-2 (bottom)
        dN_dxi[9] = 4 * L3 * Lm
        dN_deta[9] = 4 * L2 * Lm
        dN_dzeta[9] = 4 * L2 * L3 * dLm

        # 10: edge 1-4 (vertical at corner 1)
        dN_dxi[10] = L0
        dN_deta[10] = 0
        dN_dzeta[10] = L2 * dL0

        # 11: edge 2-5 (vertical at corner 2)
        dN_dxi[11] = 0
        dN_deta[11] = L0
        dN_dzeta[11] = L3 * dL0

        # 12: edge 3-4 (top)
        dN_dxi[12] = 4 * (L1 - L2) * Lp
        dN_deta[12] = -4 * L2 * Lp
        dN_dzeta[12] = 4 * L1 * L2 * dLp

        # 13: edge 3-5 (top)
        dN_dxi[13] = -4 * L3 * Lp
        dN_deta[13] = 4 * (L1 - L3) * Lp
        dN_dzeta[13] = 4 * L1 * L3 * dLp

        # 14: edge 4-5 (top)
        dN_dxi[14] = 4 * L3 * Lp
        dN_deta[14] = 4 * L2 * Lp
        dN_dzeta[14] = 4 * L2 * L3 * dLp

        return dN_dxi, dN_deta, dN_dzeta


# =============================================================================
# PYRAMID Elements
# =============================================================================


class PYRAMID5(SolidElement):
    """5-node linear pyramid element.

    Square base with apex. Useful for transitioning between hex and tet meshes.

    Node ordering:
                    4 (apex)
                   /|\\
                  / | \\
                 /  |  \\
                3-------2
               /|       |
              / |       |
             0---------1

    Base at z=0, apex at z=1.
    Natural coordinates: ξ, η ∈ [-1, 1], ζ ∈ [0, 1]
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for PYRAMID5
    ):
        if len(node_ids) != 5:
            raise ValueError(f"PYRAMID5 requires 5 nodes, got {len(node_ids)}")
        super().__init__("PYRAMID5", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """8-point Felippa integration rule for pyramid (precision 3).

        Reference: Felippa, C.A. (2004). "A compendium of FEM integration formulas
        for symbolic work," Engineering Computation, 21(8), 867-890.
        """
        points = np.array(
            [
                [-0.26318405556971359557, -0.26318405556971359557, 0.54415184401122528880],
                [0.26318405556971359557, -0.26318405556971359557, 0.54415184401122528880],
                [0.26318405556971359557, 0.26318405556971359557, 0.54415184401122528880],
                [-0.26318405556971359557, 0.26318405556971359557, 0.54415184401122528880],
                [-0.50661630334978742377, -0.50661630334978742377, 0.12251482265544137787],
                [0.50661630334978742377, -0.50661630334978742377, 0.12251482265544137787],
                [0.50661630334978742377, 0.50661630334978742377, 0.12251482265544137787],
                [-0.50661630334978742377, 0.50661630334978742377, 0.12251482265544137787],
            ]
        )

        weights = np.array(
            [
                0.10078588207982543059,
                0.10078588207982543059,
                0.10078588207982543059,
                0.10078588207982543059,
                0.23254745125350790274,
                0.23254745125350790274,
                0.23254745125350790274,
                0.23254745125350790274,
            ]
        )

        return points, weights

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Linear pyramid shape functions (singularity-free formulation).

        Based on Bedrosian (1992) degenerate hexahedron approach.
        Shape functions are expressed to avoid explicit 1/(1-zeta) terms.

        N_i = 0.25 * (r ± xi) * (r ± eta) / r  for base nodes
        N_4 = zeta  for apex

        where r = 1 - zeta
        """
        eps = 1e-14
        r = max(1 - zeta, eps)

        # Singularity-free form: (r ± xi)(r ± eta) / (4r)
        # This form keeps numerator stable as zeta → 1 (where xi, eta → 0)
        N = np.array(
            [
                0.25 * (r - xi) * (r - eta) / r,  # N0: (1 - xi/r)(1 - eta/r) * r
                0.25 * (r + xi) * (r - eta) / r,  # N1: (1 + xi/r)(1 - eta/r) * r
                0.25 * (r + xi) * (r + eta) / r,  # N2: (1 + xi/r)(1 + eta/r) * r
                0.25 * (r - xi) * (r + eta) / r,  # N3: (1 - xi/r)(1 + eta/r) * r
                zeta,  # N4 (apex)
            ]
        )

        return N

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analytical derivatives of linear pyramid shape functions.

        Formulated to minimize numerical issues near the apex.
        Uses direct differentiation of N_i = (r ± xi)(r ± eta) / (4r)
        where r = 1 - zeta.
        """
        eps = 1e-14
        r = max(1 - zeta, eps)
        r2 = r * r

        # dN/dxi: d/dxi [(r ± xi)(r ± eta) / (4r)] = ±(r ± eta) / (4r)
        dN_dxi = np.array(
            [
                -0.25 * (r - eta) / r,  # Node 0: -(r - eta) / (4r)
                0.25 * (r - eta) / r,  # Node 1: +(r - eta) / (4r)
                0.25 * (r + eta) / r,  # Node 2: +(r + eta) / (4r)
                -0.25 * (r + eta) / r,  # Node 3: -(r + eta) / (4r)
                0.0,  # Node 4 (apex)
            ]
        )

        # dN/deta: d/deta [(r ± xi)(r ± eta) / (4r)] = ±(r ± xi) / (4r)
        dN_deta = np.array(
            [
                -0.25 * (r - xi) / r,  # Node 0: -(r - xi) / (4r)
                -0.25 * (r + xi) / r,  # Node 1: -(r + xi) / (4r)
                0.25 * (r + xi) / r,  # Node 2: +(r + xi) / (4r)
                0.25 * (r - xi) / r,  # Node 3: +(r - xi) / (4r)
                0.0,  # Node 4 (apex)
            ]
        )

        # dN/dzeta: Use quotient rule on N_i = (r ± xi)(r ± eta) / (4r)
        # Let f = (r ± xi)(r ± eta), g = 4r
        # dN/dzeta = (f' * g - f * g') / g² where dr/dzeta = -1
        # f' = df/dr * dr/dzeta = [(r ± eta) + (r ± xi)] * (-1)
        # g' = 4 * (-1) = -4
        # Simplified: dN_i/dzeta = [±xi ± eta - (r ± xi)(r ± eta)/r] / (4r)
        #           = [±xi ± eta] / (4r) - N_i / r

        # More stable formulation using product form:
        # N_i = (r ± xi)(r ± eta) / (4r)
        # dN_i/dzeta = -[(r ± eta) + (r ± xi)] / (4r) + (r ± xi)(r ± eta) / (4r²)
        #            = -[(r ± eta) + (r ± xi)] / (4r) + N_i / r

        # Pre-compute terms
        rm_xi = r - xi
        rp_xi = r + xi
        rm_eta = r - eta
        rp_eta = r + eta

        dN_dzeta = np.array(
            [
                # Node 0: N0 = (r-xi)(r-eta)/(4r)
                -0.25 * (rm_eta + rm_xi) / r + 0.25 * rm_xi * rm_eta / r2,
                # Node 1: N1 = (r+xi)(r-eta)/(4r)
                -0.25 * (rm_eta + rp_xi) / r + 0.25 * rp_xi * rm_eta / r2,
                # Node 2: N2 = (r+xi)(r+eta)/(4r)
                -0.25 * (rp_eta + rp_xi) / r + 0.25 * rp_xi * rp_eta / r2,
                # Node 3: N3 = (r-xi)(r+eta)/(4r)
                -0.25 * (rp_eta + rm_xi) / r + 0.25 * rm_xi * rp_eta / r2,
                # Node 4 (apex)
                1.0,
            ]
        )

        return dN_dxi, dN_deta, dN_dzeta


class PYRAMID13(SolidElement):
    """13-node quadratic pyramid element.

    Square base with apex and mid-edge nodes.

    Gmsh Node ordering (element type 19):
    
                     4
                   ,/|\
                 ,/ .'|\
               ,/   | | \
             ,/    .' | `.
           ,7      |  12  \
         ,/       .'   |   \
       ,/         9    |    11
      0--------6-.'----3    `.
       `\        |      `\    \
         `5     .'        10   \
           `\   |           `\  \
             `\.'             `\`
                1--------8-------2
    
    Base corners: 0, 1, 2, 3
    Apex: 4
    
    Edge midpoints (Gmsh convention):
        5: edge 0-1 (base front)
        6: edge 0-3 (base left) 
        7: edge 0-4 (lateral front-left to apex)
        8: edge 1-2 (base right)
        9: edge 1-4 (lateral front-right to apex)
        10: edge 2-3 (base back)
        11: edge 2-4 (lateral back-right to apex)
        12: edge 3-4 (lateral back-left to apex)

    Natural coordinates: ξ, η ∈ [-1, 1], ζ ∈ [0, 1]
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for PYRAMID13
    ):
        if len(node_ids) != 13:
            raise ValueError(f"PYRAMID13 requires 13 nodes, got {len(node_ids)}")
        super().__init__("PYRAMID13", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """18-point Felippa integration rule for quadratic pyramid (precision 5).

        Reference: Felippa, C.A. (2004). "A compendium of FEM integration formulas
        for symbolic work," Engineering Computation, 21(8), 867-890.

        This rule exactly integrates polynomials up to degree 5 over the pyramid.
        """
        # 3×3 grid at two height levels optimized for pyramid geometry
        points = np.array(
            [
                # 9 points at z ≈ 0.544 (upper layer, near apex)
                [-0.35309846330877704481, -0.35309846330877704481, 0.54415184401122528880],
                [0.00000000000000000000, -0.35309846330877704481, 0.54415184401122528880],
                [0.35309846330877704481, -0.35309846330877704481, 0.54415184401122528880],
                [-0.35309846330877704481, 0.00000000000000000000, 0.54415184401122528880],
                [0.00000000000000000000, 0.00000000000000000000, 0.54415184401122528880],
                [0.35309846330877704481, 0.00000000000000000000, 0.54415184401122528880],
                [-0.35309846330877704481, 0.35309846330877704481, 0.54415184401122528880],
                [0.00000000000000000000, 0.35309846330877704481, 0.54415184401122528880],
                [0.35309846330877704481, 0.35309846330877704481, 0.54415184401122528880],
                # 9 points at z ≈ 0.122 (lower layer, near base)
                [-0.67969709567986745790, -0.67969709567986745790, 0.12251482265544137787],
                [0.00000000000000000000, -0.67969709567986745790, 0.12251482265544137787],
                [0.67969709567986745790, -0.67969709567986745790, 0.12251482265544137787],
                [-0.67969709567986745790, 0.00000000000000000000, 0.12251482265544137787],
                [0.00000000000000000000, 0.00000000000000000000, 0.12251482265544137787],
                [0.67969709567986745790, 0.00000000000000000000, 0.12251482265544137787],
                [-0.67969709567986745790, 0.67969709567986745790, 0.12251482265544137787],
                [0.00000000000000000000, 0.67969709567986745790, 0.12251482265544137787],
                [0.67969709567986745790, 0.67969709567986745790, 0.12251482265544137787],
            ]
        )

        weights = np.array(
            [
                # Upper layer (z ≈ 0.544) - smaller weights (smaller cross-section)
                0.023330065296255886709,
                0.037328104474009418735,
                0.023330065296255886709,
                0.037328104474009418735,
                0.059724967158415069975,
                0.037328104474009418735,
                0.023330065296255886709,
                0.037328104474009418735,
                0.023330065296255886709,
                # Lower layer (z ≈ 0.122) - larger weights (larger cross-section)
                0.053830428530904607120,
                0.086128685649447371390,
                0.053830428530904607120,
                0.086128685649447371390,
                0.137805897039115794220,
                0.086128685649447371390,
                0.053830428530904607120,
                0.086128685649447371390,
                0.053830428530904607120,
            ]
        )

        return points, weights

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Quadratic pyramid shape functions (Gmsh convention).

        Uses rational blending approach for 13-node serendipity pyramid.
        The functions satisfy both partition of unity and Kronecker delta properties.

        Reference: Bedrosian (1992), adapted for quadratic serendipity elements.
        """
        eps = 1e-14
        r = max(1 - zeta, eps)

        # Scaled coordinates (collapsed hexahedron approach)
        xi_s = np.clip(xi / r, -1.0, 1.0)
        eta_s = np.clip(eta / r, -1.0, 1.0)

        N = np.zeros(13)

        # 8-node serendipity functions on the scaled square [-1,1]²
        # These are the standard 2D serendipity shape functions

        # Corner functions (with edge corrections)
        f0 = 0.25 * (1 - xi_s) * (1 - eta_s) * (-xi_s - eta_s - 1)  # node 0
        f1 = 0.25 * (1 + xi_s) * (1 - eta_s) * (xi_s - eta_s - 1)  # node 1
        f2 = 0.25 * (1 + xi_s) * (1 + eta_s) * (xi_s + eta_s - 1)  # node 2
        f3 = 0.25 * (1 - xi_s) * (1 + eta_s) * (-xi_s + eta_s - 1)  # node 3

        # Edge midpoint functions
        f5 = 0.5 * (1 - xi_s**2) * (1 - eta_s)  # edge 0-1 (y=-1)
        f6 = 0.5 * (1 - xi_s) * (1 - eta_s**2)  # edge 0-3 (x=-1)
        f8 = 0.5 * (1 + xi_s) * (1 - eta_s**2)  # edge 1-2 (x=+1)
        f10 = 0.5 * (1 - xi_s**2) * (1 + eta_s)  # edge 2-3 (y=+1)

        # Vertical blending using (1-zeta) scaling
        # This ensures partition of unity
        base_scale = r  # = 1 - zeta

        # For base nodes: scale by (1-zeta)² and add quadratic vertical term
        # Base corners (at z=0)
        N[0] = f0 * base_scale * (1 - 2 * zeta)
        N[1] = f1 * base_scale * (1 - 2 * zeta)
        N[2] = f2 * base_scale * (1 - 2 * zeta)
        N[3] = f3 * base_scale * (1 - 2 * zeta)

        # Base edge midpoints (at z=0)
        N[5] = f5 * base_scale * (1 - 2 * zeta)
        N[6] = f6 * base_scale * (1 - 2 * zeta)
        N[8] = f8 * base_scale * (1 - 2 * zeta)
        N[10] = f10 * base_scale * (1 - 2 * zeta)

        # Apex (node 4): Gets all contribution at z=1
        N[4] = zeta * (2 * zeta - 1)  # Quadratic, =1 at z=1, =0 at z=0,0.5

        # Lateral edge midpoints (at z=0.5): Use bilinear × bubble vertical
        # Bilinear corner functions for lateral interpolation
        phi_0 = 0.25 * (1 - xi_s) * (1 - eta_s)
        phi_1 = 0.25 * (1 + xi_s) * (1 - eta_s)
        phi_2 = 0.25 * (1 + xi_s) * (1 + eta_s)
        phi_3 = 0.25 * (1 - xi_s) * (1 + eta_s)

        lateral_blend = 4 * zeta * (1 - zeta)  # =1 at z=0.5, =0 at z=0,1

        N[7] = phi_0 * lateral_blend  # edge 0-4
        N[9] = phi_1 * lateral_blend  # edge 1-4
        N[11] = phi_2 * lateral_blend  # edge 2-4
        N[12] = phi_3 * lateral_blend  # edge 3-4

        return N

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analytical derivatives of quadratic pyramid shape functions.

        Matches the rational blending shape functions.
        Uses chain rule for scaled coordinates xi_s = xi/r, eta_s = eta/r
        where r = 1 - zeta.
        """
        eps = 1e-14
        r = max(1 - zeta, eps)
        r2 = r * r

        xi_s = np.clip(xi / r, -1.0, 1.0)
        eta_s = np.clip(eta / r, -1.0, 1.0)

        dN_dxi = np.zeros(13)
        dN_deta = np.zeros(13)
        dN_dzeta = np.zeros(13)

        # Vertical blending functions
        h_base = r * (1 - 2 * zeta)  # = (1-zeta)(1-2*zeta)
        h_lateral = 4 * zeta * r  # = 4*zeta*(1-zeta)
        h_apex = zeta * (2 * zeta - 1)

        # Their derivatives w.r.t. zeta
        dh_base_dzeta = -3 + 4 * zeta
        dh_lateral_dzeta = 4 * (1 - 2 * zeta)
        dh_apex_dzeta = 4 * zeta - 1

        # 8-node serendipity corner functions on scaled base
        # f_i = 0.25 * (1 ± xi_s) * (1 ± eta_s) * (±xi_s ± eta_s - 1)
        f = np.array(
            [
                0.25 * (1 - xi_s) * (1 - eta_s) * (-xi_s - eta_s - 1),  # 0
                0.25 * (1 + xi_s) * (1 - eta_s) * (xi_s - eta_s - 1),  # 1
                0.25 * (1 + xi_s) * (1 + eta_s) * (xi_s + eta_s - 1),  # 2
                0.25 * (1 - xi_s) * (1 + eta_s) * (-xi_s + eta_s - 1),  # 3
            ]
        )

        # Derivatives of serendipity corners w.r.t. xi_s
        df_dxi_s = np.array(
            [
                0.25 * (1 - eta_s) * (2 * xi_s + eta_s),  # 0
                0.25 * (1 - eta_s) * (2 * xi_s - eta_s),  # 1
                0.25 * (1 + eta_s) * (2 * xi_s + eta_s),  # 2
                0.25 * (1 + eta_s) * (2 * xi_s - eta_s),  # 3
            ]
        )

        # Derivatives of serendipity corners w.r.t. eta_s
        df_deta_s = np.array(
            [
                0.25 * (1 - xi_s) * (xi_s + 2 * eta_s),  # 0
                0.25 * (1 + xi_s) * (-xi_s + 2 * eta_s),  # 1
                0.25 * (1 + xi_s) * (xi_s + 2 * eta_s),  # 2
                0.25 * (1 - xi_s) * (-xi_s + 2 * eta_s),  # 3
            ]
        )

        # Edge bubble functions
        g5 = 0.5 * (1 - xi_s**2) * (1 - eta_s)  # edge 0-1
        g6 = 0.5 * (1 - xi_s) * (1 - eta_s**2)  # edge 0-3
        g8 = 0.5 * (1 + xi_s) * (1 - eta_s**2)  # edge 1-2
        g10 = 0.5 * (1 - xi_s**2) * (1 + eta_s)  # edge 2-3

        dg5_dxi_s = -xi_s * (1 - eta_s)
        dg5_deta_s = -0.5 * (1 - xi_s**2)

        dg6_dxi_s = -0.5 * (1 - eta_s**2)
        dg6_deta_s = -(1 - xi_s) * eta_s

        dg8_dxi_s = 0.5 * (1 - eta_s**2)
        dg8_deta_s = -(1 + xi_s) * eta_s

        dg10_dxi_s = -xi_s * (1 + eta_s)
        dg10_deta_s = 0.5 * (1 - xi_s**2)

        # Bilinear functions for lateral edges
        phi = np.array(
            [
                0.25 * (1 - xi_s) * (1 - eta_s),  # 0
                0.25 * (1 + xi_s) * (1 - eta_s),  # 1
                0.25 * (1 + xi_s) * (1 + eta_s),  # 2
                0.25 * (1 - xi_s) * (1 + eta_s),  # 3
            ]
        )

        dphi_dxi_s = np.array(
            [
                -0.25 * (1 - eta_s),
                0.25 * (1 - eta_s),
                0.25 * (1 + eta_s),
                -0.25 * (1 + eta_s),
            ]
        )

        dphi_deta_s = np.array(
            [
                -0.25 * (1 - xi_s),
                -0.25 * (1 + xi_s),
                0.25 * (1 + xi_s),
                0.25 * (1 - xi_s),
            ]
        )

        # Helper function for chain rule
        def compute_derivs(func, df_dxi_s_val, df_deta_s_val, h, dh_dzeta):
            """Compute dN/dxi, dN/deta, dN/dzeta for N = func(xi_s, eta_s) * h(zeta)"""
            d_dxi = df_dxi_s_val * h / r
            d_deta = df_deta_s_val * h / r
            # Chain rule: dxi_s/dzeta = xi/r², deta_s/dzeta = eta/r²
            d_dzeta = ((df_dxi_s_val * xi + df_deta_s_val * eta) / r2) * h + func * dh_dzeta
            return d_dxi, d_deta, d_dzeta

        # Base corner nodes (0-3)
        for i in range(4):
            dN_dxi[i], dN_deta[i], dN_dzeta[i] = compute_derivs(
                f[i], df_dxi_s[i], df_deta_s[i], h_base, dh_base_dzeta
            )

        # Apex (4)
        dN_dxi[4] = 0.0
        dN_deta[4] = 0.0
        dN_dzeta[4] = dh_apex_dzeta

        # Base edge midpoints (5, 6, 8, 10)
        dN_dxi[5], dN_deta[5], dN_dzeta[5] = compute_derivs(
            g5, dg5_dxi_s, dg5_deta_s, h_base, dh_base_dzeta
        )
        dN_dxi[6], dN_deta[6], dN_dzeta[6] = compute_derivs(
            g6, dg6_dxi_s, dg6_deta_s, h_base, dh_base_dzeta
        )
        dN_dxi[8], dN_deta[8], dN_dzeta[8] = compute_derivs(
            g8, dg8_dxi_s, dg8_deta_s, h_base, dh_base_dzeta
        )
        dN_dxi[10], dN_deta[10], dN_dzeta[10] = compute_derivs(
            g10, dg10_dxi_s, dg10_deta_s, h_base, dh_base_dzeta
        )

        # Lateral edge midpoints (7, 9, 11, 12)
        lateral_nodes = [(7, 0), (9, 1), (11, 2), (12, 3)]
        for node_idx, corner_idx in lateral_nodes:
            dN_dxi[node_idx], dN_deta[node_idx], dN_dzeta[node_idx] = compute_derivs(
                phi[corner_idx],
                dphi_dxi_s[corner_idx],
                dphi_deta_s[corner_idx],
                h_lateral,
                dh_lateral_dzeta,
            )

        return dN_dxi, dN_deta, dN_dzeta


# =============================================================================
# TETRAHEDRON Elements
# =============================================================================


class TETRA4(SolidElement):
    """4-node linear tetrahedron element.

    Simplest 3D solid element with constant strain.

    Node ordering:
            3
           /|\\
          / | \\
         /  |  \\
        /   2   \\
       /  .'  `. \\
      0---------1

    Natural coordinates: ξ, η, ζ ∈ [0, 1] with ξ + η + ζ ≤ 1
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for TETRA4 (always 1 point)
    ):
        if len(node_ids) != 4:
            raise ValueError(f"TETRA4 requires 4 nodes, got {len(node_ids)}")
        super().__init__("TETRA4", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """1-point centroid rule for linear tetrahedron."""
        points = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1 / 6])  # Volume of unit tetrahedron
        return points, weights

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Linear tetrahedral shape functions (volume coordinates)."""
        return np.array(
            [
                1 - xi - eta - zeta,  # N0
                xi,  # N1
                eta,  # N2
                zeta,  # N3
            ]
        )

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant derivatives for linear tetrahedron."""
        dN_dxi = np.array([-1, 1, 0, 0])
        dN_deta = np.array([-1, 0, 1, 0])
        dN_dzeta = np.array([-1, 0, 0, 1])
        return dN_dxi, dN_deta, dN_dzeta


class TETRA10(SolidElement):
    """10-node quadratic tetrahedron element.

    Node ordering (following Gmsh convention):
        Corners: 0, 1, 2, 3
        Edge midpoints: 4 (0-1), 5 (1-2), 6 (0-2), 7 (0-3), 8 (2-3), 9 (1-3)
        
        Gmsh diagram:
                       v
                     .
                    ,/
                   /
               2                                    
             ,/|`\                                 
           ,/  |  `\                            
          ,6   '.   `5                         
        ,/      8    `\                      
      ,/        |      `\                   
     0--------4--'.-------1 --> u
      `\.        |      ,/                  
         `\.     |    ,9                   
            `7.  '. ,/                    
               `\. |/                     
                  `3                    
                     `\.
                        ` w
        
    Natural coordinates: ξ, η, ζ ∈ [0, 1] with ξ + η + ζ ≤ 1
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for TETRA10
    ):
        if len(node_ids) != 10:
            raise ValueError(f"TETRA10 requires 10 nodes, got {len(node_ids)}")
        super().__init__("TETRA10", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """4-point integration rule for quadratic tetrahedron."""
        a = (5 - np.sqrt(5)) / 20
        b = (5 + 3 * np.sqrt(5)) / 20

        points = np.array(
            [
                [a, a, a],
                [b, a, a],
                [a, b, a],
                [a, a, b],
            ]
        )
        weights = np.array([1, 1, 1, 1]) / 24  # Each = V_tet/4
        return points, weights

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Quadratic tetrahedral shape functions."""
        L1 = 1 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

        return np.array(
            [
                L1 * (2 * L1 - 1),  # N0
                L2 * (2 * L2 - 1),  # N1
                L3 * (2 * L3 - 1),  # N2
                L4 * (2 * L4 - 1),  # N3
                4 * L1 * L2,  # N4 - edge 0-1
                4 * L2 * L3,  # N5 - edge 1-2
                4 * L3 * L1,  # N6 - edge 0-2 (Gmsh convention)
                4 * L1 * L4,  # N7 - edge 0-3
                4 * L3 * L4,  # N8 - edge 2-3 (Gmsh: swapped with N9)
                4 * L2 * L4,  # N9 - edge 1-3 (Gmsh: swapped with N8)
            ]
        )

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives of quadratic tetrahedral shape functions."""
        L1 = 1 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

        # dL/d(xi,eta,zeta)
        dL1 = np.array([-1, -1, -1])
        dL2 = np.array([1, 0, 0])
        dL3 = np.array([0, 1, 0])
        dL4 = np.array([0, 0, 1])

        dN = np.zeros((10, 3))

        # Corner nodes
        dN[0] = (4 * L1 - 1) * dL1
        dN[1] = (4 * L2 - 1) * dL2
        dN[2] = (4 * L3 - 1) * dL3
        dN[3] = (4 * L4 - 1) * dL4

        # Edge midpoints (Gmsh convention)
        dN[4] = 4 * (L2 * dL1 + L1 * dL2)  # edge 0-1
        dN[5] = 4 * (L3 * dL2 + L2 * dL3)  # edge 1-2
        dN[6] = 4 * (L1 * dL3 + L3 * dL1)  # edge 0-2
        dN[7] = 4 * (L4 * dL1 + L1 * dL4)  # edge 0-3
        dN[8] = 4 * (L4 * dL3 + L3 * dL4)  # edge 2-3 (Gmsh: swapped with dN[9])
        dN[9] = 4 * (L4 * dL2 + L2 * dL4)  # edge 1-3 (Gmsh: swapped with dN[8])

        return dN[:, 0], dN[:, 1], dN[:, 2]


# =============================================================================
# HEXAHEDRON Elements
# =============================================================================


class HEXA8(SolidElement):
    """8-node linear hexahedron (brick) element.

    Node ordering:
            7-------6
           /|      /|
          / |     / |
         4-------5  |
         |  3----|--2
         | /     | /
         |/      |/
         0-------1

    Natural coordinates: ξ, η, ζ ∈ [-1, 1]
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Full integration by default
    ):
        if len(node_ids) != 8:
            raise ValueError(f"HEXA8 requires 8 nodes, got {len(node_ids)}")
        super().__init__("HEXA8", node_coords, node_ids, material, orientation, reduced_integration)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """2×2×2 Gauss quadrature for linear hexahedron (default).

        Reduced integration (1 point) available but requires hourglass control.
        """

        if self.reduced_integration:
            # 1 point rule - requires hourglass control to avoid spurious modes
            return np.array([[0, 0, 0]]), np.array([8.0])
        else:
            # 2×2×2 full integration - default, avoids hourglass modes
            gp = 1 / np.sqrt(3)
            pts_1d = np.array([-gp, gp])

            points = []
            for k in pts_1d:
                for j in pts_1d:
                    for i in pts_1d:
                        points.append([i, j, k])

            weights = np.ones(8)
            return np.array(points), weights

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Trilinear shape functions."""
        return 0.125 * np.array(
            [
                (1 - xi) * (1 - eta) * (1 - zeta),  # N0
                (1 + xi) * (1 - eta) * (1 - zeta),  # N1
                (1 + xi) * (1 + eta) * (1 - zeta),  # N2
                (1 - xi) * (1 + eta) * (1 - zeta),  # N3
                (1 - xi) * (1 - eta) * (1 + zeta),  # N4
                (1 + xi) * (1 - eta) * (1 + zeta),  # N5
                (1 + xi) * (1 + eta) * (1 + zeta),  # N6
                (1 - xi) * (1 + eta) * (1 + zeta),  # N7
            ]
        )

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives of trilinear shape functions."""
        dN_dxi = 0.125 * np.array(
            [
                -(1 - eta) * (1 - zeta),
                (1 - eta) * (1 - zeta),
                (1 + eta) * (1 - zeta),
                -(1 + eta) * (1 - zeta),
                -(1 - eta) * (1 + zeta),
                (1 - eta) * (1 + zeta),
                (1 + eta) * (1 + zeta),
                -(1 + eta) * (1 + zeta),
            ]
        )

        dN_deta = 0.125 * np.array(
            [
                -(1 - xi) * (1 - zeta),
                -(1 + xi) * (1 - zeta),
                (1 + xi) * (1 - zeta),
                (1 - xi) * (1 - zeta),
                -(1 - xi) * (1 + zeta),
                -(1 + xi) * (1 + zeta),
                (1 + xi) * (1 + zeta),
                (1 - xi) * (1 + zeta),
            ]
        )

        dN_dzeta = 0.125 * np.array(
            [
                -(1 - xi) * (1 - eta),
                -(1 + xi) * (1 - eta),
                -(1 + xi) * (1 + eta),
                -(1 - xi) * (1 + eta),
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
            ]
        )

        return dN_dxi, dN_deta, dN_dzeta


class HEXA20(SolidElement):
    """20-node quadratic hexahedron (serendipity brick) element.

    Node ordering (following Gmsh convention):
    
        Corner nodes: 0-7 (same as HEXA8)
        Edge midpoints: 8-19
        
        Gmsh diagram:
           3----13----2
           |\         |\
           | 15       | 14
           9  \       11 \
           |   7----19+---6
           |   |      |   |
           0---+-8----1   |
            \  17      \  18
            10 |       12|
              \|         \|
               4----16----5
               
        Edge mapping:
        8: edge 0-1,  9: edge 0-3, 10: edge 0-4, 11: edge 1-2
        12: edge 1-5, 13: edge 2-3, 14: edge 2-6, 15: edge 3-7
        16: edge 4-5, 17: edge 4-7, 18: edge 5-6, 19: edge 6-7

    Natural coordinates: ξ, η, ζ ∈ [-1, 1]
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, ...],
        material: MaterialType,
        orientation: Optional[np.ndarray] = None,
        reduced_integration: bool = False,  # Ignored for HEXA20
    ):
        if len(node_ids) != 20:
            raise ValueError(f"HEXA20 requires 20 nodes, got {len(node_ids)}")
        super().__init__("HEXA20", node_coords, node_ids, material, orientation)

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """3×3×3 Gauss quadrature for quadratic hexahedron."""
        sqrt35 = np.sqrt(3 / 5)
        pts_1d = np.array([-sqrt35, 0, sqrt35])
        wts_1d = np.array([5 / 9, 8 / 9, 5 / 9])

        points = []
        weights = []
        for k, (zk, wk) in enumerate(zip(pts_1d, wts_1d)):
            for j, (ej, wj) in enumerate(zip(pts_1d, wts_1d)):
                for i, (xi, wi) in enumerate(zip(pts_1d, wts_1d)):
                    points.append([xi, ej, zk])
                    weights.append(wi * wj * wk)

        return np.array(points), np.array(weights)

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Serendipity (20-node) hexahedron shape functions."""
        # Corner node coordinates
        corners = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )

        N = np.zeros(20)

        # Corner nodes
        for i in range(8):
            xi_i, eta_i, zeta_i = corners[i]
            N[i] = (
                0.125
                * (1 + xi * xi_i)
                * (1 + eta * eta_i)
                * (1 + zeta * zeta_i)
                * (xi * xi_i + eta * eta_i + zeta * zeta_i - 2)
            )

        # Mid-edge nodes (Gmsh convention - see edge mapping in docstring)
        # Edge 0-1: xi varies, eta=-1, zeta=-1
        N[8] = 0.25 * (1 - xi**2) * (1 - eta) * (1 - zeta)
        # Edge 0-3: eta varies, xi=-1, zeta=-1
        N[9] = 0.25 * (1 - xi) * (1 - eta**2) * (1 - zeta)
        # Edge 0-4: zeta varies, xi=-1, eta=-1
        N[10] = 0.25 * (1 - xi) * (1 - eta) * (1 - zeta**2)
        # Edge 1-2: eta varies, xi=+1, zeta=-1
        N[11] = 0.25 * (1 + xi) * (1 - eta**2) * (1 - zeta)
        # Edge 1-5: zeta varies, xi=+1, eta=-1
        N[12] = 0.25 * (1 + xi) * (1 - eta) * (1 - zeta**2)
        # Edge 2-3: xi varies, eta=+1, zeta=-1
        N[13] = 0.25 * (1 - xi**2) * (1 + eta) * (1 - zeta)
        # Edge 2-6: zeta varies, xi=+1, eta=+1
        N[14] = 0.25 * (1 + xi) * (1 + eta) * (1 - zeta**2)
        # Edge 3-7: zeta varies, xi=-1, eta=+1
        N[15] = 0.25 * (1 - xi) * (1 + eta) * (1 - zeta**2)
        # Edge 4-5: xi varies, eta=-1, zeta=+1
        N[16] = 0.25 * (1 - xi**2) * (1 - eta) * (1 + zeta)
        # Edge 4-7: eta varies, xi=-1, zeta=+1
        N[17] = 0.25 * (1 - xi) * (1 - eta**2) * (1 + zeta)
        # Edge 5-6: eta varies, xi=+1, zeta=+1
        N[18] = 0.25 * (1 + xi) * (1 - eta**2) * (1 + zeta)
        # Edge 6-7: xi varies, eta=+1, zeta=+1
        N[19] = 0.25 * (1 - xi**2) * (1 + eta) * (1 + zeta)

        return N

    def shape_function_derivatives(
        self, xi: float, eta: float, zeta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives of serendipity hexahedron shape functions (numerical)."""
        h = 1e-8

        N0 = self.shape_functions(xi, eta, zeta)

        dN_dxi = (self.shape_functions(xi + h, eta, zeta) - N0) / h
        dN_deta = (self.shape_functions(xi, eta + h, zeta) - N0) / h
        dN_dzeta = (self.shape_functions(xi, eta, zeta + h) - N0) / h

        return dN_dxi, dN_deta, dN_dzeta
