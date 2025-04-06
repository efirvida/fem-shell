"""Quadrilateral Elements for Plane Elasticity Analysis (QUAD4, QUAD8, QUAD9)

Implements isoparametric quadrilateral elements for linear elasticity problems
using consistent formulation and numerical integration.

Elements supported:
- QUAD4: 4-node bilinear quadrilateral
- QUAD8: 8-node serendipity quadrilateral
- QUAD9: 9-node biquadratic quadrilateral

Formulation:
    Stiffness matrix: K = ∫BᵀCB dΩ
    Mass matrix: M = ∫ρNᵀN dΩ
    Body forces: f = ∫Nᵀb dΩ

where:
    B: Strain-displacement matrix
    C: Constitutive matrix (plane stress)
    N: Shape functions matrix
    ρ: Material density
    b: Body force vector
"""

from typing import Tuple

import numpy as np

from fem_shell.core.material import IsotropicMaterial
from fem_shell.elements.elements import PlaneElement


class QUAD(PlaneElement):
    """Base class for quadrilateral elements

    Node numbering convention:

    QUAD4          QUAD8          QUAD9
    3---2         3---6---2       3---6---2
    |   |         |       |       |   |   |
    0---1         7       5       7   8   5
                  |       |       |   |   |
                  0---4---1       0---4---1
    """

    vector_form = {"U": ("Ux", "Uy")}

    def __init__(
        self, node_coords: np.ndarray, node_ids: Tuple[int, ...], material: IsotropicMaterial
    ):
        """Initialize quadrilateral element

        Parameters
        ----------
        node_coords : np.ndarray
            Array of nodal coordinates (n_nodes x 2)
        node_ids : Tuple[int, ...]
            Global node IDs for connectivity
        material : IsotropicMaterial
            Material properties instance
        """
        super().__init__(
            "QUAD",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=2,
        )

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss quadrature points and weights

        Default: 2x2 integration for bilinear elements

        Returns
        -------
        points : np.ndarray
            Array of (xi, eta) coordinates (n_points x 2)
        weights : np.ndarray
            Integration weights (n_points,)
        """
        gp = 1 / np.sqrt(3)
        points = np.array([(gp, gp), (-gp, gp), (-gp, -gp), (gp, -gp)])
        weights = np.ones(4)
        return points, weights

    def compute_B_matrix(self, xi: float, eta: float) -> np.ndarray:
        """Compute strain-displacement matrix B

        Strain components:
        [ε_xx, ε_yy, γ_xy]ᵀ = B * u

        Parameters
        ----------
        xi, eta : float
            Natural coordinates in [-1, 1]

        Returns
        -------
        np.ndarray
            B matrix (3 x n_dofs)
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        J, det_J, inv_J = self._compute_jacobian(xi, eta)

        if det_J <= 1e-12:
            raise ValueError(f"Non-positive Jacobian at ({xi}, {eta})")

        # Transform to global coordinates
        dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
        dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta

        # Construct B matrix
        B = np.zeros((3, self.dofs_count))
        for i in range(len(dN_dx)):
            B[0, 2 * i] = dN_dx[i]  # ε_xx
            B[1, 2 * i + 1] = dN_dy[i]  # ε_yy
            B[2, 2 * i] = dN_dy[i]  # γ_xy
            B[2, 2 * i + 1] = dN_dx[i]  # γ_yx

        return B

    @property
    def C(self) -> np.ndarray:
        """Constitutive matrix for plane strain

        Returns
        -------
        np.ndarray
            Material matrix (3x3)
        """
        E = self.material.E
        nu = self.material.nu
        # Calcular los coeficientes de Lamé
        lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        return np.array([
            [lambd + 2 * mu, lambd, 0],
            [lambd, lambd + 2 * mu, 0],
            [0, 0, mu],
        ])

    @property
    def K(self) -> np.ndarray:
        """Stiffness matrix with numerical stabilization

        Calculated using:
            K = ∫BᵀCB dΩ

        Includes stabilization to handle rigid body modes

        Returns
        -------
        np.ndarray
            Symmetric stiffness matrix (n_dofs x n_dofs)
        """
        points, weights = self.integration_points
        K = np.zeros((self.dofs_count, self.dofs_count))

        # Numerical integration
        for (xi, eta), w in zip(points, weights):
            B = self.compute_B_matrix(xi, eta)
            _, det_J, _ = self._compute_jacobian(xi, eta)
            K += (B.T @ self.C @ B) * det_J * w

        # Ensure symmetry and handle rigid modes
        K = 0.5 * (K + K.T)  # Symmetrization
        K += 1e-10 * np.eye(self.dofs_count)  # Stabilization

        return K

    @property
    def M(self) -> np.ndarray:
        """Consistent mass matrix

        Calculated using:
            M = ∫ρNᵀN dΩ

        Returns
        -------
        np.ndarray
            Mass matrix (n_dofs x n_dofs)
        """
        points, weights = self.integration_points
        M = np.zeros((self.dofs_count, self.dofs_count))

        for (xi, eta), w in zip(points, weights):
            N = self.shape_functions(xi, eta)
            _, det_J, _ = self._compute_jacobian(xi, eta)

            N_mat = np.zeros((2, self.dofs_count))
            N_mat[0, 0::2] = N  # u-components
            N_mat[1, 1::2] = N  # v-components

            M += self.material.rho * (N_mat.T @ N_mat) * det_J * w

        return 0.5 * (M + M.T)

    def body_load(self, body_force: np.ndarray) -> np.ndarray:
        """Body force vector

        Parameters
        ----------
        body_force : np.ndarray
            Body force components [bx, by]

        Returns
        -------
        np.ndarray
            Force vector (n_dofs,)
        """
        points, weights = self.integration_points
        f = np.zeros(self.dofs_count)

        for (xi, eta), w in zip(points, weights):
            N = self.shape_functions(xi, eta)
            _, det_J, _ = self._compute_jacobian(xi, eta)

            N_mat = np.zeros((2, self.dofs_count))
            N_mat[0, 0::2] = N  # u-components
            N_mat[1, 1::2] = N  # v-components

            f += (N_mat.T @ body_force[:2]) * det_J * w

        return f

    def _compute_jacobian(self, xi: float, eta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute Jacobian matrix components

        Parameters
        ----------
        xi, eta : float
            Natural coordinates

        Returns
        -------
        J : np.ndarray
            Jacobian matrix (2x2)
        det_J : float
            Jacobian determinant
        inv_J : np.ndarray
            Inverse of Jacobian matrix
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)

        J = np.array([
            [dN_dxi @ self.node_coords[:, 0], dN_deta @ self.node_coords[:, 0]],
            [dN_dxi @ self.node_coords[:, 1], dN_deta @ self.node_coords[:, 1]],
        ])

        det_J = np.linalg.det(J)
        inv_J = np.linalg.inv(J)

        return J, det_J, inv_J


class QUAD4(QUAD):
    """4-node Bilinear Quadrilateral Element

    Node ordering:
    3-------2
    |       |
    |       |
    0-------1
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        material: IsotropicMaterial,
    ):
        super().__init__(node_coords, node_ids, material)
        self.name = "QUAD4"

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Bilinear shape functions

        N0 = 0.25(1 - xi)(1 - eta)
        N1 = 0.25(1 + xi)(1 - eta)
        N2 = 0.25(1 + xi)(1 + eta)
        N3 = 0.25(1 - xi)(1 + eta)
        """
        return 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ])

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Shape function derivatives

        Returns
        -------
        dN_dxi : np.ndarray
            Derivatives with respect to xi
        dN_deta : np.ndarray
            Derivatives with respect to eta
        """
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])

        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])

        return dN_dxi, dN_deta


class QUAD8(QUAD):
    """8-node Serendipity Element

    Node ordering:
    3---6---2
    |       |
    7       5
    |       |
    0---4---1
    """

    def __init__(
        self, node_coords: np.ndarray, node_ids: Tuple[int, ...], material: IsotropicMaterial
    ):
        super().__init__(node_coords, node_ids, material)
        self.name = "QUAD8"

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """3x3 Gauss quadrature for exact integration"""
        sqrt3_5 = np.sqrt(3 / 5)
        points = np.array([
            (-sqrt3_5, -sqrt3_5),
            (0, -sqrt3_5),
            (sqrt3_5, -sqrt3_5),
            (-sqrt3_5, 0),
            (0, 0),
            (sqrt3_5, 0),
            (-sqrt3_5, sqrt3_5),
            (0, sqrt3_5),
            (sqrt3_5, sqrt3_5),
        ])
        weights = np.array([25, 40, 25, 40, 64, 40, 25, 40, 25]) / 81
        return points, weights

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Serendipity shape functions"""
        return np.array([
            0.25 * (1 - xi) * (1 - eta) * (-1 - xi - eta),  # N0
            0.25 * (1 + xi) * (1 - eta) * (-1 + xi - eta),  # N1
            0.25 * (1 + xi) * (1 + eta) * (-1 + xi + eta),  # N2
            0.25 * (1 - xi) * (1 + eta) * (-1 - xi + eta),  # N3
            0.5 * (1 - xi**2) * (1 - eta),  # N4
            0.5 * (1 + xi) * (1 - eta**2),  # N5
            0.5 * (1 - xi**2) * (1 + eta),  # N6
            0.5 * (1 - xi) * (1 - eta**2),  # N7
        ])

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical derivatives of shape functions"""
        dN_dxi = np.array([
            0.25 * (eta * (2 * xi + eta) + (1 - eta) * (xi + eta)),  # N0
            0.25 * (eta * (2 * xi - eta) + (1 - eta) * (xi - eta)),  # N1
            0.25 * (eta * (2 * xi + eta) + (1 + eta) * (xi + eta)),  # N2
            0.25 * (eta * (2 * xi - eta) + (1 + eta) * (xi - eta)),  # N3
            -xi * (1 - eta),  # N4
            0.5 * (1 - eta**2),  # N5
            -xi * (1 + eta),  # N6
            -0.5 * (1 - eta**2),  # N7
        ])

        dN_deta = np.array([
            0.25 * (xi * (xi + 2 * eta) + (1 - xi) * (xi + eta)),  # N0
            0.25 * (-xi * (xi - 2 * eta) + (1 + xi) * (-xi + eta)),  # N1
            0.25 * (xi * (xi + 2 * eta) + (1 + xi) * (xi + eta)),  # N2
            0.25 * (-xi * (xi - 2 * eta) + (1 - xi) * (-xi + eta)),  # N3
            -0.5 * (1 - xi**2),  # N4
            -(1 + xi) * eta,  # N5
            0.5 * (1 - xi**2),  # N6
            -(1 - xi) * eta,  # N7
        ])

        return dN_dxi, dN_deta


class QUAD9(QUAD):
    """9-node Lagrange Element

    Node ordering:
    3---6---2
    |   |   |
    7---8---5
    |   |   |
    0---4---1
    """

    def __init__(
        self, node_coords: np.ndarray, node_ids: Tuple[int, ...], material: IsotropicMaterial
    ):
        super().__init__(node_coords, node_ids, material)
        self.name = "QUAD9"

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return QUAD8.integration_points.fget(self)

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Biquadratic shape functions"""
        return np.array([
            0.25 * xi * eta * (xi - 1) * (eta - 1),  # N0
            0.25 * xi * eta * (xi + 1) * (eta - 1),  # N1
            0.25 * xi * eta * (xi + 1) * (eta + 1),  # N2
            0.25 * xi * eta * (xi - 1) * (eta + 1),  # N3
            0.5 * (1 - xi**2) * eta * (eta - 1),  # N4
            0.5 * xi * (xi + 1) * (1 - eta**2),  # N5
            0.5 * (1 - xi**2) * eta * (eta + 1),  # N6
            0.5 * xi * (xi - 1) * (1 - eta**2),  # N7
            (1 - xi**2) * (1 - eta**2),  # N8
        ])

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical derivatives of shape functions"""
        dN_dxi = np.array([
            0.25 * eta * (eta - 1) * (2 * xi - 1),  # N0
            0.25 * eta * (eta - 1) * (2 * xi + 1),  # N1
            0.25 * eta * (eta + 1) * (2 * xi + 1),  # N2
            0.25 * eta * (eta + 1) * (2 * xi - 1),  # N3
            -xi * eta * (eta - 1),  # N4
            0.5 * (1 - eta**2) * (2 * xi + 1),  # N5
            -xi * eta * (eta + 1),  # N6
            0.5 * (1 - eta**2) * (2 * xi - 1),  # N7
            -2 * xi * (1 - eta**2),  # N8
        ])

        dN_deta = np.array([
            0.25 * xi * (xi - 1) * (2 * eta - 1),  # N0
            0.25 * xi * (xi + 1) * (2 * eta - 1),  # N1
            0.25 * xi * (xi + 1) * (2 * eta + 1),  # N2
            0.25 * xi * (xi - 1) * (2 * eta + 1),  # N3
            0.5 * (1 - xi**2) * (2 * eta - 1),  # N4
            -xi * (xi + 1) * eta,  # N5
            0.5 * (1 - xi**2) * (2 * eta + 1),  # N6
            -xi * (xi - 1) * eta,  # N7
            -2 * eta * (1 - xi**2),  # N8
        ])

        return dN_dxi, dN_deta
