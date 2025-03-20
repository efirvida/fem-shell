from typing import Tuple

import numpy as np

from fem_shell.core.material import Material
from fem_shell.elements.elements import PlaneElement


class QUAD(PlaneElement):
    """
    Abstract base class for 2D finite element types.

    This class provides formulations for the stiffness matrix, the consistent mass matrix,
    and the load vector (body force) using an isoparametric finite element approach.

    The stiffness matrix is computed as:

        .. math::
            K_e = \\int_{\\Omega} B^T C B \\; d\\Omega,

    the consistent mass matrix as:

        .. math::
            M_e = \\int_{\\Omega} \\rho N^T N \\; d\\Omega,

    and the load vector as:

        .. math::
            f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

    where:
        - \\(B\\) is the strain-displacement matrix,
        - \\(C\\) is the constitutive (elasticity) matrix,
        - \\(N\\) is the shape function matrix,
        - \\(\\rho\\) is the material density,
        - \\(\\mathbf{b}\\) is the body force vector,
        - \\(\\Omega\\) is the element domain.
    """

    vector_form = {"U": ("Ux", "Uy")}

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        material: Material,
    ):
        """
        Initialize the element with nodal coordinates and material properties.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates array of shape (n_nodes, 2).
        material : Material
        """
        super().__init__(
            "QUAD",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=2,
        )

    @property
    def integration_points(self):
        """
        Get integration points and weights for numerical integration.

        Assumes that the function `gauss_legendre_quadrature` returns a tuple:
            (points, weights)

        Returns
        -------
        tuple
            A tuple containing:
                - points: an array of integration points,
                - weights: an array of corresponding weights.
        """
        gp = 1 / np.sqrt(3)
        points = [(gp, gp), (-gp, gp), (-gp, -gp), (gp, -gp)]
        weigths = [1.0, 1.0, 1.0, 1.0]

        return points, weigths

    @property
    def C(self) -> np.ndarray:
        """
        Constitutive (elasticity) matrix in plane stress conditions.

        Returns
        -------
        np.ndarray
            Elasticity matrix of shape (3, 3).

        Notes
        -----
        The constitutive matrix is given by:

            .. math::
                C = \\begin{bmatrix}
                    \\lambda + 2\\mu & \\lambda & 0 \\\\
                    \\lambda & \\lambda + 2\\mu & 0 \\\\
                    0 & 0 & \\mu
                \\end{bmatrix},

        where:
            - \\(\\lambda = \\frac{E \\nu}{(1+\\nu)(1-2\\nu)}\\),
            - \\(\\mu = \\frac{E}{2(1+\\nu)}\\).
        """
        E = self.material.E
        nu = self.material.nu

        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        return np.array(
            [
                [lambda_ + 2 * mu, lambda_, 0],
                [lambda_, lambda_ + 2 * mu, 0],
                [0, 0, mu],
            ]
        )

    @property
    def K(self) -> np.ndarray:
        """
        Compute the element stiffness matrix.

        The stiffness matrix is computed as:

            .. math::
                K_e = \\int_{\\Omega} B^T C B \\; d\\Omega,

        where:
            - \\(B\\) is the strain-displacement matrix,
            - \\(C\\) is the constitutive matrix.

        Returns
        -------
        np.ndarray
            Element stiffness matrix of shape (dofs x dofs).
        """
        points, weights = self.integration_points
        K = np.zeros((self.dofs_count, self.dofs_count))

        for (xi, eta), w in zip(points, weights):
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            _, det_J, inv_J = self._compute_jacobian(xi, eta)
            dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
            dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta

            B = np.zeros((3, self.dofs_count))
            B[0, 0::2] = dN_dx  # ε_xx
            B[1, 1::2] = dN_dy  # ε_yy
            B[2, 0::2] = dN_dy  # γ_xy
            B[2, 1::2] = dN_dx  # γ_xy

            K += (B.T @ self.C @ B) * det_J * w

        return K

    @property
    def M(self) -> np.ndarray:
        """
        Compute the consistent mass matrix.

        The mass matrix is computed as:

            .. math::
                M_e = \\int_{\\Omega} \\rho N^T N \\; d\\Omega,

        where:
            - \\(N\\) is the shape function matrix assembled for two degrees of freedom per node,
            - \\(\\rho\\) is the material density,
            - \\(\\Omega\\) is the element domain.

        Returns
        -------
        np.ndarray
            Element mass matrix of shape (dofs x dofs).
        """
        points, weights = self.integration_points
        M = np.zeros((self.dofs_count, self.dofs_count))
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N_values = self.shape_functions(xi, eta)

            N_matrix = np.zeros((2, self.dofs_count))
            N_matrix[0, 0::2] = N_values
            N_matrix[1, 1::2] = N_values

            M += self.material.rho * (N_matrix.T @ N_matrix) * det_J * w

        return M

    def body_load(self, body_force: np.ndarray) -> np.ndarray:
        """
        Assemble the load vector for the element due to a constant body force.

        The load vector is computed as:

            .. math::
                f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

        where:
            - \\(N\\) is the shape function matrix assembled for two degrees of freedom per node,
            - \\(\\mathbf{b}\\) is the constant body force vector (shape (2,)).

        Parameters
        ----------
        body_force : np.ndarray
            A 2-element array representing the body force per unit volume (or area in 2D).

        Returns
        -------
        np.ndarray
            Element load vector of shape (dofs,).
        """
        points, weights = self.integration_points
        f = np.zeros(self.dofs_count)
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N_values = self.shape_functions(xi, eta)
            N_matrix = np.zeros((2, self.dofs_count))
            N_matrix[0, 0::2] = N_values
            N_matrix[1, 1::2] = N_values
            f += (N_matrix.T @ body_force[:2]) * det_J * w
        return f

    def _compute_jacobian(self, xi: float, eta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute the Jacobian matrix, its determinant, and its inverse at given natural coordinates.

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        tuple
            A tuple (J, det_J, inv_J) where:
                - J is the Jacobian matrix,
                - det_J is the determinant of J,
                - inv_J is the inverse of J.

        Raises
        ------
        ValueError
            If the determinant of the Jacobian is non-positive.
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        J = np.array(
            [
                [dN_dxi @ self.node_coords[:, 0], dN_deta @ self.node_coords[:, 0]],
                [dN_dxi @ self.node_coords[:, 1], dN_deta @ self.node_coords[:, 1]],
            ]
        )
        det_J = np.linalg.det(J)
        if det_J <= 0:
            raise ValueError("Jacobian determinant is non-positive. Check the node orientation.")
        inv_J = np.linalg.inv(J)
        return J, det_J, inv_J

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute bilinear shape functions for quadrilateral element.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1]
        eta : float
            Natural coordinate in [-1, 1]

        Returns
        -------
        np.ndarray
            Array of shape functions [N0, N1, N2, N3]
        """
        return 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),  # N0
                (1 + xi) * (1 - eta),  # N1
                (1 + xi) * (1 + eta),  # N2
                (1 - xi) * (1 + eta),  # N3
            ]
        )

    def shape_function_derivatives(self, xi: float, eta: float):
        """
        Compute derivatives of bilinear shape functions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Derivatives w.r.t. xi and eta as (dN_dxi, dN_deta)
        """
        dN_dxi = [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]
        dN_deta = [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]

        return 0.25 * np.array([dN_dxi, dN_deta])

    @property
    def area(self) -> float:
        """
        Compute the area of a quadrilateral element.

        Supports 4-node (bilinear), 8-node (serendipity), and 9-node (Lagrange) elements.
        The 4-node element uses the shoelace formula, while the 8-node and 9-node elements
        are approximated by dividing them into 4 triangles.

        Returns
        -------
        float
            The approximate area of the quadrilateral element.

        Raises
        ------
        ValueError
            If `coords` does not have a valid shape.
        """
        x, y = self.node_coords[:, 0], self.node_coords[:, 1]

        return 0.5 * abs(
            x[0] * y[1]
            + x[1] * y[2]
            + x[2] * y[3]
            + x[3] * y[0]
            - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
        )
