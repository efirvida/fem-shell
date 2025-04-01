"""
MITC4 Shell Element Implementation.

This module provides an implementation of the MITC4 (Mixed Interpolation of Tensorial Components)
shell element formulation, adapted from the JaxSSO project (https://github.com/GaoyuanWu/JaxSSO).
The original project is licensed under MIT License - see NOTICE for details.

The MITC4 element formulation prevents shear locking through mixed interpolation techniques,
making it suitable for both thin and thick shell analyses. The element has four nodes with
six degrees of freedom per node (3 translations, 3 rotations).

Adapted from JaxSSO which is Copyright (c) 2021 Gaoyuan Wu under MIT License.
This adaptation maintains the same MIT License terms.
"""

from typing import List, Tuple

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC4(ShellElement):
    """
    MITC4 quadrilateral shell element class.

    This element formulation uses mixed interpolation of tensorial components to
    avoid shear locking. Each node has 6 DOFs (3 translations + 3 rotations).

    Parameters
    ----------
    node_coords : np.ndarray
        Array of nodal coordinates in global system [4x3]
    node_ids : Tuple[int, int, int, int]
        Global node IDs for element connectivity
    material : Material
        Material properties object
    thickness : float
        Shell thickness
    kx_mod : float, optional
        Stiffness modification factor in x-direction, by default 1.0
    ky_mod : float, optional
        Stiffness modification factor in y-direction, by default 1.0

    Attributes
    ----------
    kx_mod : float
        X-direction stiffness modifier
    ky_mod : float
        Y-direction stiffness modifier
    node_coords : np.ndarray
        Element nodal coordinates [4x3]
    thickness : float
        Element thickness

    Notes
    -----
    The element formulation follows the MITC4 procedure described in:
    - Bathe, K.J., and Dvorkin, E.N. (1985). "A four-node plate bending element
      based on Mindlin/Reissner plate theory and a mixed interpolation."
    """

    vector_form = {"U": ("Ux", "Uy", "Uz"), "W": ("Wx", "Wy", "Wz")}

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        material: Material,
        thickness: float,
        kx_mod: float = 1.0,
        ky_mod: float = 1.0,
    ):
        super().__init__(
            "MITC4",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=6,
            thickness=thickness,
        )

        self.kx_mod = kx_mod
        self.ky_mod = ky_mod

        gp = 1 / np.sqrt(3)
        self._gauss_points = [(gp, gp), (-gp, gp), (-gp, -gp), (gp, -gp)]
        self._local_coordinates = self._compute_local_coordinates()

    def area(self):
        """
        Calculate element reference area.

        Returns
        -------
        float
            Total area of the quadrilateral element calculated as sum of two triangles

        Notes
        -----
        The quadrilateral is divided into two triangles (1-2-3 and 3-4-1) for area calculation.
        """
        X1, Y1, Z1 = self.node_coords[0]
        X2, Y2, Z2 = self.node_coords[1]
        X3, Y3, Z3 = self.node_coords[2]
        X4, Y4, Z4 = self.node_coords[3]

        edge_12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1])
        edge_14 = np.array([X4 - X1, Y4 - Y1, Z4 - Z1])
        edge_32 = np.array([X2 - X3, Y2 - Y3, Z2 - Z3])
        edge_34 = np.array([X4 - X3, Y4 - Y3, Z4 - Z3])
        area_1 = np.linalg.norm(np.cross(edge_12, edge_14)) * 0.5
        area_2 = np.linalg.norm(np.cross(edge_34, edge_32)) * 0.5
        return area_1 + area_2

    def _compute_local_coordinates(self):
        """
        Compute the local coordinate system and normal vectors.

        Returns
        -------
        np.ndarray
            Local coordinates of nodes as [x1, y1, x2, y2, x3, y3, x4, y4].
            Node 3 is used as the reference, so its local coordinates are (0,0).
        """
        # Convert node coordinates to a numpy array (shape: [4, 3])
        nodes = np.asarray(self.node_coords)

        # Define reference node (node 3, index 2)
        ref = nodes[2]

        # Compute vectors from the reference node to nodes 1, 2, and 4
        vec_31 = nodes[0] - ref  # Node 1 relative to node 3
        vec_32 = nodes[1] - ref  # Node 2 relative to node 3
        vec_34 = nodes[3] - ref  # Node 4 relative to node 3

        # Additional vector from node 4 to node 2 (used to define the coordinate system)
        vec_42 = nodes[1] - nodes[3]

        # Define the local axes:
        # - Local x-axis is the direction from node 3 to node 1.
        x_axis = vec_31
        # - Local z-axis is perpendicular to the plane formed by x_axis and vec_42.
        z_axis = np.cross(x_axis, vec_42)
        # - Local y-axis is perpendicular to both z_axis and x_axis.
        y_axis = np.cross(z_axis, x_axis)

        # Normalize the local axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Compute the local coordinates along the x- and y-axes using dot products.
        # For node 3, the coordinates are (0, 0) by definition.
        x1 = np.dot(vec_31, x_axis)
        x2 = np.dot(vec_32, x_axis)
        x3 = 0.0
        x4 = np.dot(vec_34, x_axis)

        y1 = np.dot(vec_31, y_axis)
        y2 = np.dot(vec_32, y_axis)
        y3 = 0.0
        y4 = np.dot(vec_34, y_axis)

        # Combine the local coordinates into a single array: [x1, y1, x2, y2, x3, y3, x4, y4]
        local_coordinates = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
        return local_coordinates

    def T(self):
        """
        Construct the local-to-global transformation matrix.

        Returns
        -------
        np.ndarray
            24x24 transformation matrix for element DOFs

        Notes
        -----
        The transformation matrix follows the convention:
        [u', v', w', θx', θy', θz'] = T @ [u, v, w, θx, θy, θz],
        where the prime denotes local coordinates.
        """
        # Convert node coordinates to a numpy array (if they aren't already)
        nodes = np.asarray(self.node_coords)  # shape: (4, 3)

        # Compute directional vectors from nodes:
        # vector_31: from node 3 to node 1
        # vector_42: from node 4 to node 2 (following the given order)
        vector_31 = nodes[0] - nodes[2]
        vector_42 = nodes[1] - nodes[3]

        # Define the local axis system
        x_axis = vector_31
        z_axis = np.cross(x_axis, vector_42)
        y_axis = np.cross(z_axis, x_axis)

        # Normalize the axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Create the direction cosine matrix (3x3)
        dirCos = np.array([x_axis, y_axis, z_axis])

        # The transformation matrix T is block-diagonal with 8 blocks of dirCos.
        # Each node has 2 blocks (translational and rotational) of size 3x3.
        # Using np.kron, we build T = kron(I_8, dirCos), resulting in a (8*3)x(8*3) = 24x24 matrix.
        T = np.kron(np.eye(8), dirCos)

        return T

    def J(self, r, s):
        """
        Calculate Jacobian matrix at parametric coordinates (r,s).

        Parameters
        ----------
        r : float
            Parametric coordinate in first direction [-1, 1]
        s : float
            Parametric coordinate in second direction [-1, 1]

        Returns
        -------
        np.ndarray
            2x2 Jacobian matrix
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self._local_coordinates
        return (
            1
            / 4
            * np.array([
                [
                    x1 * (s + 1) - x2 * (s + 1) + x3 * (s - 1) - x4 * (s - 1),
                    y1 * (s + 1) - y2 * (s + 1) + y3 * (s - 1) - y4 * (s - 1),
                ],
                [
                    x1 * (r + 1) - x2 * (r - 1) + x3 * (r - 1) - x4 * (r + 1),
                    y1 * (r + 1) - y2 * (r - 1) + y3 * (r - 1) - y4 * (r + 1),
                ],
            ])
        )

    def B_kappa(self, r, s):
        """
        Compute bending strain-displacement matrix.

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            3x12 bending strain-displacement matrix
        """
        J_val = self.J(r, s)
        dH = np.linalg.solve(
            J_val,
            1 / 4 * np.array([[1 + s, -1 - s, -1 + s, 1 - s], [1 + r, 1 - r, -1 + r, -1 - r]]),
        )
        return np.array([
            [0, 0, -dH[0, 0], 0, 0, -dH[0, 1], 0, 0, -dH[0, 2], 0, 0, -dH[0, 3]],
            [0, dH[1, 0], 0, 0, dH[1, 1], 0, 0, dH[1, 2], 0, 0, dH[1, 3], 0],
            [
                0,
                dH[0, 0],
                -dH[1, 0],
                0,
                dH[0, 1],
                -dH[1, 1],
                0,
                dH[0, 2],
                -dH[1, 2],
                0,
                dH[0, 3],
                -dH[1, 3],
            ],
        ])

    def B_gamma(self, r, s):
        """
        Compute shear strain-displacement matrix using MITC4 interpolation.

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            2x12 shear strain-displacement matrix
        """
        J_val = self.J(r, s)
        x1, y1, x2, y2, x3, y3, x4, y4 = self._local_coordinates
        Ax = x1 - x2 - x3 + x4
        Bx = x1 - x2 + x3 - x4
        Cx = x1 + x2 - x3 - x4
        Ay = y1 - y2 - y3 + y4
        By = y1 - y2 + y3 - y4
        Cy = y1 + y2 - y3 - y4

        r_axis = np.array([(x1 + x4) / 2 - (x2 + x3) / 2, (y1 + y4) / 2 - (y2 + y3) / 2, 0.0])
        s_axis = np.array([(x1 + x2) / 2 - (x3 + x4) / 2, (y1 + y2) / 2 - (y3 + y4) / 2, 0.0])
        r_axis /= np.linalg.norm(r_axis)
        s_axis /= np.linalg.norm(s_axis)

        det_J = np.linalg.det(J_val)
        gr = np.sqrt((Cx + r * Bx) ** 2 + (Cy + r * By) ** 2) / (8 * det_J)
        gs = np.sqrt((Ax + s * Bx) ** 2 + (Ay + s * By) ** 2) / (8 * det_J)

        gamma_rz = gr * np.array([
            [
                (1 + s) / 2,
                -(y1 - y2) * (1 + s) / 4,
                (x1 - x2) * (1 + s) / 4,
                -(1 + s) / 2,
                -(y1 - y2) * (1 + s) / 4,
                (x1 - x2) * (1 + s) / 4,
                -(1 - s) / 2,
                -(y4 - y3) * (1 - s) / 4,
                (x4 - x3) * (1 - s) / 4,
                (1 - s) / 2,
                -(y4 - y3) * (1 - s) / 4,
                (x4 - x3) * (1 - s) / 4,
            ]
        ])

        gamma_sz = gs * np.array([
            [
                (1 + r) / 2,
                -(y1 - y4) * (1 + r) / 4,
                (x1 - x4) * (1 + r) / 4,
                (1 - r) / 2,
                -(y2 - y3) * (1 - r) / 4,
                (x2 - x3) * (1 - r) / 4,
                -(1 - r) / 2,
                -(y2 - y3) * (1 - r) / 4,
                (x2 - x3) * (1 - r) / 4,
                -(1 + r) / 2,
                -(y1 - y4) * (1 + r) / 4,
                (x1 - x4) * (1 + r) / 4,
            ]
        ])

        cos_alpha = np.dot(r_axis, [1, 0, 0])
        cos_beta = np.dot(s_axis, [1, 0, 0])
        sin_alpha = -np.linalg.norm(np.cross(r_axis, [1, 0, 0]))
        sin_beta = np.linalg.norm(np.cross(s_axis, [1, 0, 0]))

        return np.vstack((
            gamma_rz * sin_beta - gamma_sz * sin_alpha,
            -gamma_rz * cos_beta + gamma_sz * cos_alpha,
        ))

    def B_m(self, r, s):
        """
        Compute membrane strain-displacement matrix.

        Parameters
        ----------
        r : float
            Parametric coordinate in first direction [-1, 1]
        s : float
            Parametric coordinate in second direction [-1, 1]

        Returns
        -------
        np.ndarray
            3x8 membrane strain-displacement matrix

        Notes
        -----
        The matrix relates membrane strains to nodal displacements:
        ε_membrane = B_m @ u_elem
        where u_elem contains membrane DOFs [u1, v1, u2, v2, u3, v3, u4, v4]
        """
        J_val = self.J(r, s)
        dH = np.linalg.solve(
            J_val,
            1 / 4 * np.array([[s + 1, -s - 1, s - 1, -s + 1], [r + 1, -r + 1, r - 1, -r - 1]]),
        )
        return np.array([
            [dH[0, 0], 0, dH[0, 1], 0, dH[0, 2], 0, dH[0, 3], 0],
            [0, dH[1, 0], 0, dH[1, 1], 0, dH[1, 2], 0, dH[1, 3]],
            [dH[1, 0], dH[0, 0], dH[1, 1], dH[0, 1], dH[1, 2], dH[0, 2], dH[1, 3], dH[0, 3]],
        ])

    def Cb(self) -> np.ndarray:
        """
        Compute bending constitutive matrix for isotropic or orthotropic materials.

        Returns
        -------
        np.ndarray
            A 3x3 bending constitutive matrix.

        Notes
        -----
        For isotropic materials, the matrix follows the plane stress constitutive law:
        \\[
        C_b = \frac{E h^3}{12(1-\nu^2)} \begin{bmatrix}
        1 & \nu & 0 \\
        \nu & 1 & 0 \\
        0 & 0 & \frac{1-\nu}{2}
        \\end{bmatrix}
        \\]
        where \\( h \\) is the thickness, \\( E \\) is Young's modulus, and \\( \nu \\) is Poisson's ratio.

        For orthotropic materials, the matrix is:
        \\[
        C_b = \begin{bmatrix}
        D_{11} & D_{12} & 0 \\
        D_{12} & D_{22} & 0 \\
        0 & 0 & D_{66}
        \\end{bmatrix}
        \\]
        where:
        - \\( D_{11} = \frac{E_1 h^3}{12(1-\nu_{12}\nu_{21})} \\)
        - \\( D_{12} = \frac{\nu_{12} E_2 h^3}{12(1-\nu_{12}\nu_{21})} \\)
        - \\( D_{22} = \frac{E_2 h^3}{12(1-\nu_{12}\nu_{21})} \\)
        - \\( D_{66} = \frac{G_{12} h^3}{12} \\)
        and \\( \nu_{21} = \nu_{12} \frac{E_2}{E_1} \\).
        """
        h = self.thickness
        if isinstance(self.material, OrthotropicMaterial):
            E1, E2, _ = self.material.E
            nu12, _, _ = self.material.nu
            G12, _, _ = self.material.G

            # Reciprocal Poisson's ratio
            nu21 = nu12 * E2 / E1

            denominator = 1 - nu12 * nu21
            factor = h**3 / (12 * denominator)

            D11 = E1 * factor
            D12 = nu12 * E2 * factor
            D22 = E2 * factor
            D66 = G12 * h**3 / 12

            return np.array([[D11, D12, 0], [D12, D22, 0], [0, 0, D66]])

        else:
            E = self.material.E
            nu = self.material.nu
            factor = E * h**3 / (12 * (1 - nu**2))
            return factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    def Cs(self) -> np.ndarray:
        """
        Compute shear constitutive matrix for isotropic or orthotropic materials.

        Returns
        -------
        np.ndarray
            A 2x2 shear constitutive matrix.

        Notes
        -----
        For isotropic materials, the matrix is:
        \\[
        C_s = \frac{E h k}{2(1+\nu)} \\mathbf{I}
        \\]
        where \\( k = \frac{5}{6} \\) is the shear correction factor, \\( h \\) is the thickness,
        \\( E \\) is Young's modulus, and \\( \nu \\) is Poisson's ratio.

        For orthotropic materials, the matrix is:
        \\[
        C_s = \begin{bmatrix}
        G_{13} h k & 0 \\
        0 & G_{23} h k
        \\end{bmatrix}
        \\]
        where \\( G_{13} \\) and \\( G_{23} \\) are the transverse shear moduli.
        """
        k = 5 / 6
        h = self.thickness

        if isinstance(self.material, OrthotropicMaterial):
            _, G23, G13 = self.material.G
            return np.diag([G13 * h * k, G23 * h * k])

        else:  # Isotropic material
            E = self.material.E
            nu = self.material.nu
            G = E / (2 * (1 + nu))
            return G * h * k * np.eye(2)

    def Cm(self) -> np.ndarray:
        """
        Compute membrane constitutive matrix for isotropic or orthotropic materials.

        Returns
        -------
        np.ndarray
            A 3x3 membrane constitutive matrix.

        Notes
        -----
        For isotropic materials, the matrix is:
        \\[
        C_m = \frac{1}{1-\nu^2} \begin{bmatrix}
        E_x & \nu E_y & 0 \\
        \nu E_x & E_y & 0 \\
        0 & 0 & G (1-\nu^2)
        \\end{bmatrix}
        \\]
        where \\( E_x = E \\cdot k_{x\\_mod} \\), \\( E_y = E \\cdot k_{y\\_mod} \\), and \\( G = \frac{E}{2(1+\nu)} \\).

        For orthotropic materials, the matrix is:
        \\[
        C_m = \begin{bmatrix}
        \frac{E_1}{1-\nu_{12}\nu_{21}} & \frac{\nu_{12} E_2}{1-\nu_{12}\nu_{21}} & 0 \\
        \frac{\nu_{21} E_1}{1-\nu_{12}\nu_{21}} & \frac{E_2}{1-\nu_{12}\nu_{21}} & 0 \\
        0 & 0 & G_{12}
        \\end{bmatrix}
        \\]
        where \\( \nu_{21} = \nu_{12} \frac{E_2}{E_1} \\), and stiffness modifiers \\( k_{x\\_mod} \\) and \\( k_{y\\_mod} \\)
        are applied to \\( E_1 \\) and \\( E_2 \\), respectively.
        """
        if isinstance(self.material, OrthotropicMaterial):
            E1 = self.material.E[0] * self.kx_mod
            E2 = self.material.E[1] * self.ky_mod
            nu12, _, _ = self.material.nu
            G12, _, _ = self.material.G

            nu21 = nu12 * E2 / E1

            denominator = 1 - nu12 * nu21
            inv_denom = 1 / denominator

            return np.array([
                [E1 * inv_denom, nu12 * E2 * inv_denom, 0],
                [nu21 * E1 * inv_denom, E2 * inv_denom, 0],
                [0, 0, G12],
            ])

        else:
            E = self.material.E
            nu = self.material.nu
            Ex = E * self.kx_mod
            Ey = E * self.ky_mod
            G = E / (2 * (1 + nu))

            inv_denom = 1 / (1 - nu**2)
            return inv_denom * np.array([
                [Ex, nu * Ey, 0],
                [nu * Ex, Ey, 0],
                [0, 0, G * (1 - nu**2)],
            ])

    @staticmethod
    def index_k_b():
        """
        Get index mapping for bending stiffness matrix assembly.

        Returns
        -------
        Tuple containing:
        - m_arr : np.ndarray
            Row indices in expanded 24x24 matrix
        - n_arr : np.ndarray
            Column indices in expanded 24x24 matrix
        - i_arr : np.ndarray
            Row indices in local 12x12 matrix
        - j_arr : np.ndarray
            Column indices in local 12x12 matrix

        Notes
        -----
        Maps local bending DOFs (rotations θz) to global DOF positions:
        Nodes 1-4 rotation_z DOFs at positions [5, 11, 17, 23]
        """
        global_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22], dtype=int)

        # Create meshgrid for local indices
        i_arr, j_arr = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")

        # Create expanded indices using broadcasting
        m_arr = np.broadcast_to(global_dofs[:, None], (12, 12))
        n_arr = np.broadcast_to(global_dofs, (12, 12))

        return m_arr, n_arr, i_arr, j_arr

    def k_b(self):
        """
        Assemble element bending stiffness matrix.

        Returns
        -------
        np.ndarray
            24x24 bending stiffness matrix in local coordinates

        Notes
        -----
        Construction process:
        1. Numerical integration with 2x2 Gauss points
        2. Combines bending (B_kappa) and shear (B_gamma) contributions
        3. Adds drilling stiffness (1e-3 of smallest diagonal entry)
        4. Uses index mapping from index_k_b() for DOF expansion
        """
        Cb_val = self.Cb()
        Cs_val = self.Cs()

        # Initialize contributions.
        k1 = 0.0
        k2 = 0.0

        # Loop over the Gauss points only once to compute both contributions.
        for r, s in self._gauss_points:
            detJ = np.linalg.det(self.J(r, s))
            B_kappa = self.B_kappa(r, s)
            B_gamma = self.B_gamma(r, s)
            k1 += B_kappa.T @ Cb_val @ B_kappa * detJ
            k2 += B_gamma.T @ Cs_val @ B_gamma * detJ

        # Sum the bending and shear stiffness matrices.
        k = k1 + k2

        # Expand the stiffness matrix using precomputed index arrays.
        k_exp = np.zeros((24, 24))
        m_arr, n_arr, i_arr, j_arr = self.index_k_b()
        k_exp[m_arr, n_arr] = k[i_arr, j_arr]

        # Adjust drilling degrees of freedom to avoid singularities.
        drilling_dofs = np.array([5, 11, 17, 23])
        k_exp[drilling_dofs, drilling_dofs] = np.min(np.abs(np.diag(k))) / 1000

        return k_exp

    @staticmethod
    def index_k_m():
        """
        Get index mapping for membrane stiffness matrix assembly.

        Returns
        -------
        Tuple containing:
        - m_arr : np.ndarray
            Row indices in expanded 24x24 matrix
        - n_arr : np.ndarray
            Column indices in expanded 24x24 matrix
        - i_arr : np.ndarray
            Row indices in local 8x8 matrix
        - j_arr : np.ndarray
            Column indices in local 8x8 matrix

        Notes
        -----
        Maps local membrane DOFs (translations u,v) to global DOF positions:
        Nodes 1-4 translation DOFs at positions [0,1,6,7,12,13,18,19]
        """
        global_dofs = np.array([0, 1, 6, 7, 12, 13, 18, 19], dtype=int)

        # Create meshgrid for local indices
        i_arr, j_arr = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")

        # Create expanded indices using broadcasting
        m_arr = np.broadcast_to(global_dofs[:, None], (8, 8))
        n_arr = np.broadcast_to(global_dofs, (8, 8))

        return m_arr, n_arr, i_arr, j_arr

    def k_m(self):
        """
        Assemble element membrane stiffness matrix.

        Returns
        -------
        np.ndarray
            24x24 membrane stiffness matrix in local coordinates

        Notes
        -----
        Construction process:
        1. Numerical integration with 2x2 Gauss points
        2. Uses membrane strain-displacement matrix B_m
        3. Applies membrane constitutive matrix Cm
        4. Uses index mapping from index_k_m() for DOF expansion
        """
        # Compute the material matrix (assumed constant over the element)
        Cm_val = self.Cm()

        # Accumulate the integrated stiffness contributions at each Gauss point
        contributions = []
        for r, s in self._gauss_points:
            B = self.B_m(r, s)
            detJ = np.linalg.det(self.J(r, s))
            contributions.append(B.T @ Cm_val @ B * detJ)

        # Sum the contributions and multiply by the thickness
        k_local = self.thickness * np.sum(contributions, axis=0)

        # Expand the stiffness matrix using precomputed index arrays
        m_arr, n_arr, i_arr, j_arr = self.index_k_m()
        k_expanded = np.zeros((24, 24))
        k_expanded[m_arr, n_arr] = k_local[i_arr, j_arr]

        return k_expanded

    @property
    def K(self):
        """
        Global stiffness matrix of the element.

        Returns
        -------
        np.ndarray
            24x24 element stiffness matrix in global coordinates

        Notes
        -----
        The stiffness matrix is calculated as:
        K_global = T.T @ (K_membrane + K_bending) @ T
        where T is the transformation matrix.
        """
        ele_K = self.k_m() + self.k_b()
        T = self.T()
        return np.linalg.solve(T, ele_K) @ T

    @property
    def M(self) -> np.ndarray:
        """
        Compute the mass matrix including all 6 DOFs per node.
        """
        rho = self.material.rho
        h = self.thickness
        assert rho > 0 and h > 0, "Material properties must be positive"

        M_local = np.zeros((24, 24))

        for r, s in self._gauss_points:
            detJ = np.linalg.det(self.J(r, s))
            N = self.shape_functions(r, s)

            # Componentes de traslación (u, v, w)
            M_trans = rho * h * (N[:3].T @ N[:3]) * detJ

            # Inercia rotacional para θx, θy (usando h^3/12) y θz (pequeño valor)
            I_rot_xy = (rho * h**3) / 12
            I_rot_z = 1e-6 * rho * h

            # Ensamblar contribuciones rotacionales
            M_rot = np.zeros((24, 24))

            # Contribuciones de θx y θy (suma de productos externos)
            for dof in [3, 4]:  # θx (dof=3), θy (dof=4)
                M_rot += I_rot_xy * np.outer(N[dof], N[dof]) * detJ

            # Contribución de θz usando producto externo
            M_rot += I_rot_z * np.outer(N[5], N[5]) * detJ  # θz

            M_local += M_trans + M_rot

        T = self.T()
        M = T.T @ M_local @ T
        assert np.all(np.linalg.eigvalsh(M) > -1e-10), "Matriz no positiva semi-definida"
        return M

    def shape_functions(self, r: float, s: float) -> np.ndarray:
        """
        Funciones de forma unificadas para 6 GDL por nodo.
        Returns:
            np.ndarray (6×24): [u, v, w, θx, θy, θz]^T
        """
        N1 = 0.25 * (1 - r) * (1 - s)
        N2 = 0.25 * (1 + r) * (1 - s)
        N3 = 0.25 * (1 + r) * (1 + s)
        N4 = 0.25 * (1 - r) * (1 + s)

        N = np.zeros((6, 24))
        for i in range(6):
            N[i, i::6] = [N1, N2, N3, N4]

        return N

    def body_load(self, body_force: np.ndarray) -> np.ndarray:
        """
        Vector de carga actualizado para 6 GDL.
        """
        f = np.zeros(24)

        for r, s in self._gauss_points:
            J = self.J(r, s)
            det_J = np.linalg.det(J)
            N = self.shape_functions(r, s)[:3]  # Solo componentes de desplazamiento

            f += (N.T @ body_force[:3]) * det_J * self.thickness

        return f


class MITC4Layered(MITC4):
    """
    MITC4 quadrilateral shell element with multiple material layers.

    This element extends the MITC4 formulation to support multiple layers of
    isotropic materials, each with their own thickness and orientation.

    Parameters
    ----------
    node_coords : np.ndarray
        Array of nodal coordinates in global system [4x3]
    node_ids : Tuple[int, int, int, int]
        Global node IDs for element connectivity
    layers : List[Tuple[Material, float, float]]
        List of layers, each defined by a material, thickness (float), and
        orientation angle in radians (float)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        layers: List[Tuple[Material, float, float]],
    ):
        # Extract thickness from each layer and calculate total thickness
        total_thickness = sum(thickness for material, thickness, angle in layers)

        # Initialize the base class with dummy material (first layer's material)
        super().__init__(
            node_coords=node_coords,
            node_ids=node_ids,
            material=layers[0][0],  # Dummy material, not used in overridden methods
            thickness=total_thickness,
            kx_mod=1.0,  # Not used in this implementation
            ky_mod=1.0,  # Not used in this implementation
        )

        self.layers = layers

        # Precompute z positions for each layer relative to mid-surface
        self.layer_info = self._compute_layer_z_positions(total_thickness)

    def _compute_layer_z_positions(self, total_thickness: float) -> List[dict]:
        """
        Compute each layer's z-coordinates relative to the mid-surface.

        Returns
        -------
        List[dict]
            List of layer information dictionaries containing:
            - material: IsotropicMaterial
            - thickness: float
            - angle: float (orientation in radians)
            - z_centroid: float (distance from mid-surface)
            - z_bottom: float (bottom coordinate relative to mid-surface)
            - z_top: float (top coordinate relative to mid-surface)
        """
        mid_surface_z = total_thickness / 2.0
        current_z_bottom = -mid_surface_z  # Start from the bottom of the shell
        layer_info = []

        for material, thickness, angle in self.layers:
            z_top = current_z_bottom + thickness
            z_centroid = (current_z_bottom + z_top) / 2.0

            layer_info.append({
                "material": material,
                "thickness": thickness,
                "angle": angle,
                "z_centroid": z_centroid,
                "z_bottom": current_z_bottom,
                "z_top": z_top,
            })

            current_z_bottom = z_top  # Move to next layer

        return layer_info

    def Cm(self) -> np.ndarray:
        """
        Compute membrane constitutive matrix considering all layers.

        Returns
        -------
        np.ndarray
            3x3 membrane constitutive matrix
        """
        Cm = np.zeros((3, 3))
        for layer in self.layer_info:
            mat = layer["material"]
            h = layer["thickness"]
            E = mat.E
            nu = mat.nu

            # Membrane constitutive matrix for isotropic material
            Q = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

            Cm += Q * h

        return Cm

    def Cb(self) -> np.ndarray:
        """
        Compute bending constitutive matrix considering all layers.

        Returns
        -------
        np.ndarray
            3x3 bending constitutive matrix
        """
        Cb = np.zeros((3, 3))
        for layer in self.layer_info:
            mat = layer["material"]
            h = layer["thickness"]
            z_centroid = layer["z_centroid"]
            E = mat.E
            nu = mat.nu

            # Bending constitutive matrix for isotropic material
            Q = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

            # Contribution from layer's bending and position
            I_term = (h**3) / 12 + h * z_centroid**2
            Cb += Q * I_term

        return Cb

    def Cs(self) -> np.ndarray:
        """
        Compute shear constitutive matrix considering all layers.

        Returns
        -------
        np.ndarray
            2x2 shear constitutive matrix
        """
        total_shear = 0.0
        for layer in self.layer_info:
            mat = layer["material"]
            h = layer["thickness"]
            G = mat.E / (2 * (1 + mat.nu))
            total_shear += G * h

        k = 5 / 6  # Shear correction factor
        return (total_shear * k) * np.eye(2)


# Copyright notice and license information as required by MIT

"""
Adapted from JaxSSO (https://github.com/GaoyuanWu/JaxSSO), Copyright (c) 2021 Gaoyuan Wu.
Original work licensed under MIT License. This adapted version maintains the same license terms.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
