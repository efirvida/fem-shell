"""
MITC4+ Four-Node Shell Element Implementation.

This implementation uses the MITC4+ formulation exclusively, which provides
improved performance for warped (non-planar) elements through bubble function
enrichment and static condensation. The MITC4+ formulation guarantees symmetric
stiffness matrices even for highly distorted geometries.

References:
    [1] Dvorkin & Bathe (1984). Engineering Computations, 1, 77-88.
    [2] Ko, Lee & Bathe (2017). Computers and Structures, 182, 404-418.
    [3] Bathe (2006). Finite Element Procedures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC4(ShellElement):
    """
    MITC4+ four-node quadrilateral shell element.

    Each corner node has 6 DOFs (u, v, w, θx, θy, θz). Uses mixed interpolation
    for transverse shear strains with bubble function enrichment to prevent shear
    locking and ensure symmetric stiffness matrices for warped elements.

    Parameters
    ----------
    node_coords : np.ndarray
        Nodal coordinates in global system [4x3]
    node_ids : Tuple[int, int, int, int]
        Global node IDs for element connectivity
    material : Material
        Material properties object
    thickness : float
        Shell thickness
    shear_correction_factor : float, optional
        Shear correction factor (default: 5/6)
    nonlinear : bool, optional
        Enable geometric nonlinear analysis (default: False)
    """

    vector_form = {"U": ("Ux", "Uy", "Uz"), "θ": ("θx", "θy", "θz")}

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        material: Material,
        thickness: float,
        shear_correction_factor: Optional[float] = None,
        nonlinear: bool = False,
    ):
        super().__init__(
            "MITC4",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=6,
            thickness=thickness,
            nonlinear=nonlinear,
        )

        self.element_type = "MITC4"

        self._initial_coords = np.array(node_coords, dtype=float).copy()
        self._current_coords = np.array(node_coords, dtype=float).copy()
        self._current_displacements = np.zeros(24)

        if shear_correction_factor is not None:
            self._shear_correction_factor = shear_correction_factor
        elif (
            hasattr(material, "shear_correction_factor")
            and material.shear_correction_factor is not None
        ):
            self._shear_correction_factor = material.shear_correction_factor
        else:
            self._shear_correction_factor = 5.0 / 6.0

        gp = 1.0 / np.sqrt(3.0)
        self._gauss_points = [(-gp, -gp), (+gp, -gp), (+gp, +gp), (-gp, +gp)]
        self._gauss_weights = [1.0, 1.0, 1.0, 1.0]

        self._local_coordinates = self._compute_local_coordinates()
        self._initial_directors = self._compute_initial_directors()
        self._current_directors = self._initial_directors.copy()

        self._dH_cache = {}
        self._N_cache = {}
        self._jacobian_cache = {}

        self._init_tying_points()

    def _init_tying_points(self) -> None:
        """Initialize tying points and precompute matrices."""
        self._tying_points_edge = {
            "A": (0.0, 1.0),
            "B": (-1.0, 0.0),
            "C": (0.0, -1.0),
            "D": (1.0, 0.0),
        }
        self._tying_point_center = (0.0, 0.0)

        self._membrane_tying_points = {
            "A": (0.0, 1.0),
            "B": (0.0, -1.0),
            "C": (1.0, 0.0),
            "D": (-1.0, 0.0),
            "E": (0.0, 0.0),
        }

        self._bubble_cache = {}
        for xi, eta in self._gauss_points:
            Nb = self._bubble_function(xi, eta)
            dNb_dxi, dNb_deta = self._bubble_derivatives(xi, eta)
            self._bubble_cache[(xi, eta)] = (Nb, dNb_dxi, dNb_deta)

        self._compute_characteristic_vectors()
        self._compute_membrane_coefficients()
        self._precompute_membrane_tying_matrices()

    # =========================================================================
    # SHAPE FUNCTIONS
    # =========================================================================

    def _shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Compute bilinear shape functions at (xi, eta)."""
        N0 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N1 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N3 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return np.array([N0, N1, N2, N3])

    def _shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute shape function derivatives (dN/dxi, dN/deta)."""
        dN_dxi = np.array(
            [
                -0.25 * (1.0 - eta),
                +0.25 * (1.0 - eta),
                +0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ]
        )
        dN_deta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                +0.25 * (1.0 + xi),
                +0.25 * (1.0 - xi),
            ]
        )
        return dN_dxi, dN_deta

    def _compute_characteristic_vectors(self) -> None:
        """Compute characteristic vectors (Ko et al. 2017, Eqs. 9-12)."""
        nodes = self._initial_coords

        self._x_r = 0.25 * (-nodes[0] + nodes[1] + nodes[2] - nodes[3])
        self._x_s = 0.25 * (-nodes[0] - nodes[1] + nodes[2] + nodes[3])
        self._x_d = 0.25 * (nodes[0] - nodes[1] + nodes[2] - nodes[3])

        n_vec = np.cross(self._x_r, self._x_s)
        norm_n = np.linalg.norm(n_vec)
        if norm_n < 1e-12:
            n_vec = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
            norm_n = np.linalg.norm(n_vec)
        self._n = n_vec / norm_n

        A = np.array(
            [
                [
                    np.dot(self._x_r, self._x_r),
                    np.dot(self._x_r, self._x_s),
                    np.dot(self._x_r, self._n),
                ],
                [
                    np.dot(self._x_s, self._x_r),
                    np.dot(self._x_s, self._x_s),
                    np.dot(self._x_s, self._n),
                ],
                [np.dot(self._n, self._x_r), np.dot(self._n, self._x_s), np.dot(self._n, self._n)],
            ]
        )

        try:
            coeff_r = np.linalg.solve(A, np.array([1, 0, 0]))
            coeff_s = np.linalg.solve(A, np.array([0, 1, 0]))
            self._m_r = coeff_r[0] * self._x_r + coeff_r[1] * self._x_s + coeff_r[2] * self._n
            self._m_s = coeff_s[0] * self._x_r + coeff_s[1] * self._x_s + coeff_s[2] * self._n
        except np.linalg.LinAlgError:
            self._m_r = self._x_r / np.dot(self._x_r, self._x_r)
            self._m_s = self._x_s / np.dot(self._x_s, self._x_s)

    def _compute_membrane_coefficients(self) -> None:
        """Compute MITC4+ membrane coefficients (Ko et al. 2017, Eqs. 24-28)."""
        c_r = np.dot(self._x_d, self._m_r)
        c_s = np.dot(self._x_d, self._m_s)

        d = c_r**2 + c_s**2 - 1.0
        if abs(d) < 1e-12:
            d = 1e-12 if d >= 0 else -1e-12

        self._a_A = c_r * (c_r - 1.0) / (2.0 * d)
        self._a_B = c_r * (c_r + 1.0) / (2.0 * d)
        self._a_C = c_s * (c_s - 1.0) / (2.0 * d)
        self._a_D = c_s * (c_s + 1.0) / (2.0 * d)
        self._a_E = -2.0 * c_r * c_s / d

    # =========================================================================
    # MITC4+ BUBBLE FUNCTIONS
    # =========================================================================

    def _bubble_function(self, xi: float, eta: float) -> float:
        """Compute bubble function Nb = (1 - xi^2)(1 - eta^2)."""
        return (1.0 - xi * xi) * (1.0 - eta * eta)

    def _bubble_derivatives(self, xi: float, eta: float) -> Tuple[float, float]:
        """Compute bubble function derivatives (dNb/dxi, dNb/deta)."""
        dNb_dxi = -2.0 * xi * (1.0 - eta * eta)
        dNb_deta = -2.0 * eta * (1.0 - xi * xi)
        return dNb_dxi, dNb_deta

    def _bubble_second_derivatives(self, xi: float, eta: float) -> Tuple[float, float, float]:
        """Compute bubble function second derivatives."""
        d2Nb_dxi2 = -2.0 * (1.0 - eta * eta)
        d2Nb_deta2 = -2.0 * (1.0 - xi * xi)
        d2Nb_dxideta = 4.0 * xi * eta
        return d2Nb_dxi2, d2Nb_deta2, d2Nb_dxideta

    # =========================================================================
    # MITC4+ ASSUMED MEMBRANE STRAIN INTERPOLATION
    # =========================================================================

    def _precompute_membrane_tying_matrices(self) -> None:
        """Precompute membrane strain B-matrix components at tying points."""
        self._membrane_tying_data = {}

        for name, (xi, eta) in self._membrane_tying_points.items():
            J_mat, detJ = self.J(xi, eta)
            J_inv = np.linalg.inv(J_mat)
            dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)
            dH = self._get_dH(xi, eta)

            g_r = J_mat[0, :]
            g_s = J_mat[1, :]

            self._membrane_tying_data[name] = {
                "xi": xi,
                "eta": eta,
                "J": J_mat,
                "J_inv": J_inv,
                "detJ": detJ,
                "dN_dxi": dN_dxi,
                "dN_deta": dN_deta,
                "dH": dH,
                "g_r": g_r,
                "g_s": g_s,
            }

    def _compute_covariant_membrane_strain_B(
        self, xi: float, eta: float, component: str
    ) -> np.ndarray:
        """Compute covariant membrane strain-displacement relation for a component.

        Uses 3D tangent vectors computed from global coordinates to properly
        handle warped (non-planar) elements. The covariant membrane strains are
        (Ko et al. 2017, Eq. 24):

            e_rr = g_r · (∂u/∂r)
            e_ss = g_s · (∂u/∂s)
            e_rs = 0.5 * (g_r · (∂u/∂s) + g_s · (∂u/∂r))

        where u = (u, v, w) is the 3D displacement vector in global coordinates
        and g_r, g_s are the 3D covariant base vectors.

        The resulting B matrix relates displacements in the LOCAL coordinate
        system (after transformation by T) to covariant strains.
        """
        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)

        # Get 3D tangent vectors from global coordinates
        g_r_3D, g_s_3D = self.J3D(xi, eta)

        # Project 3D tangent vectors to local coordinate system
        # The local system has basis vectors e1, e2, e3
        g_r_local = np.array(
            [
                np.dot(g_r_3D, self._e1),
                np.dot(g_r_3D, self._e2),
                np.dot(g_r_3D, self._e3),
            ]
        )
        g_s_local = np.array(
            [
                np.dot(g_s_3D, self._e1),
                np.dot(g_s_3D, self._e2),
                np.dot(g_s_3D, self._e3),
            ]
        )

        B_row = np.zeros(24)

        for i in range(4):
            u_idx = 6 * i  # u displacement (local x)
            v_idx = 6 * i + 1  # v displacement (local y)
            w_idx = 6 * i + 2  # w displacement (local z = normal)

            if component == "rr":
                # e_rr = g_r · (∂u/∂r) in local coordinates
                B_row[u_idx] = g_r_local[0] * dN_dxi[i]
                B_row[v_idx] = g_r_local[1] * dN_dxi[i]
                B_row[w_idx] = g_r_local[2] * dN_dxi[i]
            elif component == "ss":
                # e_ss = g_s · (∂u/∂s) in local coordinates
                B_row[u_idx] = g_s_local[0] * dN_deta[i]
                B_row[v_idx] = g_s_local[1] * dN_deta[i]
                B_row[w_idx] = g_s_local[2] * dN_deta[i]
            elif component == "rs":
                # e_rs = 0.5 * (g_r · (∂u/∂s) + g_s · (∂u/∂r))
                B_row[u_idx] = 0.5 * (g_r_local[0] * dN_deta[i] + g_s_local[0] * dN_dxi[i])
                B_row[v_idx] = 0.5 * (g_r_local[1] * dN_deta[i] + g_s_local[1] * dN_dxi[i])
                B_row[w_idx] = 0.5 * (g_r_local[2] * dN_deta[i] + g_s_local[2] * dN_dxi[i])

        return B_row

    def _covariant_to_cartesian_strain_transform(self, xi: float, eta: float) -> np.ndarray:
        """Transform covariant strains to Cartesian strains."""
        J_mat, detJ = self.J(xi, eta)
        if abs(detJ) < 1e-12:
            J_inv = np.linalg.pinv(J_mat)
        else:
            J_inv = np.linalg.inv(J_mat)

        if abs(detJ) < 1e-12:
            return np.eye(3)

        dxi_dx = J_inv[0, 0]
        dxi_dy = J_inv[0, 1]
        deta_dx = J_inv[1, 0]
        deta_dy = J_inv[1, 1]

        T = np.array(
            [
                [dxi_dx**2, deta_dx**2, dxi_dx * deta_dx],
                [dxi_dy**2, deta_dy**2, dxi_dy * deta_dy],
                [
                    2.0 * dxi_dx * dxi_dy,
                    2.0 * deta_dx * deta_dy,
                    dxi_dx * deta_dy + dxi_dy * deta_dx,
                ],
            ]
        )

        return T

    def B_m_MITC4_plus(self, xi: float, eta: float) -> np.ndarray:
        """MITC4+ membrane B-matrix (Ko et al. 2017, Eqs. 27a-c)."""
        tp = self._membrane_tying_points

        B_rr_A = self._compute_covariant_membrane_strain_B(tp["A"][0], tp["A"][1], "rr")
        B_rr_B = self._compute_covariant_membrane_strain_B(tp["B"][0], tp["B"][1], "rr")
        B_ss_C = self._compute_covariant_membrane_strain_B(tp["C"][0], tp["C"][1], "ss")
        B_ss_D = self._compute_covariant_membrane_strain_B(tp["D"][0], tp["D"][1], "ss")
        B_rs_E = self._compute_covariant_membrane_strain_B(tp["E"][0], tp["E"][1], "rs")

        a_A, a_B, a_C, a_D, a_E = self._a_A, self._a_B, self._a_C, self._a_D, self._a_E
        r, s = xi, eta

        B_rr = (
            (0.5 * (1.0 - 2.0 * a_A + s + 2.0 * a_A * s * s)) * B_rr_A
            + (0.5 * (1.0 - 2.0 * a_B - s + 2.0 * a_B * s * s)) * B_rr_B
            + a_C * (-1.0 + s * s) * B_ss_C
            + a_D * (-1.0 + s * s) * B_ss_D
            + a_E * (-1.0 + s * s) * B_rs_E
        )

        B_ss = (
            a_A * (-1.0 + r * r) * B_rr_A
            + a_B * (-1.0 + r * r) * B_rr_B
            + (0.5 * (1.0 - 2.0 * a_C + r + 2.0 * a_C * r * r)) * B_ss_C
            + (0.5 * (1.0 - 2.0 * a_D - r + 2.0 * a_D * r * r)) * B_ss_D
            + a_E * (-1.0 + r * r) * B_rs_E
        )

        B_rs = (
            0.25 * (r + 4.0 * a_A * r * s) * B_rr_A
            + 0.25 * (-r + 4.0 * a_B * r * s) * B_rr_B
            + 0.25 * (s + 4.0 * a_C * r * s) * B_ss_C
            + 0.25 * (-s + 4.0 * a_D * r * s) * B_ss_D
            + (1.0 + a_E * r * s) * B_rs_E
        )

        B_covariant = np.vstack([B_rr, B_ss, 2.0 * B_rs])
        T = self._covariant_to_cartesian_strain_transform(xi, eta)
        B_cartesian = T @ B_covariant

        return B_cartesian

    # =========================================================================
    # LOCAL COORDINATE SYSTEM
    # =========================================================================

    def _compute_element_warping(self) -> float:
        """Compute warping metric (0 = flat, higher = more warped)."""
        nodes = self._current_coords
        normals = np.zeros((4, 3))

        v01 = nodes[1] - nodes[0]
        v03 = nodes[3] - nodes[0]
        normals[0] = np.cross(v01, v03)

        v12 = nodes[2] - nodes[1]
        v10 = nodes[0] - nodes[1]
        normals[1] = np.cross(v12, v10)

        v23 = nodes[3] - nodes[2]
        v21 = nodes[1] - nodes[2]
        normals[2] = np.cross(v23, v21)

        v30 = nodes[0] - nodes[3]
        v32 = nodes[2] - nodes[3]
        normals[3] = np.cross(v30, v32)

        for i in range(4):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm

        mean_normal = np.mean(normals, axis=0)
        mean_norm = np.linalg.norm(mean_normal)
        if mean_norm > 1e-12:
            mean_normal /= mean_norm

        min_dot = 1.0
        for i in range(4):
            dot = np.dot(normals[i], mean_normal)
            min_dot = min(min_dot, dot)

        return 1.0 - min_dot

    def _compute_local_coordinates(self) -> np.ndarray:
        """Compute local coordinates for thin plates."""
        nodes = self._initial_coords

        normals = []
        v1 = nodes[1] - nodes[0]
        v2 = nodes[2] - nodes[0]
        n1 = np.cross(v1, v2)
        if np.linalg.norm(n1) > 1e-12:
            normals.append(n1 / np.linalg.norm(n1))

        v1 = nodes[2] - nodes[0]
        v2 = nodes[3] - nodes[0]
        n2 = np.cross(v1, v2)
        if np.linalg.norm(n2) > 1e-12:
            normals.append(n2 / np.linalg.norm(n2))

        if normals:
            e3 = np.mean(normals, axis=0)
            e3 /= np.linalg.norm(e3)
        else:
            e3 = np.array([0, 0, 1])

        e1 = nodes[1] - nodes[0]
        e1 = e1 - np.dot(e1, e3) * e3
        if np.linalg.norm(e1) < 1e-12:
            e1 = nodes[2] - nodes[0]
            e1 = e1 - np.dot(e1, e3) * e3
        e1 /= np.linalg.norm(e1)

        e2 = np.cross(e3, e1)
        e2 /= np.linalg.norm(e2)

        self._e1, self._e2, self._e3 = e1, e2, e3

        local_coords = np.zeros((4, 2))
        for i in range(4):
            local_coords[i, 0] = np.dot(nodes[i], e1)
            local_coords[i, 1] = np.dot(nodes[i], e2)

        return local_coords

    def _compute_local_coordinates_standard(self) -> np.ndarray:
        """Standard local coordinate computation for nearly flat elements."""
        nodes = self._current_coords

        v1 = 0.5 * (nodes[2] + nodes[1] - nodes[3] - nodes[0])
        v2 = 0.5 * (nodes[3] + nodes[2] - nodes[1] - nodes[0])

        length = np.linalg.norm(v1)
        if length < 1e-12:
            raise ValueError("Degenerate element: v1 has zero length")
        e1 = v1 / length

        alpha = np.dot(v2, e1)
        v2_orth = v2 - alpha * e1

        length = np.linalg.norm(v2_orth)
        if length < 1e-12:
            raise ValueError("Degenerate element: orthogonalized v2 has zero length")
        e2 = v2_orth / length

        e3 = np.cross(e1, e2)
        self._e1, self._e2, self._e3 = e1, e2, e3

        local_coords = np.zeros((4, 2))
        for i in range(4):
            local_coords[i, 0] = np.dot(nodes[i], e1)
            local_coords[i, 1] = np.dot(nodes[i], e2)

        return local_coords

    def _compute_local_coordinates_robust(self) -> np.ndarray:
        """Robust local coordinate computation for warped elements."""
        nodes = self._current_coords
        normals = np.zeros((4, 3))

        v01 = nodes[1] - nodes[0]
        v03 = nodes[3] - nodes[0]
        normals[0] = np.cross(v01, v03)

        v12 = nodes[2] - nodes[1]
        v10 = nodes[0] - nodes[1]
        normals[1] = np.cross(v12, v10)

        v23 = nodes[3] - nodes[2]
        v21 = nodes[1] - nodes[2]
        normals[2] = np.cross(v23, v21)

        v30 = nodes[0] - nodes[3]
        v32 = nodes[2] - nodes[3]
        normals[3] = np.cross(v30, v32)

        for i in range(4):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm
            else:
                d1 = nodes[2] - nodes[0]
                d2 = nodes[3] - nodes[1]
                normals[i] = np.cross(d1, d2)
                norm = np.linalg.norm(normals[i])
                if norm > 1e-12:
                    normals[i] /= norm

        e3 = np.mean(normals, axis=0)
        norm_e3 = np.linalg.norm(e3)
        if norm_e3 < 1e-12:
            d1 = nodes[2] - nodes[0]
            d2 = nodes[3] - nodes[1]
            e3 = np.cross(d1, d2)
            norm_e3 = np.linalg.norm(e3)
            if norm_e3 < 1e-12:
                raise ValueError("Degenerate element: cannot determine normal")
        e3 /= norm_e3

        v01 = nodes[1] - nodes[0]
        e1 = v01 - np.dot(v01, e3) * e3
        norm_e1 = np.linalg.norm(e1)

        if norm_e1 < 1e-12:
            v02 = nodes[2] - nodes[0]
            e1 = v02 - np.dot(v02, e3) * e3
            norm_e1 = np.linalg.norm(e1)
            if norm_e1 < 1e-12:
                raise ValueError("Degenerate element: cannot determine local x-axis")
        e1 /= norm_e1

        e2 = np.cross(e3, e1)
        self._e1, self._e2, self._e3 = e1, e2, e3

        local_coords = np.zeros((4, 2))
        ref_node = nodes[0]
        for i in range(4):
            rel_pos = nodes[i] - ref_node
            local_coords[i, 0] = np.dot(rel_pos, e1)
            local_coords[i, 1] = np.dot(rel_pos, e2)

        center_x = np.mean(local_coords[:, 0])
        center_y = np.mean(local_coords[:, 1])
        local_coords[:, 0] -= center_x
        local_coords[:, 1] -= center_y

        return local_coords

    def _compute_initial_directors(self) -> np.ndarray:
        """Compute initial nodal directors (normals)."""
        return np.tile(self._e3, (4, 1))

    # =========================================================================
    # JACOBIAN AND TRANSFORMATION
    # =========================================================================

    def J3D(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3D tangent vectors g_r and g_s at (xi, eta).

        For warped elements, these vectors have components in all 3 directions.
        This is essential for correctly computing membrane strains in warped elements.

        Returns
        -------
        g_r : np.ndarray
            Tangent vector in r-direction (3D)
        g_s : np.ndarray
            Tangent vector in s-direction (3D)
        """
        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)
        x = self._initial_coords

        # g_r = dx/dr = sum(dN_i/dr * x_i)
        g_r = np.zeros(3)
        g_s = np.zeros(3)
        for i in range(4):
            g_r += dN_dxi[i] * x[i]
            g_s += dN_deta[i] * x[i]

        return g_r, g_s

    def J(self, xi: float, eta: float) -> Tuple[np.ndarray, float]:
        """Compute Jacobian matrix and determinant at (xi, eta)."""
        cache_key = (xi, eta)
        if cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]

        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)
        xl = self._local_coordinates

        dx_dxi = np.dot(dN_dxi, xl[:, 0])
        dy_dxi = np.dot(dN_dxi, xl[:, 1])
        dx_deta = np.dot(dN_deta, xl[:, 0])
        dy_deta = np.dot(dN_deta, xl[:, 1])

        J_mat = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
        detJ = dx_dxi * dy_deta - dy_dxi * dx_deta

        self._jacobian_cache[cache_key] = (J_mat, detJ)
        return J_mat, detJ

    def T(self) -> np.ndarray:
        """Compute 24x24 transformation matrix from global to local coordinates."""
        T3 = np.vstack([self._e1, self._e2, self._e3])
        T = np.zeros((24, 24))
        for i in range(8):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = T3
        return T

    def _get_dH(self, xi: float, eta: float) -> np.ndarray:
        """Get shape function derivatives dN/dx, dN/dy at (xi, eta)."""
        cache_key = (xi, eta)
        if cache_key in self._dH_cache:
            return self._dH_cache[cache_key]

        J_mat, detJ = self.J(xi, eta)
        J_inv = np.linalg.inv(J_mat)
        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)

        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

        dH = np.vstack([dN_dx, dN_dy])
        self._dH_cache[cache_key] = dH
        return dH

    def area(self) -> float:
        """Compute element area using Gauss integration."""
        area = 0.0
        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            area += w * detJ
        return area

    # =========================================================================
    # CONSTITUTIVE MATRICES
    # =========================================================================

    def Cm(self) -> np.ndarray:
        """Membrane constitutive matrix (3x3)."""
        if isinstance(self.material, OrthotropicMaterial):
            raise NotImplementedError("Orthotropic material not yet supported in MITC4")

        E = self.material.E
        nu = self.material.nu
        factor = E * self.thickness / (1 - nu**2)
        return factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    def Cb(self) -> np.ndarray:
        """Bending constitutive matrix (3x3)."""
        E = self.material.E
        nu = self.material.nu
        h = self.thickness
        factor = E * h**3 / (12 * (1 - nu**2))
        return factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    def Cs(self) -> np.ndarray:
        """Transverse shear constitutive matrix (2x2)."""
        E = self.material.E
        nu = self.material.nu
        G = E / (2 * (1 + nu))
        k = self._shear_correction_factor
        return k * G * self.thickness * np.eye(2)

    # =========================================================================
    # STRAIN-DISPLACEMENT MATRICES
    # =========================================================================

    def B_m(self, xi: float, eta: float) -> np.ndarray:
        """Membrane strain-displacement matrix (3x24)."""
        dH = self._get_dH(xi, eta)
        Bm = np.zeros((3, 24))

        for i in range(4):
            u_idx = 6 * i
            v_idx = 6 * i + 1
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            Bm[0, u_idx] = dNi_dx
            Bm[1, v_idx] = dNi_dy
            Bm[2, u_idx] = dNi_dy
            Bm[2, v_idx] = dNi_dx

        return Bm

    def B_kappa(self, xi: float, eta: float) -> np.ndarray:
        """Bending curvature-displacement matrix (3x24)."""
        dH = self._get_dH(xi, eta)
        Bk = np.zeros((3, 24))

        for i in range(4):
            thx_idx = 6 * i + 3
            thy_idx = 6 * i + 4
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            Bk[0, thy_idx] = dNi_dx
            Bk[1, thx_idx] = -dNi_dy
            Bk[2, thy_idx] = dNi_dy
            Bk[2, thx_idx] = -dNi_dx

        return Bk

    def B_gamma_MITC4(self, xi: float, eta: float) -> np.ndarray:
        """MITC4 transverse shear strain-displacement matrix (2x24)."""
        xl = self._local_coordinates

        dx34 = xl[2, 0] - xl[3, 0]
        dy34 = xl[2, 1] - xl[3, 1]
        dx21 = xl[1, 0] - xl[0, 0]
        dy21 = xl[1, 1] - xl[0, 1]
        dx32 = xl[2, 0] - xl[1, 0]
        dy32 = xl[2, 1] - xl[1, 1]
        dx41 = xl[3, 0] - xl[0, 0]
        dy41 = xl[3, 1] - xl[0, 1]

        one_over_four = 0.25
        G = np.zeros((4, 12))

        G[0, 0] = -0.5
        G[0, 1] = -dy41 * one_over_four
        G[0, 2] = dx41 * one_over_four
        G[0, 9] = 0.5
        G[0, 10] = -dy41 * one_over_four
        G[0, 11] = dx41 * one_over_four

        G[1, 0] = -0.5
        G[1, 1] = -dy21 * one_over_four
        G[1, 2] = dx21 * one_over_four
        G[1, 3] = 0.5
        G[1, 4] = -dy21 * one_over_four
        G[1, 5] = dx21 * one_over_four

        G[2, 3] = -0.5
        G[2, 4] = -dy32 * one_over_four
        G[2, 5] = dx32 * one_over_four
        G[2, 6] = 0.5
        G[2, 7] = -dy32 * one_over_four
        G[2, 8] = dx32 * one_over_four

        G[3, 6] = 0.5
        G[3, 7] = -dy34 * one_over_four
        G[3, 8] = dx34 * one_over_four
        G[3, 9] = -0.5
        G[3, 10] = -dy34 * one_over_four
        G[3, 11] = dx34 * one_over_four

        Ax = -xl[0, 0] + xl[1, 0] + xl[2, 0] - xl[3, 0]
        Bx = xl[0, 0] - xl[1, 0] + xl[2, 0] - xl[3, 0]
        Cx = -xl[0, 0] - xl[1, 0] + xl[2, 0] + xl[3, 0]

        Ay = -xl[0, 1] + xl[1, 1] + xl[2, 1] - xl[3, 1]
        By = xl[0, 1] - xl[1, 1] + xl[2, 1] - xl[3, 1]
        Cy = -xl[0, 1] - xl[1, 1] + xl[2, 1] + xl[3, 1]

        alph = np.arctan2(Ay, Ax)
        beta = np.pi / 2.0 - np.arctan2(Cx, Cy)

        Rot = np.array(
            [
                [np.sin(beta), -np.sin(alph)],
                [-np.cos(beta), np.cos(alph)],
            ]
        )

        Ms = np.zeros((2, 4))
        Ms[1, 0] = 1.0 - xi
        Ms[0, 1] = 1.0 - eta
        Ms[1, 2] = 1.0 + xi
        Ms[0, 3] = 1.0 + eta

        Bsv = Ms @ G

        _, detJ = self.J(xi, eta)
        r1_vec = np.array([Cx + xi * Bx, Cy + xi * By])
        r1 = np.linalg.norm(r1_vec)
        r2_vec = np.array([Ax + eta * Bx, Ay + eta * By])
        r2 = np.linalg.norm(r2_vec)

        for j in range(12):
            Bsv[0, j] = Bsv[0, j] * r1 / (8.0 * detJ)
            Bsv[1, j] = Bsv[1, j] * r2 / (8.0 * detJ)

        Bs_12 = Rot @ Bsv

        Bs = np.zeros((2, 24))
        for i in range(4):
            Bs[:, 6 * i + 2] = Bs_12[:, 3 * i + 0]
            Bs[:, 6 * i + 3] = Bs_12[:, 3 * i + 1]
            Bs[:, 6 * i + 4] = Bs_12[:, 3 * i + 2]

        return Bs

    def B_gamma_MITC4_plus(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """MITC4+ transverse shear matrix with bubble enrichment."""
        Bs_nodal = self.B_gamma_MITC4(xi, eta)
        Nb = self._bubble_function(xi, eta)

        Bs_bubble = np.zeros((2, 2))
        Bs_bubble[0, 1] = Nb
        Bs_bubble[1, 0] = -Nb

        return Bs_nodal, Bs_bubble

    def B_kappa_bubble(self, xi: float, eta: float) -> np.ndarray:
        """Bending curvature contribution from bubble rotation enrichment (3x2)."""
        J_mat, detJ = self.J(xi, eta)
        J_inv = np.linalg.inv(J_mat)

        dNb_dxi, dNb_deta = self._bubble_derivatives(xi, eta)

        dNb_dx = J_inv[0, 0] * dNb_dxi + J_inv[0, 1] * dNb_deta
        dNb_dy = J_inv[1, 0] * dNb_dxi + J_inv[1, 1] * dNb_deta

        Bk_bubble = np.zeros((3, 2))
        Bk_bubble[0, 1] = dNb_dx
        Bk_bubble[1, 0] = -dNb_dy
        Bk_bubble[2, 0] = -dNb_dx
        Bk_bubble[2, 1] = dNb_dy

        return Bk_bubble

    def _compute_B_drill(self, xi: float, eta: float) -> np.ndarray:
        """Compute drilling rotation B matrix (Hughes & Brezzi 1989)."""
        dH = self._get_dH(xi, eta)
        N = self._shape_functions(xi, eta)

        B_drill = np.zeros(24)

        for i in range(4):
            u_idx = 6 * i
            v_idx = 6 * i + 1
            thz_idx = 6 * i + 5

            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            B_drill[u_idx] = -0.5 * dNi_dy
            B_drill[v_idx] = 0.5 * dNi_dx
            B_drill[thz_idx] = -N[i]

        return B_drill

    # =========================================================================
    # STIFFNESS MATRIX ASSEMBLY
    # =========================================================================

    def k_m(self) -> np.ndarray:
        """Membrane stiffness matrix (24x24) using MITC4+ assumed strain interpolation."""
        Cm = self.Cm()
        Km = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            Bm = self.B_m_MITC4_plus(xi, eta)
            Km += w * detJ * Bm.T @ Cm @ Bm

        return Km

    def _k_bending_shear(self) -> np.ndarray:
        """
        Combined bending+shear stiffness with bubble condensation.

        Uses MITC4+ formulation with bubble function enrichment for the
        transverse shear strain field. The bubble DOFs are statically
        condensed out, ensuring a symmetric stiffness matrix even for
        warped (non-planar) elements.

        The static condensation formula K = Knn - Knb @ Kbb^{-1} @ Knb^T
        is mathematically guaranteed to produce a symmetric result.

        Returns
        -------
        np.ndarray
            24x24 combined bending+shear stiffness matrix
        """
        Cb = self.Cb()
        Cs = self.Cs()

        Knn_b = np.zeros((24, 24))
        Knb_b = np.zeros((24, 2))
        Kbb_b = np.zeros((2, 2))

        Knn_s = np.zeros((24, 24))
        Knb_s = np.zeros((24, 2))
        Kbb_s = np.zeros((2, 2))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)

            Bk_nodal = self.B_kappa(xi, eta)
            Bk_bubble = self.B_kappa_bubble(xi, eta)

            Knn_b += w * detJ * Bk_nodal.T @ Cb @ Bk_nodal
            Knb_b += w * detJ * Bk_nodal.T @ Cb @ Bk_bubble
            Kbb_b += w * detJ * Bk_bubble.T @ Cb @ Bk_bubble

            Bs_nodal, Bs_bubble = self.B_gamma_MITC4_plus(xi, eta)

            Knn_s += w * detJ * Bs_nodal.T @ Cs @ Bs_nodal
            Knb_s += w * detJ * Bs_nodal.T @ Cs @ Bs_bubble
            Kbb_s += w * detJ * Bs_bubble.T @ Cs @ Bs_bubble

        Knn = Knn_b + Knn_s
        Knb = Knb_b + Knb_s
        Kbb = Kbb_b + Kbb_s

        try:
            Kbb_inv = np.linalg.inv(Kbb)
            K_condensed = Knn - Knb @ Kbb_inv @ Knb.T
        except np.linalg.LinAlgError:
            K_condensed = Knn

        return K_condensed

    def k_drill(self) -> np.ndarray:
        """Drilling stiffness matrix (24x24) for stabilization."""
        # Stabilization factor (small compared to membrane stiffness)
        # Scaled with t^2 to be consistent with rotational stiffness dimensions (Force*Length)
        # alpha=0.15 works well for both Twisted Beam (t=0.32) and Hook (t=2.0)
        k_drill_stab = self.material.E * (self.thickness ** 2) * 0.15

        K_drill = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            B_drill = self._compute_B_drill(xi, eta)
            K_drill += w * detJ * k_drill_stab * np.outer(B_drill, B_drill)

        return K_drill

    @property
    def K(self) -> np.ndarray:
        """
        Global stiffness matrix (24x24).

        Uses the MITC4+ formulation with static condensation for bending
        and shear, which guarantees symmetry even for warped elements.
        """
        K_local = self.k_m() + self._k_bending_shear() + self.k_drill()

        T = self.T()
        K_global = T.T @ K_local @ T
        K_global = 0.5 * (K_global + K_global.T)

        return K_global

    @property
    def M(self) -> np.ndarray:
        """Consistent mass matrix (24x24)."""
        rho = self.material.rho
        h = self.thickness
        M = np.zeros((24, 24))

        m_trans = rho * h
        m_rot = rho * h**3 / 12.0

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            N = self._shape_functions(xi, eta)

            for i in range(4):
                for j in range(4):
                    val_t = N[i] * N[j] * m_trans * w * detJ
                    for k in range(3):
                        M[6 * i + k, 6 * j + k] += val_t

                    val_r = N[i] * N[j] * m_rot * w * detJ
                    for k in range(3, 6):
                        M[6 * i + k, 6 * j + k] += val_r

        T = self.T()
        return T.T @ M @ T

    # =========================================================================
    # GEOMETRIC NONLINEAR ANALYSIS
    # =========================================================================

    def update_configuration(self, displacements: np.ndarray) -> None:
        """Update element configuration with new nodal displacements."""
        if len(displacements) != 24:
            raise ValueError(f"Expected 24 DOFs, got {len(displacements)}")

        self._current_displacements = np.array(displacements, dtype=float)

        for i in range(4):
            u_vec = displacements[6 * i : 6 * i + 3]
            self._current_coords[i] = self._initial_coords[i] + u_vec

            theta = displacements[6 * i + 3 : 6 * i + 6]
            if np.linalg.norm(theta) > 1e-12:
                cross_theta = np.cross(theta, self._initial_directors[i])
                self._current_directors[i] = self._initial_directors[i] + cross_theta
                self._current_directors[i] /= np.linalg.norm(self._current_directors[i])
            else:
                self._current_directors[i] = self._initial_directors[i].copy()

    def reset_configuration(self) -> None:
        """Reset element to initial (undeformed) configuration."""
        self._current_coords = self._initial_coords.copy()
        self._current_displacements = np.zeros(24)
        self._current_directors = self._initial_directors.copy()

    def get_displacement_gradient(self, xi: float, eta: float) -> np.ndarray:
        """Compute displacement gradient H = du/dX at (xi, eta)."""
        dH = self._get_dH(xi, eta)

        u_nodes = np.zeros((4, 3))
        for i in range(4):
            u_nodes[i, 0] = self._current_displacements[6 * i]
            u_nodes[i, 1] = self._current_displacements[6 * i + 1]
            u_nodes[i, 2] = self._current_displacements[6 * i + 2]

        H = np.zeros((3, 3))
        for j in range(2):
            for comp in range(3):
                H[comp, j] = np.dot(u_nodes[:, comp], dH[j, :])

        return H

    def compute_green_lagrange_strain(self, xi: float, eta: float) -> np.ndarray:
        """Compute Green-Lagrange strain tensor E = 0.5*(H + H^T + H^T @ H)."""
        H = self.get_displacement_gradient(xi, eta)
        return 0.5 * (H + H.T + H.T @ H)

    def _compute_B_geometric(self, xi: float, eta: float) -> np.ndarray:
        """Compute geometric B matrix for geometric stiffness."""
        dH = self._get_dH(xi, eta)
        B_G = np.zeros((8, 24))

        for i in range(4):
            col = 6 * i
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            B_G[0, col] = dNi_dx
            B_G[1, col] = dNi_dy
            B_G[2, col + 1] = dNi_dx
            B_G[3, col + 1] = dNi_dy
            B_G[4, col + 2] = dNi_dx
            B_G[5, col + 2] = dNi_dy
            B_G[6, col + 3] = dNi_dx
            B_G[7, col + 3] = dNi_dy

        return B_G

    def compute_geometric_stiffness(
        self,
        sigma_membrane: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Compute geometric (stress) stiffness matrix."""
        sigma = np.asarray(sigma_membrane)

        S_m = (
            np.array(
                [
                    [sigma[0], sigma[2]],
                    [sigma[2], sigma[1]],
                ]
            )
            * self.thickness
        )

        S_tilde = np.zeros((8, 8))
        S_tilde[0:2, 0:2] = S_m
        S_tilde[2:4, 2:4] = S_m
        S_tilde[4:6, 4:6] = S_m
        S_tilde[6:8, 6:8] = S_m

        K_sigma = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            B_G = self._compute_B_geometric(xi, eta)
            K_sigma += w * detJ * B_G.T @ S_tilde @ B_G

        K_sigma = 0.5 * (K_sigma + K_sigma.T)

        if transform_to_global:
            T = self.T()
            K_sigma = T.T @ K_sigma @ T

        return K_sigma

    def compute_membrane_stress_from_displacement(
        self,
        u_local: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Compute membrane stress from local displacement vector."""
        E = self.material.E
        nu = self.material.nu
        factor = E / (1 - nu**2)
        C_m = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

        Bm = self.B_m(xi, eta)
        epsilon_m = Bm @ u_local
        sigma_m = C_m @ epsilon_m

        return sigma_m

    def _get_local_stiffness(self) -> np.ndarray:
        """Get local stiffness matrix (cached for efficiency in nonlinear methods)."""
        return self.k_m() + self._k_bending_shear() + self.k_drill()

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Compute tangent stiffness matrix for nonlinear analysis."""
        K0 = self._get_local_stiffness()

        if not self.nonlinear:
            if transform_to_global:
                T = self.T()
                return T.T @ K0 @ T
            return K0

        if sigma is None:
            T = self.T()
            u_local = T @ self._current_displacements
            sigma = self.compute_membrane_stress_from_displacement(u_local)

        K_sigma = self.compute_geometric_stiffness(sigma, transform_to_global=False)
        K_T = K0 + K_sigma
        K_T = 0.5 * (K_T + K_T.T)

        if transform_to_global:
            T = self.T()
            K_T = T.T @ K_T @ T

        return K_T

    def compute_internal_forces(
        self,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Compute internal force vector."""
        T = self.T()
        u_local = T @ self._current_displacements

        K_local = self._get_local_stiffness()
        f_int = K_local @ u_local

        if transform_to_global:
            f_int = T.T @ f_int

        return f_int

    def compute_residual(
        self,
        f_ext: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Compute residual force vector R = f_ext - f_int."""
        f_int = self.compute_internal_forces(transform_to_global=transform_to_global)
        return f_ext - f_int

    def compute_strain_energy(self) -> float:
        """Compute total strain energy stored in the element."""
        T = self.T()
        u_local = T @ self._current_displacements
        K_local = self._get_local_stiffness()
        return 0.5 * float(u_local @ K_local @ u_local)

    # =========================================================================
    # STRESS AND STRAIN OUTPUT
    # =========================================================================

    def compute_strains(
        self,
        u_global: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict:
        """Compute strains at a parametric point."""
        T = self.T()
        u_local = T @ u_global

        eps_membrane = self.B_m(xi, eta) @ u_local
        kappa = self.B_kappa(xi, eta) @ u_local
        gamma = self.B_gamma_MITC4(xi, eta) @ u_local

        return {
            "membrane": eps_membrane,
            "curvature": kappa,
            "shear": gamma,
        }

    def compute_stresses(
        self,
        u_global: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict:
        """Compute stresses at a parametric point."""
        strains = self.compute_strains(u_global, xi, eta)

        Cm = self.Cm()
        Cb = self.Cb()
        Cs = self.Cs()

        sigma_membrane = Cm @ strains["membrane"]
        moments = Cb @ strains["curvature"]
        shear = Cs @ strains["shear"]

        return {
            "membrane": sigma_membrane,
            "moments": moments,
            "shear": shear,
        }
        return f_ext - f_int

    def compute_strain_energy(self) -> float:
        """Compute total strain energy stored in the element."""
        T = self.T()
        u_local = T @ self._current_displacements
        K_local = self._get_local_stiffness()
        return 0.5 * float(u_local @ K_local @ u_local)

    # =========================================================================
    # STRESS AND STRAIN OUTPUT
    # =========================================================================

    def compute_strains(
        self,
        u_global: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict:
        """Compute strains at a parametric point."""
        T = self.T()
        u_local = T @ u_global

        eps_membrane = self.B_m(xi, eta) @ u_local
        kappa = self.B_kappa(xi, eta) @ u_local
        gamma = self.B_gamma_MITC4(xi, eta) @ u_local

        return {
            "membrane": eps_membrane,
            "curvature": kappa,
            "shear": gamma,
        }

    def compute_stresses(
        self,
        u_global: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict:
        """Compute stresses at a parametric point."""
        strains = self.compute_strains(u_global, xi, eta)

        Cm = self.Cm()
        Cb = self.Cb()
        Cs = self.Cs()

        sigma_membrane = Cm @ strains["membrane"]
        moments = Cb @ strains["curvature"]
        shear = Cs @ strains["shear"]

        return {
            "membrane": sigma_membrane,
            "moments": moments,
            "shear": shear,
        }
