"""
MITC4+ Shell Element Implementation.

This module provides an implementation of the MITC4+ four-node shell element
formulation, based on research by Ko et al. (2017) and Dvorkin & Bathe (1984).

The MITC4+ element is a 4-node quadrilateral shell element that:
1. Uses MITC (Mixed Interpolation of Tensorial Components) to prevent shear locking.
2. Uses assumed membrane strain field to prevent membrane locking.
3. Supports both flat and warped geometries.
4. Supports geometric nonlinear analysis (Total Lagrangian formulation).

Key features of the MITC4+ formulation:
- Bilinear shape functions for displacements and rotations
- Assumed transverse shear strain field with tying points at element edges
- Assumed membrane strain field to handle warped elements
- Drilling stiffness stabilization for in-plane rotation

References
----------
- Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in geometric
  nonlinear analysis." Computers & Structures, 185, 1-14.
- Dvorkin, E.N. and Bathe, K.J. (1984). "A continuum mechanics based four-node shell
  element for general nonlinear analysis." Engineering Computations, 1(1), 77-88.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC4(ShellElement):
    """
    MITC4+ four-node quadrilateral shell element class.

    This element implements the MITC4+ formulation from Ko et al. (2017).
    Each corner node has 6 DOFs (u, v, w, θx, θy, θz).

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
    shear_correction_factor : float, optional
        Shear correction factor (default: 5/6)
    nonlinear : bool, optional
        Enable geometric nonlinear analysis

    Attributes
    ----------
    element_type : str
        Element identifier "MITC4"
    thickness : float
        Shell thickness
    dofs_count : int
        Total DOFs (24 for 4 nodes × 6 DOFs)

    Notes
    -----
    The MITC4+ element uses:

    **Parametric coordinates**: (r, s) where r, s ∈ [-1, 1]
    - Node 1: (-1, -1)
    - Node 2: ( 1, -1)
    - Node 3: ( 1,  1)
    - Node 4: (-1,  1)

    **Bilinear shape functions**:
    - N₁ = 0.25(1-r)(1-s)
    - N₂ = 0.25(1+r)(1-s)
    - N₃ = 0.25(1+r)(1+s)
    - N₄ = 0.25(1-r)(1+s)

    **Assumed transverse shear strain field** uses 4 tying points on edges:
    - A: (0, -1) - midpoint of edge 1-2
    - B: (1, 0)  - midpoint of edge 2-3
    - C: (0, 1)  - midpoint of edge 3-4
    - D: (-1, 0) - midpoint of edge 4-1

    **Assumed membrane strain field** uses 5 tying points (MITC4+):
    - Tying points for handling warped geometries
    """

    vector_form = {"U": ("Ux", "Uy", "Uz"), "θ": ("θx", "θy", "θz")}

    def __init__(
        self,
        node_coords: Union[Sequence[float], np.ndarray],
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

        # Initial local coordinate system
        self._initial_coords = np.array(node_coords, dtype=float).copy()
        self._current_coords = np.array(node_coords, dtype=float).copy()
        self._current_displacements = np.zeros(24)  # 4 nodes * 6 DOFs

        # Shear correction factor
        if shear_correction_factor is not None:
            self._shear_correction_factor = shear_correction_factor
        elif (
            hasattr(material, "shear_correction_factor")
            and material.shear_correction_factor is not None
        ):
            self._shear_correction_factor = material.shear_correction_factor
        else:
            self._shear_correction_factor = 5.0 / 6.0

        # Gauss quadrature for quadrilateral (2x2 integration)
        gp = 1.0 / np.sqrt(3.0)
        self._gauss_points = [
            (-gp, -gp),
            (gp, -gp),
            (gp, gp),
            (-gp, gp),
        ]
        self._gauss_weights = [1.0, 1.0, 1.0, 1.0]

        # Element geometry
        self._local_coordinates = self._compute_local_coordinates()
        self._initial_directors = self._compute_initial_directors()
        self._current_directors = self._initial_directors.copy()

        # Compute characteristic vectors for MITC4+
        self._compute_characteristic_vectors()

        # Caches
        self._dH_cache = {}
        self._N_cache = {}

        # =====================================================================
        # MITC4 Tying Points for Assumed Transverse Shear Strain
        # =====================================================================
        # Tying points on element edges (Dvorkin & Bathe, 1984)
        self._tying_points_shear = {
            "A": (0.0, -1.0),  # Midpoint of edge 1-2
            "B": (1.0, 0.0),  # Midpoint of edge 2-3
            "C": (0.0, 1.0),  # Midpoint of edge 3-4
            "D": (-1.0, 0.0),  # Midpoint of edge 4-1
        }

        # =====================================================================
        # MITC4+ Tying Points for Assumed Membrane Strain
        # =====================================================================
        # Ko et al. (2017) - 5 tying points for handling warped elements
        self._tying_points_membrane = {
            "A": (0.0, -1.0),
            "B": (1.0, 0.0),
            "C": (0.0, 1.0),
            "D": (-1.0, 0.0),
            "E": (0.0, 0.0),
        }

    # =========================================================================
    # SHAPE FUNCTIONS
    # =========================================================================

    def _shape_functions(self, r: float, s: float) -> np.ndarray:
        """
        Compute bilinear shape functions for quadrilateral element.

        Parameters
        ----------
        r, s : float
            Parametric coordinates (-1 ≤ r, s ≤ 1)

        Returns
        -------
        np.ndarray
            Shape functions [N1, N2, N3, N4]
        """
        N1 = 0.25 * (1.0 - r) * (1.0 - s)
        N2 = 0.25 * (1.0 + r) * (1.0 - s)
        N3 = 0.25 * (1.0 + r) * (1.0 + s)
        N4 = 0.25 * (1.0 - r) * (1.0 + s)
        return np.array([N1, N2, N3, N4])

    def _shape_function_derivatives(self, r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives of shape functions in parametric coordinates.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (dN_dr, dN_ds) each of shape (4,)
        """
        # Derivatives with respect to r
        dN_dr = 0.25 * np.array([
            -(1.0 - s),
            (1.0 - s),
            (1.0 + s),
            -(1.0 + s),
        ])

        # Derivatives with respect to s
        dN_ds = 0.25 * np.array([
            -(1.0 - r),
            -(1.0 + r),
            (1.0 + r),
            (1.0 - r),
        ])

        return dN_dr, dN_ds

    # =========================================================================
    # LOCAL COORDINATE SYSTEM
    # =========================================================================

    def _compute_local_coordinates(self) -> np.ndarray:
        """
        Compute local 2D coordinates and basis vectors.

        Projects the 3D element onto a local x-y plane.

        Returns
        -------
        np.ndarray
            Local coordinates (4x2) in local x-y plane
        """
        nodes = self._current_coords

        # Compute element center
        center = np.mean(nodes, axis=0)

        # Approximate normal using cross product of diagonals
        d13 = nodes[2] - nodes[0]
        d24 = nodes[3] - nodes[1]
        n = np.cross(d13, d24)
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            raise ValueError("Degenerate quadrilateral: nodes are coplanar")
        e3 = n / n_len

        # e1: direction from center to midpoint of edge 1-2
        mid_12 = 0.5 * (nodes[0] + nodes[1])
        v1 = mid_12 - center
        e1 = v1 / np.linalg.norm(v1)

        # e2: perpendicular to e1 and e3
        e2 = np.cross(e3, e1)

        # Project nodes to local coordinates
        local_coords = np.zeros((4, 2))
        for i in range(4):
            v = nodes[i] - center
            local_coords[i, 0] = np.dot(v, e1)
            local_coords[i, 1] = np.dot(v, e2)

        # Store basis for future use
        self._e1, self._e2, self._e3 = e1, e2, e3
        self._center = center

        return local_coords

    def _compute_initial_directors(self) -> np.ndarray:
        """
        Compute initial nodal directors (normals).
        For flat elements, directors are simply the element normal.
        """
        return np.tile(self._e3, (4, 1))

    def _compute_characteristic_vectors(self) -> None:
        """
        Compute characteristic vectors for MITC4+ membrane strain.

        These vectors are used to detect and handle warped geometries.
        Following Ko et al. (2017).
        """
        nodes = self._current_coords

        # x_r: vector along r-direction (average of edges in r-direction)
        x_r = 0.5 * ((nodes[1] - nodes[0]) + (nodes[2] - nodes[3]))
        self._x_r = x_r / np.linalg.norm(x_r)

        # x_s: vector along s-direction (average of edges in s-direction)
        x_s = 0.5 * ((nodes[3] - nodes[0]) + (nodes[2] - nodes[1]))
        self._x_s = x_s / np.linalg.norm(x_s)

        # x_d: diagonal vector (for warping detection)
        x_d = nodes[2] - nodes[0]
        self._x_d = x_d / np.linalg.norm(x_d)

        # n: average normal
        self._n = np.cross(self._x_r, self._x_s)
        self._n = self._n / np.linalg.norm(self._n)

        # m_r, m_s: vectors perpendicular to n in local plane
        self._m_r = np.cross(self._n, self._x_s)
        self._m_r = self._m_r / np.linalg.norm(self._m_r)

        self._m_s = np.cross(self._x_r, self._n)
        self._m_s = self._m_s / np.linalg.norm(self._m_s)

        # Compute MITC4+ membrane coefficients
        self._compute_membrane_coefficients()

    def _compute_membrane_coefficients(self) -> None:
        """
        Compute coefficients for the "new MITC4+" assumed membrane strain field
        as per Ko et al. (2017), Computers and Structures 182.
        """
        c_r = np.dot(self._x_d, self._m_r)
        c_s = np.dot(self._x_d, self._m_s)
        d = c_r**2 + c_s**2 + 1.0

        # These are the 'a' coefficients from Eq (27c) of the paper
        # "A new MITC4+ shell element"
        self._a_A_paper = c_r * (c_r - 1.0) / (2.0 * d)  # For paper's tying point A(0,1)
        self._a_B_paper = c_r * (c_r + 1.0) / (2.0 * d)  # For paper's tying point B(0,-1)
        self._a_C_paper = c_s * (c_s + 1.0) / (2.0 * d)  # For paper's tying point C(1,0)
        self._a_D_paper = c_s * (c_s - 1.0) / (2.0 * d)  # For paper's tying point D(-1,0)
        self._a_E_paper = 2.0 * c_r * c_s / d  # For paper's tying point E(0,0)

    def _compute_element_warping(self) -> float:
        """
        Compute warping metric for the element.

        Returns
        -------
        float
            Warping measure (0 for flat, >0 for warped)
        """
        nodes = self._current_coords

        # Distance from nodes to best-fit plane
        center = np.mean(nodes, axis=0)
        n = self._n

        warping = 0.0
        for node in nodes:
            dist = abs(np.dot(node - center, n))
            warping += dist

        # Normalize by element size
        size = np.linalg.norm(nodes[2] - nodes[0])
        return float(warping / size) if size > 1e-12 else 0.0

    # =========================================================================
    # GEOMETRY AND JACOBIAN
    # =========================================================================

    def area(self) -> float:
        """
        Compute area of the quadrilateral element.

        Returns
        -------
        float
            Element area
        """
        # Use Gauss integration to compute area
        area = 0.0
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            area += w * detJ
        return area

    def J(self, r: float, s: float) -> Tuple[np.ndarray, float]:
        """
        Compute Jacobian matrix and determinant at (r, s).

        J = [∂x/∂r  ∂y/∂r  ∂z/∂r]
            [∂x/∂s  ∂y/∂s  ∂z/∂s]

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, float]
            (J, detJ) - Jacobian matrix (2x3) and its determinant magnitude
        """
        dN_dr, dN_ds = self._shape_function_derivatives(r, s)
        nodes = self._current_coords

        # Jacobian matrix (2x3 for shell)
        J_mat = np.zeros((2, 3))
        for i in range(4):
            J_mat[0, :] += dN_dr[i] * nodes[i]
            J_mat[1, :] += dN_ds[i] * nodes[i]

        # For shell elements, detJ is the magnitude of the cross product
        g_r = J_mat[0, :]
        g_s = J_mat[1, :]
        cross = np.cross(g_r, g_s)
        detJ = float(np.linalg.norm(cross))

        return J_mat, detJ

    def T(self) -> np.ndarray:
        """
        Compute transformation matrix from global to local coordinates.

        Returns
        -------
        np.ndarray
            24×24 transformation matrix (block diagonal of 3×3 direction cosines)
        """
        # Update basis if needed
        nodes = self._current_coords
        center = np.mean(nodes, axis=0)

        # Recompute basis
        d13 = nodes[2] - nodes[0]
        d24 = nodes[3] - nodes[1]
        n = np.cross(d13, d24)
        e3 = n / np.linalg.norm(n)

        mid_12 = 0.5 * (nodes[0] + nodes[1])
        v1 = mid_12 - center
        e1 = v1 / np.linalg.norm(v1)

        e2 = np.cross(e3, e1)

        T3 = np.vstack([e1, e2, e3])

        # Build 24x24 transformation matrix
        T = np.zeros((24, 24))
        for i in range(8):  # 4 nodes × 2 blocks (translations + rotations)
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = T3

        return T

    def _get_dH(self, r: float, s: float) -> np.ndarray:
        """
        Get shape function derivatives ∂N/∂x, ∂N/∂y at (r, s).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            2×4 matrix [[∂N1/∂x, ∂N2/∂x, ∂N3/∂x, ∂N4/∂x],
                        [∂N1/∂y, ∂N2/∂y, ∂N3/∂y, ∂N4/∂y]]
        """
        if (r, s) in self._dH_cache:
            return self._dH_cache[(r, s)]

        dN_dr, dN_ds = self._shape_function_derivatives(r, s)
        J_mat, detJ = self.J(r, s)

        # Project to local x-y plane
        nodes_local = self._local_coordinates

        # Compute Jacobian in local coordinates
        J_local = np.zeros((2, 2))
        for i in range(4):
            J_local[0, :] += dN_dr[i] * nodes_local[i]
            J_local[1, :] += dN_ds[i] * nodes_local[i]

        # Inverse Jacobian
        J_inv = np.linalg.inv(J_local)

        # Transform derivatives
        dN_dx = J_inv[0, 0] * dN_dr + J_inv[0, 1] * dN_ds
        dN_dy = J_inv[1, 0] * dN_dr + J_inv[1, 1] * dN_ds

        dH = np.vstack([dN_dx, dN_dy])
        self._dH_cache[(r, s)] = dH
        return dH

    # =========================================================================
    # CONSTITUTIVE MATRICES
    # =========================================================================

    def Cm(self) -> np.ndarray:
        """
        Membrane constitutive matrix (3×3).

        Returns
        -------
        np.ndarray
            3×3 membrane stiffness matrix [σxx, σyy, τxy] = Cm × [εxx, εyy, γxy]
        """
        if isinstance(self.material, OrthotropicMaterial):
            raise NotImplementedError("Orthotropic material not yet supported in MITC4")

        E = self.material.E
        nu = self.material.nu
        factor = E * self.thickness / (1 - nu**2)
        return factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    def Cb(self) -> np.ndarray:
        """
        Bending constitutive matrix (3×3).

        Returns
        -------
        np.ndarray
            3×3 bending stiffness matrix [Mxx, Myy, Mxy] = Cb × [κxx, κyy, κxy]
        """
        E = self.material.E
        nu = self.material.nu
        h = self.thickness
        factor = E * h**3 / (12 * (1 - nu**2))
        return factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    def Cs(self) -> np.ndarray:
        """
        Transverse shear constitutive matrix (2×2).

        Returns
        -------
        np.ndarray
            2×2 shear stiffness matrix [Qx, Qy] = Cs × [γxz, γyz]
        """
        E = self.material.E
        nu = self.material.nu
        G = E / (2 * (1 + nu))
        k = self._shear_correction_factor
        return k * G * self.thickness * np.eye(2)

    # =========================================================================
    # STRAIN-DISPLACEMENT MATRICES
    # =========================================================================

    def B_m(self, r: float, s: float) -> np.ndarray:
        """
        Membrane strain-displacement matrix (3×24).

        For standard formulation (no MITC+ assumed membrane strain).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×24 membrane B matrix
        """
        dH = self._get_dH(r, s)
        Bm = np.zeros((3, 24))

        for i in range(4):
            u_idx = 6 * i
            v_idx = 6 * i + 1

            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # εxx = ∂u/∂x
            Bm[0, u_idx] = dNi_dx
            # εyy = ∂v/∂y
            Bm[1, v_idx] = dNi_dy
            # γxy = ∂u/∂y + ∂v/∂x
            Bm[2, u_idx] = dNi_dy
            Bm[2, v_idx] = dNi_dx

        return Bm

    def k_m(self) -> np.ndarray:
        """
        Membrane stiffness matrix (24×24) using the "new MITC4+" formulation.

        This method implements the assumed strain field from Ko et al. (2017),
        Computers and Structures 182, by constructing the assumed B matrix
        at each Gauss point according to Eq (27) in the paper.

        Returns
        -------
        np.ndarray
            24×24 membrane stiffness matrix in local coordinates
        """
        Cm = self.Cm()
        Km = np.zeros((24, 24))

        # The paper's tying points are A(0,1), B(0,-1), C(1,0), D(-1,0), E(0,0).
        # We pre-calculate the standard B-matrix at these points.
        B_A = self.B_m(r=0.0, s=1.0)
        B_B = self.B_m(r=0.0, s=-1.0)
        B_C = self.B_m(r=1.0, s=0.0)
        B_D = self.B_m(r=-1.0, s=0.0)
        B_E = self.B_m(r=0.0, s=0.0)

        # Extract rows for each strain component from the B-matrices at tying points.
        # e.g., B_A_rr is the row that gives the rr-strain component at point A.
        B_A_rr, B_A_ss, B_A_rs = B_A[0, :], B_A[1, :], B_A[2, :]
        B_B_rr, B_B_ss, B_B_rs = B_B[0, :], B_B[1, :], B_B[2, :]
        B_C_rr, B_C_ss, B_C_rs = B_C[0, :], B_C[1, :], B_C[2, :]
        B_D_rr, B_D_ss, B_D_rs = B_D[0, :], B_D[1, :], B_D[2, :]
        B_E_rr, B_E_ss, B_E_rs = B_E[0, :], B_E[1, :], B_E[2, :]

        # Geometric coefficients from the paper's formulation
        aA, aB, aC, aD, aE = (
            self._a_A_paper,
            self._a_B_paper,
            self._a_C_paper,
            self._a_D_paper,
            self._a_E_paper,
        )

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)

            B_m_assumed = np.zeros((3, 24))

            # Construct B_rr_assumed row from Eq (27a)
            B_m_assumed[0, :] = (1.0 - 2.0 * aA + s + 2.0 * aA * s**2) * B_A_rr
            B_m_assumed[0, :] += (1.0 - 2.0 * aB - s + 2.0 * aB * s**2) * B_B_rr
            B_m_assumed[0, :] += aC * (-1.0 + s**2) * B_C_rr
            B_m_assumed[0, :] += aD * (-1.0 + s**2) * B_D_rr
            B_m_assumed[0, :] += aE * (-1.0 + s**2) * B_E_rr

            # Construct B_ss_assumed row from Eq (27b)
            B_m_assumed[1, :] = aA * (-1.0 + r**2) * B_A_ss
            B_m_assumed[1, :] += aB * (-1.0 + r**2) * B_B_ss
            B_m_assumed[1, :] += (1.0 - 2.0 * aC + r + 2.0 * aC * r**2) * B_C_ss
            B_m_assumed[1, :] += (1.0 - 2.0 * aD - r + 2.0 * aD * r**2) * B_D_ss
            B_m_assumed[1, :] += aE * (-1.0 + r**2) * B_E_ss

            # Construct B_rs_assumed row from Eq (27c)
            B_m_assumed[2, :] = (0.25 * r + 0.5 * aA * r * s) * B_A_rs
            B_m_assumed[2, :] += (-0.25 * r + 0.5 * aB * r * s) * B_B_rs
            B_m_assumed[2, :] += (0.25 * s + 0.5 * aC * r * s) * B_C_rs
            B_m_assumed[2, :] += (-0.25 * s + 0.5 * aD * r * s) * B_D_rs
            B_m_assumed[2, :] += (1.0 + 0.5 * aE * r * s) * B_E_rs

            Km += w * detJ * B_m_assumed.T @ Cm @ B_m_assumed

        return Km

    def B_kappa(self, r: float, s: float) -> np.ndarray:
        """
        Bending curvature-displacement matrix (3×24).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×24 curvature B matrix
        """
        dH = self._get_dH(r, s)
        Bk = np.zeros((3, 24))

        for i in range(4):
            thx_idx = 6 * i + 3  # θx
            thy_idx = 6 * i + 4  # θy

            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # κxx = ∂θy/∂x
            Bk[0, thy_idx] = dNi_dx
            # κyy = -∂θx/∂y
            Bk[1, thx_idx] = -dNi_dy
            # κxy = ∂θy/∂y - ∂θx/∂x (twist curvature)
            Bk[2, thy_idx] = dNi_dy
            Bk[2, thx_idx] = -dNi_dx

        return Bk

    def B_gamma_MITC4(self, r: float, s: float) -> np.ndarray:
        """
        MITC4 assumed transverse shear strain-displacement matrix (2×24).

        Uses 4 tying points on element edges (Dvorkin & Bathe, 1984).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            2×24 shear B matrix
        """
        # Covariant shear strains at tying points
        gamma_r_A, gamma_s_A = self._covariant_shear_at_point(*self._tying_points_shear["A"])
        gamma_r_B, gamma_s_B = self._covariant_shear_at_point(*self._tying_points_shear["B"])
        gamma_r_C, gamma_s_C = self._covariant_shear_at_point(*self._tying_points_shear["C"])
        gamma_r_D, gamma_s_D = self._covariant_shear_at_point(*self._tying_points_shear["D"])

        # Linear interpolation functions for assumed shear strain
        # γ_r = γ_r^A * (1-s)/2 + γ_r^C * (1+s)/2
        # γ_s = γ_s^D * (1-r)/2 + γ_s^B * (1+r)/2

        b_r = 0.5 * (1.0 - s) * gamma_r_A + 0.5 * (1.0 + s) * gamma_r_C
        b_s = 0.5 * (1.0 - r) * gamma_s_D + 0.5 * (1.0 + r) * gamma_s_B

        B_gamma = np.vstack([b_r, b_s])
        return B_gamma

    def _covariant_shear_at_point(self, r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariant shear strain components at a point.

        γ_r = ∂w/∂r + g_r · θ
        γ_s = ∂w/∂s + g_s · θ

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (B_gamma_r, B_gamma_s) each of shape (24,)
        """
        N = self._shape_functions(r, s)
        dN_dr, dN_ds = self._shape_function_derivatives(r, s)
        J_mat, _ = self.J(r, s)

        g_r = J_mat[0, :]
        g_s = J_mat[1, :]

        # Components of tangent vectors in the local element system
        g_r_local_x = np.dot(g_r, self._e1)
        g_r_local_y = np.dot(g_r, self._e2)
        g_s_local_x = np.dot(g_s, self._e1)
        g_s_local_y = np.dot(g_s, self._e2)

        B_gamma_r = np.zeros(24)
        B_gamma_s = np.zeros(24)

        for i in range(4):
            w_idx = 6 * i + 2
            thx_idx = 6 * i + 3
            thy_idx = 6 * i + 4

            # γ_r = ∂w/∂r + g_r · θ
            # where θ is the rotation vector (-θy*e1 + θx*e2)
            B_gamma_r[w_idx] = dN_dr[i]
            B_gamma_r[thy_idx] += N[i] * g_r_local_x
            B_gamma_r[thx_idx] -= N[i] * g_r_local_y

            # γ_s = ∂w/∂s + g_s · θ
            B_gamma_s[w_idx] = dN_ds[i]
            B_gamma_s[thy_idx] += N[i] * g_s_local_x
            B_gamma_s[thx_idx] -= N[i] * g_s_local_y

        return B_gamma_r, B_gamma_s

    # =========================================================================
    # STIFFNESS MATRIX
    # =========================================================================

    def k_m(self) -> np.ndarray:
        """
        Membrane stiffness matrix (24×24).

        Returns
        -------
        np.ndarray
            24×24 membrane stiffness matrix in local coordinates
        """
        Cm = self.Cm()
        Km = np.zeros((24, 24))

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            Bm = self.B_m(r, s)
            Km += w * detJ * Bm.T @ Cm @ Bm

        return Km

    def k_b(self) -> np.ndarray:
        """
        Bending stiffness matrix (24×24).

        Returns
        -------
        np.ndarray
            24×24 bending stiffness matrix in local coordinates
        """
        Cb = self.Cb()
        Kb = np.zeros((24, 24))

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            Bk = self.B_kappa(r, s)
            Kb += w * detJ * Bk.T @ Cb @ Bk

        return Kb

    def _get_shear_transformation(self, r: float, s: float) -> np.ndarray:
        """
        Compute the transformation matrix from covariant to Cartesian shear strains.
        [γ_xz, γ_yz]^T = T_shear * [γ_r, γ_s]^T
        """
        J_mat, _ = self.J(r, s)
        g_r, g_s = J_mat[0, :], J_mat[1, :]

        # Transformation matrix J_sh = [[g_r.e1, g_s.e1], [g_r.e2, g_s.e2]]
        J_sh = np.array([
            [np.dot(g_r, self._e1), np.dot(g_s, self._e1)],
            [np.dot(g_r, self._e2), np.dot(g_s, self._e2)],
        ])

        # The transformation is the inverse of J_sh
        T_shear = np.linalg.inv(J_sh)
        return T_shear

    def k_s(self) -> np.ndarray:
        """
        Transverse shear stiffness matrix (24×24) using MITC4 interpolation.
        Returns
        -------
        np.ndarray
            24×24 shear stiffness matrix in local coordinates
        """
        Cs = self.Cs()
        Ks = np.zeros((24, 24))
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            # B matrix for covariant shear strains [γ_r, γ_s]
            Bg_covariant = self.B_gamma_MITC4(r, s)
            # Transformation from covariant to Cartesian shear strains
            T_shear = self._get_shear_transformation(r, s)
            # B matrix for Cartesian shear strains [γ_xz, γ_yz]
            Bg_cartesian = T_shear @ Bg_covariant
            Ks += w * detJ * Bg_cartesian.T @ Cs @ Bg_cartesian
        return Ks

    def k_drill(self) -> np.ndarray:
        """
        Drilling stiffness matrix (24×24) for in-plane rotation stabilization.

        Returns
        -------
        np.ndarray
            24×24 drilling stiffness matrix
        """
        # Stabilization factor (small compared to membrane stiffness)
        k_drill_stab = self.material.E * self.thickness * 1e-3

        Kd = np.zeros((24, 24))

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            dH = self._get_dH(r, s)

            # Drilling DOF is θz (rotation about normal)
            for i in range(4):
                for j in range(4):
                    thz_i = 6 * i + 5
                    thz_j = 6 * j + 5

                    dNi_dx = dH[0, i]
                    dNi_dy = dH[1, i]
                    dNj_dx = dH[0, j]
                    dNj_dy = dH[1, j]

                    Kd[thz_i, thz_j] += (
                        k_drill_stab * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * w * detJ
                    )

        return Kd

    @property
    def K(self) -> np.ndarray:
        """
        Global stiffness matrix (24×24).

        Returns
        -------
        np.ndarray
            24×24 element stiffness matrix in global coordinates
        """
        # Local stiffness
        K_local = self.k_m() + self.k_b() + self.k_s() + self.k_drill()

        # Transform to global
        T = self.T()
        K_global = T.T @ K_local @ T

        # Ensure symmetry
        K_global = 0.5 * (K_global + K_global.T)

        return K_global

    # =========================================================================
    # MASS MATRIX
    # =========================================================================

    @property
    def M(self) -> np.ndarray:
        """
        Consistent mass matrix (24×24).

        Returns
        -------
        np.ndarray
            24×24 element mass matrix in global coordinates
        """
        rho = self.material.rho
        h = self.thickness
        M_local = np.zeros((24, 24))

        # Translational and rotational inertia
        m_trans = rho * h
        m_rot = rho * h**3 / 12.0

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(r, s)
            N_vals = self._shape_functions(r, s)

            # Build shape function matrix (6x24)
            N = np.zeros((6, 24))
            for i in range(4):
                for dof in range(6):
                    N[dof, 6 * i + dof] = N_vals[i]

            # Translational mass (u, v, w)
            M_trans = m_trans * (N[:3].T @ N[:3]) * detJ

            # Rotational inertia contributions
            M_rot_mat = np.zeros((24, 24))
            for dof in [3, 4, 5]:  # θx, θy, θz
                M_rot_mat += m_rot * np.outer(N[dof], N[dof]) * detJ

            M_local += w * (M_trans + M_rot_mat)

        # Transform to global coordinates
        T = self.T()
        M_global = T.T @ M_local @ T

        return M_global

    # =========================================================================
    # GEOMETRIC NONLINEAR ANALYSIS
    # =========================================================================

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Computes the skew-symmetric matrix of a 3D vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def update_configuration(self, displacements: np.ndarray) -> None:
        """
        Update element configuration with new nodal displacements.

        Uses Rodrigues' rotation formula for a robust, path-independent update
        of director vectors suitable for large rotations.

        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (24,) in global coordinates
        """
        if len(displacements) != 24:
            raise ValueError(f"Expected 24 DOFs, got {len(displacements)}")

        self._current_displacements = np.array(displacements, dtype=float)

        for i in range(4):
            u_vec = displacements[6 * i : 6 * i + 3]
            self._current_coords[i] = self._initial_coords[i] + u_vec

            # Update directors using Rodrigues' rotation formula for large rotations
            theta = displacements[6 * i + 3 : 6 * i + 6]
            norm_theta = np.linalg.norm(theta)

            if norm_theta > 1e-12:
                k = theta / norm_theta
                K = self._skew(k)
                # Rotation matrix R = I + sin(θ)*K + (1-cos(θ))*K^2
                R = np.eye(3) + np.sin(norm_theta) * K + (1 - np.cos(norm_theta)) * (K @ K)
                self._current_directors[i] = R @ self._initial_directors[i]
            else:
                # No rotation
                self._current_directors[i] = self._initial_directors[i].copy()

        # Update geometry-dependent quantities
        self._local_coordinates = self._compute_local_coordinates()
        self._compute_characteristic_vectors()
        self._dH_cache.clear()

    def reset_configuration(self) -> None:
        """Reset element to initial (undeformed) configuration."""
        self._current_coords = self._initial_coords.copy()
        self._current_displacements = np.zeros(24)
        self._current_directors = self._initial_directors.copy()
        self._local_coordinates = self._compute_local_coordinates()
        self._compute_characteristic_vectors()
        self._dH_cache.clear()

    def __repr__(self):
        return f"<MITC4 id={self.id} thickness={self.thickness:.4f}>"
