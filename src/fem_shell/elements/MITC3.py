"""
MITC3+ Shell Element Implementation.

This module provides an implementation of the MITC3+ triangular shell element
formulation, based on the research by Lee, Lee, and Bathe (2014).

The MITC3+ element is a 3-node triangular shell element that:
1. Uses MITC (Mixed Interpolation of Tensorial Components) to prevent shear locking.
2. Uses a cubic bubble function to enrich the rotation field for improved bending behavior.
3. Uses an assumed transverse shear strain field with internal tying points.
4. Supports geometric nonlinear analysis (Total Lagrangian formulation).

Key features of the MITC3+ formulation:
- Bubble function enrichment: f₄ = 27rs(1-r-s) enriches the rotation interpolation
- Internal tying points for transverse shear: 6 points (A,B,C,D,E,F) inside the element
- Assumed transverse shear strain field prevents shear locking even for thin shells
- Membrane strains are computed directly (constant strain triangle)

References
----------
- Lee, Y., Lee, P.S., and Bathe, K.J. (2014). "The MITC3+ shell element and its performance."
  Computers & Structures, 138, 12-23.
- Lee, Y., Lee, P.S., and Bathe, K.J. (2015). "The MITC3+ shell element in geometric nonlinear analysis."
  Computers & Structures, 146, 91-104.
"""

from typing import List, Optional, Tuple, Dict

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial, IsotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC3(ShellElement):
    """
    MITC3+ triangular shell element class.

    This element implements the MITC3+ formulation from Lee, Lee & Bathe (2014).
    Each corner node has 6 DOFs (u, v, w, θx, θy, θz). The element internally uses
    a bubble node for rotation enrichment, which is statically condensed out.

    Parameters
    ----------
    node_coords : np.ndarray
        Array of nodal coordinates in global system [3x3]
    node_ids : Tuple[int, int, int]
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
        Element identifier "MITC3"
    thickness : float
        Shell thickness
    dofs_count : int
        Total DOFs (18 for 3 nodes × 6 DOFs)

    Notes
    -----
    The MITC3+ element uses:

    **Parametric coordinates**: (r, s) where
    - Node 1: (0, 0)
    - Node 2: (1, 0)
    - Node 3: (0, 1)

    **Linear shape functions**:
    - h₁ = 1 - r - s
    - h₂ = r
    - h₃ = s

    **Bubble function for rotation enrichment**:
    - f₄ = 27rs(1-r-s)

    **Enriched shape functions** (for rotations only):
    - f₁ = h₁ - f₄/3
    - f₂ = h₂ - f₄/3
    - f₃ = h₃ - f₄/3

    **Assumed transverse shear strain field** uses 6 internal tying points:
    - A: (1/6, 2/3) - on line from center to node 3
    - B: (2/3, 1/6) - on line from center to node 2
    - C: (1/6, 1/6) - on line from center to node 1
    - D: (1/3+d, 1/3-2d) - near center, direction to edge 1-2 midpoint
    - E: (1/3-2d, 1/3+d) - near center, direction to edge 1-3 midpoint
    - F: (1/3+d, 1/3+d) - near center, direction to edge 2-3 midpoint

    where d = 10⁻⁴ is a small parameter to reduce in-plane twisting stiffness.
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int],
        material: Material,
        thickness: float,
        shear_correction_factor: Optional[float] = None,
        nonlinear: bool = False,
    ):
        super().__init__(
            "MITC3",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=6,
            thickness=thickness,
        )

        self.element_type = "MITC3"
        self.nonlinear = nonlinear

        # Initial local coordinate system
        self._initial_coords = np.array(node_coords, dtype=float).copy()
        self._current_coords = np.array(node_coords, dtype=float).copy()
        self._current_displacements = np.zeros(18)  # 3 nodes * 6 DOFs

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

        # Gauss quadrature for triangle (3 points, degree 2)
        # Using Hammer points in (r, s) coordinates
        # L1 = 1-r-s, L2 = r, L3 = s
        self._gauss_points = [
            (1.0 / 6.0, 1.0 / 6.0),  # L1=2/3, L2=1/6, L3=1/6
            (2.0 / 3.0, 1.0 / 6.0),  # L1=1/6, L2=2/3, L3=1/6
            (1.0 / 6.0, 2.0 / 3.0),  # L1=1/6, L2=1/6, L3=2/3
        ]
        # Weights: each 1/3 of total area
        self._gauss_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        # Element geometry
        self._local_coordinates = self._compute_local_coordinates()
        self._initial_directors = self._compute_initial_directors()
        self._current_directors = self._initial_directors.copy()

        # Caches
        self._dH_cache = {}
        self._N_cache = {}
        self._covariant_cache = {}

        # =====================================================================
        # MITC3+ Internal Tying Points for Assumed Transverse Shear Strain
        # =====================================================================
        # According to Lee, Lee & Bathe (2014) "The MITC3+ shell element"

        # Parameter for linear part tying points (small value to reduce in-plane twisting)
        self._d_param = 1.0e-4

        # Tying points for constant part of assumed shear strain (A, B, C)
        # These are located on internal lines from centroid to vertices
        self._tying_points_constant = {
            "A": (1.0 / 6.0, 2.0 / 3.0),  # On line from centroid to node 3
            "B": (2.0 / 3.0, 1.0 / 6.0),  # On line from centroid to node 2
            "C": (1.0 / 6.0, 1.0 / 6.0),  # On line from centroid to node 1
        }

        # Tying points for linear part (D, E, F) - near centroid
        d = self._d_param
        self._tying_points_linear = {
            "D": (1.0 / 3.0 + d, 1.0 / 3.0 - 2.0 * d),  # Direction to edge 1-2 midpoint
            "E": (1.0 / 3.0 - 2.0 * d, 1.0 / 3.0 + d),  # Direction to edge 1-3 midpoint
            "F": (1.0 / 3.0 + d, 1.0 / 3.0 + d),  # Direction to edge 2-3 midpoint
        }

    # =========================================================================
    # SHAPE FUNCTIONS AND BUBBLE ENRICHMENT
    # =========================================================================

    def _shape_functions(self, r: float, s: float) -> np.ndarray:
        """
        Compute linear shape functions for triangular element.

        Parameters
        ----------
        r, s : float
            Parametric coordinates (0 ≤ r+s ≤ 1)

        Returns
        -------
        np.ndarray
            Shape functions [h1, h2, h3]
        """
        h1 = 1.0 - r - s
        h2 = r
        h3 = s
        return np.array([h1, h2, h3])

    def _bubble_function(self, r: float, s: float) -> float:
        """
        Compute cubic bubble function for rotation enrichment.

        The bubble function is f₄ = 27rs(1-r-s), which:
        - Equals zero at all corner nodes
        - Has maximum value at centroid (r=s=1/3)
        - Provides internal rotation enrichment

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        float
            Bubble function value
        """
        return 27.0 * r * s * (1.0 - r - s)

    def _bubble_derivatives(self, r: float, s: float) -> Tuple[float, float]:
        """
        Compute derivatives of bubble function.

        df₄/dr = 27s(1 - 2r - s)
        df₄/ds = 27r(1 - r - 2s)

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        Tuple[float, float]
            (df4_dr, df4_ds)
        """
        df4_dr = 27.0 * s * (1.0 - 2.0 * r - s)
        df4_ds = 27.0 * r * (1.0 - r - 2.0 * s)
        return df4_dr, df4_ds

    def _enriched_shape_functions(self, r: float, s: float) -> np.ndarray:
        """
        Compute enriched shape functions for rotation interpolation.

        f₁ = h₁ - f₄/3 = (1-r-s) - 9rs(1-r-s)
        f₂ = h₂ - f₄/3 = r - 9rs(1-r-s)
        f₃ = h₃ - f₄/3 = s - 9rs(1-r-s)
        f₄ = 27rs(1-r-s)

        Note: f₁ + f₂ + f₃ + f₄ = h₁ + h₂ + h₃ = 1 (partition of unity)

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            Enriched shape functions [f1, f2, f3, f4]
        """
        h = self._shape_functions(r, s)
        f4 = self._bubble_function(r, s)
        f1 = h[0] - f4 / 3.0
        f2 = h[1] - f4 / 3.0
        f3 = h[2] - f4 / 3.0
        return np.array([f1, f2, f3, f4])

    def _enriched_shape_derivatives(self, r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives of enriched shape functions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (df_dr, df_ds) each of shape (4,) for [f1, f2, f3, f4]
        """
        # Linear shape function derivatives (constant)
        dh_dr = np.array([-1.0, 1.0, 0.0])
        dh_ds = np.array([-1.0, 0.0, 1.0])

        # Bubble function derivatives
        df4_dr, df4_ds = self._bubble_derivatives(r, s)

        # Enriched derivatives
        df_dr = np.array(
            [dh_dr[0] - df4_dr / 3.0, dh_dr[1] - df4_dr / 3.0, dh_dr[2] - df4_dr / 3.0, df4_dr]
        )

        df_ds = np.array(
            [dh_ds[0] - df4_ds / 3.0, dh_ds[1] - df4_ds / 3.0, dh_ds[2] - df4_ds / 3.0, df4_ds]
        )

        return df_dr, df_ds

    # =========================================================================
    # COVARIANT BASE VECTORS
    # =========================================================================

    def _compute_covariant_base_vectors(self, r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariant base vectors g_r and g_s at parametric point (r, s).

        For a triangle with linear geometry interpolation:
            g_r = ∂x/∂r = Σ (∂hi/∂r) * xi = x2 - x1
            g_s = ∂x/∂s = Σ (∂hi/∂s) * xi = x3 - x1

        Since shape functions are linear, base vectors are constant over element.

        Parameters
        ----------
        r, s : float
            Parametric coordinates (not used for linear element, included for interface)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            g_r : (2,) covariant base vector in r-direction (local x,y components)
            g_s : (2,) covariant base vector in s-direction (local x,y components)
        """
        if (0, 0) in self._covariant_cache:
            return self._covariant_cache[(0, 0)]

        x1, y1, x2, y2, x3, y3 = self._local_coordinates

        # For linear triangle: base vectors are constant
        # g_r = x2 - x1, g_s = x3 - x1
        g_r = np.array([x2 - x1, y2 - y1])
        g_s = np.array([x3 - x1, y3 - y1])

        self._covariant_cache[(0, 0)] = (g_r, g_s)
        return g_r, g_s

    def _compute_metric_tensor(self) -> np.ndarray:
        """
        Compute the covariant metric tensor g_ij.

        For a linear triangle, the metric tensor is constant:
            g_rr = g_r · g_r
            g_ss = g_s · g_s
            g_rs = g_r · g_s

        Returns
        -------
        np.ndarray
            (2, 2) covariant metric tensor [[g_rr, g_rs], [g_rs, g_ss]]
        """
        g_r, g_s = self._compute_covariant_base_vectors(0, 0)

        g_rr = np.dot(g_r, g_r)
        g_ss = np.dot(g_s, g_s)
        g_rs = np.dot(g_r, g_s)

        return np.array([[g_rr, g_rs], [g_rs, g_ss]])

    # =========================================================================
    # LOCAL COORDINATE SYSTEM
    # =========================================================================

    def _compute_local_coordinates(self) -> Tuple[float, ...]:
        """
        Compute local 2D coordinates of the element nodes.

        Projects the 3D element onto a local x-y plane defined by the element.
        Node 1 is at origin (0,0).
        Node 2 is on x-axis (x2, 0).
        Node 3 is in x-y plane (x3, y3).

        Returns
        -------
        Tuple[float, ...]
            Local coordinates (x1, y1, x2, y2, x3, y3)
        """
        nodes = self._current_coords

        # Vector 1-2
        v12 = nodes[1] - nodes[0]
        L12 = np.linalg.norm(v12)
        e1 = v12 / L12

        # Vector 1-3
        v13 = nodes[2] - nodes[0]

        # Normal vector (z-axis)
        n = np.cross(v12, v13)
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            raise ValueError("Degenerate triangle: nodes are collinear")
        e3 = n / n_len

        # y-axis
        e2 = np.cross(e3, e1)

        # Project nodes to local coordinates
        x1, y1 = 0.0, 0.0
        x2, y2 = L12, 0.0
        x3 = np.dot(v13, e1)
        y3 = np.dot(v13, e2)

        # Store basis for future use
        self._e1, self._e2, self._e3 = e1, e2, e3

        return (x1, y1, x2, y2, x3, y3)

    def _compute_initial_directors(self) -> np.ndarray:
        """
        Compute initial nodal directors (normals).
        For flat elements, directors are simply the element normal.
        """
        return np.tile(self._e3, (3, 1))

    def area(self) -> float:
        """
        Compute area of the triangular element.

        Returns
        -------
        float
            Element area
        """
        x1, y1, x2, y2, x3, y3 = self._local_coordinates
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    def J(self, r: float, s: float) -> Tuple[np.ndarray, float]:
        """
        Compute Jacobian matrix and determinant at (r, s).

        For a linear triangle, the Jacobian is constant:
        J = [∂x/∂r  ∂y/∂r] = [x2-x1  y2-y1]
            [∂x/∂s  ∂y/∂s]   [x3-x1  y3-y1]

        Note: detJ = 2 × Area

        Parameters
        ----------
        r, s : float
            Parametric coordinates (not used for linear element)

        Returns
        -------
        Tuple[np.ndarray, float]
            (J, detJ) - Jacobian matrix and determinant
        """
        x1, y1, x2, y2, x3, y3 = self._local_coordinates

        dx_dr = x2 - x1
        dy_dr = y2 - y1
        dx_ds = x3 - x1
        dy_ds = y3 - y1

        J_mat = np.array([[dx_dr, dy_dr], [dx_ds, dy_ds]])
        detJ = dx_dr * dy_ds - dy_dr * dx_ds

        return J_mat, detJ

    def T(self) -> np.ndarray:
        """
        Compute transformation matrix from global to local coordinates.

        Returns
        -------
        np.ndarray
            18×18 transformation matrix (block diagonal of 3×3 direction cosines)
        """
        nodes = self._current_coords
        v12 = nodes[1] - nodes[0]
        e1 = v12 / np.linalg.norm(v12)
        v13 = nodes[2] - nodes[0]
        n = np.cross(v12, v13)
        e3 = n / np.linalg.norm(n)
        e2 = np.cross(e3, e1)

        T3 = np.vstack([e1, e2, e3])

        T = np.zeros((18, 18))
        for i in range(6):  # 3 nodes × 2 blocks (translations + rotations)
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = T3

        return T

    # =========================================================================
    # STANDARD SHAPE FUNCTIONS (for displacements)
    # =========================================================================

    def _compute_N(self, r: float, s: float) -> np.ndarray:
        """
        Compute shape function matrix N at (r, s).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            6×18 shape function matrix
        """
        h = self._shape_functions(r, s)

        N = np.zeros((6, 18))
        for i, hi in enumerate(h):
            idx = 6 * i
            N[0, idx] = hi  # u
            N[1, idx + 1] = hi  # v
            N[2, idx + 2] = hi  # w
            N[3, idx + 3] = hi  # θx
            N[4, idx + 4] = hi  # θy
            N[5, idx + 5] = hi  # θz
        return N

    def _get_dH(self, r: float, s: float) -> np.ndarray:
        """
        Get shape function derivatives ∂h/∂x, ∂h/∂y at (r, s).

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            2×3 matrix [[∂h1/∂x, ∂h2/∂x, ∂h3/∂x],
                        [∂h1/∂y, ∂h2/∂y, ∂h3/∂y]]
        """
        if (r, s) in self._dH_cache:
            return self._dH_cache[(r, s)]

        J_mat, _ = self.J(r, s)
        J_inv = np.linalg.inv(J_mat)

        # Shape function derivatives in natural coordinates
        dh_dr = np.array([-1.0, 1.0, 0.0])
        dh_ds = np.array([-1.0, 0.0, 1.0])

        # Transform to Cartesian
        dh_dx = J_inv[0, 0] * dh_dr + J_inv[0, 1] * dh_ds
        dh_dy = J_inv[1, 0] * dh_dr + J_inv[1, 1] * dh_ds

        dH = np.vstack([dh_dx, dh_dy])
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
            raise NotImplementedError("Orthotropic material not yet supported in MITC3")

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
        Membrane strain-displacement matrix (3×18).

        For MITC3+, membrane strains use standard displacement-based formulation
        (constant strain triangle). No assumed strain field for membrane.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×18 membrane B matrix
        """
        dH = self._get_dH(r, s)
        Bm = np.zeros((3, 18))

        for i in range(3):
            u_idx = 6 * i
            v_idx = 6 * i + 1

            dhi_dx = dH[0, i]
            dhi_dy = dH[1, i]

            # εxx = ∂u/∂x
            Bm[0, u_idx] = dhi_dx
            # εyy = ∂v/∂y
            Bm[1, v_idx] = dhi_dy
            # γxy = ∂u/∂y + ∂v/∂x
            Bm[2, u_idx] = dhi_dy
            Bm[2, v_idx] = dhi_dx

        return Bm

    def B_kappa(self, r: float, s: float) -> np.ndarray:
        """
        Bending curvature-displacement matrix (3×18).

        Uses 3D covariant formulation. Curvatures relate to the derivatives
        of the director vector rotations.
        """
        # Linear shape function derivatives in natural coordinates
        dh_dr = np.array([-1.0, 1.0, 0.0])
        dh_ds = np.array([-1.0, 0.0, 1.0])

        # Jacobian and basis
        J_mat, _ = self.J(r, s)
        J_inv = np.linalg.inv(J_mat)

        # Cartesian shape function derivatives
        dh_dx = J_inv[0, 0] * dh_dr + J_inv[0, 1] * dh_ds
        dh_dy = J_inv[1, 0] * dh_dr + J_inv[1, 1] * dh_ds

        Bk = np.zeros((3, 18))

        for i in range(3):
            thx_idx = 6 * i + 3  # θx
            thy_idx = 6 * i + 4  # θy

            # κxx = ∂θy/∂x, κyy = -∂θx/∂y, κxy = ∂θy/∂y - ∂θx/∂x
            # (Standard plate bending approximation, but consistent with 3D covariant basis for flat shell)
            Bk[0, thy_idx] = dh_dx[i]
            Bk[1, thx_idx] = -dh_dy[i]
            Bk[2, thy_idx] = dh_dy[i]
            Bk[2, thx_idx] = -dh_dx[i]

        return Bk

    def _evaluate_covariant_shear_at_point(
        self, r: float, s: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate covariant transverse shear strains (e_rt, e_st) at a point.

        These are the displacement-based (not assumed) shear strains:
            e_rt = g_r · (∂w/∂r * e_z + θ)
            e_st = g_s · (∂w/∂s * e_z + θ)

        For a flat shell in Mindlin-Reissner theory:
            e_rt = ∂w/∂r + g_r · θ
            e_st = ∂w/∂s + g_s · θ

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            B_ert : (18,) contributions to e_rt
            B_est : (18,) contributions to e_st
        """
        # Shape functions and derivatives in natural coordinates
        h = self._shape_functions(r, s)
        dh_dr = np.array([-1.0, 1.0, 0.0])
        dh_ds = np.array([-1.0, 0.0, 1.0])

        # Covariant base vectors (in local xy plane)
        g_r, g_s = self._compute_covariant_base_vectors(r, s)

        B_ert = np.zeros(18)
        B_est = np.zeros(18)

        for i in range(3):
            w_idx = 6 * i + 2  # w
            thx_idx = 6 * i + 3  # θx (rotation about local x)
            thy_idx = 6 * i + 4  # θy (rotation about local y)

            # e_rt contribution: ∂w/∂r + g_r·θ
            # θ = [θx, θy] in local frame contributes as θx*(-g_r[1]) + θy*(g_r[0])
            # (cross product logic: θ×z where z is out of plane)
            B_ert[w_idx] = dh_dr[i]
            B_ert[thy_idx] = h[i] * g_r[0]  # θy contribution
            B_ert[thx_idx] = -h[i] * g_r[1]  # θx contribution (negative)

            # e_st contribution: ∂w/∂s + g_s·θ
            B_est[w_idx] = dh_ds[i]
            B_est[thy_idx] = h[i] * g_s[0]
            B_est[thx_idx] = -h[i] * g_s[1]

        return B_ert, B_est

    def _evaluate_covariant_shear_at_point_ext(
        self, r: float, s: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extended covariant shear evaluation including MITC3+ bubble rotations.

        The MITC3+ element enriches the *rotation field* using the cubic bubble
        function (Lee et al., 2014). The internal bubble node contributes only
        two rotational DOFs (about local x and y), which are statically condensed.

        This routine returns strain-displacement *vectors* for the covariant
        transverse shear strains (e_rt, e_st) in an extended DOF space:
        - first 18 entries: nodal DOFs (3 nodes × 6)
        - last 2 entries: bubble rotations [θx4, θy4]
        """
        # Standard shape functions (for w)
        h = self._shape_functions(r, s)
        dh_dr = np.array([-1.0, 1.0, 0.0])
        dh_ds = np.array([-1.0, 0.0, 1.0])

        # Enriched shape functions (for rotations)
        f = self._enriched_shape_functions(r, s)  # [f1, f2, f3, f4]

        # Covariant base vectors (in local xy plane)
        g_r, g_s = self._compute_covariant_base_vectors(r, s)

        B_ert = np.zeros(20)
        B_est = np.zeros(20)

        # Corner nodes (1..3)
        for i in range(3):
            w_idx = 6 * i + 2
            thx_idx = 6 * i + 3
            thy_idx = 6 * i + 4

            # ∂w/∂r and ∂w/∂s (w uses linear interpolation)
            B_ert[w_idx] = dh_dr[i]
            B_est[w_idx] = dh_ds[i]

            # Rotation contributions use enriched interpolation
            B_ert[thy_idx] = f[i] * g_r[0]
            B_ert[thx_idx] = -f[i] * g_r[1]

            B_est[thy_idx] = f[i] * g_s[0]
            B_est[thx_idx] = -f[i] * g_s[1]

        # Bubble node rotations (internal DOFs, indices 18 and 19)
        thx4_idx = 18
        thy4_idx = 19

        B_ert[thy4_idx] = f[3] * g_r[0]
        B_ert[thx4_idx] = -f[3] * g_r[1]

        B_est[thy4_idx] = f[3] * g_s[0]
        B_est[thx4_idx] = -f[3] * g_s[1]

        return B_ert, B_est

    def _B_kappa_ext(self, r: float, s: float) -> np.ndarray:
        """Extended bending B-matrix including bubble rotations (3x20)."""
        # Enriched derivatives in natural coordinates
        df_dr, df_ds = self._enriched_shape_derivatives(r, s)
        J_mat, _ = self.J(r, s)
        J_inv = np.linalg.inv(J_mat)

        # Convert to local Cartesian derivatives
        df_dx = J_inv[0, 0] * df_dr + J_inv[0, 1] * df_ds
        df_dy = J_inv[1, 0] * df_dr + J_inv[1, 1] * df_ds

        Bk = np.zeros((3, 20))

        # Corner nodes
        for i in range(3):
            thx_idx = 6 * i + 3
            thy_idx = 6 * i + 4

            # κxx = ∂θy/∂x
            Bk[0, thy_idx] = df_dx[i]
            # κyy = -∂θx/∂y
            Bk[1, thx_idx] = -df_dy[i]
            # κxy = ∂θy/∂y - ∂θx/∂x
            Bk[2, thy_idx] = df_dy[i]
            Bk[2, thx_idx] = -df_dx[i]

        # Bubble rotations
        thx4_idx = 18
        thy4_idx = 19
        Bk[0, thy4_idx] = df_dx[3]
        Bk[1, thx4_idx] = -df_dy[3]
        Bk[2, thy4_idx] = df_dy[3]
        Bk[2, thx4_idx] = -df_dx[3]

        return Bk

    def _B_gamma_ext(self, r: float, s: float) -> np.ndarray:
        """Extended MITC3+ transverse shear B-matrix including bubble rotations (2x20)."""
        # Constant part tying points
        B_ert_A, B_est_A = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_constant["A"]
        )
        B_ert_B, B_est_B = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_constant["B"]
        )
        B_ert_C, B_est_C = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_constant["C"]
        )

        # Linear part tying points
        B_ert_D, B_est_D = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_linear["D"]
        )
        B_ert_E, B_est_E = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_linear["E"]
        )
        B_ert_F, B_est_F = self._evaluate_covariant_shear_at_point_ext(
            *self._tying_points_linear["F"]
        )

        # Constant part (Eq. 15)
        B_ert_const = (2.0 / 3.0) * (B_ert_B - 0.5 * B_est_B) + (1.0 / 3.0) * (B_ert_C + B_est_C)
        B_est_const = (2.0 / 3.0) * (B_est_A - 0.5 * B_ert_A) + (1.0 / 3.0) * (B_ert_C + B_est_C)

        # Linear part (Eq. 16)
        B_c_hat = (B_ert_F - B_ert_D) - (B_est_F - B_est_E)
        B_ert_linear = (1.0 / 3.0) * B_c_hat * (3.0 * s - 1.0)
        B_est_linear = (1.0 / 3.0) * B_c_hat * (1.0 - 3.0 * r)

        # Total assumed covariant shear (Eq. 17)
        B_ert_assumed = B_ert_const + B_ert_linear
        B_est_assumed = B_est_const + B_est_linear

        # Transform to local Cartesian shear components
        J_mat, _ = self.J(r, s)
        J_inv = np.linalg.inv(J_mat)

        B_gamma_xz = J_inv[0, 0] * B_ert_assumed + J_inv[0, 1] * B_est_assumed
        B_gamma_yz = J_inv[1, 0] * B_ert_assumed + J_inv[1, 1] * B_est_assumed

        return np.vstack([B_gamma_xz, B_gamma_yz])

    def _k_bs_condensed(self) -> np.ndarray:
        """Combined bending+shear stiffness with MITC3+ bubble condensation (18x18)."""
        Cb = self.Cb()
        Cs = self.Cs()
        area = self.area()

        K_ext = np.zeros((20, 20))
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            Bk_ext = self._B_kappa_ext(r, s)
            Bg_ext = self._B_gamma_ext(r, s)
            K_ext += w * area * (Bk_ext.T @ Cb @ Bk_ext + Bg_ext.T @ Cs @ Bg_ext)

        # Static condensation of bubble rotations (last 2 DOFs)
        K_uu = K_ext[:18, :18]
        K_uq = K_ext[:18, 18:]
        K_qu = K_ext[18:, :18]
        K_qq = K_ext[18:, 18:]

        # No external load on internal bubble DOFs -> q = -Kqq^{-1} Kqu u
        K_cond = K_uu - K_uq @ np.linalg.solve(K_qq, K_qu)
        K_cond = 0.5 * (K_cond + K_cond.T)
        return K_cond

    def B_gamma(self, r: float, s: float) -> np.ndarray:
        """
        Transverse shear strain-displacement matrix (2×18) using MITC3+ interpolation.
        Includes bubble condensation implicitly if called from K.
        Otherwise provides the 2x18 matrix for nodal DOFs.
        """
        # If we are in nonlinear mode or need bubble, we should use the condensed version
        # For simplicity in the API, we keep B_gamma as 2x18 and handle bubble condensation in K and f_int.

        # Tying points evaluation
        B_ert_A, B_est_A = self._evaluate_covariant_shear_at_point(*self._tying_points_constant["A"])
        B_ert_B, B_est_B = self._evaluate_covariant_shear_at_point(*self._tying_points_constant["B"])
        B_ert_C, B_est_C = self._evaluate_covariant_shear_at_point(*self._tying_points_constant["C"])

        B_ert_D, B_est_D = self._evaluate_covariant_shear_at_point(*self._tying_points_linear["D"])
        B_ert_E, B_est_E = self._evaluate_covariant_shear_at_point(*self._tying_points_linear["E"])
        B_ert_F, B_est_F = self._evaluate_covariant_shear_at_point(*self._tying_points_linear["F"])

        # Constant part (Eq. 15)
        B_ert_const = (2.0 / 3.0) * (B_ert_B - 0.5 * B_est_B) + (1.0 / 3.0) * (B_ert_C + B_est_C)
        B_est_const = (2.0 / 3.0) * (B_est_A - 0.5 * B_ert_A) + (1.0 / 3.0) * (B_ert_C + B_est_C)

        # Linear part (Eq. 16)
        B_c_hat = (B_ert_F - B_ert_D) - (B_est_F - B_est_E)
        B_ert_linear = (1.0 / 3.0) * B_c_hat * (3.0 * s - 1.0)
        B_est_linear = (1.0 / 3.0) * B_c_hat * (1.0 - 3.0 * r)

        B_ert_assumed = B_ert_const + B_ert_linear
        B_est_assumed = B_est_const + B_est_linear

        # Transform to local Cartesian
        J_mat, _ = self.J(r, s)
        J_inv = np.linalg.inv(J_mat)

        B_gamma_xz = J_inv[0, 0] * B_ert_assumed + J_inv[0, 1] * B_est_assumed
        B_gamma_yz = J_inv[1, 0] * B_ert_assumed + J_inv[1, 1] * B_est_assumed

        return np.vstack([B_gamma_xz, B_gamma_yz])

    # B_gamma_standard and other legacy methods removed for cleanliness

    # =========================================================================
    # STIFFNESS MATRIX ASSEMBLY
    # =========================================================================

    def k_m(self) -> np.ndarray:
        """
        Membrane stiffness matrix (18×18).

        Returns
        -------
        np.ndarray
            18×18 membrane stiffness matrix in local coordinates
        """
        Cm = self.Cm()
        Km = np.zeros((18, 18))
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            Bm = self.B_m(r, s)
            Km += w * area * Bm.T @ Cm @ Bm

        # Add drilling stiffness to stabilize the rotation about the normal (local z).
        # We use an integrated formulation: K_drill = integral( B_drill.T * k_stab * B_drill ) dA
        # This respects rigid body rotation unlike grounded springs.
        k_drill_stab = self.material.E * self.thickness * 1e-3
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            B_drill = self._compute_B_drill(r, s)
            # Use only the 5-th row which corresponds to d_theta_z
            B_d = B_drill[5:6, :]
            Km += w * area * k_drill_stab * B_d.T @ B_d

        return Km

    def k_b(self) -> np.ndarray:
        """
        Bending stiffness matrix (18×18).

        Returns
        -------
        np.ndarray
            18×18 bending stiffness matrix in local coordinates
        """
        Cb = self.Cb()
        Kb = np.zeros((18, 18))
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            Bk = self.B_kappa(r, s)
            Kb += w * area * Bk.T @ Cb @ Bk

        return Kb

    def k_s(self) -> np.ndarray:
        """
        Transverse shear stiffness matrix (18×18) using MITC3+ interpolation.

        Returns
        -------
        np.ndarray
            18×18 shear stiffness matrix in local coordinates
        """
        Cs = self.Cs()
        Ks = np.zeros((18, 18))
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            Bg = self.B_gamma(r, s)
            Ks += w * area * Bg.T @ Cs @ Bg

        return Ks

    @property
    def K(self) -> np.ndarray:
        """
        Global stiffness matrix (18×18).

        Returns
        -------
        np.ndarray
            18×18 element stiffness matrix in global coordinates
        """
        # Local stiffness
        # Membrane is standard CST; bending+shear include MITC3+ bubble enrichment
        # condensed at element level.
        K_local = self.k_m() + self._k_bs_condensed()

        # Transform to global
        T = self.T()
        K_global = T.T @ K_local @ T

        # Ensure symmetry (numerical cleanup)
        K_global = 0.5 * (K_global + K_global.T)

        return K_global

    @property
    def M(self) -> np.ndarray:
        """
        Consistent mass matrix (18×18).

        Returns
        -------
        np.ndarray
            18×18 element mass matrix in global coordinates
        """
        rho = self.material.rho
        h = self.thickness
        M = np.zeros((18, 18))
        area = self.area()

        # Translational and rotational inertia
        m_trans = rho * h
        m_rot = rho * h**3 / 12.0

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            N = self._compute_N(r, s)

            for i in range(3):
                for j in range(3):
                    # Translational (u, v, w)
                    val_t = N[0, 6 * i] * N[0, 6 * j] * m_trans * w * area
                    for k in range(3):
                        M[6 * i + k, 6 * j + k] += val_t

                    # Rotational (θx, θy, θz)
                    val_r = N[0, 6 * i] * N[0, 6 * j] * m_rot * w * area
                    for k in range(3, 6):
                        M[6 * i + k, 6 * j + k] += val_r

        T = self.T()
        return T.T @ M @ T

    # =========================================================================
    # GEOMETRIC NONLINEAR ANALYSIS (Total Lagrangian)
    # =========================================================================

    def update_configuration(self, displacements: np.ndarray) -> None:
        """
        Update element configuration with new nodal displacements.

        This method updates the current nodal positions based on displacement
        increments. Used in nonlinear analysis to track the deformed configuration.

        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (18,) in global coordinates.
            Order: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2, u3,v3,w3,θx3,θy3,θz3]

        Notes
        -----
        For Total Lagrangian formulation, the reference configuration remains
        the initial undeformed state. The current coordinates are computed as:

            x_current = x_initial + u

        The local coordinate system and cached values are NOT updated here
        since TL formulation uses the initial configuration for all derivatives.
        """
        if len(displacements) != 18:
            raise ValueError(f"Expected 18 DOFs, got {len(displacements)}")

        self._current_displacements = np.array(displacements, dtype=float)

        for i in range(3):
            u_vec = displacements[6 * i : 6 * i + 3]
            self._current_coords[i] = self._initial_coords[i] + u_vec

            # Update directors for large rotations (simplified for small rotation increments)
            # θ = [θx, θy, θz]
            theta = displacements[6 * i + 3 : 6 * i + 6]
            if np.linalg.norm(theta) > 1e-12:
                # Small rotation approximation for V_n = R(theta) * V_n0
                # In more advanced formulations, we'd use a better rotation update
                cross_theta = np.cross(theta, self._initial_directors[i])
                self._current_directors[i] = self._initial_directors[i] + cross_theta
                # Re-normalize to maintain director length (for shells)
                self._current_directors[i] /= np.linalg.norm(self._current_directors[i])
            else:
                self._current_directors[i] = self._initial_directors[i].copy()

    def reset_configuration(self) -> None:
        """
        Reset element to initial (undeformed) configuration.

        Clears all displacement history and restores initial coordinates.
        """
        self._current_coords = self._initial_coords.copy()
        self._current_displacements = np.zeros(18)

    def get_displacement_gradient(self, r: float, s: float) -> np.ndarray:
        """
        Compute displacement gradient H = ∂u/∂X.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×3 displacement gradient tensor
        """
        dH = self._get_dH(r, s)

        u_nodes = np.zeros((3, 3))
        for i in range(3):
            u_nodes[i, 0] = self._current_displacements[6 * i]
            u_nodes[i, 1] = self._current_displacements[6 * i + 1]
            u_nodes[i, 2] = self._current_displacements[6 * i + 2]

        H = np.zeros((3, 3))
        for j in range(2):  # x, y derivatives only (flat shell)
            for comp in range(3):  # u, v, w components
                H[comp, j] = np.dot(u_nodes[:, comp], dH[j, :])

        return H

    def compute_green_lagrange_strain(self, r: float, s: float) -> np.ndarray:
        """
        Compute Green-Lagrange strain tensor E.

        The Green-Lagrange strain tensor is defined as:

            E = 0.5 * (F^T F - I) = 0.5 * (H + H^T + H^T H)

        where F = I + H is the deformation gradient.

        This strain measure is work-conjugate to the Second Piola-Kirchhoff
        stress and is valid for large displacements and rotations.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×3 Green-Lagrange strain tensor

        Notes
        -----
        The strain tensor has the form:
            [E_xx  E_xy  E_xz]
            [E_xy  E_yy  E_yz]
            [E_xz  E_yz  E_zz]
        """
        H = self.get_displacement_gradient(r, s)
        return 0.5 * (H + H.T + H.T @ H)

    def compute_green_lagrange_strain_voigt(self, r: float = 0.0, s: float = 0.0) -> np.ndarray:
        """
        Compute Green-Lagrange strain in Voigt notation.

        Parameters
        ----------
        r : float, optional
            Parametric coordinate, default 0 (approximately centroid for triangle)
        s : float, optional
            Parametric coordinate, default 0 (approximately centroid for triangle)

        Returns
        -------
        np.ndarray
            (6,) strain vector [E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_xz]

        Notes
        -----
        Engineering shear strains (γ = 2E) are used in Voigt notation.
        For centroid evaluation in a triangle, use r=s=1/3.
        """
        E = self.compute_green_lagrange_strain(r, s)

        return np.array(
            [
                E[0, 0],  # E_xx
                E[1, 1],  # E_yy
                E[2, 2],  # E_zz
                2 * E[0, 1],  # γ_xy = 2*E_xy
                2 * E[1, 2],  # γ_yz = 2*E_yz
                2 * E[0, 2],  # γ_xz = 2*E_xz
            ]
        )

    def _compute_G_matrix(self, r: float, s: float) -> np.ndarray:
        """
        Compute G matrix for geometric stiffness.

        The G matrix relates displacement gradient variations to nodal DOF
        variations. Used in the geometric stiffness computation:

            K_σ = ∫ G^T · S̃ · G dA

        where S̃ is the stress matrix in appropriate form.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            6×18 gradient matrix [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y]
        """
        dH = self._get_dH(r, s)

        G = np.zeros((6, 18))

        for i in range(3):
            col = 6 * i
            dhi_dx = dH[0, i]
            dhi_dy = dH[1, i]

            # ∂u/∂x, ∂u/∂y
            G[0, col] = dhi_dx
            G[1, col] = dhi_dy
            # ∂v/∂x, ∂v/∂y
            G[2, col + 1] = dhi_dx
            G[3, col + 1] = dhi_dy
            # ∂w/∂x, ∂w/∂y
            G[4, col + 2] = dhi_dx
            G[5, col + 2] = dhi_dy

        return G

    def _compute_B_L(self, r: float, s: float) -> np.ndarray:
        """
        Compute the linear part of the strain-displacement matrix for TL formulation.

        This is the standard B matrix relating infinitesimal strain increments
        to displacement increments:

            δε = B_L · δu

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            (6, 18) linear strain-displacement matrix
        """
        dH = self._get_dH(r, s)  # 2x3: [dN/dx; dN/dy]

        # Build B_L matrix (6 strain components x 18 DOFs)
        # Strain order: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]
        B_L = np.zeros((6, 18))

        for i in range(3):
            col = 6 * i  # Starting column for node i
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # ε_xx = ∂u/∂x
            B_L[0, col] = dNi_dx

            # ε_yy = ∂v/∂y
            B_L[1, col + 1] = dNi_dy

            # ε_zz = 0 for membrane (would include ∂w/∂z for 3D)
            # B_L[2, :] = 0

            # γ_xy = ∂u/∂y + ∂v/∂x
            B_L[3, col] = dNi_dy  # ∂u/∂y
            B_L[3, col + 1] = dNi_dx  # ∂v/∂x

            # γ_yz = ∂v/∂z + ∂w/∂y (shell: includes rotation contributions)
            # γ_xz = ∂u/∂z + ∂w/∂x (shell: includes rotation contributions)
            # These are handled by B_gamma for shell elements

        return B_L

    def _compute_B_NL(self, r: float, s: float) -> np.ndarray:
        """
        Compute the nonlinear part of the strain-displacement matrix.

        The nonlinear B matrix captures the quadratic terms in the
        Green-Lagrange strain tensor. It depends on the current displacement
        state through the displacement gradient H.

        For the Green-Lagrange strain:
            E = ε_linear + ε_nonlinear
            ε_nonlinear = 0.5 * H^T @ H

        The variation gives:
            δE = B_L · δu + B_NL(u) · δu

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            (6, 18) nonlinear strain-displacement matrix

        Notes
        -----
        B_NL depends on the current displacement state and must be
        recomputed after each configuration update.
        """
        dH_shape = self._get_dH(r, s)  # 2x3: [dN/dx; dN/dy]
        H = self.get_displacement_gradient(r, s)  # 3x3 displacement gradient

        # Build B_NL matrix
        # The nonlinear strain terms involve products of displacement gradients
        B_NL = np.zeros((6, 18))

        for i in range(3):
            col = 6 * i
            dNi_dx = dH_shape[0, i]
            dNi_dy = dH_shape[1, i]

            # E_xx nonlinear: 0.5 * [(∂u/∂x)² + (∂v/∂x)² + (∂w/∂x)²]
            # δE_xx = (∂u/∂x)·δ(∂u/∂x) + (∂v/∂x)·δ(∂v/∂x) + (∂w/∂x)·δ(∂w/∂x)
            B_NL[0, col] = H[0, 0] * dNi_dx  # (∂u/∂x) * dNi/dx
            B_NL[0, col + 1] = H[1, 0] * dNi_dx  # (∂v/∂x) * dNi/dx
            B_NL[0, col + 2] = H[2, 0] * dNi_dx  # (∂w/∂x) * dNi/dx

            # E_yy nonlinear: 0.5 * [(∂u/∂y)² + (∂v/∂y)² + (∂w/∂y)²]
            B_NL[1, col] = H[0, 1] * dNi_dy  # (∂u/∂y) * dNi/dy
            B_NL[1, col + 1] = H[1, 1] * dNi_dy  # (∂v/∂y) * dNi/dy
            B_NL[1, col + 2] = H[2, 1] * dNi_dy  # (∂w/∂y) * dNi/dy

            # E_zz nonlinear: 0 for membrane

            # 2*E_xy nonlinear: (∂u/∂x)(∂u/∂y) + (∂v/∂x)(∂v/∂y) + (∂w/∂x)(∂w/∂y)
            B_NL[3, col] = H[0, 0] * dNi_dy + H[0, 1] * dNi_dx
            B_NL[3, col + 1] = H[1, 0] * dNi_dy + H[1, 1] * dNi_dx
            B_NL[3, col + 2] = H[2, 0] * dNi_dy + H[2, 1] * dNi_dx

        return B_NL

    def _compute_B_geometric(self, r: float, s: float) -> np.ndarray:
        """
        Compute the geometric B matrix for geometric stiffness computation.

        This matrix relates the symmetric gradient of displacements to the
        nodal DOFs and is used in computing the geometric stiffness matrix:

            K_G = ∫ B_G^T · S_m · B_G dA

        where S_m is the membrane stress matrix.

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            (6, 18) geometric strain-displacement matrix

        Notes
        -----
        For the triangular element with 3 nodes:
        - Rows: [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y]
        - Cols: 18 DOFs (6 per node × 3 nodes)
        """
        dH = self._get_dH(r, s)  # 2x3

        B_G = np.zeros((6, 18))

        for i in range(3):
            col = 6 * i  # Starting column for node i
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # ∂u/∂x, ∂u/∂y (row 0, 1)
            B_G[0, col] = dNi_dx  # ∂u/∂x
            B_G[1, col] = dNi_dy  # ∂u/∂y

            # ∂v/∂x, ∂v/∂y (row 2, 3)
            B_G[2, col + 1] = dNi_dx  # ∂v/∂x
            B_G[3, col + 1] = dNi_dy  # ∂v/∂y

            # ∂w/∂x, ∂w/∂y (row 4, 5)
            B_G[4, col + 2] = dNi_dx  # ∂w/∂x
            B_G[5, col + 2] = dNi_dy  # ∂w/∂y

        return B_G

    def compute_geometric_stiffness(
        self,
        sigma_membrane: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute geometric (stress) stiffness matrix K_σ.

        The geometric stiffness captures the effect of membrane stress on
        out-of-plane stiffness. It is essential for:
        - Buckling analysis
        - Nonlinear static analysis (Newton-Raphson)
        - Prestressed modal analysis

        For the Total Lagrangian formulation:

            K_σ = ∫ B_G^T · S̃_m · B_G · dA

        where B_G is the geometric strain-displacement matrix and S̃_m is
        the membrane stress matrix in block diagonal form.

        Parameters
        ----------
        sigma_membrane : np.ndarray
            Membrane stress state [σ_xx, σ_yy, σ_xy] (in local coordinates)
        transform_to_global : bool, optional
            If True, transform result to global coordinates. Default True.

        Returns
        -------
        np.ndarray
            (18, 18) geometric stiffness matrix

        Notes
        -----
        The stress matrix S̃_m has the block diagonal structure:

            S̃_m = [S_m   0    0  ]
                   [0    S_m   0  ]
                   [0     0   S_m ]

        where S_m = [[σ_xx, σ_xy], [σ_xy, σ_yy]]

        References
        ----------
        Lee, Y., Lee, P.S., and Bathe, K.J. (2015). "The MITC3+ shell element
        in geometric nonlinear analysis." Computers & Structures, 146, 91-104.
        """
        sigma = np.asarray(sigma_membrane)

        # Build 2x2 membrane stress block
        S_m = (
            np.array(
                [
                    [sigma[0], sigma[2]],  # [σ_xx, σ_xy]
                    [sigma[2], sigma[1]],  # [σ_xy, σ_yy]
                ]
            )
            * self.thickness
        )

        # Build 6x6 block diagonal stress matrix
        S_tilde = np.zeros((6, 6))
        S_tilde[0:2, 0:2] = S_m  # For u derivatives
        S_tilde[2:4, 2:4] = S_m  # For v derivatives
        S_tilde[4:6, 4:6] = S_m  # For w derivatives

        K_sigma = np.zeros((18, 18))
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            B_G = self._compute_B_geometric(r, s)
            K_sigma += w * area * B_G.T @ S_tilde @ B_G

        # Force symmetry
        K_sigma = 0.5 * (K_sigma + K_sigma.T)

        if transform_to_global:
            T = self.T()
            K_sigma = T.T @ K_sigma @ T

        return K_sigma

    def compute_membrane_stress_from_displacement(
        self,
        u_local: np.ndarray,
        r: float = 1.0 / 3.0,
        s: float = 1.0 / 3.0,
    ) -> np.ndarray:
        """
        Compute membrane stress from local displacement vector.

        Uses the constitutive relation:
            σ = C_m · ε_m

        where ε_m is the membrane strain computed from displacements.

        Parameters
        ----------
        u_local : np.ndarray
            (18,) displacement vector in local coordinates
        r : float, optional
            Parametric coordinate for strain evaluation. Default 1/3 (centroid).
        s : float, optional
            Parametric coordinate for strain evaluation. Default 1/3 (centroid).

        Returns
        -------
        np.ndarray
            (3,) membrane stress [σ_xx, σ_yy, σ_xy]

        Notes
        -----
        For the constant strain triangle (CST) used in MITC3+, the membrane
        strain is constant over the element, so the evaluation point doesn't
        matter for linear analysis.
        """
        # Extract membrane DOFs (u, v at each node)
        mem_dofs = np.array([0, 1, 6, 7, 12, 13])
        u_mem = u_local[mem_dofs]

        # Membrane B matrix
        B_m = self.B_m(r, s)

        # Membrane strain
        epsilon_m = B_m[:, mem_dofs] @ u_mem

        # Stress from constitutive relation
        # Note: Cm already includes thickness for stiffness, need raw C
        E = self.material.E
        nu = self.material.nu
        factor = E / (1 - nu**2)
        C_m = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

        sigma_m = C_m @ epsilon_m

        return sigma_m

    def compute_centrifugal_prestress(
        self,
        omega: float,
        rotation_axis: np.ndarray,
        rotation_center: np.ndarray,
    ) -> np.ndarray:
        """
        Compute centrifugal prestress for rotating structures.

        Calculates the membrane stress state due to centrifugal forces in
        rotating machinery (e.g., wind turbine blades, helicopter rotors).

        Parameters
        ----------
        omega : float
            Angular velocity (rad/s)
        rotation_axis : np.ndarray
            Unit vector defining the rotation axis in global coordinates
        rotation_center : np.ndarray
            Point on the rotation axis in global coordinates

        Returns
        -------
        np.ndarray
            (3,) centrifugal membrane stress [σ_xx, σ_yy, σ_xy] in local coords

        Notes
        -----
        The centrifugal acceleration is a_c = ω² × r, where r is the radial
        distance from the rotation axis.

        The resulting stress is approximated as:
            σ_cf ≈ ρ × ω² × r × L_char

        where L_char is a characteristic element length and ρ is the density.
        """
        # Normalize rotation axis
        axis = np.asarray(rotation_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        center = np.asarray(rotation_center, dtype=float)

        # Compute element centroid in global coordinates
        centroid_global = np.mean(self.node_coords, axis=0)

        # Vector from rotation center to centroid
        r_vec = centroid_global - center

        # Project out the component along rotation axis to get radial distance
        r_parallel = np.dot(r_vec, axis) * axis
        r_radial_vec = r_vec - r_parallel
        r_radial = np.linalg.norm(r_radial_vec)

        if r_radial < 1e-10:
            # Element is on the rotation axis - no centrifugal stress
            return np.zeros(3)

        # Radial direction (unit vector pointing outward from axis)
        radial_dir = r_radial_vec / r_radial

        # Centrifugal acceleration: a_c = ω² · r
        # Centrifugal stress (simplified): σ ≈ ρ · ω² · r · L_char
        # where L_char is a characteristic length (use element dimension)
        rho = self.material.rho
        L_char = np.sqrt(self.area())  # Characteristic element length

        # Simplified centrifugal stress magnitude
        sigma_cf = rho * omega**2 * r_radial * L_char

        # Transform radial direction to local coordinates
        T = self.T()
        T_3x3 = T[:3, :3]  # Extract 3x3 rotation part
        radial_local = T_3x3 @ radial_dir

        # Project stress into local membrane plane (x-y plane)
        # σ_xx = σ_cf · cos²θ, σ_yy = σ_cf · sin²θ, σ_xy = σ_cf · sinθcosθ
        cos_theta = radial_local[0]  # x-component
        sin_theta = radial_local[1]  # y-component

        sigma_xx = sigma_cf * cos_theta**2
        sigma_yy = sigma_cf * sin_theta**2
        sigma_xy = sigma_cf * cos_theta * sin_theta

        return np.array([sigma_xx, sigma_yy, sigma_xy])

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute tangent stiffness matrix for nonlinear analysis.
        """
        # Linear/Reference stiffness
        K0 = self.k_m() + self._k_bs_condensed()

        if not self.nonlinear:
            if transform_to_global:
                T = self.T()
                return T.T @ K0 @ T
            return K0

        # Nonlinear terms (K_L + K_sigma)
        # For TL formulation, K_T = K_0 + K_L + K_sigma
        # Here we follow the structure of MITC4 for consistency

        # Compute stresses if not provided
        if sigma is None:
            T = self.T()
            u_local = T @ self._current_displacements
            sigma = self.compute_membrane_stress_from_displacement(u_local)

        # Geometric stiffness K_sigma
        K_sigma = self.compute_geometric_stiffness(sigma, transform_to_global=False)

        # Initial displacement stiffness K_L
        K_L = np.zeros((18, 18))
        E_mat = self.material.E
        nu = self.material.nu
        factor = E_mat / (1 - nu**2)
        Cm_raw = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            B_L = self._compute_B_L(r, s)
            B_NL = self._compute_B_NL(r, s)

            B_m_L = B_L[[0, 1, 3], :]
            B_m_NL = B_NL[[0, 1, 3], :]

            # K_L = integral( (BL^T @ C @ BNL + BNL^T @ C @ BL + BNL^T @ C @ BNL) dA )
            # The last term is needed for full TL consistency
            K_L += (
                (
                    B_m_L.T @ Cm_raw @ B_m_NL
                    + B_m_NL.T @ Cm_raw @ B_m_L
                    + B_m_NL.T @ Cm_raw @ B_m_NL
                )
                * w
                * area
                * self.thickness
            )

        K_T = K0 + K_L + K_sigma

        # Ensure symmetry
        K_T = 0.5 * (K_T + K_T.T)

        if transform_to_global:
            T = self.T()
            K_T = T.T @ K_T @ T

        return K_T

    def compute_internal_forces(
        self,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute internal force vector for nonlinear analysis.
        """
        T = self.T()
        u_local = T @ self._current_displacements
        
        f_int = np.zeros(18)
        
        # Consistent integration for all components
        E_mat = self.material.E
        nu = self.material.nu
        factor = E_mat / (1 - nu**2)
        Cm_raw = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        
        area = self.area()
        
        # Bending and Shear (statically condensed bubble)
        # In linear range, f = K_cond @ u
        # For non-linear, we should integrate B^T @ sigma
        # But we assume bending/shear are linear in local frame for flat shell TL
        f_bs = self._k_bs_condensed() @ u_local
        f_int += f_bs

        # Membrane integration (with GL strain if nonlinear)
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            if self.nonlinear:
                E_GL = self.compute_green_lagrange_strain(r, s)
                # Voigt: [Exx, Eyy, 2Exy]
                eps_m = np.array([E_GL[0, 0], E_GL[1, 1], 2 * E_GL[0, 1]])
                sigma_m = Cm_raw @ eps_m
                
                B_L = self._compute_B_L(r, s)
                B_NL = self._compute_B_NL(r, s)
                B_total = B_L[[0, 1, 3], :] + B_NL[[0, 1, 3], :]
                
                f_int += B_total.T @ sigma_m * w * area * self.thickness
            else:
                Bm = self.B_m(r, s)
                eps_m = Bm @ u_local
                sigma_m = Cm_raw @ eps_m
                f_int += Bm.T @ sigma_m * w * area * self.thickness

        if transform_to_global:
            f_int = T.T @ f_int
            
        return f_int

    def compute_residual(
        self,
        f_ext: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute residual force vector for Newton-Raphson iteration.

        The residual is the out-of-balance force:

            R = f_ext - f_int

        Convergence is achieved when ||R|| < tolerance.

        Parameters
        ----------
        f_ext : np.ndarray
            (18,) external force vector in global coordinates
        transform_to_global : bool, optional
            Whether internal forces are in global coordinates. Default True.

        Returns
        -------
        np.ndarray
            (18,) residual force vector
        """
        f_int = self.compute_internal_forces(transform_to_global=transform_to_global)
        return f_ext - f_int

    def compute_strain_energy(self) -> float:
        """
        Compute total strain energy stored in the element.
        """
        T = self.T()
        u_local = T @ self._current_displacements

        # Bending and Shear energy (using condensed stiffness)
        U = 0.5 * u_local @ self._k_bs_condensed() @ u_local

        # Membrane energy
        E_mat = self.material.E
        nu = self.material.nu
        factor = E_mat / (1 - nu**2)
        Cm_raw = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        area = self.area()

        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            if self.nonlinear:
                E_GL = self.compute_green_lagrange_strain(r, s)
                eps_m = np.array([E_GL[0, 0], E_GL[1, 1], 2 * E_GL[0, 1]])
            else:
                Bm = self.B_m(r, s)
                eps_m = Bm @ u_local

            U += 0.5 * eps_m @ Cm_raw @ eps_m * w * area * self.thickness

        return float(U)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_B_drill(self, r: float, s: float) -> np.ndarray:
        """
        Compute drilling rotation B matrix (for drilling DOF stabilization).

        Based on Hughes & Brezzi (1989) drilling rotation formulation.

        Parameters
        ----------
        r, s : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            6×18 drilling B matrix
        """
        dH = self._get_dH(r, s)
        h = self._shape_functions(r, s)

        B_drill = np.zeros((6, 18))

        for i in range(3):
            u_idx = 6 * i
            v_idx = 6 * i + 1
            thz_idx = 6 * i + 5

            dhi_dx = dH[0, i]
            dhi_dy = dH[1, i]

            # Drilling rotation: (∂v/∂x - ∂u/∂y)/2 - θz
            B_drill[5, u_idx] = -0.5 * dhi_dy
            B_drill[5, v_idx] = 0.5 * dhi_dx
            B_drill[5, thz_idx] = -h[i]

        return B_drill
