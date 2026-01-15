"""
MITC4/MITC4+ Four-Node Shell Element Implementation

This module implements both the MITC4 (Mixed Interpolation of Tensorial Components)
and the enhanced MITC4+ four-node quadrilateral shell elements.

References:
    [1] Dvorkin, E.N. and Bathe, K.J. (1984). "A continuum mechanics based four-node
        shell element for general nonlinear analysis." Engineering Computations, 1, 77-88.
    [2] Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "A new MITC4+ shell element."
        Computers and Structures, 182, 404-418.
    [3] Bathe, K.J. (2006). "Finite Element Procedures." Klaus-Jurgen Bathe.
    [4] OpenSees ShellMITC4 implementation.

The MITC4 formulation uses assumed strain fields for transverse shear to eliminate
shear locking in thin shells. The element combines:
- Bilinear displacement interpolation for membrane behavior
- Bilinear rotation interpolation for bending
- Mixed interpolation for transverse shear (MITC method)
- Hughes-Brezzi drilling DOF stabilization

The MITC4+ enhancement (Ko, Lee & Bathe 2017) adds:
- Bubble function enrichment for rotations (2 internal DOFs)
- Modified assumed strain field for transverse shear
- Static condensation of internal bubble DOFs
- Improved performance in warped/distorted elements

Node ordering convention (counter-clockwise):
    3-------2
    |       |
    |       |
    0-------1

Natural coordinates: ξ (xi) ∈ [-1,1], η (eta) ∈ [-1,1]

DOFs per node: 6 (u, v, w, θx, θy, θz)
    - u, v, w: displacements in global x, y, z
    - θx, θy, θz: rotations about global x, y, z axes

Tying points for MITC4 transverse shear interpolation:
    - Point A: (0, +1) - top edge midpoint
    - Point B: (-1, 0) - left edge midpoint
    - Point C: (0, -1) - bottom edge midpoint
    - Point D: (+1, 0) - right edge midpoint

MITC4+ additional tying points (internal):
    - Point E: (0, 0) - element center
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC4(ShellElement):
    """
    MITC4/MITC4+ four-node quadrilateral shell element.

    This element implements both the MITC4 formulation from Dvorkin & Bathe (1984)
    and the enhanced MITC4+ formulation from Ko, Lee & Bathe (2017).
    Each corner node has 6 DOFs (u, v, w, θx, θy, θz). The element uses
    mixed interpolation for transverse shear strains to prevent shear locking.

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
        Enable geometric nonlinear analysis (default: False)
    use_mitc4_plus : bool, optional
        Use MITC4+ formulation with bubble enrichment (default: False).
        When True, uses the enhanced formulation from Ko, Lee & Bathe (2017)
        which provides improved performance for warped/distorted elements.

    Attributes
    ----------
    element_type : str
        Element identifier "MITC4" or "MITC4+"
    thickness : float
        Shell thickness
    dofs_count : int
        Total DOFs (24 for 4 nodes × 6 DOFs)
    use_mitc4_plus : bool
        Whether MITC4+ formulation is active

    Notes
    -----
    The MITC4 element uses:

    **Parametric coordinates**: (ξ, η) where ξ, η ∈ [-1, 1]
    - Node 0: (-1, -1)
    - Node 1: (+1, -1)
    - Node 2: (+1, +1)
    - Node 3: (-1, +1)

    **Bilinear shape functions**:
    - N₀ = (1-ξ)(1-η)/4
    - N₁ = (1+ξ)(1-η)/4
    - N₂ = (1+ξ)(1+η)/4
    - N₃ = (1-ξ)(1+η)/4

    **MITC4 Transverse Shear Interpolation**:
    The assumed transverse shear strain field uses values at four tying points:
    - γxz is sampled at points A (η=+1) and C (η=-1), interpolated linearly in η
    - γyz is sampled at points B (ξ=-1) and D (ξ=+1), interpolated linearly in ξ

    **MITC4+ Enhancement** (when use_mitc4_plus=True):
    The MITC4+ formulation adds:
    - Bubble function Nb = (1-ξ²)(1-η²) for rotation enrichment
    - Two internal rotation DOFs (α₁, α₂) that are statically condensed
    - Modified assumed shear strain field with center tying point
    - Improved accuracy for warped elements and thin shells

    This eliminates shear locking while maintaining consistency with the
    displacement-based membrane and bending formulations.
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
        use_mitc4_plus: bool = False,
    ):
        super().__init__(
            "MITC4+" if use_mitc4_plus else "MITC4",
            node_coords=node_coords,
            node_ids=node_ids,
            material=material,
            dofs_per_node=6,
            thickness=thickness,
            nonlinear=nonlinear,
        )

        self.use_mitc4_plus = use_mitc4_plus
        self.element_type = "MITC4+" if use_mitc4_plus else "MITC4"

        # Store initial and current coordinates
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

        # 2x2 Gauss quadrature points and weights
        gp = 1.0 / np.sqrt(3.0)
        self._gauss_points = [
            (-gp, -gp),
            (+gp, -gp),
            (+gp, +gp),
            (-gp, +gp),
        ]
        self._gauss_weights = [1.0, 1.0, 1.0, 1.0]

        # Compute local coordinate system and project nodes
        self._local_coordinates = self._compute_local_coordinates()
        self._initial_directors = self._compute_initial_directors()
        self._current_directors = self._initial_directors.copy()

        # Caches for performance
        self._dH_cache = {}
        self._N_cache = {}
        self._jacobian_cache = {}

        # MITC4+ specific initialization
        if self.use_mitc4_plus:
            self._init_mitc4_plus()

    def _init_mitc4_plus(self) -> None:
        """
        Initialize MITC4+ specific parameters.

        Sets up the tying points and parameters for the enhanced MITC4+
        formulation from Ko, Lee & Bathe (2017).
        """
        # Tying points for MITC4+ assumed shear strain field
        # Edge midpoints (same as MITC4)
        self._tying_points_edge = {
            "A": (0.0, 1.0),  # Top edge midpoint
            "B": (-1.0, 0.0),  # Left edge midpoint
            "C": (0.0, -1.0),  # Bottom edge midpoint
            "D": (1.0, 0.0),  # Right edge midpoint
        }

        # Center tying point (new in MITC4+)
        self._tying_point_center = (0.0, 0.0)

        # Tying points for membrane assumed strain field (Ko et al. 2017)
        # e_rr sampled at A(0,+1) and B(0,-1), interpolated linearly in eta
        # e_ss sampled at C(+1,0) and D(-1,0), interpolated linearly in xi
        # e_rs sampled ONLY at E(0,0) - this removes the bilinear locking term!
        self._membrane_tying_points = {
            "A": (0.0, 1.0),   # for e_rr
            "B": (0.0, -1.0),  # for e_rr
            "C": (1.0, 0.0),   # for e_ss
            "D": (-1.0, 0.0),  # for e_ss
            "E": (0.0, 0.0),   # for e_rs (center only!)
        }

        # Precompute bubble function contributions at Gauss points
        self._bubble_cache = {}
        for xi, eta in self._gauss_points:
            Nb = self._bubble_function(xi, eta)
            dNb_dxi, dNb_deta = self._bubble_derivatives(xi, eta)
            self._bubble_cache[(xi, eta)] = (Nb, dNb_dxi, dNb_deta)

        # Precompute membrane B matrices at tying points for efficiency
        self._precompute_membrane_tying_matrices()

    # =========================================================================
    # SHAPE FUNCTIONS
    # =========================================================================

    def _shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute bilinear shape functions for quadrilateral element.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates in [-1, 1]

        Returns
        -------
        np.ndarray
            Shape functions [N0, N1, N2, N3]
        """
        N0 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N1 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N3 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return np.array([N0, N1, N2, N3])

    def _shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives of shape functions with respect to natural coordinates.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (dN_dxi, dN_deta) each of shape (4,)
        """
        # ∂N/∂ξ
        dN_dxi = np.array([
            -0.25 * (1.0 - eta),  # ∂N0/∂ξ
            +0.25 * (1.0 - eta),  # ∂N1/∂ξ
            +0.25 * (1.0 + eta),  # ∂N2/∂ξ
            -0.25 * (1.0 + eta),  # ∂N3/∂ξ
        ])

        # ∂N/∂η
        dN_deta = np.array([
            -0.25 * (1.0 - xi),  # ∂N0/∂η
            -0.25 * (1.0 + xi),  # ∂N1/∂η
            +0.25 * (1.0 + xi),  # ∂N2/∂η
            +0.25 * (1.0 - xi),  # ∂N3/∂η
        ])

        return dN_dxi, dN_deta

    # =========================================================================
    # MITC4+ BUBBLE FUNCTIONS
    # =========================================================================

    def _bubble_function(self, xi: float, eta: float) -> float:
        """
        Compute bubble function for MITC4+ rotation enrichment.

        The bubble function is Nb = (1 - ξ²)(1 - η²), which:
        - Equals zero at all corner nodes and edges
        - Has maximum value (1.0) at the element center (ξ=η=0)
        - Provides internal rotation enrichment

        This is used in MITC4+ to enrich the rotation field and improve
        the element's performance for warped geometries.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates in [-1, 1]

        Returns
        -------
        float
            Bubble function value
        """
        return (1.0 - xi * xi) * (1.0 - eta * eta)

    def _bubble_derivatives(self, xi: float, eta: float) -> Tuple[float, float]:
        """
        Compute derivatives of the bubble function.

        ∂Nb/∂ξ = -2ξ(1 - η²)
        ∂Nb/∂η = -2η(1 - ξ²)

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        Tuple[float, float]
            (dNb_dxi, dNb_deta)
        """
        dNb_dxi = -2.0 * xi * (1.0 - eta * eta)
        dNb_deta = -2.0 * eta * (1.0 - xi * xi)
        return dNb_dxi, dNb_deta

    def _bubble_second_derivatives(self, xi: float, eta: float) -> Tuple[float, float, float]:
        """
        Compute second derivatives of the bubble function.

        ∂²Nb/∂ξ² = -2(1 - η²)
        ∂²Nb/∂η² = -2(1 - ξ²)
        ∂²Nb/∂ξ∂η = 4ξη

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        Tuple[float, float, float]
            (d2Nb_dxi2, d2Nb_deta2, d2Nb_dxideta)
        """
        d2Nb_dxi2 = -2.0 * (1.0 - eta * eta)
        d2Nb_deta2 = -2.0 * (1.0 - xi * xi)
        d2Nb_dxideta = 4.0 * xi * eta
        return d2Nb_dxi2, d2Nb_deta2, d2Nb_dxideta

    # =========================================================================
    # MITC4+ ASSUMED MEMBRANE STRAIN INTERPOLATION
    # =========================================================================

    def _precompute_membrane_tying_matrices(self) -> None:
        """
        Precompute membrane strain B-matrix components at tying points.

        For MITC4+ assumed membrane strain interpolation (Ko, Lee & Bathe 2017):
        - e_rr is sampled at points A(0,+1) and B(0,-1)
        - e_ss is sampled at points C(+1,0) and D(-1,0)
        - e_rs is sampled ONLY at center E(0,0) to remove bilinear locking term
        """
        # Store covariant basis vectors and metric coefficients at tying points
        self._membrane_tying_data = {}

        for name, (xi, eta) in self._membrane_tying_points.items():
            # Get Jacobian and its inverse at this tying point
            J_mat, detJ = self.J(xi, eta)
            J_inv = np.linalg.inv(J_mat)

            # Get shape function derivatives in natural coordinates
            dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)

            # Get shape function derivatives in Cartesian coordinates
            dH = self._get_dH(xi, eta)

            # Covariant basis vectors g_r = ∂x/∂r, g_s = ∂x/∂s
            # In 2D local coordinates: g_r = [dx/dxi, dy/dxi], g_s = [dx/deta, dy/deta]
            g_r = J_mat[0, :]  # [dx/dxi, dy/dxi]
            g_s = J_mat[1, :]  # [dx/deta, dy/deta]

            self._membrane_tying_data[name] = {
                'xi': xi,
                'eta': eta,
                'J': J_mat,
                'J_inv': J_inv,
                'detJ': detJ,
                'dN_dxi': dN_dxi,
                'dN_deta': dN_deta,
                'dH': dH,
                'g_r': g_r,
                'g_s': g_s,
            }

    def _compute_covariant_membrane_strain_B(
        self, xi: float, eta: float, component: str
    ) -> np.ndarray:
        """
        Compute the covariant membrane strain-displacement relation for a specific component.

        The covariant membrane strains are:
        - e_rr = g_r · (∂u/∂r) = ∂u/∂ξ · g_r
        - e_ss = g_s · (∂u/∂s) = ∂u/∂η · g_s
        - e_rs = 0.5 * (g_r · ∂u/∂s + g_s · ∂u/∂r)

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates of tying point
        component : str
            Strain component: 'rr', 'ss', or 'rs'

        Returns
        -------
        np.ndarray
            1×24 row vector relating this strain component to nodal DOFs
        """
        J_mat, detJ = self.J(xi, eta)
        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)

        # Covariant basis vectors
        g_r = J_mat[0, :]  # [dx/dxi, dy/dxi]
        g_s = J_mat[1, :]  # [dx/deta, dy/deta]

        B_row = np.zeros(24)

        for i in range(4):
            u_idx = 6 * i      # u displacement
            v_idx = 6 * i + 1  # v displacement

            if component == 'rr':
                # e_rr = g_r · (∂u/∂ξ) = g_rx * ∂u/∂ξ + g_ry * ∂v/∂ξ
                B_row[u_idx] = g_r[0] * dN_dxi[i]
                B_row[v_idx] = g_r[1] * dN_dxi[i]

            elif component == 'ss':
                # e_ss = g_s · (∂u/∂η) = g_sx * ∂u/∂η + g_sy * ∂v/∂η
                B_row[u_idx] = g_s[0] * dN_deta[i]
                B_row[v_idx] = g_s[1] * dN_deta[i]

            elif component == 'rs':
                # e_rs = 0.5 * (g_r · ∂u/∂η + g_s · ∂u/∂ξ)
                # = 0.5 * (g_rx * ∂u/∂η + g_ry * ∂v/∂η + g_sx * ∂u/∂ξ + g_sy * ∂v/∂ξ)
                B_row[u_idx] = 0.5 * (g_r[0] * dN_deta[i] + g_s[0] * dN_dxi[i])
                B_row[v_idx] = 0.5 * (g_r[1] * dN_deta[i] + g_s[1] * dN_dxi[i])

        return B_row

    def _covariant_to_cartesian_strain_transform(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute transformation matrix from covariant to Cartesian strains.

        The transformation relates covariant strains [e_rr, e_ss, 2*e_rs] to
        Cartesian strains [e_xx, e_yy, gamma_xy] via:

        [e_xx  ]   [T11  T12  T13] [e_rr  ]
        [e_yy  ] = [T21  T22  T23] [e_ss  ]
        [gamma_xy] [T31  T32  T33] [2*e_rs]

        where the transformation depends on the Jacobian inverse.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×3 strain transformation matrix
        """
        J_mat, detJ = self.J(xi, eta)
        J_inv = np.linalg.inv(J_mat)

        # J_inv = [[∂ξ/∂x, ∂ξ/∂y], [∂η/∂x, ∂η/∂y]]
        dxi_dx = J_inv[0, 0]
        dxi_dy = J_inv[0, 1]
        deta_dx = J_inv[1, 0]
        deta_dy = J_inv[1, 1]

        # Strain transformation matrix
        # e_xx = e_rr * (∂ξ/∂x)² + e_ss * (∂η/∂x)² + 2*e_rs * (∂ξ/∂x)(∂η/∂x)
        # e_yy = e_rr * (∂ξ/∂y)² + e_ss * (∂η/∂y)² + 2*e_rs * (∂ξ/∂y)(∂η/∂y)
        # gamma_xy = 2*e_rr*(∂ξ/∂x)(∂ξ/∂y) + 2*e_ss*(∂η/∂x)(∂η/∂y) + 2*e_rs*((∂ξ/∂x)(∂η/∂y)+(∂ξ/∂y)(∂η/∂x))

        T = np.array([
            [dxi_dx**2,           deta_dx**2,           dxi_dx * deta_dx],
            [dxi_dy**2,           deta_dy**2,           dxi_dy * deta_dy],
            [2*dxi_dx*dxi_dy,     2*deta_dx*deta_dy,    dxi_dx*deta_dy + dxi_dy*deta_dx]
        ])

        return T

    def B_m_MITC4_plus(self, xi: float, eta: float) -> np.ndarray:
        """
        MITC4+ membrane strain-displacement matrix with assumed strain interpolation (3×24).

        This implements the assumed membrane strain field from Ko, Lee & Bathe (2017).
        The key innovation is that the in-plane shear strain e_rs is sampled ONLY at the
        element center, which removes the bilinear term that causes membrane locking.

        Assumed strain interpolation:
        - ẽ_rr(r,s) = (1+s)/2 * e_rr^A + (1-s)/2 * e_rr^B  (linear in eta)
        - ẽ_ss(r,s) = (1+r)/2 * e_ss^C + (1-r)/2 * e_ss^D  (linear in xi)
        - ẽ_rs(r,s) = e_rs^E  (constant - bilinear term removed!)

        Tying points:
        - A: (0, +1), B: (0, -1) for e_rr
        - C: (+1, 0), D: (-1, 0) for e_ss
        - E: (0, 0) for e_rs (center only)

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×24 membrane B matrix in local Cartesian coordinates
        """
        # Get covariant strain B-matrices at tying points
        tp = self._membrane_tying_points

        # e_rr at points A and B
        B_rr_A = self._compute_covariant_membrane_strain_B(tp["A"][0], tp["A"][1], 'rr')
        B_rr_B = self._compute_covariant_membrane_strain_B(tp["B"][0], tp["B"][1], 'rr')

        # e_ss at points C and D
        B_ss_C = self._compute_covariant_membrane_strain_B(tp["C"][0], tp["C"][1], 'ss')
        B_ss_D = self._compute_covariant_membrane_strain_B(tp["D"][0], tp["D"][1], 'ss')

        # e_rs at center point E only (this removes bilinear locking term!)
        B_rs_E = self._compute_covariant_membrane_strain_B(tp["E"][0], tp["E"][1], 'rs')

        # Interpolate covariant strains to current point (xi, eta)
        # ẽ_rr = (1+eta)/2 * e_rr^A + (1-eta)/2 * e_rr^B
        B_rr_interp = 0.5 * (1.0 + eta) * B_rr_A + 0.5 * (1.0 - eta) * B_rr_B

        # ẽ_ss = (1+xi)/2 * e_ss^C + (1-xi)/2 * e_ss^D
        B_ss_interp = 0.5 * (1.0 + xi) * B_ss_C + 0.5 * (1.0 - xi) * B_ss_D

        # ẽ_rs = e_rs^E (constant - no bilinear term!)
        B_rs_interp = B_rs_E

        # Stack covariant strain rows: [e_rr, e_ss, 2*e_rs]
        B_covariant = np.vstack([B_rr_interp, B_ss_interp, 2.0 * B_rs_interp])

        # Transform from covariant to Cartesian strains
        T = self._covariant_to_cartesian_strain_transform(xi, eta)
        B_cartesian = T @ B_covariant

        return B_cartesian

    # =========================================================================
    # LOCAL COORDINATE SYSTEM
    # =========================================================================

    def _compute_element_warping(self) -> float:
        """
        Compute warping metric for the element.

        Measures how far the element deviates from being planar by computing
        the variation in nodal normals. A value close to 0 indicates a flat
        element, while larger values indicate significant warping.

        Returns
        -------
        float
            Warping metric (0 = flat, higher = more warped)
        """
        nodes = self._current_coords

        # Compute normals at each node using adjacent triangles
        # Node ordering: 0-1-2-3 (counter-clockwise)
        normals = np.zeros((4, 3))

        # Node 0: average of triangles 3-0-1
        v01 = nodes[1] - nodes[0]
        v03 = nodes[3] - nodes[0]
        normals[0] = np.cross(v01, v03)

        # Node 1: average of triangles 0-1-2
        v12 = nodes[2] - nodes[1]
        v10 = nodes[0] - nodes[1]
        normals[1] = np.cross(v12, v10)

        # Node 2: average of triangles 1-2-3
        v23 = nodes[3] - nodes[2]
        v21 = nodes[1] - nodes[2]
        normals[2] = np.cross(v23, v21)

        # Node 3: average of triangles 2-3-0
        v30 = nodes[0] - nodes[3]
        v32 = nodes[2] - nodes[3]
        normals[3] = np.cross(v30, v32)

        # Normalize all normals
        for i in range(4):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm

        # Compute warping as maximum deviation from mean normal
        mean_normal = np.mean(normals, axis=0)
        mean_norm = np.linalg.norm(mean_normal)
        if mean_norm > 1e-12:
            mean_normal /= mean_norm

        # Warping metric: 1 - minimum dot product with mean normal
        min_dot = 1.0
        for i in range(4):
            dot = np.dot(normals[i], mean_normal)
            min_dot = min(min_dot, dot)

        return 1.0 - min_dot

    def _compute_local_coordinates(self) -> np.ndarray:
        """
        Compute local 2D coordinates of the element nodes.

        Projects the 3D element onto a local x-y plane defined by the element.
        Automatically selects between standard and robust methods based on
        element warping.

        For nearly flat elements, uses the standard midplane vector method.
        For significantly warped elements, uses averaged nodal normals for
        improved accuracy.

        Returns
        -------
        np.ndarray
            Local coordinates shape (4, 2) - [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        """
        # Check element warping to select appropriate method
        warping = self._compute_element_warping()

        # Use robust method if warping exceeds threshold (about 5 degrees deviation)
        warping_threshold = 0.004  # corresponds to ~5° normal deviation
        if warping > warping_threshold:
            return self._compute_local_coordinates_robust()
        else:
            return self._compute_local_coordinates_standard()

    def _compute_local_coordinates_standard(self) -> np.ndarray:
        """
        Standard local coordinate computation for nearly flat elements.

        Uses vectors along the ξ and η directions to define the local plane.
        This is the classic approach from OpenSees and is efficient for
        elements with minimal warping.

        Returns
        -------
        np.ndarray
            Local coordinates shape (4, 2)
        """
        nodes = self._current_coords

        # Compute in-plane vectors (similar to OpenSees computeBasis)
        # v1 = 0.5 * (coor2 + coor1 - coor3 - coor0)  -- roughly along ξ direction
        # v2 = 0.5 * (coor3 + coor2 - coor1 - coor0)  -- roughly along η direction
        v1 = 0.5 * (nodes[2] + nodes[1] - nodes[3] - nodes[0])
        v2 = 0.5 * (nodes[3] + nodes[2] - nodes[1] - nodes[0])

        # Normalize v1
        length = np.linalg.norm(v1)
        if length < 1e-12:
            raise ValueError("Degenerate element: v1 has zero length")
        e1 = v1 / length

        # Gram-Schmidt orthogonalization for v2
        alpha = np.dot(v2, e1)
        v2_orth = v2 - alpha * e1

        length = np.linalg.norm(v2_orth)
        if length < 1e-12:
            raise ValueError("Degenerate element: orthogonalized v2 has zero length")
        e2 = v2_orth / length

        # Normal vector (e3 = e1 × e2)
        e3 = np.cross(e1, e2)

        # Store basis vectors for later use
        self._e1, self._e2, self._e3 = e1, e2, e3

        # Project nodes to local coordinates
        local_coords = np.zeros((4, 2))
        for i in range(4):
            local_coords[i, 0] = np.dot(nodes[i], e1)
            local_coords[i, 1] = np.dot(nodes[i], e2)

        return local_coords

    def _compute_local_coordinates_robust(self) -> np.ndarray:
        """
        Robust local coordinate computation for warped elements.

        Uses averaged nodal normals to define the local plane, which provides
        better accuracy for elements with significant out-of-plane warping
        (e.g., twisted beam elements).

        The method computes normals at each node using adjacent edges, then
        averages them to get a representative element normal. This approach
        is more accurate for warped elements than the standard midplane method.

        Returns
        -------
        np.ndarray
            Local coordinates shape (4, 2)

        Notes
        -----
        This method is based on recommendations for MITC4 elements in warped
        configurations. It helps reduce "warped element locking" by providing
        a more representative local coordinate system.

        References
        ----------
        - Dvorkin, E.N. and Bathe, K.J. (1984). "A continuum mechanics based
          four-node shell element for general nonlinear analysis."
        - MacNeal, R.H. (1994). "Finite Elements: Their Design and Performance."
        """
        nodes = self._current_coords

        # Compute normals at each node using adjacent triangles
        # Node ordering: 0-1-2-3 (counter-clockwise)
        normals = np.zeros((4, 3))

        # Node 0: triangle 3-0-1
        v01 = nodes[1] - nodes[0]
        v03 = nodes[3] - nodes[0]
        normals[0] = np.cross(v01, v03)

        # Node 1: triangle 0-1-2
        v12 = nodes[2] - nodes[1]
        v10 = nodes[0] - nodes[1]
        normals[1] = np.cross(v12, v10)

        # Node 2: triangle 1-2-3
        v23 = nodes[3] - nodes[2]
        v21 = nodes[1] - nodes[2]
        normals[2] = np.cross(v23, v21)

        # Node 3: triangle 2-3-0
        v30 = nodes[0] - nodes[3]
        v32 = nodes[2] - nodes[3]
        normals[3] = np.cross(v30, v32)

        # Normalize each nodal normal
        for i in range(4):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm
            else:
                # Fallback: use cross product of diagonals
                d1 = nodes[2] - nodes[0]
                d2 = nodes[3] - nodes[1]
                normals[i] = np.cross(d1, d2)
                norm = np.linalg.norm(normals[i])
                if norm > 1e-12:
                    normals[i] /= norm

        # Average normal of the element (e3)
        e3 = np.mean(normals, axis=0)
        norm_e3 = np.linalg.norm(e3)
        if norm_e3 < 1e-12:
            # Fallback: use cross product of diagonals
            d1 = nodes[2] - nodes[0]
            d2 = nodes[3] - nodes[1]
            e3 = np.cross(d1, d2)
            norm_e3 = np.linalg.norm(e3)
            if norm_e3 < 1e-12:
                raise ValueError("Degenerate element: cannot determine normal")
        e3 /= norm_e3

        # Define e1 as vector from node 0 to node 1, projected onto the plane
        v01 = nodes[1] - nodes[0]
        e1 = v01 - np.dot(v01, e3) * e3
        norm_e1 = np.linalg.norm(e1)

        if norm_e1 < 1e-12:
            # Fallback: try vector from node 0 to node 2
            v02 = nodes[2] - nodes[0]
            e1 = v02 - np.dot(v02, e3) * e3
            norm_e1 = np.linalg.norm(e1)
            if norm_e1 < 1e-12:
                raise ValueError("Degenerate element: cannot determine local x-axis")
        e1 /= norm_e1

        # e2 = e3 × e1 (right-handed coordinate system)
        e2 = np.cross(e3, e1)

        # Store basis vectors for later use
        self._e1, self._e2, self._e3 = e1, e2, e3

        # Project nodes to local coordinates usando un nodo de referencia
        local_coords = np.zeros((4, 2))
        ref_node = nodes[0]  # Usar nodo 0 como referencia
        for i in range(4):
            rel_pos = nodes[i] - ref_node
            local_coords[i, 0] = np.dot(rel_pos, e1)
            local_coords[i, 1] = np.dot(rel_pos, e2)

        # Opcional: trasladar al centroide para mejor condicionamiento numérico
        center_x = np.mean(local_coords[:, 0])
        center_y = np.mean(local_coords[:, 1])
        local_coords[:, 0] -= center_x
        local_coords[:, 1] -= center_y

        return local_coords

    def _compute_initial_directors(self) -> np.ndarray:
        """
        Compute initial nodal directors (normals).
        For flat elements, directors are simply the element normal.
        """
        return np.tile(self._e3, (4, 1))

    # =========================================================================
    # JACOBIAN AND TRANSFORMATION
    # =========================================================================

    def J(self, xi: float, eta: float) -> Tuple[np.ndarray, float]:
        """
        Compute Jacobian matrix and determinant at (ξ, η).

        The Jacobian maps derivatives from natural to local Cartesian coordinates:
        J = [∂x/∂ξ  ∂y/∂ξ]
            [∂x/∂η  ∂y/∂η]

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, float]
            (J, detJ) - Jacobian matrix (2×2) and determinant
        """
        cache_key = (xi, eta)
        if cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]

        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)
        xl = self._local_coordinates

        # Jacobian components
        dx_dxi = np.dot(dN_dxi, xl[:, 0])
        dy_dxi = np.dot(dN_dxi, xl[:, 1])
        dx_deta = np.dot(dN_deta, xl[:, 0])
        dy_deta = np.dot(dN_deta, xl[:, 1])

        J_mat = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
        detJ = dx_dxi * dy_deta - dy_dxi * dx_deta

        self._jacobian_cache[cache_key] = (J_mat, detJ)
        return J_mat, detJ

    def T(self) -> np.ndarray:
        """
        Compute transformation matrix from global to local coordinates.

        Returns
        -------
        np.ndarray
            24×24 transformation matrix (block diagonal of 3×3 direction cosines)
        """
        # Build 3×3 rotation matrix
        T3 = np.vstack([self._e1, self._e2, self._e3])

        # Build 24×24 block diagonal matrix
        T = np.zeros((24, 24))
        for i in range(8):  # 4 nodes × 2 blocks (translations + rotations)
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = T3

        return T

    def _get_dH(self, xi: float, eta: float) -> np.ndarray:
        """
        Get shape function derivatives ∂N/∂x, ∂N/∂y at (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            2×4 matrix [[∂N0/∂x, ∂N1/∂x, ∂N2/∂x, ∂N3/∂x],
                        [∂N0/∂y, ∂N1/∂y, ∂N2/∂y, ∂N3/∂y]]
        """
        cache_key = (xi, eta)
        if cache_key in self._dH_cache:
            return self._dH_cache[cache_key]

        J_mat, detJ = self.J(xi, eta)
        J_inv = np.linalg.inv(J_mat)

        dN_dxi, dN_deta = self._shape_function_derivatives(xi, eta)

        # Transform to Cartesian derivatives
        # [∂N/∂x]   [J_inv]   [∂N/∂ξ]
        # [∂N/∂y] = [     ] × [∂N/∂η]
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

        dH = np.vstack([dN_dx, dN_dy])
        self._dH_cache[cache_key] = dH
        return dH

    def area(self) -> float:
        """
        Compute area of the quadrilateral element using Gauss integration.

        Returns
        -------
        float
            Element area
        """
        area = 0.0
        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            area += w * detJ
        return area

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

    def B_m(self, xi: float, eta: float) -> np.ndarray:
        """
        Membrane strain-displacement matrix (3×24).

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×24 membrane B matrix
        """
        dH = self._get_dH(xi, eta)
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

    def B_kappa(self, xi: float, eta: float) -> np.ndarray:
        """
        Bending curvature-displacement matrix (3×24).

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×24 bending B matrix
        """
        dH = self._get_dH(xi, eta)
        Bk = np.zeros((3, 24))

        for i in range(4):
            thx_idx = 6 * i + 3  # θx
            thy_idx = 6 * i + 4  # θy

            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # κxx = ∂θy/∂x (rotation θy causes curvature in x direction)
            Bk[0, thy_idx] = dNi_dx
            # κyy = -∂θx/∂y (rotation θx causes curvature in y direction, negative sign)
            Bk[1, thx_idx] = -dNi_dy
            # κxy = ∂θy/∂y - ∂θx/∂x (twist)
            Bk[2, thy_idx] = dNi_dy
            Bk[2, thx_idx] = -dNi_dx

        return Bk

    def B_gamma_MITC4(self, xi: float, eta: float) -> np.ndarray:
        """
        MITC4 transverse shear strain-displacement matrix (2×24).

        Uses assumed strain field interpolated from tying points to prevent
        shear locking. The transverse shear strains are sampled at edge
        midpoints and interpolated:
        - γxz: sampled at A(0,+1) and C(0,-1), linear in η
        - γyz: sampled at B(-1,0) and D(+1,0), linear in ξ

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            2×24 transverse shear B matrix [γxz, γyz]
        """
        xl = self._local_coordinates

        # Edge vectors in local coordinates
        dx34 = xl[2, 0] - xl[3, 0]
        dy34 = xl[2, 1] - xl[3, 1]
        dx21 = xl[1, 0] - xl[0, 0]
        dy21 = xl[1, 1] - xl[0, 1]
        dx32 = xl[2, 0] - xl[1, 0]
        dy32 = xl[2, 1] - xl[1, 1]
        dx41 = xl[3, 0] - xl[0, 0]
        dy41 = xl[3, 1] - xl[0, 1]

        # Build G matrix (4×12) - relates edge rotations to nodal DOFs (w, θx, θy)
        # Following OpenSees convention
        one_over_four = 0.25
        G = np.zeros((4, 12))

        # Edge 4-1 (connects nodes 3 and 0)
        G[0, 0] = -0.5
        G[0, 1] = -dy41 * one_over_four
        G[0, 2] = dx41 * one_over_four
        G[0, 9] = 0.5
        G[0, 10] = -dy41 * one_over_four
        G[0, 11] = dx41 * one_over_four

        # Edge 1-2 (connects nodes 0 and 1)
        G[1, 0] = -0.5
        G[1, 1] = -dy21 * one_over_four
        G[1, 2] = dx21 * one_over_four
        G[1, 3] = 0.5
        G[1, 4] = -dy21 * one_over_four
        G[1, 5] = dx21 * one_over_four

        # Edge 2-3 (connects nodes 1 and 2)
        G[2, 3] = -0.5
        G[2, 4] = -dy32 * one_over_four
        G[2, 5] = dx32 * one_over_four
        G[2, 6] = 0.5
        G[2, 7] = -dy32 * one_over_four
        G[2, 8] = dx32 * one_over_four

        # Edge 3-4 (connects nodes 2 and 3)
        G[3, 6] = 0.5
        G[3, 7] = -dy34 * one_over_four
        G[3, 8] = dx34 * one_over_four
        G[3, 9] = -0.5
        G[3, 10] = -dy34 * one_over_four
        G[3, 11] = dx34 * one_over_four

        # Geometry coefficients for transformation
        Ax = -xl[0, 0] + xl[1, 0] + xl[2, 0] - xl[3, 0]
        Bx = xl[0, 0] - xl[1, 0] + xl[2, 0] - xl[3, 0]
        Cx = -xl[0, 0] - xl[1, 0] + xl[2, 0] + xl[3, 0]

        Ay = -xl[0, 1] + xl[1, 1] + xl[2, 1] - xl[3, 1]
        By = xl[0, 1] - xl[1, 1] + xl[2, 1] - xl[3, 1]
        Cy = -xl[0, 1] - xl[1, 1] + xl[2, 1] + xl[3, 1]

        # Compute rotation angles for transformation
        alph = np.arctan2(Ay, Ax)
        beta = np.pi / 2.0 - np.arctan2(Cx, Cy)

        # Rotation matrix
        Rot = np.array([
            [np.sin(beta), -np.sin(alph)],
            [-np.cos(beta), np.cos(alph)],
        ])

        # Interpolation matrix Ms (2×4)
        Ms = np.zeros((2, 4))
        Ms[1, 0] = 1.0 - xi
        Ms[0, 1] = 1.0 - eta
        Ms[1, 2] = 1.0 + xi
        Ms[0, 3] = 1.0 + eta

        # Intermediate shear matrix Bsv = Ms * G
        Bsv = Ms @ G

        # Compute scaling factors (lengths along coordinate directions)
        _, detJ = self.J(xi, eta)

        r1_vec = np.array([Cx + xi * Bx, Cy + xi * By])
        r1 = np.linalg.norm(r1_vec)

        r2_vec = np.array([Ax + eta * Bx, Ay + eta * By])
        r2 = np.linalg.norm(r2_vec)

        # Scale the B matrix
        for j in range(12):
            Bsv[0, j] = Bsv[0, j] * r1 / (8.0 * detJ)
            Bsv[1, j] = Bsv[1, j] * r2 / (8.0 * detJ)

        # Apply rotation to get shear in local Cartesian coordinates
        Bs_12 = Rot @ Bsv

        # Expand from 12 DOFs (w, θx, θy per node) to 24 DOFs (u, v, w, θx, θy, θz per node)
        Bs = np.zeros((2, 24))
        for i in range(4):
            # Map columns: (w, θx, θy) at node i -> (u, v, w, θx, θy, θz) at node i
            # w is at position 2, θx at 3, θy at 4 in the full DOF vector
            Bs[:, 6 * i + 2] = Bs_12[:, 3 * i + 0]  # w
            Bs[:, 6 * i + 3] = Bs_12[:, 3 * i + 1]  # θx
            Bs[:, 6 * i + 4] = Bs_12[:, 3 * i + 2]  # θy

        return Bs

    def B_gamma_MITC4_plus(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        MITC4+ transverse shear strain-displacement matrix with bubble enrichment.

        This implements the enhanced assumed strain field from Ko, Lee & Bathe (2017).
        The MITC4+ formulation adds:
        1. Bubble function enrichment for the rotation field
        2. Modified interpolation that includes center point contribution
        3. Returns both nodal and bubble contributions separately for static condensation

        The enriched rotation field is:
            θ̃α = θα + Nb * αα  (α = x, y)

        where Nb = (1-ξ²)(1-η²) is the bubble function and αα are internal DOFs.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Bs_nodal, Bs_bubble) where:
            - Bs_nodal: 2×24 matrix for nodal DOFs
            - Bs_bubble: 2×2 matrix for internal bubble DOFs [α1, α2]
        """
        # Get standard MITC4 shear matrix first
        Bs_nodal = self.B_gamma_MITC4(xi, eta)

        # Now compute bubble contribution to shear
        # The bubble function only affects rotations, so shear contribution comes from:
        # γxz_bubble = Nb * α2  (α2 couples to γxz via θy contribution)
        # γyz_bubble = -Nb * α1  (α1 couples to γyz via θx contribution)

        Nb = self._bubble_function(xi, eta)

        # Build bubble contribution matrix (2×2)
        # [γxz]   [0    Nb  ] [α1]
        # [γyz] = [-Nb  0   ] [α2]
        #
        # More precisely, following the formulation where bubble enriches rotations:
        # γxz = ∂w/∂x + θy  =>  bubble contrib: Nb * α2 (from enriched θy)
        # γyz = ∂w/∂y - θx  =>  bubble contrib: -Nb * α1 (from enriched θx)

        Bs_bubble = np.zeros((2, 2))
        Bs_bubble[0, 1] = Nb  # γxz contribution from α2 (θy enrichment)
        Bs_bubble[1, 0] = -Nb  # γyz contribution from α1 (θx enrichment)

        return Bs_nodal, Bs_bubble

    def B_kappa_bubble(self, xi: float, eta: float) -> np.ndarray:
        """
        Bending curvature contribution from bubble rotation enrichment (3×2).

        For MITC4+, the enriched rotation field adds curvature terms:
            κxx_bubble = ∂(Nb*α2)/∂x
            κyy_bubble = -∂(Nb*α1)/∂y
            κxy_bubble = ∂(Nb*α2)/∂y - ∂(Nb*α1)/∂x

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×2 bending B matrix for bubble DOFs [α1, α2]
        """
        J_mat, detJ = self.J(xi, eta)
        J_inv = np.linalg.inv(J_mat)

        dNb_dxi, dNb_deta = self._bubble_derivatives(xi, eta)

        # Transform to physical coordinates
        dNb_dx = J_inv[0, 0] * dNb_dxi + J_inv[0, 1] * dNb_deta
        dNb_dy = J_inv[1, 0] * dNb_dxi + J_inv[1, 1] * dNb_deta

        Bk_bubble = np.zeros((3, 2))
        # κxx = ∂θy/∂x  => contribution from α2
        Bk_bubble[0, 1] = dNb_dx
        # κyy = -∂θx/∂y  => contribution from α1
        Bk_bubble[1, 0] = -dNb_dy
        # κxy = ∂θy/∂y - ∂θx/∂x  => contributions from both
        Bk_bubble[2, 0] = -dNb_dx  # from α1 (θx enrichment)
        Bk_bubble[2, 1] = dNb_dy  # from α2 (θy enrichment)

        return Bk_bubble

    def _compute_B_drill(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute drilling rotation B matrix (for drilling DOF stabilization).

        Based on Hughes & Brezzi (1989) drilling rotation formulation.
        The drilling strain is defined as:
            ε_drill = (∂v/∂x - ∂u/∂y)/2 - θz

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            1×24 drilling B matrix
        """
        dH = self._get_dH(xi, eta)
        N = self._shape_functions(xi, eta)

        B_drill = np.zeros(24)

        for i in range(4):
            u_idx = 6 * i
            v_idx = 6 * i + 1
            thz_idx = 6 * i + 5

            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # Drilling rotation: (∂v/∂x - ∂u/∂y)/2 - θz
            B_drill[u_idx] = -0.5 * dNi_dy
            B_drill[v_idx] = 0.5 * dNi_dx
            B_drill[thz_idx] = -N[i]

        return B_drill

    # =========================================================================
    # STIFFNESS MATRIX ASSEMBLY
    # =========================================================================

    def k_m(self) -> np.ndarray:
        """
        Membrane stiffness matrix (24×24).

        For MITC4+ (use_mitc4_plus=True), uses assumed strain interpolation
        to prevent membrane locking as described in Ko, Lee & Bathe (2017).

        Returns
        -------
        np.ndarray
            24×24 membrane stiffness matrix in local coordinates
        """
        Cm = self.Cm()
        Km = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            # Use MITC4+ assumed strain membrane B-matrix if enabled
            if self.use_mitc4_plus:
                Bm = self.B_m_MITC4_plus(xi, eta)
            else:
                Bm = self.B_m(xi, eta)
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

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            Bk = self.B_kappa(xi, eta)
            Kb += w * detJ * Bk.T @ Cb @ Bk

        return Kb

    def _k_bending_shear_mitc4_plus(self) -> np.ndarray:
        """
        Combined bending+shear stiffness with MITC4+ bubble condensation (24×24).

        This method implements the static condensation of internal bubble DOFs
        as described in Ko, Lee & Bathe (2017). The enriched element has:
        - 24 nodal DOFs (standard)
        - 2 internal bubble DOFs (α1, α2) for rotation enrichment

        The stiffness is partitioned as:
        [Knn  Knb] [un]   [fn]
        [Kbn  Kbb] [ub] = [0 ]

        where n=nodal, b=bubble. Since fb=0 (no external load on internal DOFs),
        static condensation gives:
            K_condensed = Knn - Knb @ Kbb^(-1) @ Kbn

        Returns
        -------
        np.ndarray
            24×24 condensed bending+shear stiffness matrix
        """
        Cb = self.Cb()
        Cs = self.Cs()

        # Initialize partitioned matrices
        Knn_b = np.zeros((24, 24))  # Nodal-nodal bending
        Knb_b = np.zeros((24, 2))  # Nodal-bubble bending
        Kbb_b = np.zeros((2, 2))  # Bubble-bubble bending

        Knn_s = np.zeros((24, 24))  # Nodal-nodal shear
        Knb_s = np.zeros((24, 2))  # Nodal-bubble shear
        Kbb_s = np.zeros((2, 2))  # Bubble-bubble shear

        # Full 2x2 Gauss integration for bending
        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)

            # Bending contributions
            Bk_nodal = self.B_kappa(xi, eta)
            Bk_bubble = self.B_kappa_bubble(xi, eta)

            Knn_b += w * detJ * Bk_nodal.T @ Cb @ Bk_nodal
            Knb_b += w * detJ * Bk_nodal.T @ Cb @ Bk_bubble
            Kbb_b += w * detJ * Bk_bubble.T @ Cb @ Bk_bubble

            # Shear contributions (using MITC4+ interpolation)
            Bs_nodal, Bs_bubble = self.B_gamma_MITC4_plus(xi, eta)

            Knn_s += w * detJ * Bs_nodal.T @ Cs @ Bs_nodal
            Knb_s += w * detJ * Bs_nodal.T @ Cs @ Bs_bubble
            Kbb_s += w * detJ * Bs_bubble.T @ Cs @ Bs_bubble

        # Combine bending and shear
        Knn = Knn_b + Knn_s
        Knb = Knb_b + Knb_s
        Kbb = Kbb_b + Kbb_s

        # Static condensation: K = Knn - Knb @ Kbb^(-1) @ Kbn
        # Check for well-conditioned Kbb
        try:
            Kbb_inv = np.linalg.inv(Kbb)
            K_condensed = Knn - Knb @ Kbb_inv @ Knb.T
        except np.linalg.LinAlgError:
            # Fallback to standard if condensation fails
            K_condensed = Knn

        return K_condensed

    def k_s(self) -> np.ndarray:
        """
        Transverse shear stiffness matrix (24×24) using MITC4 interpolation.

        Returns
        -------
        np.ndarray
            24×24 shear stiffness matrix in local coordinates
        """
        # For MITC4+, shear is computed together with bending in condensed form
        if self.use_mitc4_plus:
            # Return zero here; shear is included in K via _k_bending_shear_mitc4_plus
            return np.zeros((24, 24))

        # Standard MITC4: shear stiffness with reduced integration
        Cs = self.Cs()
        Ks = np.zeros((24, 24))

        # Reduced 1-point integration for shear
        xi, eta = 0.0, 0.0  # Center point
        _, detJ = self.J(xi, eta)
        Bg = self.B_gamma_MITC4(xi, eta)
        Ks = 4.0 * detJ * Bg.T @ Cs @ Bg  # Weight = 4 for reduced integration

        return Ks

    def k_drill(self) -> np.ndarray:
        """
        Drilling stiffness matrix (24×24) for stabilization.

        Uses a penalty approach based on Hughes & Brezzi (1989).
        The drilling stiffness prevents spurious zero-energy modes
        associated with the drilling DOF (θz).

        Returns
        -------
        np.ndarray
            24×24 drilling stiffness matrix in local coordinates
        """
        # Drilling stiffness coefficient (similar to OpenSees Ktt)
        # Use minimum eigenvalue of membrane tangent as suggested by OpenSees
        E = self.material.E
        nu = self.material.nu
        G = E / (2 * (1 + nu))
        # k_tt = G * self.thickness  # Drilling stiffness parameter
        k_tt = G * self.thickness**3 / 24.0

        K_drill = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            B_drill = self._compute_B_drill(xi, eta)
            K_drill += w * detJ * k_tt * np.outer(B_drill, B_drill)

        return K_drill

    @property
    def K(self) -> np.ndarray:
        """
        Global stiffness matrix (24×24).

        Combines membrane, bending, shear, and drilling stiffness matrices
        and transforms to global coordinates.

        For MITC4+ (use_mitc4_plus=True), uses the enhanced formulation with
        bubble function enrichment and static condensation of internal DOFs.

        Returns
        -------
        np.ndarray
            24×24 element stiffness matrix in global coordinates
        """
        if self.use_mitc4_plus:
            # MITC4+: Use condensed bending+shear stiffness
            K_local = self.k_m() + self._k_bending_shear_mitc4_plus() + self.k_drill()
        else:
            # Standard MITC4: Sum all components
            K_local = self.k_m() + self.k_b() + self.k_s() + self.k_drill()

        # Transform to global coordinates
        T = self.T()
        K_global = T.T @ K_local @ T

        # Ensure symmetry (numerical cleanup)
        K_global = 0.5 * (K_global + K_global.T)

        return K_global

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
        M = np.zeros((24, 24))

        # Translational and rotational inertia
        m_trans = rho * h
        m_rot = rho * h**3 / 12.0

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            N = self._shape_functions(xi, eta)

            for i in range(4):
                for j in range(4):
                    # Translational mass (u, v, w)
                    val_t = N[i] * N[j] * m_trans * w * detJ
                    for k in range(3):
                        M[6 * i + k, 6 * j + k] += val_t

                    # Rotational inertia (θx, θy, θz)
                    val_r = N[i] * N[j] * m_rot * w * detJ
                    for k in range(3, 6):
                        M[6 * i + k, 6 * j + k] += val_r

        # Transform to global coordinates
        T = self.T()
        return T.T @ M @ T

    # =========================================================================
    # GEOMETRIC NONLINEAR ANALYSIS
    # =========================================================================

    def update_configuration(self, displacements: np.ndarray) -> None:
        """
        Update element configuration with new nodal displacements.

        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (24,) in global coordinates.
        """
        if len(displacements) != 24:
            raise ValueError(f"Expected 24 DOFs, got {len(displacements)}")

        self._current_displacements = np.array(displacements, dtype=float)

        for i in range(4):
            u_vec = displacements[6 * i : 6 * i + 3]
            self._current_coords[i] = self._initial_coords[i] + u_vec

            # Update directors for rotations
            theta = displacements[6 * i + 3 : 6 * i + 6]
            if np.linalg.norm(theta) > 1e-12:
                cross_theta = np.cross(theta, self._initial_directors[i])
                self._current_directors[i] = self._initial_directors[i] + cross_theta
                self._current_directors[i] /= np.linalg.norm(self._current_directors[i])
            else:
                self._current_directors[i] = self._initial_directors[i].copy()

    def reset_configuration(self) -> None:
        """
        Reset element to initial (undeformed) configuration.
        """
        self._current_coords = self._initial_coords.copy()
        self._current_displacements = np.zeros(24)
        self._current_directors = self._initial_directors.copy()

    def get_displacement_gradient(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute displacement gradient H = ∂u/∂X at parametric point (xi, eta).

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×3 displacement gradient tensor
        """
        dH = self._get_dH(xi, eta)

        u_nodes = np.zeros((4, 3))
        for i in range(4):
            u_nodes[i, 0] = self._current_displacements[6 * i]
            u_nodes[i, 1] = self._current_displacements[6 * i + 1]
            u_nodes[i, 2] = self._current_displacements[6 * i + 2]

        H = np.zeros((3, 3))
        for j in range(2):  # x, y derivatives (flat shell)
            for comp in range(3):
                H[comp, j] = np.dot(u_nodes[:, comp], dH[j, :])

        return H

    def compute_green_lagrange_strain(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute Green-Lagrange strain tensor E at parametric point.

        E = 0.5 * (H + H^T + H^T @ H)

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            3×3 Green-Lagrange strain tensor
        """
        H = self.get_displacement_gradient(xi, eta)
        return 0.5 * (H + H.T + H.T @ H)

    def _compute_B_geometric(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute geometric B matrix for geometric stiffness computation.

        Parameters
        ----------
        xi, eta : float
            Parametric coordinates

        Returns
        -------
        np.ndarray
            8×24 geometric strain-displacement matrix
        """
        dH = self._get_dH(xi, eta)

        B_G = np.zeros((8, 24))

        for i in range(4):
            col = 6 * i
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]

            # ∂u/∂x, ∂u/∂y
            B_G[0, col] = dNi_dx
            B_G[1, col] = dNi_dy
            # ∂v/∂x, ∂v/∂y
            B_G[2, col + 1] = dNi_dx
            B_G[3, col + 1] = dNi_dy
            # ∂w/∂x, ∂w/∂y
            B_G[4, col + 2] = dNi_dx
            B_G[5, col + 2] = dNi_dy
            # θx and θy could contribute for thick shells (omitted for thin shell)
            B_G[6, col + 3] = dNi_dx
            B_G[7, col + 3] = dNi_dy

        return B_G

    def compute_geometric_stiffness(
        self,
        sigma_membrane: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute geometric (stress) stiffness matrix K_σ.

        Parameters
        ----------
        sigma_membrane : np.ndarray
            Membrane stress state [σ_xx, σ_yy, σ_xy] (in local coordinates)
        transform_to_global : bool, optional
            If True, transform result to global coordinates. Default True.

        Returns
        -------
        np.ndarray
            24×24 geometric stiffness matrix
        """
        sigma = np.asarray(sigma_membrane)

        # Build 2×2 membrane stress block
        S_m = (
            np.array([
                [sigma[0], sigma[2]],  # [σ_xx, σ_xy]
                [sigma[2], sigma[1]],  # [σ_xy, σ_yy]
            ])
            * self.thickness
        )

        # Build 8×8 block diagonal stress matrix (for 4 displacement gradients)
        S_tilde = np.zeros((8, 8))
        S_tilde[0:2, 0:2] = S_m  # For u derivatives
        S_tilde[2:4, 2:4] = S_m  # For v derivatives
        S_tilde[4:6, 4:6] = S_m  # For w derivatives
        S_tilde[6:8, 6:8] = S_m  # For rotation derivatives

        K_sigma = np.zeros((24, 24))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            B_G = self._compute_B_geometric(xi, eta)
            K_sigma += w * detJ * B_G.T @ S_tilde @ B_G

        # Force symmetry
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
        """
        Compute membrane stress from local displacement vector.

        Parameters
        ----------
        u_local : np.ndarray
            (24,) displacement vector in local coordinates
        xi, eta : float, optional
            Parametric coordinates for strain evaluation. Default (0,0) = center.

        Returns
        -------
        np.ndarray
            (3,) membrane stress [σ_xx, σ_yy, σ_xy]
        """
        E = self.material.E
        nu = self.material.nu
        factor = E / (1 - nu**2)
        C_m = factor * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

        Bm = self.B_m(xi, eta)
        epsilon_m = Bm @ u_local
        sigma_m = C_m @ epsilon_m

        return sigma_m

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute tangent stiffness matrix for nonlinear analysis.

        Parameters
        ----------
        sigma : np.ndarray, optional
            Membrane stress state. If None, computed from current displacements.
        transform_to_global : bool, optional
            Whether to transform to global coordinates. Default True.

        Returns
        -------
        np.ndarray
            24×24 tangent stiffness matrix
        """
        # Linear/Reference stiffness
        K0 = self.k_m() + self.k_b() + self.k_s() + self.k_drill()

        if not self.nonlinear:
            if transform_to_global:
                T = self.T()
                return T.T @ K0 @ T
            return K0

        # Compute stresses if not provided
        if sigma is None:
            T = self.T()
            u_local = T @ self._current_displacements
            sigma = self.compute_membrane_stress_from_displacement(u_local)

        # Geometric stiffness
        K_sigma = self.compute_geometric_stiffness(sigma, transform_to_global=False)

        K_T = K0 + K_sigma

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
        Compute internal force vector.

        Parameters
        ----------
        transform_to_global : bool, optional
            Whether to transform to global coordinates. Default True.

        Returns
        -------
        np.ndarray
            24-element internal force vector
        """
        T = self.T()
        u_local = T @ self._current_displacements

        # For linear analysis: f_int = K_local @ u_local
        K_local = self.k_m() + self.k_b() + self.k_s() + self.k_drill()
        f_int = K_local @ u_local

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

        R = f_ext - f_int

        Parameters
        ----------
        f_ext : np.ndarray
            (24,) external force vector
        transform_to_global : bool, optional
            Whether internal forces are in global coordinates. Default True.

        Returns
        -------
        np.ndarray
            (24,) residual force vector
        """
        f_int = self.compute_internal_forces(transform_to_global=transform_to_global)
        return f_ext - f_int

    def compute_strain_energy(self) -> float:
        """
        Compute total strain energy stored in the element.

        Returns
        -------
        float
            Strain energy
        """
        T = self.T()
        u_local = T @ self._current_displacements

        K_local = self.k_m() + self.k_b() + self.k_s() + self.k_drill()
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
        """
        Compute strains at a parametric point.

        Parameters
        ----------
        u_global : np.ndarray
            (24,) displacement vector in global coordinates
        xi, eta : float, optional
            Parametric coordinates. Default (0,0) = center.

        Returns
        -------
        dict
            Dictionary with membrane, bending, and shear strains
        """
        T = self.T()
        u_local = T @ u_global

        eps_membrane = self.B_m(xi, eta) @ u_local
        kappa = self.B_kappa(xi, eta) @ u_local
        gamma = self.B_gamma_MITC4(xi, eta) @ u_local

        return {
            "membrane": eps_membrane,  # [εxx, εyy, γxy]
            "curvature": kappa,  # [κxx, κyy, κxy]
            "shear": gamma,  # [γxz, γyz]
        }

    def compute_stresses(
        self,
        u_global: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict:
        """
        Compute stresses at a parametric point.

        Parameters
        ----------
        u_global : np.ndarray
            (24,) displacement vector in global coordinates
        xi, eta : float, optional
            Parametric coordinates. Default (0,0) = center.

        Returns
        -------
        dict
            Dictionary with membrane forces, bending moments, and shear forces
        """
        strains = self.compute_strains(u_global, xi, eta)

        Cm = self.Cm()
        Cb = self.Cb()
        Cs = self.Cs()

        sigma_membrane = Cm @ strains["membrane"]
        moments = Cb @ strains["curvature"]
        shear = Cs @ strains["shear"]

        return {
            "membrane": sigma_membrane,  # [Nxx, Nyy, Nxy] (force per unit length)
            "moments": moments,  # [Mxx, Myy, Mxy] (moment per unit length)
            "shear": shear,  # [Qxz, Qyz] (shear force per unit length)
        }
