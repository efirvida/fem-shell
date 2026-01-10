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

from typing import List, Optional, Tuple

import numpy as np

from fem_shell.core.material import Material, OrthotropicMaterial
from fem_shell.elements.elements import ShellElement


class MITC4(ShellElement):
    """
    MITC4+ quadrilateral shell element class with geometric nonlinear capability.

    This element formulation uses mixed interpolation of tensorial components to
    avoid shear locking, and assumed membrane strain interpolation to avoid
    membrane locking (MITC4+ formulation). Supports both linear and geometric
    nonlinear analysis (large displacements/rotations).

    Each node has 6 DOFs (3 translations + 3 rotations).

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
    shear_correction_factor : float, optional
        Shear correction factor for transverse shear stiffness
    nonlinear : bool, optional
        Enable geometric nonlinear analysis (Total Lagrangian formulation).
        Default is False (linear analysis).

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
    nonlinear : bool
        Whether geometric nonlinearity is enabled

    Notes
    -----
    **Formulation Assumptions**:

    - **Linear Elasticity**: Material behavior is linear elastic. Geometric
      nonlinearity (large displacements/rotations) is supported when 
      ``nonlinear=True``.

    - **Total Lagrangian (TL) Formulation**: For nonlinear analysis, all quantities
      are referred to the initial (undeformed) configuration. The Green-Lagrange
      strain tensor and Second Piola-Kirchhoff stress are used.

    - **Reissner-Mindlin Theory**: Suitable for thin to moderately thick shells.
      Includes transverse shear deformation effects.

    - **Shear Locking Free**: Mixed interpolation of transverse shear strains
      prevents spurious shear locking in thin shell limit.

    - **Membrane Locking Free**: MITC4+ assumed strain interpolation prevents
      membrane locking in curved shells and distorted meshes.

    **Nonlinear Analysis Methods**:
    
    - ``update_configuration(u)``: Update nodal positions with displacements
    - ``compute_tangent_stiffness()``: Get tangent stiffness K_T = K_0 + K_L + K_σ
    - ``compute_internal_forces()``: Get internal force vector f_int
    - ``compute_green_lagrange_strain()``: Get Green-Lagrange strain tensor

    The element formulation follows the MITC4+ procedure described in:

    - Ko, Y., Lee, P.S., and Bathe, K.J. (2016). "A new MITC4+ shell element."
      Computers & Structures, 182, 404-418.
    
    - Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
      geometric nonlinear analysis." Computers & Structures, 185, 1-14.
    
    - Bathe, K.J., and Dvorkin, E.N. (1985). "A four-node plate bending element
      based on Mindlin/Reissner plate theory and a mixed interpolation."
    """

    vector_form = {"U": ("Ux", "Uy", "Uz"), "θ": ("θx", "θy", "θz")}

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        material: Material,
        thickness: float,
        kx_mod: float = 1.0,
        ky_mod: float = 1.0,
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
        )

        self.element_type = "MITC4"
        self.kx_mod = kx_mod
        self.ky_mod = ky_mod
        self.nonlinear = nonlinear
        
        # Store initial (reference) configuration and current configuration
        # For Total Lagrangian formulation, reference is always the initial config
        self._initial_coords = np.array(node_coords, dtype=float).copy()
        self._current_coords = np.array(node_coords, dtype=float).copy()
        self._current_displacements = np.zeros(24)  # Current nodal displacements

        # Shear correction factor priority:
        # 1. Element constructor parameter (highest)
        # 2. Material property
        # 3. Default 5/6 (exact for isotropic rectangular sections)
        if shear_correction_factor is not None:
            self._shear_correction_factor = shear_correction_factor
        elif (
            hasattr(material, "shear_correction_factor")
            and material.shear_correction_factor is not None
        ):
            self._shear_correction_factor = material.shear_correction_factor
        else:
            self._shear_correction_factor = 5.0 / 6.0

        gp = 1 / np.sqrt(3)
        self._gauss_points = [(gp, gp), (-gp, gp), (-gp, -gp), (gp, -gp)]
        self._local_coordinates = self._compute_local_coordinates()
        self._J_cache = {}
        self._B_gamma_cache = {}
        # Performance optimization: cache derivatives and shape functions
        self._dH_cache = {}  # Cache shape function derivatives dH
        self._N_cache = {}  # Cache shape functions N
        # Pre-populate N cache during initialization for Gauss points
        for r, s in self._gauss_points:
            self._N_cache[(r, s)] = self._compute_N(r, s)

        # Setup membrane tying points for MITC+ interpolation
        # Tying points for ε_xx: 4 points along edges parallel to η-direction
        self._tying_points_eps_xx = [
            (-1.0, -gp),  # Point 1: left edge, bottom
            (-1.0, +gp),  # Point 2: left edge, top
            (+1.0, -gp),  # Point 3: right edge, bottom
            (+1.0, +gp),  # Point 4: right edge, top
        ]

        # Tying points for ε_yy: 4 points along edges parallel to ξ-direction
        self._tying_points_eps_yy = [
            (-gp, -1.0),  # Point 1: bottom edge, left
            (+gp, -1.0),  # Point 2: bottom edge, right
            (-gp, +1.0),  # Point 3: top edge, left
            (+gp, +1.0),  # Point 4: top edge, right
        ]

        # Tying points for γ_xy: 5 points (center + 4 corners)
        self._tying_points_gamma_xy = [
            (0.0, 0.0),  # Point 0: center (uses bubble function)
            (-1.0, -1.0),  # Point 1: corner 1
            (+1.0, -1.0),  # Point 2: corner 2
            (+1.0, +1.0),  # Point 3: corner 3
            (-1.0, +1.0),  # Point 4: corner 4
        ]

        # Cache for tying point evaluations
        self._eps_rr_cache = {}
        self._eps_ss_cache = {}
        self._eps_rs_cache = {}
        
        # Cache for covariant base vectors
        self._covariant_cache = {}

    def _compute_covariant_base_vectors(self, r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariant base vectors g_r and g_s at parametric point (r, s).
        
        According to Ko, Lee & Bathe (2016) "A new MITC4+ shell element", the covariant
        base vectors are defined as:
            g_r = ∂x/∂r = Σ (∂Ni/∂r) * xi
            g_s = ∂x/∂s = Σ (∂Ni/∂s) * xi
        
        where xi are the nodal coordinates in the local coordinate system.
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            g_r : (2,) covariant base vector in r-direction (local x,y components)
            g_s : (2,) covariant base vector in s-direction (local x,y components)
            
        Notes
        -----
        For a 2D membrane element in local coordinates, the base vectors are 2D.
        The z-component is zero for flat elements.
        """
        if (r, s) in self._covariant_cache:
            return self._covariant_cache[(r, s)]
        
        x1, y1, x2, y2, x3, y3, x4, y4 = self._local_coordinates
        
        # Shape function derivatives with respect to natural coordinates (r, s)
        # dN/dr = [∂N1/∂r, ∂N2/∂r, ∂N3/∂r, ∂N4/∂r]
        # dN/ds = [∂N1/∂s, ∂N2/∂s, ∂N3/∂s, ∂N4/∂s]
        dN_dr = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
        dN_ds = 0.25 * np.array([-(1 - r), -(1 + r), (1 + r), (1 - r)])
        
        x_coords = np.array([x1, x2, x3, x4])
        y_coords = np.array([y1, y2, y3, y4])
        
        # Covariant base vectors: g_r = ∂x/∂r, g_s = ∂x/∂s
        g_r = np.array([np.dot(dN_dr, x_coords), np.dot(dN_dr, y_coords)])
        g_s = np.array([np.dot(dN_ds, x_coords), np.dot(dN_ds, y_coords)])
        
        self._covariant_cache[(r, s)] = (g_r, g_s)
        return g_r, g_s

    def _compute_metric_tensor(self, r: float, s: float) -> np.ndarray:
        """
        Compute the covariant metric tensor g_ij at parametric point (r, s).
        
        The metric tensor components are:
            g_rr = g_r · g_r
            g_ss = g_s · g_s
            g_rs = g_r · g_s
            
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (2, 2) covariant metric tensor [[g_rr, g_rs], [g_rs, g_ss]]
        """
        g_r, g_s = self._compute_covariant_base_vectors(r, s)
        
        g_rr = np.dot(g_r, g_r)
        g_ss = np.dot(g_s, g_s)
        g_rs = np.dot(g_r, g_s)
        
        return np.array([[g_rr, g_rs], [g_rs, g_ss]])

    def _evaluate_B_m_covariant(self, r: float, s: float) -> np.ndarray:
        """
        Evaluate membrane strain-displacement matrix in covariant coordinates.
        
        According to Ko, Lee & Bathe (2016), the covariant membrane strains are:
            e_rr = ∂u/∂r · g_r  (strain in r-direction)
            e_ss = ∂u/∂s · g_s  (strain in s-direction)  
            e_rs = 0.5 * (∂u/∂r · g_s + ∂u/∂s · g_r)  (shear strain)
            
        where g_r and g_s are the covariant base vectors.
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (3, 8) covariant strain-displacement matrix [e_rr, e_ss, 2*e_rs]^T
            relating covariant strains to membrane DOFs [u1,v1,u2,v2,u3,v3,u4,v4]
        """
        g_r, g_s = self._compute_covariant_base_vectors(r, s)
        
        # Shape function derivatives with respect to natural coordinates
        dN_dr = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
        dN_ds = 0.25 * np.array([-(1 - r), -(1 + r), (1 + r), (1 - r)])
        
        # Build covariant B matrix (3 x 8)
        # DOF order: [u1, v1, u2, v2, u3, v3, u4, v4]
        B_cov = np.zeros((3, 8))
        
        for i in range(4):
            # e_rr = ∂u/∂r · g_r = (∂N/∂r * u) * g_r[0] + (∂N/∂r * v) * g_r[1]
            B_cov[0, 2*i] = dN_dr[i] * g_r[0]      # u contribution to e_rr
            B_cov[0, 2*i + 1] = dN_dr[i] * g_r[1]  # v contribution to e_rr
            
            # e_ss = ∂u/∂s · g_s
            B_cov[1, 2*i] = dN_ds[i] * g_s[0]      # u contribution to e_ss
            B_cov[1, 2*i + 1] = dN_ds[i] * g_s[1]  # v contribution to e_ss
            
            # 2*e_rs = ∂u/∂r · g_s + ∂u/∂s · g_r
            B_cov[2, 2*i] = dN_dr[i] * g_s[0] + dN_ds[i] * g_r[0]
            B_cov[2, 2*i + 1] = dN_dr[i] * g_s[1] + dN_ds[i] * g_r[1]
        
        return B_cov

    def _covariant_to_cartesian_strain_transform(self, r: float, s: float) -> np.ndarray:
        """
        Compute transformation matrix from covariant to Cartesian strains.
        
        According to Ko, Lee & Bathe (2016), the transformation from covariant 
        strains (e_rr, e_ss, 2*e_rs) to Cartesian strains (ε_xx, ε_yy, γ_xy) 
        involves the Jacobian matrix.
        
        The Cartesian strains are related to covariant strains by:
            [ε_xx]     [T]   [e_rr  ]
            [ε_yy]  =       [e_ss  ]
            [γ_xy]          [2*e_rs]
            
        where T is the strain transformation matrix based on the inverse Jacobian.
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (3, 3) transformation matrix from covariant to Cartesian strains
        """
        J, detJ = self.J(r, s)
        
        # Inverse Jacobian: maps from natural to local Cartesian
        # J = [∂x/∂r  ∂y/∂r]    J_inv = [∂r/∂x  ∂s/∂x]
        #     [∂x/∂s  ∂y/∂s]            [∂r/∂y  ∂s/∂y]
        J_inv = np.linalg.inv(J)
        
        # Components of inverse Jacobian
        dr_dx = J_inv[0, 0]
        ds_dx = J_inv[1, 0]
        dr_dy = J_inv[0, 1]
        ds_dy = J_inv[1, 1]
        
        # Strain transformation matrix
        # ε_xx = e_rr * (∂r/∂x)² + e_ss * (∂s/∂x)² + 2*e_rs * (∂r/∂x)(∂s/∂x)
        # ε_yy = e_rr * (∂r/∂y)² + e_ss * (∂s/∂y)² + 2*e_rs * (∂r/∂y)(∂s/∂y)
        # γ_xy = 2*e_rr*(∂r/∂x)(∂r/∂y) + 2*e_ss*(∂s/∂x)(∂s/∂y) + 2*e_rs*((∂r/∂x)(∂s/∂y) + (∂s/∂x)(∂r/∂y))
        T = np.array([
            [dr_dx**2, ds_dx**2, dr_dx * ds_dx],
            [dr_dy**2, ds_dy**2, dr_dy * ds_dy],
            [2*dr_dx*dr_dy, 2*ds_dx*ds_dy, dr_dx*ds_dy + ds_dx*dr_dy]
        ])
        
        return T

    def _get_dH(self, r: float, s: float) -> np.ndarray:
        """
        Get shape function derivatives dH at (r,s), cached for performance.

        Parameters
        ----------
        r : float
            Parametric coordinate in r direction
        s : float
            Parametric coordinate in s direction

        Returns
        -------
        np.ndarray
            2x4 matrix of shape function derivatives

        Notes
        -----
        Caches results to avoid redundant matrix inversions.
        Eliminates ~5-8% computational overhead from duplicate derivative calculations.
        """
        if (r, s) not in self._dH_cache:
            J_val, _ = self.J(r, s)
            dH = np.linalg.solve(
                J_val,
                1 / 4 * np.array([[1 + s, -1 - s, -1 + s, 1 - s], [1 + r, 1 - r, -1 + r, -1 - r]]),
            )
            self._dH_cache[(r, s)] = dH
        return self._dH_cache[(r, s)]

    def _get_N(self, r: float, s: float) -> np.ndarray:
        """
        Get shape functions N at (r,s), cached for performance.

        Parameters
        ----------
        r : float
            Parametric coordinate in r direction
        s : float
            Parametric coordinate in s direction

        Returns
        -------
        np.ndarray
            6x24 matrix of shape functions

        Notes
        -----
        Cache hit eliminates shape function computation (saves ~2-3% of M matrix assembly).
        """
        return self._N_cache.get((r, s), self._compute_N(r, s))

    def _compute_N(self, r: float, s: float) -> np.ndarray:
        """
        Compute shape functions N at (r,s) - internal method (cached externally).

        Returns
        -------
        np.ndarray
            6x24 matrix [u, v, w, θx, θy, θz]
        """
        N1 = 0.25 * (1 - r) * (1 - s)
        N2 = 0.25 * (1 + r) * (1 - s)
        N3 = 0.25 * (1 + r) * (1 + s)
        N4 = 0.25 * (1 - r) * (1 + s)
        N = np.zeros((6, 24))
        # Diagonals for each node's translations and rotations
        for i, Ni in enumerate([N1, N2, N3, N4]):
            N[0, 6 * i] = Ni  # u
            N[1, 6 * i + 1] = Ni  # v
            N[2, 6 * i + 2] = Ni  # w
            N[3, 6 * i + 3] = Ni  # θx
            N[4, 6 * i + 4] = Ni  # θy
            N[5, 6 * i + 5] = Ni  # θz
        return N

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
        if (r, s) not in self._J_cache:
            x1, y1, x2, y2, x3, y3, x4, y4 = self._local_coordinates
            J = 0.25 * np.array([
                [
                    x1 * (s + 1) - x2 * (s + 1) + x3 * (s - 1) - x4 * (s - 1),
                    y1 * (s + 1) - y2 * (s + 1) + y3 * (s - 1) - y4 * (s - 1),
                ],
                [
                    x1 * (r + 1) - x2 * (r - 1) + x3 * (r - 1) - x4 * (r + 1),
                    y1 * (r + 1) - y2 * (r - 1) + y3 * (r - 1) - y4 * (r + 1),
                ],
            ])
            detJ = np.linalg.det(J)
            self._J_cache[(r, s)] = (J, detJ)
        else:
            J, detJ = self._J_cache[(r, s)]

        return J, detJ

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

        Notes
        -----
        Uses cached shape function derivatives (_get_dH) to avoid redundant matrix inversions.
        Performance optimization: ~5-8% faster than direct computation.
        """
        dH = self._get_dH(r, s)  # ← Uses cache instead of recomputing
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

        This implements the MITC4 (Mixed Interpolation of Tensorial Components) selective
        reduced integration for shear strains, preventing shear locking for thick shells.
        Based on Dvorkin & Bathe (1984) and OpenSees ShellMITC4 implementation.

        References
        ----------
        - Dvorkin, E.N., and Bathe, K.J. (1984). "A continuum mechanics based four node
          shell element for general nonlinear analysis." Engineering with Computers, 1, 77-88.
        - OpenSees ShellMITC4: Pacific Earthquake Engineering Research Center implementation

        Parameters
        ----------
        r : float
            Parametric coordinate in ξ direction [-1, 1]
        s : float
            Parametric coordinate in η direction [-1, 1]

        Returns
        -------
        np.ndarray
            2x12 shear strain-displacement matrix relating transverse shear strains
            γ_13 and γ_23 to nodal degrees of freedom
        """
        J_val, _ = self.J(r, s)
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

    def _evaluate_B_m_at_point(self, r: float, s: float) -> np.ndarray:
        """
        Evaluate standard (non-interpolated) membrane B matrix at a single point.
        
        This method is kept for backward compatibility but is not used in
        the MITC4+ formulation. Use _evaluate_B_m_covariant instead.
        """
        J_val, _ = self.J(r, s)
        dH = np.linalg.solve(
            J_val,
            1 / 4 * np.array([[s + 1, -s - 1, s - 1, -s + 1], [r + 1, -r + 1, r - 1, -r - 1]]),
        )
        return np.array([
            [dH[0, 0], 0, dH[0, 1], 0, dH[0, 2], 0, dH[0, 3], 0],
            [0, dH[1, 0], 0, dH[1, 1], 0, dH[1, 2], 0, dH[1, 3]],
            [dH[1, 0], dH[0, 0], dH[1, 1], dH[0, 1], dH[1, 2], dH[0, 2], dH[1, 3], dH[0, 3]],
        ])

    def _get_eps_rr_at_tying_points(self) -> List[np.ndarray]:
        """
        Evaluate covariant strain e_rr at tying points for MITC4+.
        
        According to Ko, Lee & Bathe (2016), e_rr is sampled at 4 points
        on the edges r = ±1, at s = ±1/√3.
        
        Returns
        -------
        List[np.ndarray]
            List of 4 B-matrix rows (each 8,) for e_rr at each tying point
        """
        eps_rr_list = []
        for r_t, s_t in self._tying_points_eps_xx:  # r = ±1, s = ±g
            B_cov = self._evaluate_B_m_covariant(r_t, s_t)
            eps_rr_list.append(B_cov[0, :])  # First row is e_rr
        return eps_rr_list

    def _get_eps_ss_at_tying_points(self) -> List[np.ndarray]:
        """
        Evaluate covariant strain e_ss at tying points for MITC4+.
        
        According to Ko, Lee & Bathe (2016), e_ss is sampled at 4 points
        on the edges s = ±1, at r = ±1/√3.
        
        Returns
        -------
        List[np.ndarray]
            List of 4 B-matrix rows (each 8,) for e_ss at each tying point
        """
        eps_ss_list = []
        for r_t, s_t in self._tying_points_eps_yy:  # r = ±g, s = ±1
            B_cov = self._evaluate_B_m_covariant(r_t, s_t)
            eps_ss_list.append(B_cov[1, :])  # Second row is e_ss
        return eps_ss_list

    def _get_eps_rs_at_tying_points(self) -> List[np.ndarray]:
        """
        Evaluate covariant shear strain 2*e_rs at tying points for MITC4+.
        
        According to Ko, Lee & Bathe (2016), e_rs is sampled at 5 points:
        center (0,0) and 4 corners (±1, ±1).
        
        Returns
        -------
        List[np.ndarray]
            List of 5 B-matrix rows (each 8,) for 2*e_rs at each tying point
        """
        eps_rs_list = []
        for r_t, s_t in self._tying_points_gamma_xy:  # center + corners
            B_cov = self._evaluate_B_m_covariant(r_t, s_t)
            eps_rs_list.append(B_cov[2, :])  # Third row is 2*e_rs
        return eps_rs_list

    # Legacy methods for backward compatibility
    def _get_eps_xx_at_tying_points(self) -> List[np.ndarray]:
        """Legacy method - now uses covariant formulation."""
        return self._get_eps_rr_at_tying_points()

    def _get_eps_yy_at_tying_points(self) -> List[np.ndarray]:
        """Legacy method - now uses covariant formulation."""
        return self._get_eps_ss_at_tying_points()

    def _get_gamma_xy_at_tying_points(self) -> List[np.ndarray]:
        """Legacy method - now uses covariant formulation."""
        return self._get_eps_rs_at_tying_points()

    def _interpolate_eps_rr(self, r: float, s: float, eps_rr_tied: List[np.ndarray]) -> np.ndarray:
        """
        Interpolate covariant strain e_rr using MITC4+ assumed strain field.
        
        According to Ko, Lee & Bathe (2016) Eq. (39), the assumed e_rr field is:
        
        ẽ_rr(r,s) = (1-r)/2 * [(g-s)/(2g) * e_rr^A + (g+s)/(2g) * e_rr^B]
                  + (1+r)/2 * [(g-s)/(2g) * e_rr^C + (g+s)/(2g) * e_rr^D]
        
        where g = 1/√3 and A,B,C,D are tying points at (r,s) = (-1,-g), (-1,g), (1,-g), (1,g).
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
        eps_rr_tied : List[np.ndarray]
            Covariant strain rows at the 4 tying points
            
        Returns
        -------
        np.ndarray
            Interpolated B-matrix row (8,) for e_rr at (r, s)
        """
        gp = 1 / np.sqrt(3)
        
        # Interpolation weights in s-direction (linear between ±g)
        ws_minus = (gp - s) / (2 * gp)
        ws_plus = (s + gp) / (2 * gp)

        # Interpolation weights in r-direction (linear between ±1)
        wr_minus = (1 - r) / 2.0
        wr_plus = (1 + r) / 2.0

        # Bilinear interpolation: Points 0:(-1,-g), 1:(-1,g), 2:(1,-g), 3:(1,g)
        val_left = ws_minus * eps_rr_tied[0] + ws_plus * eps_rr_tied[1]
        val_right = ws_minus * eps_rr_tied[2] + ws_plus * eps_rr_tied[3]

        return wr_minus * val_left + wr_plus * val_right

    def _interpolate_eps_ss(self, r: float, s: float, eps_ss_tied: List[np.ndarray]) -> np.ndarray:
        """
        Interpolate covariant strain e_ss using MITC4+ assumed strain field.
        
        According to Ko, Lee & Bathe (2016) Eq. (40), the assumed e_ss field is:
        
        ẽ_ss(r,s) = (1-s)/2 * [(g-r)/(2g) * e_ss^E + (g+r)/(2g) * e_ss^F]
                  + (1+s)/2 * [(g-r)/(2g) * e_ss^G + (g+r)/(2g) * e_ss^H]
        
        where g = 1/√3 and E,F,G,H are tying points at (r,s) = (-g,-1), (g,-1), (-g,1), (g,1).
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
        eps_ss_tied : List[np.ndarray]
            Covariant strain rows at the 4 tying points
            
        Returns
        -------
        np.ndarray
            Interpolated B-matrix row (8,) for e_ss at (r, s)
        """
        gp = 1 / np.sqrt(3)
        
        # Interpolation weights in r-direction (linear between ±g)
        wr_minus = (gp - r) / (2 * gp)
        wr_plus = (r + gp) / (2 * gp)

        # Interpolation weights in s-direction (linear between ±1)
        ws_minus = (1 - s) / 2.0
        ws_plus = (1 + s) / 2.0

        # Bilinear interpolation: Points 0:(-g,-1), 1:(g,-1), 2:(-g,1), 3:(g,1)
        val_bottom = wr_minus * eps_ss_tied[0] + wr_plus * eps_ss_tied[1]
        val_top = wr_minus * eps_ss_tied[2] + wr_plus * eps_ss_tied[3]

        return ws_minus * val_bottom + ws_plus * val_top

    def _interpolate_eps_rs(
        self, r: float, s: float, eps_rs_tied: List[np.ndarray]
    ) -> np.ndarray:
        """
        Interpolate covariant shear strain 2*e_rs using MITC4+ bubble function.
        
        According to Ko, Lee & Bathe (2016) Eq. (41), the assumed 2*e_rs field is:
        
        2*ẽ_rs(r,s) = Σ h̄_i * (2*e_rs)^(c_i) + h̄_0 * (2*e_rs)^(c_0)
        
        where:
        - h̄_0 = (1-r²)(1-s²) is the bubble function (maximum at center, zero at edges)
        - h̄_i = h_i - (1/4)*h̄_0 are modified bilinear functions for corners
        - h_i are standard bilinear shape functions
        - c_0 is center point (0,0), c_i are corner points (±1, ±1)
        
        This can be rewritten as:
        2*ẽ_rs = Σ h_i * (2*e_rs)^(c_i) + h̄_0 * [(2*e_rs)^(c_0) - (1/4)Σ(2*e_rs)^(c_i)]
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
        eps_rs_tied : List[np.ndarray]
            Covariant shear strain rows at the 5 tying points (center + 4 corners)
            
        Returns
        -------
        np.ndarray
            Interpolated B-matrix row (8,) for 2*e_rs at (r, s)
        """
        # Bubble function: h̄_0 = (1-r²)(1-s²)
        # Vanishes on all edges, maximum value 1 at center
        h_bubble = (1 - r**2) * (1 - s**2)

        # Standard bilinear shape functions for corners
        h1 = 0.25 * (1 - r) * (1 - s)  # corner (-1, -1)
        h2 = 0.25 * (1 + r) * (1 - s)  # corner (+1, -1)
        h3 = 0.25 * (1 + r) * (1 + s)  # corner (+1, +1)
        h4 = 0.25 * (1 - r) * (1 + s)  # corner (-1, +1)

        # Bilinear interpolation of corner values
        eps_rs_bilinear = (
            h1 * eps_rs_tied[1]  # corner 1: (-1, -1)
            + h2 * eps_rs_tied[2]  # corner 2: (+1, -1)
            + h3 * eps_rs_tied[3]  # corner 3: (+1, +1)
            + h4 * eps_rs_tied[4]  # corner 4: (-1, +1)
        )
        
        # Center value and bilinear average at center (h_i = 0.25 at center)
        eps_rs_center = eps_rs_tied[0]  # center (0, 0)
        eps_rs_bilinear_center = 0.25 * (
            eps_rs_tied[1] + eps_rs_tied[2] + eps_rs_tied[3] + eps_rs_tied[4]
        )
        
        # Final MITC4+ interpolation with bubble enrichment
        return eps_rs_bilinear + h_bubble * (eps_rs_center - eps_rs_bilinear_center)

    # Legacy aliases for backward compatibility
    def _interpolate_eps_xx(self, r: float, s: float, eps_xx_tied: List[np.ndarray]) -> np.ndarray:
        """Legacy method - now uses covariant formulation."""
        return self._interpolate_eps_rr(r, s, eps_xx_tied)

    def _interpolate_eps_yy(self, r: float, s: float, eps_yy_tied: List[np.ndarray]) -> np.ndarray:
        """Legacy method - now uses covariant formulation."""
        return self._interpolate_eps_ss(r, s, eps_yy_tied)

    def _interpolate_gamma_xy(
        self, r: float, s: float, gamma_xy_tied: List[np.ndarray]
    ) -> np.ndarray:
        """Legacy method - now uses covariant formulation."""
        return self._interpolate_eps_rs(r, s, gamma_xy_tied)

    def B_m(self, r: float, s: float) -> np.ndarray:
        """
        Compute MITC4+ membrane strain-displacement matrix with assumed strain interpolation.

        This implements the full MITC4+ formulation from Ko, Lee & Bathe (2016)
        "A new MITC4+ shell element" using covariant strain interpolation and
        proper coordinate transformation.
        
        The procedure is:
        1. Evaluate covariant strains (e_rr, e_ss, 2*e_rs) at strategic tying points
        2. Interpolate covariant strains to evaluation point using assumed strain fields
        3. Transform interpolated covariant strains to Cartesian strains (ε_xx, ε_yy, γ_xy)

        This eliminates membrane locking that occurs in curved shells and distorted meshes.

        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]

        Returns
        -------
        np.ndarray
            3x8 MITC4+ membrane strain-displacement matrix in Cartesian coordinates
            
        References
        ----------
        Ko, Y., Lee, P.S., and Bathe, K.J. (2016). "A new MITC4+ shell element."
        Computers & Structures, 182, 404-418.
        """
        # Step 1: Evaluate covariant strains at tying points
        eps_rr_tied = self._get_eps_rr_at_tying_points()
        eps_ss_tied = self._get_eps_ss_at_tying_points()
        eps_rs_tied = self._get_eps_rs_at_tying_points()

        # Step 2: Interpolate covariant strains to evaluation point (r, s)
        eps_rr_interp = self._interpolate_eps_rr(r, s, eps_rr_tied)
        eps_ss_interp = self._interpolate_eps_ss(r, s, eps_ss_tied)
        eps_rs_interp = self._interpolate_eps_rs(r, s, eps_rs_tied)

        # Assemble interpolated covariant B matrix (3x8)
        B_covariant = np.array([
            eps_rr_interp,
            eps_ss_interp,
            eps_rs_interp,  # This is 2*e_rs
        ])
        
        # Step 3: Transform from covariant to Cartesian strains
        T = self._covariant_to_cartesian_strain_transform(r, s)
        
        # B_cartesian = T @ B_covariant
        # This gives [ε_xx, ε_yy, γ_xy]^T = T @ [e_rr, e_ss, 2*e_rs]^T
        B_cartesian = T @ B_covariant

        return B_cartesian

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
        where \\( k \\) is the shear correction factor (default 5/6), \\( h \\) is the thickness,
        \\( E \\) is Young's modulus, and \\( \nu \\) is Poisson's ratio.

        For orthotropic materials, the matrix is:
        \\[
        C_s = \begin{bmatrix}
        G_{13} h k & 0 \\
        0 & G_{23} h k
        \\end{bmatrix}
        \\]
        where \\( G_{13} \\) and \\( G_{23} \\) are the transverse shear moduli.

        The shear correction factor \\( k \\) can be specified at:
        1. Element construction (highest priority)
        2. Material property
        3. Default value of 5/6 (exact for isotropic rectangular sections)

        Typical values:
        - Isotropic: 5/6 ≈ 0.833
        - Unidirectional composite: 0.78-0.85
        - Quasi-isotropic laminate: 0.70-0.78
        - Sandwich with soft core: 0.15-0.40
        """
        k = self._shear_correction_factor
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
        Assemble element bending stiffness matrix with drilling stiffness.

        Returns
        -------
        np.ndarray
            24x24 bending stiffness matrix in local coordinates

        Notes
        -----
        Construction process:
        1. Numerical integration with 2x2 Gauss points
        2. Combines bending (B_kappa) and shear (B_gamma) contributions
        3. Adds drilling stiffness using theoretical B_drill matrix (OpenSees approach)
        4. Uses index mapping from index_k_b() for DOF expansion

        References
        ----------
        - OpenSees ShellMITC4: formResidAndTangent() method
        - Bathe, K.J. (1996). "Finite Element Procedures", Prentice Hall
        """
        Cb_val = self.Cb()
        Cs_val = self.Cs()

        # Initialize contributions.
        k1 = 0.0
        k2 = 0.0
        k_drill = 0.0

        # Compute drilling stiffness penalty (from OpenSees ShellMITC4::setDomain)
        # Using minimum membrane stiffness as reference for drilling penalty
        E = self.material.E if hasattr(self.material, "E") else 1.0
        nu = self.material.nu if hasattr(self.material, "nu") else 0.3
        G = E / (2 * (1 + nu))
        Ktt = G  # Drilling stiffness penalty parameter

        # Loop over the Gauss points to compute contributions.
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)
            B_kappa = self.B_kappa(r, s)
            B_gamma = self.B_gamma(r, s)
            B_drill = self._compute_B_drill(r, s)

            k1 += B_kappa.T @ Cb_val @ B_kappa * detJ
            k2 += B_gamma.T @ Cs_val @ B_gamma * detJ
            # Drilling stiffness (only affects drilling DOF θz)
            k_drill += B_drill[5, :].reshape(1, -1).T @ B_drill[5, :].reshape(1, -1) * Ktt * detJ

        # Sum the bending, shear, and drilling stiffness matrices.
        k = k1 + k2

        # Expand the stiffness matrix using precomputed index arrays.
        k_exp = np.zeros((24, 24))
        m_arr, n_arr, i_arr, j_arr = self.index_k_b()
        k_exp[m_arr, n_arr] = k[i_arr, j_arr]

        # Add drilling stiffness to drilling DOF (θz at each node)
        drilling_dofs = np.array([5, 11, 17, 23])
        for i_drill in drilling_dofs:
            # Ensure drilling stiffness is non-zero and reasonable
            drill_val = max(Ktt * self.thickness, np.min(np.abs(np.diag(k_exp)[:5])) / 100)
            k_exp[i_drill, i_drill] = max(k_exp[i_drill, i_drill], drill_val)

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
            _, detJ = self.J(r, s)
            contributions.append(B.T @ Cm_val @ B * detJ)

        # Sum the contributions and multiply by the thickness
        k_local = self.thickness * np.sum(contributions, axis=0)

        # Expand the stiffness matrix using precomputed index arrays
        m_arr, n_arr, i_arr, j_arr = self.index_k_m()
        k_expanded = np.zeros((24, 24))
        k_expanded[m_arr, n_arr] = k_local[i_arr, j_arr]

        return k_expanded

    def _is_T_orthogonal(self, tol: float = 1e-10) -> bool:
        """
        Check if transformation matrix T is orthogonal (T^T @ T = I).

        Parameters
        ----------
        tol : float
            Tolerance for orthogonality check

        Returns
        -------
        bool
            True if T is orthogonal within tolerance
        """
        T = self.T()
        orth_check = T.T @ T
        return np.allclose(orth_check, np.eye(24), atol=tol)

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
        where T is the transformation matrix constructed from the local coordinate system.

        Since T is orthogonal (direction cosine matrix), we use T.T @ K_local @ T
        instead of T^(-1) @ K_local @ T for numerical stability.
        """
        ele_K = self.k_m() + self.k_b()
        T = self.T()
        # T is orthogonal (direction cosines), so use T.T instead of inv(T)
        K = T.T @ ele_K @ T

        K = 0.5 * (K + K.T)  # Force Symmetrization
        K += 1e-10 * np.eye(self.dofs_count)  # Stabilization
        return K

    @property
    def M(self) -> np.ndarray:
        """
        Compute the mass matrix including all 6 DOFs per node.

        The mass matrix includes both translational and rotational inertia contributions.
        All rotational inertia terms use the consistent moment of inertia for a uniform
        plate: I = ρ*h³/12 for both in-plane (θx, θy) and drilling (θz) rotations.

        Returns
        -------
        np.ndarray
            24x24 mass matrix in global coordinates, positive semi-definite

        Notes
        -----
        Translational mass per unit area: ρ*h
        Rotational inertia per unit area: ρ*h³/12 (consistent with Mindlin plate theory)
        Uses cached shape functions (_get_N) for 2-3% performance improvement.
        """
        rho = self.material.rho
        h = self.thickness
        assert rho > 0 and h > 0, "Material properties must be positive"

        M_local = np.zeros((24, 24))

        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)
            N = self._get_N(r, s)  # ← Uses cache instead of computing shape_functions

            # Translational mass (u, v, w)
            M_trans = rho * h * (N[:3].T @ N[:3]) * detJ

            # Rotational inertia (consistent for all rotations: θx, θy, θz)
            # Using moment of inertia of uniform plate: I = ρ*h³/12
            I_rot = (rho * h**3) / 12

            # Assemble rotational contributions
            M_rot = np.zeros((24, 24))

            # Contributions from θx, θy, θz (all use same I_rot for consistency)
            for dof in [3, 4, 5]:  # θx (dof=3), θy (dof=4), θz (dof=5)
                M_rot += I_rot * np.outer(N[dof], N[dof]) * detJ

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

    def _compute_B_drill(self, r: float, s: float) -> np.ndarray:
        """
        Compute drilling (membrane rotation) strain-displacement matrix.

        Implements the drilling DOF formulation from OpenSees ShellMITC4.
        The drilling strain relates the rotation θz to in-plane displacement derivatives.

        Based on OpenSees implementation (ShellMITC4.cpp, computeBdrill method).

        Parameters
        ----------
        r : float
            Parametric coordinate
        s : float
            Parametric coordinate

        Returns
        -------
        np.ndarray
            6x12 matrix relating drilling DOF to displacement derivatives
        """
        J_val, _ = self.J(r, s)
        x1, y1, x2, y2, x3, y3, x4, y4 = self._local_coordinates

        # Shape function derivatives with respect to local coordinates
        dH = np.linalg.solve(
            J_val,
            1 / 4 * np.array([[s + 1, -s - 1, s - 1, -s + 1], [r + 1, -r + 1, r - 1, -r - 1]]),
        )

        # Drilling B matrix (following OpenSees ShellMITC4::computeBdrill)
        # B_drill relates ω = 0.5*(∂v/∂x - ∂u/∂y) to nodal displacements
        B_drill = np.zeros((6, 12))

        for i in range(4):
            # u, v components for each node
            B_drill[0, 3 * i] = -0.5 * dH[1, i]  # ∂u/∂y term
            B_drill[0, 3 * i + 1] = +0.5 * dH[0, i]  # ∂v/∂x term

            # Rotation contributions (drilling effect on rotations)
            B_drill[3, 3 * i] = -dH[1, i]  # ∂²u/∂y²
            B_drill[4, 3 * i + 1] = -dH[0, i]  # ∂²v/∂x²
            B_drill[5, 3 * i] = -dH[1, i]  # θz component from ∂u/∂y
            B_drill[5, 3 * i + 1] = +dH[0, i]  # θz component from ∂v/∂x

        return B_drill

    def body_load(self, body_force: np.ndarray) -> np.ndarray:
        """
        Vector de carga actualizado para 6 GDL.
        """
        f = np.zeros(24)

        for r, s in self._gauss_points:
            _, det_J = self.J(r, s)
            N = self.shape_functions(r, s)[:3]  # Solo componentes de desplazamiento

            f += (N.T @ body_force[:3]) * det_J * self.thickness

        return f

    def validate_element(self, verbose: bool = False) -> bool:
        """
        Validate element geometric and numerical properties.

        Checks:
        - Jacobian is strictly positive at all Gauss points
        - Element aspect ratio is reasonable (< 1000)
        - Stiffness matrix is positive semi-definite
        - Mass matrix is positive semi-definite

        Parameters
        ----------
        verbose : bool
            Print validation results

        Returns
        -------
        bool
            True if element is valid, False otherwise
        """
        try:
            # Check Jacobian at all Gauss points
            min_detJ = float("inf")
            for r, s in self._gauss_points:
                _, detJ = self.J(r, s)
                min_detJ = min(min_detJ, detJ)
                if detJ <= 1e-12:
                    if verbose:
                        print(f"ERROR: Non-positive Jacobian at ({r:.4f}, {s:.4f}): detJ={detJ}")
                    return False

            # Check aspect ratio (max/min edge length)
            edges = [
                np.linalg.norm(self.node_coords[1] - self.node_coords[0]),
                np.linalg.norm(self.node_coords[2] - self.node_coords[1]),
                np.linalg.norm(self.node_coords[3] - self.node_coords[2]),
                np.linalg.norm(self.node_coords[0] - self.node_coords[3]),
            ]
            aspect_ratio = max(edges) / min(edges)
            if aspect_ratio > 1000:
                if verbose:
                    print(f"WARNING: High aspect ratio: {aspect_ratio:.2f}")

            # Check stiffness matrix positive semi-definite (allow tiny negative noise)
            K = self.K
            eigs_K = np.linalg.eigvalsh(K)
            tol_K = max(1e-6, 1e-12 * np.max(np.abs(eigs_K)))
            if np.any(eigs_K < -tol_K):
                if verbose:
                    print(
                        "ERROR: Stiffness matrix not positive semi-definite. "
                        f"Min eigenvalue: {eigs_K[0]:.6e}, tol: {tol_K:.2e}"
                    )
                return False

            # Check mass matrix positive semi-definite (allow tiny negative noise)
            M = self.M
            eigs_M = np.linalg.eigvalsh(M)
            tol_M = max(1e-8, 1e-12 * np.max(np.abs(eigs_M)))
            if np.any(eigs_M < -tol_M):
                if verbose:
                    print(
                        "ERROR: Mass matrix not positive semi-definite. "
                        f"Min eigenvalue: {eigs_M[0]:.6e}, tol: {tol_M:.2e}"
                    )
                return False

            if verbose:
                print("Element validation OK:")
                print(f"  Min Jacobian: {min_detJ:.6e}")
                print(f"  Aspect ratio: {aspect_ratio:.2f}")
                print(f"  K eigenvalues: [{eigs_K[0]:.3e}, ..., {eigs_K[-1]:.3e}]")
                print(f"  M eigenvalues: [{eigs_M[0]:.3e}, ..., {eigs_M[-1]:.3e}]")

            return True

        except Exception as e:
            if verbose:
                print(f"Validation error: {str(e)}")
            return False

    def compute_geometric_stiffness(
        self,
        sigma_membrane: np.ndarray,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute the geometric stiffness matrix (stress stiffening) for the shell element.

        This method implements the geometric stiffness matrix K_G arising from membrane
        prestress, which captures the stress-stiffening effect. The formulation follows
        Ko, Lee & Bathe (2017) "The MITC4+ shell element in geometric nonlinear analysis".

        The geometric stiffness accounts for the effect of in-plane (membrane) stresses
        on out-of-plane (bending) stiffness. This is essential for:
        - Rotating structures (centrifugal stiffening)
        - Prestressed structures
        - Buckling analysis
        - Large rotation problems

        Mathematical formulation:
        K_G = ∫_A B_G^T · S_m · B_G · dA

        where:
        - B_G: Geometric strain-displacement matrix (relates displacement gradients
               to nodal DOFs)
        - S_m: Membrane stress matrix in block diagonal form:
               S_m = diag([σ], [σ], [σ]) where [σ] = [[σ_xx, σ_xy], [σ_xy, σ_yy]]

        Parameters
        ----------
        sigma_membrane : np.ndarray
            Membrane stress tensor at element centroid, can be:
            - Shape (3,): Voigt notation [σ_xx, σ_yy, σ_xy] (Pa)
            - Shape (2, 2): Full tensor [[σ_xx, σ_xy], [σ_xy, σ_yy]] (Pa)
        transform_to_global : bool, optional
            If True, transform result to global coordinates. Default True.

        Returns
        -------
        np.ndarray
            24x24 geometric stiffness matrix

        Notes
        -----
        The geometric stiffness matrix is symmetric and can be positive or negative
        depending on the stress state:
        - Tensile stresses (σ > 0): K_G is positive → increases effective stiffness
        - Compressive stresses (σ < 0): K_G is negative → decreases effective stiffness

        For rotating blades, centrifugal forces create tensile membrane stresses that
        stiffen the structure, raising natural frequencies.

        References
        ----------
        - Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
          geometric nonlinear analysis." Computers & Structures, 185, 1-14.
        - Bathe, K.J. (1996). "Finite Element Procedures", Prentice Hall, Chapter 6.
        - Cook, R.D. et al. (2001). "Concepts and Applications of Finite Element
          Analysis", 4th ed., Section 18.2.

        Examples
        --------
        >>> # Centrifugal stress from rotation
        >>> omega = 1.5  # rad/s
        >>> rho = 1500  # kg/m³
        >>> r_avg = 50  # m (average radius)
        >>> sigma_cf = rho * omega**2 * r_avg  # Centrifugal stress (Pa)
        >>> sigma_membrane = np.array([sigma_cf, 0, 0])  # Tension in radial direction
        >>> K_G = element.compute_geometric_stiffness(sigma_membrane)
        """
        # Convert stress to 2x2 tensor form if given in Voigt notation
        if sigma_membrane.shape == (3,):
            sigma_xx, sigma_yy, sigma_xy = sigma_membrane
            S_2x2 = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
        elif sigma_membrane.shape == (2, 2):
            S_2x2 = sigma_membrane
        else:
            raise ValueError(
                f"sigma_membrane must have shape (3,) or (2,2), got {sigma_membrane.shape}"
            )

        # Initialize geometric stiffness matrix (24x24)
        K_G = np.zeros((24, 24))

        # 2x2 Gauss integration
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)

            # Compute geometric B matrix at this Gauss point
            B_G = self._compute_B_geometric(r, s)

            # Build block diagonal stress matrix S_m (6x6)
            # S_m = diag([S_2x2], [S_2x2], [S_2x2])
            # This accounts for all 3 displacement components (u, v, w)
            S_m = np.zeros((6, 6))
            S_m[0:2, 0:2] = S_2x2  # u-displacement block
            S_m[2:4, 2:4] = S_2x2  # v-displacement block
            S_m[4:6, 4:6] = S_2x2  # w-displacement block

            # Integrate: K_G += B_G^T · S_m · B_G · detJ · thickness
            K_G += self.thickness * (B_G.T @ S_m @ B_G) * detJ

        # Force symmetry (numerical precision)
        K_G = 0.5 * (K_G + K_G.T)

        if transform_to_global:
            T = self.T()
            K_G = T.T @ K_G @ T

        return K_G

    def _compute_B_geometric(self, r: float, s: float) -> np.ndarray:
        """
        Compute the geometric strain-displacement matrix B_G.

        This matrix relates the displacement gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, etc.)
        to the nodal degrees of freedom. It's used in the geometric stiffness
        formulation to capture the nonlinear strain terms.

        The structure of B_G is:
        [∂u/∂x]     [dN1/dx,   0,      0,    0, 0, 0, dN2/dx, ...]   [u1]
        [∂u/∂y]     [dN1/dy,   0,      0,    0, 0, 0, dN2/dy, ...]   [v1]
        [∂v/∂x]  =  [  0,   dN1/dx,    0,    0, 0, 0,   0,    ...]   [w1]
        [∂v/∂y]     [  0,   dN1/dy,    0,    0, 0, 0,   0,    ...]   [θx1]
        [∂w/∂x]     [  0,      0,   dN1/dx,  0, 0, 0,   0,    ...]   [θy1]
        [∂w/∂y]     [  0,      0,   dN1/dy,  0, 0, 0,   0,    ...]   [θz1]
                                                                      ...

        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]

        Returns
        -------
        np.ndarray
            6x24 geometric strain-displacement matrix
        """
        # Get shape function derivatives in local coordinates
        dH = self._get_dH(r, s)  # 2x4 matrix: [dN/dx; dN/dy] for 4 nodes

        # Build B_G matrix (6 rows × 24 cols)
        # 6 rows: [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y]
        # 24 cols: 4 nodes × 6 DOFs = [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,...]
        B_G = np.zeros((6, 24))

        for i in range(4):  # Loop over 4 nodes
            base_col = 6 * i  # Starting column for node i

            dNi_dx = dH[0, i]  # ∂Ni/∂x
            dNi_dy = dH[1, i]  # ∂Ni/∂y

            # ∂u/∂x and ∂u/∂y (row 0, 1) → u DOF (column base_col + 0)
            B_G[0, base_col + 0] = dNi_dx  # ∂u/∂x
            B_G[1, base_col + 0] = dNi_dy  # ∂u/∂y

            # ∂v/∂x and ∂v/∂y (row 2, 3) → v DOF (column base_col + 1)
            B_G[2, base_col + 1] = dNi_dx  # ∂v/∂x
            B_G[3, base_col + 1] = dNi_dy  # ∂v/∂y

            # ∂w/∂x and ∂w/∂y (row 4, 5) → w DOF (column base_col + 2)
            B_G[4, base_col + 2] = dNi_dx  # ∂w/∂x
            B_G[5, base_col + 2] = dNi_dy  # ∂w/∂y

        return B_G

    def compute_membrane_stress_from_displacement(
        self, u_local: np.ndarray, r: float = 0.0, s: float = 0.0
    ) -> np.ndarray:
        """
        Compute membrane stress at a point given nodal displacements.

        This is a utility method to compute the membrane stress tensor from
        the current displacement state, which can then be used as input to
        compute_geometric_stiffness().

        Parameters
        ----------
        u_local : np.ndarray
            Nodal displacement vector in local coordinates (24,)
        r : float, optional
            Parametric coordinate, default 0.0 (center)
        s : float, optional
            Parametric coordinate, default 0.0 (center)

        Returns
        -------
        np.ndarray
            Membrane stress in Voigt notation [σ_xx, σ_yy, σ_xy] (Pa)

        Notes
        -----
        σ = C_m · ε_m where ε_m = B_m · u_membrane
        """
        # Extract membrane DOFs from full displacement vector
        # Membrane DOFs: u, v for each node → positions [0,1,6,7,12,13,18,19]
        membrane_dof_indices = np.array([0, 1, 6, 7, 12, 13, 18, 19])
        u_membrane = u_local[membrane_dof_indices]

        # Compute membrane strain
        B_m = self.B_m(r, s)  # 3x8
        epsilon_m = B_m @ u_membrane  # [ε_xx, ε_yy, γ_xy]

        # Compute stress using constitutive matrix
        C_m = self.Cm()  # 3x3
        sigma_m = C_m @ epsilon_m  # [σ_xx, σ_yy, σ_xy]

        return sigma_m

    def compute_centrifugal_prestress(
        self,
        omega: float,
        rotation_axis: np.ndarray,
        rotation_center: np.ndarray,
    ) -> np.ndarray:
        """
        Compute membrane prestress from centrifugal loading for a rotating element.

        This calculates the membrane stress state caused by centrifugal forces,
        which is the primary source of geometric stiffening in rotating structures
        like wind turbine blades.

        The centrifugal stress in a rotating blade varies with position:
        σ_rr(r) ≈ ρω²∫_r^R r' dr' = (ρω²/2)(R² - r²)

        For simplicity, this method computes the stress at the element centroid
        using a simplified radial stress model.

        Parameters
        ----------
        omega : float
            Angular velocity (rad/s)
        rotation_axis : np.ndarray
            Unit vector defining rotation axis (3,)
        rotation_center : np.ndarray
            Point on rotation axis (3,)

        Returns
        -------
        np.ndarray
            Membrane stress in Voigt notation [σ_xx, σ_yy, σ_xy] (Pa)

        Notes
        -----
        The returned stress is in the local coordinate system of the element.
        For accurate results in complex geometries, the stress should be computed
        from a static analysis with centrifugal body forces.

        This simplified method is suitable for:
        - Quick estimates of stress stiffening effects
        - Initialization of nonlinear analysis
        - Validation and testing
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

    # =========================================================================
    # GEOMETRIC NONLINEAR ANALYSIS METHODS
    # Following Ko, Lee & Bathe (2017) "The MITC4+ shell element in geometric
    # nonlinear analysis" - Total Lagrangian formulation
    # =========================================================================

    def update_configuration(self, displacements: np.ndarray) -> None:
        """
        Update element configuration with new nodal displacements.
        
        This method updates the current nodal positions based on displacement
        increments. Used in nonlinear analysis to track the deformed configuration.
        
        Parameters
        ----------
        displacements : np.ndarray
            Nodal displacement vector (24,) in global coordinates.
            Order: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2, ...]
            
        Notes
        -----
        For Total Lagrangian formulation, the reference configuration remains
        the initial undeformed state. The current coordinates are computed as:
        
            x_current = x_initial + u
            
        The local coordinate system and cached values are NOT updated here
        since TL formulation uses the initial configuration for all derivatives.
        
        For Updated Lagrangian (UL) formulation, you would need to recompute
        the local coordinates and clear all caches after each update.
        """
        if len(displacements) != 24:
            raise ValueError(f"Expected 24 DOFs, got {len(displacements)}")
        
        self._current_displacements = np.array(displacements, dtype=float)
        
        # Extract translational DOFs and update current coordinates
        for i in range(4):
            u = displacements[6*i]      # x-displacement
            v = displacements[6*i + 1]  # y-displacement
            w = displacements[6*i + 2]  # z-displacement
            self._current_coords[i] = self._initial_coords[i] + np.array([u, v, w])

    def reset_configuration(self) -> None:
        """
        Reset element to initial (undeformed) configuration.
        
        Clears all displacement history and restores initial coordinates.
        """
        self._current_coords = self._initial_coords.copy()
        self._current_displacements = np.zeros(24)

    def get_displacement_gradient(self, r: float, s: float) -> np.ndarray:
        """
        Compute displacement gradient tensor F - I at parametric point (r, s).
        
        The displacement gradient H = ∂u/∂X relates incremental displacements
        to the reference configuration. For TL formulation:
        
            H_ij = ∂u_i/∂X_j
            
        The deformation gradient is F = I + H.
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (3, 3) displacement gradient tensor H = ∂u/∂X
        """
        # Shape function derivatives w.r.t. local Cartesian coordinates
        dH = self._get_dH(r, s)  # 2x4: [dN/dx; dN/dy]
        
        # Extract translational displacements
        u_nodes = np.zeros((4, 3))
        for i in range(4):
            u_nodes[i, 0] = self._current_displacements[6*i]      # u
            u_nodes[i, 1] = self._current_displacements[6*i + 1]  # v
            u_nodes[i, 2] = self._current_displacements[6*i + 2]  # w
        
        # Compute displacement gradient (in-plane components)
        # H[i,j] = Σ_k (dN_k/dX_j) * u_k_i
        H = np.zeros((3, 3))
        
        # ∂u/∂x, ∂v/∂x, ∂w/∂x
        H[0, 0] = np.dot(dH[0, :], u_nodes[:, 0])  # ∂u/∂x
        H[1, 0] = np.dot(dH[0, :], u_nodes[:, 1])  # ∂v/∂x
        H[2, 0] = np.dot(dH[0, :], u_nodes[:, 2])  # ∂w/∂x
        
        # ∂u/∂y, ∂v/∂y, ∂w/∂y
        H[0, 1] = np.dot(dH[1, :], u_nodes[:, 0])  # ∂u/∂y
        H[1, 1] = np.dot(dH[1, :], u_nodes[:, 1])  # ∂v/∂y
        H[2, 1] = np.dot(dH[1, :], u_nodes[:, 2])  # ∂w/∂y
        
        # For shell elements, z-derivatives are handled through thickness integration
        # H[:, 2] remains zero for membrane behavior
        
        return H

    def compute_green_lagrange_strain(
        self, r: float = 0.0, s: float = 0.0
    ) -> np.ndarray:
        """
        Compute Green-Lagrange strain tensor at a parametric point.
        
        The Green-Lagrange strain tensor E is defined as:
        
            E = 0.5 * (F^T F - I) = 0.5 * (H + H^T + H^T H)
            
        where F = I + H is the deformation gradient.
        
        This strain measure is work-conjugate to the Second Piola-Kirchhoff
        stress and is valid for large displacements and rotations.
        
        Parameters
        ----------
        r : float, optional
            Parametric coordinate in ξ-direction, default 0 (center)
        s : float, optional
            Parametric coordinate in η-direction, default 0 (center)
            
        Returns
        -------
        np.ndarray
            (3, 3) Green-Lagrange strain tensor E
            
        Notes
        -----
        The strain tensor has the form:
            [E_xx  E_xy  E_xz]
            [E_xy  E_yy  E_yz]
            [E_xz  E_yz  E_zz]
            
        For membrane behavior of shells, E_xz, E_yz, E_zz are typically small.
        """
        H = self.get_displacement_gradient(r, s)
        
        # Green-Lagrange strain: E = 0.5 * (H + H^T + H^T @ H)
        E = 0.5 * (H + H.T + H.T @ H)
        
        return E

    def compute_green_lagrange_strain_voigt(
        self, r: float = 0.0, s: float = 0.0
    ) -> np.ndarray:
        """
        Compute Green-Lagrange strain in Voigt notation.
        
        Parameters
        ----------
        r : float, optional
            Parametric coordinate in ξ-direction, default 0 (center)
        s : float, optional
            Parametric coordinate in η-direction, default 0 (center)
            
        Returns
        -------
        np.ndarray
            (6,) strain vector [E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_xz]
            
        Notes
        -----
        Engineering shear strains (γ = 2E) are used in Voigt notation.
        """
        E = self.compute_green_lagrange_strain(r, s)
        
        return np.array([
            E[0, 0],          # E_xx
            E[1, 1],          # E_yy
            E[2, 2],          # E_zz
            2 * E[0, 1],      # γ_xy = 2*E_xy
            2 * E[1, 2],      # γ_yz = 2*E_yz
            2 * E[0, 2],      # γ_xz = 2*E_xz
        ])

    def _compute_B_L(self, r: float, s: float) -> np.ndarray:
        """
        Compute the linear part of the strain-displacement matrix for TL formulation.
        
        This is the standard B matrix evaluated at the current configuration,
        relating infinitesimal strain increments to displacement increments:
        
            δε = B_L · δu
            
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (6, 24) linear strain-displacement matrix
        """
        dH = self._get_dH(r, s)  # 2x4: [dN/dx; dN/dy]
        
        # Build B_L matrix (6 strain components x 24 DOFs)
        # Strain order: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]
        B_L = np.zeros((6, 24))
        
        for i in range(4):
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
            B_L[3, col] = dNi_dy      # ∂u/∂y
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
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (6, 24) nonlinear strain-displacement matrix
            
        Notes
        -----
        B_NL depends on the current displacement state and must be
        recomputed after each configuration update.
        """
        dH_shape = self._get_dH(r, s)  # 2x4: [dN/dx; dN/dy]
        H = self.get_displacement_gradient(r, s)  # 3x3 displacement gradient
        
        # Build B_NL matrix
        # The nonlinear strain terms involve products of displacement gradients
        B_NL = np.zeros((6, 24))
        
        for i in range(4):
            col = 6 * i
            dNi_dx = dH_shape[0, i]
            dNi_dy = dH_shape[1, i]
            
            # E_xx nonlinear: 0.5 * [(∂u/∂x)² + (∂v/∂x)² + (∂w/∂x)²]
            # δE_xx = (∂u/∂x)·δ(∂u/∂x) + (∂v/∂x)·δ(∂v/∂x) + (∂w/∂x)·δ(∂w/∂x)
            B_NL[0, col] = H[0, 0] * dNi_dx      # (∂u/∂x) * dNi/dx
            B_NL[0, col + 1] = H[1, 0] * dNi_dx  # (∂v/∂x) * dNi/dx
            B_NL[0, col + 2] = H[2, 0] * dNi_dx  # (∂w/∂x) * dNi/dx
            
            # E_yy nonlinear: 0.5 * [(∂u/∂y)² + (∂v/∂y)² + (∂w/∂y)²]
            B_NL[1, col] = H[0, 1] * dNi_dy      # (∂u/∂y) * dNi/dy
            B_NL[1, col + 1] = H[1, 1] * dNi_dy  # (∂v/∂y) * dNi/dy
            B_NL[1, col + 2] = H[2, 1] * dNi_dy  # (∂w/∂y) * dNi/dy
            
            # E_zz nonlinear: 0 for membrane
            
            # 2*E_xy nonlinear: (∂u/∂x)(∂u/∂y) + (∂v/∂x)(∂v/∂y) + (∂w/∂x)(∂w/∂y)
            B_NL[3, col] = H[0, 0] * dNi_dy + H[0, 1] * dNi_dx
            B_NL[3, col + 1] = H[1, 0] * dNi_dy + H[1, 1] * dNi_dx
            B_NL[3, col + 2] = H[2, 0] * dNi_dy + H[2, 1] * dNi_dx
        
        return B_NL

    def _compute_G_matrix(self, r: float, s: float) -> np.ndarray:
        """
        Compute the G matrix for geometric stiffness in nonlinear analysis.
        
        The G matrix relates displacement gradient variations to nodal DOF
        variations. Used in the geometric stiffness computation:
        
            K_σ = ∫ G^T · S̃ · G dA
            
        where S̃ is the stress matrix in appropriate form.
        
        Parameters
        ----------
        r : float
            Parametric coordinate in ξ-direction [-1, 1]
        s : float
            Parametric coordinate in η-direction [-1, 1]
            
        Returns
        -------
        np.ndarray
            (4, 24) G matrix for geometric stiffness
        """
        dH = self._get_dH(r, s)  # 2x4
        
        # G matrix structure: relates [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ...] to DOFs
        # For membrane: 4 gradient components (du/dx, du/dy, dv/dx, dv/dy)
        # Extended for shell to include w gradients
        G = np.zeros((6, 24))  # [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y]
        
        for i in range(4):
            col = 6 * i
            dNi_dx = dH[0, i]
            dNi_dy = dH[1, i]
            
            # ∂u/∂x, ∂u/∂y
            G[0, col] = dNi_dx
            G[1, col] = dNi_dy
            
            # ∂v/∂x, ∂v/∂y
            G[2, col + 1] = dNi_dx
            G[3, col + 1] = dNi_dy
            
            # ∂w/∂x, ∂w/∂y
            G[4, col + 2] = dNi_dx
            G[5, col + 2] = dNi_dy
        
        return G

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute tangent stiffness matrix for nonlinear analysis.
        
        The tangent stiffness matrix is:
        
            K_T = K_0 + K_L + K_σ
            
        where:
        - K_0: Initial (linear) stiffness matrix
        - K_L: Large displacement stiffness (from nonlinear strain terms)
        - K_σ: Geometric (stress) stiffness matrix
        
        Parameters
        ----------
        sigma : np.ndarray, optional
            Current stress state for geometric stiffness. If None, computed
            from current displacements. Shape (3,) for membrane stress
            [σ_xx, σ_yy, σ_xy] or (2, 2) tensor.
        transform_to_global : bool, optional
            Transform result to global coordinates. Default True.
            
        Returns
        -------
        np.ndarray
            (24, 24) tangent stiffness matrix
            
        Notes
        -----
        For linear analysis (self.nonlinear = False), this returns just K_0.
        
        The tangent stiffness is used in Newton-Raphson iteration:
            K_T · Δu = f_ext - f_int
        """
        # Get linear stiffness (always needed)
        K_0 = self.k_m() + self.k_b()
        
        if not self.nonlinear:
            if transform_to_global:
                T = self.T()
                return T.T @ K_0 @ T
            return K_0
        
        # Compute stress if not provided
        if sigma is None:
            # Transform displacements to local coordinates
            T = self.T()
            u_local = T @ self._current_displacements
            sigma = self.compute_membrane_stress_from_displacement(u_local)
        
        # Large displacement stiffness K_L
        K_L = np.zeros((24, 24))
        
        # Get constitutive matrices
        Cm = self.Cm()  # 3x3 membrane
        
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)
            
            # Nonlinear B matrix contribution
            B_NL = self._compute_B_NL(r, s)
            B_L = self._compute_B_L(r, s)
            
            # Extract membrane parts (first 4 components: xx, yy, zz, xy)
            B_m_L = B_L[[0, 1, 3], :][:, [0, 1, 6, 7, 12, 13, 18, 19]]
            B_m_NL = B_NL[[0, 1, 3], :][:, [0, 1, 6, 7, 12, 13, 18, 19]]
            
            # K_L contribution (cross terms between linear and nonlinear B)
            # This captures the coupling between linear and nonlinear strain
            k_L_local = (B_m_L.T @ Cm @ B_m_NL + B_m_NL.T @ Cm @ B_m_L) * detJ * self.thickness
            
            # Map back to full 24x24
            mem_dofs = np.array([0, 1, 6, 7, 12, 13, 18, 19])
            K_L[np.ix_(mem_dofs, mem_dofs)] += k_L_local
        
        # Geometric stiffness K_σ
        K_sigma = self.compute_geometric_stiffness(sigma, transform_to_global=False)
        
        # Total tangent stiffness
        K_T = K_0 + K_L + K_sigma
        
        # Force symmetry
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
        
        The internal force vector represents the element's resistance to
        the current deformation state:
        
            f_int = ∫ B^T · σ dV
            
        For equilibrium in nonlinear analysis:
            f_ext = f_int
            
        Or in incremental form (Newton-Raphson):
            K_T · Δu = f_ext - f_int
            
        Parameters
        ----------
        transform_to_global : bool, optional
            Transform result to global coordinates. Default True.
            
        Returns
        -------
        np.ndarray
            (24,) internal force vector
            
        Notes
        -----
        For Total Lagrangian formulation, we use Second Piola-Kirchhoff
        stress S and Green-Lagrange strain E. The internal force is:
        
            f_int = ∫ (B_L + B_NL)^T · S dA · h
            
        where S is computed from E using the constitutive relation S = C : E.
        """
        # Transform displacements to local coordinates
        T = self.T()
        u_local = T @ self._current_displacements
        
        f_int = np.zeros(24)
        
        # Get constitutive matrices
        Cm = self.Cm()  # 3x3 membrane
        Cb = self.Cb()  # 3x3 bending
        Cs = self.Cs()  # 2x2 shear
        
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)
            
            # Membrane contribution
            B_m = self.B_m(r, s)  # 3x8 MITC4+ interpolated
            
            # Extract membrane displacements
            mem_dofs = np.array([0, 1, 6, 7, 12, 13, 18, 19])
            u_mem = u_local[mem_dofs]
            
            # Green-Lagrange strain (membrane part)
            E_GL = self.compute_green_lagrange_strain(r, s)
            E_membrane = np.array([E_GL[0, 0], E_GL[1, 1], 2*E_GL[0, 1]])
            
            # Second Piola-Kirchhoff stress (for linear elastic material)
            S_membrane = Cm @ E_membrane
            
            # Internal force contribution from membrane
            if self.nonlinear:
                # Use combined B matrix for nonlinear analysis
                B_L = self._compute_B_L(r, s)
                B_NL = self._compute_B_NL(r, s)
                B_total = B_L[[0, 1, 3], :] + B_NL[[0, 1, 3], :]
                
                # Map to membrane DOFs
                B_mem_total = np.zeros((3, 8))
                for i, dof in enumerate(mem_dofs):
                    col_24 = dof
                    B_mem_total[:, i] = B_total[:, col_24]
                
                f_mem = B_mem_total.T @ S_membrane * detJ * self.thickness
            else:
                # Linear analysis - use standard B_m
                f_mem = B_m.T @ S_membrane * detJ * self.thickness
            
            f_int[mem_dofs] += f_mem
            
            # Bending contribution
            B_kappa = self.B_kappa(r, s)  # 3x12
            bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
            u_bend = u_local[bend_dofs]
            
            kappa = B_kappa @ u_bend
            M = Cb @ kappa  # Bending moments
            f_bend = B_kappa.T @ M * detJ
            f_int[bend_dofs] += f_bend
            
            # Shear contribution
            B_gamma = self.B_gamma(r, s)  # 2x12
            gamma = B_gamma @ u_bend
            Q = Cs @ gamma  # Shear forces
            f_shear = B_gamma.T @ Q * detJ
            f_int[bend_dofs] += f_shear
        
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
            (24,) external force vector in global coordinates
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
        
        For linear elastic material with Green-Lagrange strain:
        
            U = 0.5 * ∫ S : E dV = 0.5 * ∫ E^T C E dV
            
        Returns
        -------
        float
            Total strain energy (Joules)
        """
        # Transform displacements to local coordinates
        T = self.T()
        u_local = T @ self._current_displacements
        
        U = 0.0
        Cm = self.Cm()
        Cb = self.Cb()
        Cs = self.Cs()
        
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)
            
            # Membrane strain energy
            E_GL = self.compute_green_lagrange_strain(r, s)
            E_mem = np.array([E_GL[0, 0], E_GL[1, 1], 2*E_GL[0, 1]])
            U += 0.5 * E_mem @ Cm @ E_mem * detJ * self.thickness
            
            # Bending strain energy
            B_kappa = self.B_kappa(r, s)
            bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
            u_bend = u_local[bend_dofs]
            kappa = B_kappa @ u_bend
            U += 0.5 * kappa @ Cb @ kappa * detJ
            
            # Shear strain energy
            B_gamma = self.B_gamma(r, s)
            gamma = B_gamma @ u_bend
            U += 0.5 * gamma @ Cs @ gamma * detJ
        
        return U
