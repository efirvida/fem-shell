"""
MITC3+ Composite Shell Element Implementation.

This module extends the MITC3+ triangular shell element for laminated composite 
materials using Classical Lamination Theory (CLT). The element supports:

- Multi-layer composite laminates with arbitrary fiber orientations
- ABD stiffness matrix formulation
- Membrane-bending coupling (asymmetric laminates)
- Layer-by-layer stress recovery
- Failure criteria evaluation (Tsai-Wu, Hashin)

References
----------
- Lee, Y., Lee, P.S., and Bathe, K.J. (2014). "The MITC3+ shell element and its performance."
- Lee, Y., Lee, P.S., and Bathe, K.J. (2015). "The MITC3+ shell element in geometric nonlinear analysis."
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from fem_shell.core.laminate import (
    Laminate,
    Ply,
    compute_Qbar,
)
from fem_shell.core.material import OrthotropicMaterial
from fem_shell.constitutive.failure import (
    FailureMode,
    FailureResult,
    evaluate_ply_failure,
    stress_transformation_matrix,
)
from fem_shell.elements.MITC3 import MITC3


class MITC3Composite(MITC3):
    """
    MITC3+ triangular shell element with laminated composite support.

    This element extends MITC3 to handle multi-layer composite laminates
    using Classical Lamination Theory (CLT). The ABD stiffness matrices
    are computed from the laminate definition and used for constitutive
    behavior.

    Parameters
    ----------
    node_coords : np.ndarray
        Array of nodal coordinates in global system [3x3]
    node_ids : Tuple[int, int, int]
        Global node IDs for element connectivity
    laminate : Laminate
        Laminate definition with ply stack and material properties
    nonlinear : bool, optional
        Enable geometric nonlinear analysis, by default False

    Attributes
    ----------
    laminate : Laminate
        The laminate definition
    A : np.ndarray
        Extensional stiffness matrix (3x3)
    B_coup : np.ndarray
        Coupling stiffness matrix (3x3)
    D : np.ndarray
        Bending stiffness matrix (3x3)

    Notes
    -----
    **Key Differences from MITC3:**

    1. Constitutive matrices (Cm, Cb, Cs) are derived from ABD matrices
    2. Coupling between membrane and bending (B matrix) is supported
    3. Layer-by-layer stress recovery is available
    4. Failure criteria can be evaluated per ply

    **Stiffness Matrix:**

    For laminates with coupling (B ≠ 0):
    K = ∫[Bm^T·A·Bm + Bm^T·B·Bκ + Bκ^T·B·Bm + Bκ^T·D·Bκ + Bγ^T·Cs·Bγ]dA

    For symmetric laminates (B = 0), the coupling terms vanish.

    Examples
    --------
    >>> # Create quasi-isotropic laminate
    >>> material = OrthotropicMaterial(
    ...     name="Carbon/Epoxy",
    ...     E=(140e9, 10e9, 10e9),
    ...     G=(5e9, 4e9, 5e9),
    ...     nu=(0.3, 0.3, 0.3),
    ...     rho=1600
    ... )
    >>> plies = [Ply(material, 0.125e-3, angle) for angle in [0, 45, -45, 90, 90, -45, 45, 0]]
    >>> laminate = Laminate(plies)
    >>> element = MITC3Composite(coords, node_ids, laminate)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int],
        laminate: Laminate,
        nonlinear: bool = False,
    ):
        # Store laminate before calling parent __init__
        self.laminate = laminate

        # Store ABD matrices for direct access
        self._A_matrix = laminate.A.copy()
        self._B_matrix = laminate.B.copy()
        self._D_matrix = laminate.D.copy()
        self._Cs_matrix = laminate.Cs.copy()

        # Check for coupling
        self._has_coupling = not laminate.is_symmetric

        # Create equivalent material for parent class compatibility
        equiv_props = laminate.get_equivalent_properties()
        equivalent_material = self._create_equivalent_material(laminate, equiv_props)

        # Call parent constructor
        super().__init__(
            node_coords=node_coords,
            node_ids=node_ids,
            material=equivalent_material,
            thickness=laminate.total_thickness,
            shear_correction_factor=laminate.shear_correction_factor,
            nonlinear=nonlinear,
        )

        # Override element type
        self.element_type = "MITC3Composite"

    def _create_equivalent_material(
        self,
        laminate: Laminate,
        equiv_props: dict,
    ) -> OrthotropicMaterial:
        """
        Create equivalent orthotropic material for parent class.

        This is used for geometric calculations and provides approximate
        properties when needed. The actual constitutive behavior uses
        the ABD matrices.
        """
        # Use first ply material as reference for density
        ref_material = laminate.plies[0].material

        # Compute average/equivalent properties
        Ex = equiv_props['Ex_membrane']
        Ey = equiv_props['Ey_membrane']
        Gxy = equiv_props['Gxy_membrane']
        nuxy = equiv_props['nuxy_membrane']

        # Use reference material for out-of-plane properties
        _, G23, G13 = ref_material.G

        return OrthotropicMaterial(
            name=f"Equivalent_{laminate}",
            E=(Ex, Ey, Ey),
            G=(Gxy, G23, G13),
            nu=(nuxy, nuxy * Ey / Ex, nuxy),
            rho=ref_material.rho,
            shear_correction_factor=laminate.shear_correction_factor,
        )

    @property
    def A(self) -> np.ndarray:
        """Extensional stiffness matrix (3x3) [N/m]."""
        return self._A_matrix

    @property
    def B_coup(self) -> np.ndarray:
        """Coupling stiffness matrix (3x3) [N]."""
        return self._B_matrix

    @property
    def D(self) -> np.ndarray:
        """Bending stiffness matrix (3x3) [N·m]."""
        return self._D_matrix

    @property
    def has_coupling(self) -> bool:
        """True if laminate has membrane-bending coupling (B ≠ 0)."""
        return self._has_coupling

    def Cm(self) -> np.ndarray:
        """
        Membrane constitutive matrix from CLT.

        Returns the A matrix normalized by thickness for compatibility
        with the standard shell formulation:
        Cm = A / h

        Returns
        -------
        np.ndarray
            3x3 membrane stiffness matrix
        """
        return self._A_matrix / self.thickness

    def Cb(self) -> np.ndarray:
        """
        Bending constitutive matrix from CLT.

        Returns the D matrix directly. Note that for CLT:
        M = D @ κ (moments per unit length)

        Returns
        -------
        np.ndarray
            3x3 bending stiffness matrix [N·m]
        """
        return self._D_matrix

    def Cs(self) -> np.ndarray:
        """
        Transverse shear constitutive matrix from CLT.

        Returns the integrated transverse shear stiffness with
        correction factor applied.

        Returns
        -------
        np.ndarray
            2x2 shear stiffness matrix [N/m]
        """
        return self._Cs_matrix

    @property
    def K(self) -> np.ndarray:
        """
        Compute element stiffness matrix for laminated composite.

        This overrides the parent property to include membrane-bending
        coupling when the laminate is asymmetric (B ≠ 0).

        Returns
        -------
        np.ndarray
            18x18 element stiffness matrix
        """
        # Always use laminate-specific computation
        # (parent assumes thickness multiplication which is not correct for CLT)
        if self._has_coupling:
            K_local = self._K_with_coupling()
        else:
            K_local = self._K_symmetric_laminate()

        # Transform to global coordinates
        T = self.T()
        K = T.T @ K_local @ T

        # Symmetrize
        K = 0.5 * (K + K.T)

        # Stabilization for drilling DOFs (θz)
        avg_diag = np.trace(K) / 18.0
        drill_stiff = 1e-6 * max(avg_diag, 1e-10)
        for i in range(3):
            idx = 6*i + 5  # θz
            K[idx, idx] += drill_stiff

        return K

    def _K_symmetric_laminate(self) -> np.ndarray:
        """
        Compute stiffness matrix for symmetric laminate (B = 0).

        Uses ABD matrices directly without thickness multiplication
        since CLT already integrates through thickness.
        """
        K = np.zeros((18, 18))

        # Constitutive matrices (already integrated through thickness)
        A_mat = self._A_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix

        # DOF indices for MITC3 (3 nodes)
        # Membrane DOFs: u, v for each node
        mem_dofs = np.array([0, 1, 6, 7, 12, 13])
        # Bending DOFs: w, θx, θy for each node
        bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])

        area = self.area()

        # Gauss integration
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            # Strain-displacement matrices
            B_m = self.B_m(r, s)  # 3x18
            B_kappa = self.B_kappa(r, s)  # 3x18
            B_gamma = self.B_gamma(r, s)  # 2x18

            # Extract relevant columns
            B_m_mem = B_m[:, mem_dofs]  # 3x6
            B_kappa_bend = B_kappa[:, bend_dofs]  # 3x9
            B_gamma_bend = B_gamma[:, bend_dofs]  # 2x9

            # Membrane stiffness: Bm^T @ A @ Bm (A already includes thickness integration)
            K_mm = B_m_mem.T @ A_mat @ B_m_mem * w * area
            K[np.ix_(mem_dofs, mem_dofs)] += K_mm

            # Bending stiffness: Bκ^T @ D @ Bκ (D already includes h³/12 factor)
            K_bb = B_kappa_bend.T @ D_mat @ B_kappa_bend * w * area
            K[np.ix_(bend_dofs, bend_dofs)] += K_bb

            # Shear stiffness: Bγ^T @ Cs @ Bγ
            K_ss = B_gamma_bend.T @ Cs_mat @ B_gamma_bend * w * area
            K[np.ix_(bend_dofs, bend_dofs)] += K_ss

        return K

    def _K_with_coupling(self) -> np.ndarray:
        """
        Compute stiffness matrix with membrane-bending coupling.

        For asymmetric laminates:
        K = ∫[Bm^T·A·Bm + Bm^T·B·Bκ + Bκ^T·B·Bm + Bκ^T·D·Bκ + Bγ^T·Cs·Bγ]dA
        
        Returns local stiffness matrix (transformation to global is done in K property).
        """
        K = np.zeros((18, 18))

        # Constitutive matrices
        A_mat = self._A_matrix
        B_mat = self._B_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix

        # DOF indices for MITC3 (3 nodes)
        # Membrane DOFs: u, v for each node
        mem_dofs = np.array([0, 1, 6, 7, 12, 13])
        # Bending DOFs: w, θx, θy for each node
        bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])

        area = self.area()

        # Gauss integration
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            # Strain-displacement matrices
            B_m = self.B_m(r, s)  # 3x18
            B_kappa = self.B_kappa(r, s)  # 3x18
            B_gamma = self.B_gamma(r, s)  # 2x18

            # Extract relevant columns
            B_m_mem = B_m[:, mem_dofs]  # 3x6
            B_kappa_bend = B_kappa[:, bend_dofs]  # 3x9
            B_gamma_bend = B_gamma[:, bend_dofs]  # 2x9

            # Membrane stiffness: Bm^T @ A @ Bm
            K_mm = B_m_mem.T @ A_mat @ B_m_mem * w * area
            K[np.ix_(mem_dofs, mem_dofs)] += K_mm

            # Bending stiffness: Bκ^T @ D @ Bκ
            K_bb = B_kappa_bend.T @ D_mat @ B_kappa_bend * w * area
            K[np.ix_(bend_dofs, bend_dofs)] += K_bb

            # Shear stiffness: Bγ^T @ Cs @ Bγ
            K_ss = B_gamma_bend.T @ Cs_mat @ B_gamma_bend * w * area
            K[np.ix_(bend_dofs, bend_dofs)] += K_ss

            # Coupling terms: Bm^T @ B @ Bκ and Bκ^T @ B @ Bm
            K_mb = B_m_mem.T @ B_mat @ B_kappa_bend * w * area
            K[np.ix_(mem_dofs, bend_dofs)] += K_mb
            K[np.ix_(bend_dofs, mem_dofs)] += K_mb.T

        return K

    def compute_midplane_strains(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mid-plane strains and curvatures at a point.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)

        Returns
        -------
        epsilon_0 : np.ndarray
            Mid-plane strains [εxx, εyy, γxy]
        kappa : np.ndarray
            Curvatures [κxx, κyy, κxy]
        """
        # Membrane DOFs
        mem_dofs = np.array([0, 1, 6, 7, 12, 13])
        u_membrane = u_local[mem_dofs]

        # Bending DOFs
        bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
        u_bending = u_local[bend_dofs]

        # Compute strains
        B_m = self.B_m(r, s)
        epsilon_0 = B_m[:, mem_dofs] @ u_membrane

        B_kappa = self.B_kappa(r, s)
        kappa = B_kappa[:, bend_dofs] @ u_bending

        return epsilon_0, kappa

    def compute_ply_strains(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """
        Compute strains in each ply at specified through-thickness positions.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)
        z_positions : str
            Position within each ply: "mid", "top", "bottom", or "all"

        Returns
        -------
        List[Dict]
            List of dictionaries for each ply containing:
            - 'ply_index': int
            - 'angle': float (degrees)
            - 'z': float (distance from mid-plane)
            - 'epsilon_laminate': np.ndarray [εxx, εyy, γxy]
            - 'epsilon_ply': np.ndarray [ε1, ε2, γ12]
        """
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)

        results = []

        for k, ply in enumerate(self.laminate.plies):
            # Determine z-positions to evaluate
            if z_positions == "mid":
                z_list = [(ply.z_bottom + ply.z_top) / 2]
            elif z_positions == "top":
                z_list = [ply.z_top]
            elif z_positions == "bottom":
                z_list = [ply.z_bottom]
            elif z_positions == "all":
                z_list = [ply.z_bottom, (ply.z_bottom + ply.z_top) / 2, ply.z_top]
            else:
                raise ValueError(f"Unknown z_positions: {z_positions}")

            for z in z_list:
                # Total strain at z: ε = ε⁰ + z·κ
                epsilon_lam = epsilon_0 + z * kappa

                # Transform to ply coordinates
                T_eps = self._strain_transformation_matrix(ply.angle)
                epsilon_ply = T_eps @ epsilon_lam

                results.append({
                    'ply_index': k,
                    'angle': ply.angle,
                    'z': z,
                    'epsilon_laminate': epsilon_lam.copy(),
                    'epsilon_ply': epsilon_ply.copy(),
                })

        return results

    def compute_ply_stresses(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """
        Compute stresses in each ply at specified through-thickness positions.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)
        z_positions : str
            Position within each ply: "mid", "top", "bottom", or "all"

        Returns
        -------
        List[Dict]
            List of dictionaries for each ply containing:
            - 'ply_index': int
            - 'angle': float (degrees)
            - 'z': float
            - 'sigma_laminate': np.ndarray [σxx, σyy, τxy]
            - 'sigma_ply': np.ndarray [σ1, σ2, τ12]
        """
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)

        results = []

        for k, ply in enumerate(self.laminate.plies):
            # Get transformed stiffness matrix for this ply
            Qbar = compute_Qbar(ply.material, ply.angle)

            # Determine z-positions
            if z_positions == "mid":
                z_list = [(ply.z_bottom + ply.z_top) / 2]
            elif z_positions == "top":
                z_list = [ply.z_top]
            elif z_positions == "bottom":
                z_list = [ply.z_bottom]
            elif z_positions == "all":
                z_list = [ply.z_bottom, (ply.z_bottom + ply.z_top) / 2, ply.z_top]
            else:
                raise ValueError(f"Unknown z_positions: {z_positions}")

            for z in z_list:
                # Total strain at z
                epsilon_lam = epsilon_0 + z * kappa

                # Stress in laminate coordinates: σ = Qbar @ ε
                sigma_lam = Qbar @ epsilon_lam

                # Transform stress to ply coordinates
                T_stress = stress_transformation_matrix(ply.angle)
                sigma_ply = T_stress @ sigma_lam

                results.append({
                    'ply_index': k,
                    'angle': ply.angle,
                    'z': z,
                    'sigma_laminate': sigma_lam.copy(),
                    'sigma_ply': sigma_ply.copy(),
                })

        return results

    def _strain_transformation_matrix(self, theta_deg: float) -> np.ndarray:
        """
        Compute strain transformation matrix from laminate to ply coordinates.

        For engineering strains [εxx, εyy, γxy] -> [ε1, ε2, γ12]:

        T = [c²    s²     sc   ]
            [s²    c²    -sc   ]
            [-2sc  2sc   c²-s²]
        """
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        c2, s2 = c**2, s**2

        return np.array([
            [c2, s2, s * c],
            [s2, c2, -s * c],
            [-2 * s * c, 2 * s * c, c2 - s2]
        ])

    def evaluate_failure(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
        criterion: str = "tsai-wu",
    ) -> List[Dict]:
        """
        Evaluate failure criteria for all plies.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)
        criterion : str
            Failure criterion: "tsai-wu", "hashin", or "max-stress"

        Returns
        -------
        List[Dict]
            List of failure results for each ply containing:
            - 'ply_index': int
            - 'angle': float
            - 'z': float
            - 'sigma_ply': np.ndarray
            - 'failure_result': FailureResult
            - 'failed': bool
            - 'failure_index': float
            - 'mode': FailureMode

        Raises
        ------
        ValueError
            If a ply has no strength properties defined
        """
        # Get stresses at ply mid-thickness
        stresses = self.compute_ply_stresses(u_local, r, s, z_positions="mid")

        results = []

        for stress_data in stresses:
            k = stress_data['ply_index']
            ply = self.laminate.plies[k]

            if ply.strength is None:
                raise ValueError(
                    f"Ply {k} (angle={ply.angle}°) has no strength properties. "
                    "Define StrengthProperties for failure analysis."
                )

            sigma_ply = stress_data['sigma_ply']

            # Evaluate failure criterion
            failure_result = evaluate_ply_failure(
                sigma_ply, ply.strength, criterion
            )

            results.append({
                'ply_index': k,
                'angle': ply.angle,
                'z': stress_data['z'],
                'sigma_ply': sigma_ply.copy(),
                'failure_result': failure_result,
                'failed': failure_result.failed,
                'failure_index': failure_result.failure_index,
                'mode': failure_result.mode,
            })

        return results

    def get_critical_ply(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
        criterion: str = "tsai-wu",
    ) -> Dict:
        """
        Find the ply with the highest failure index.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)
        criterion : str
            Failure criterion

        Returns
        -------
        Dict
            Failure data for the critical ply
        """
        failure_results = self.evaluate_failure(u_local, r, s, criterion)

        return max(failure_results, key=lambda x: x['failure_index'])

    def compute_stress_resultants(
        self,
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute membrane forces and bending moments per unit length.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (18,)
        r, s : float
            Parametric coordinates (default: centroid)

        Returns
        -------
        N : np.ndarray
            Membrane force resultants [Nxx, Nyy, Nxy] (N/m)
        M : np.ndarray
            Moment resultants [Mxx, Myy, Mxy] (N)
        """
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)

        # Using ABD relation: [N; M] = [A B; B D] @ [ε⁰; κ]
        N = self._A_matrix @ epsilon_0 + self._B_matrix @ kappa
        M = self._B_matrix @ epsilon_0 + self._D_matrix @ kappa

        return N, M

    def compute_membrane_stress_from_displacement(
        self, 
        u_local: np.ndarray,
        r: float = 1.0/3.0,
        s: float = 1.0/3.0,
    ) -> np.ndarray:
        """
        Compute membrane stress from local displacement vector.
        
        For composite laminates, uses the A matrix constitutive relation:
            N = A · ε₀  (for symmetric laminate)
            σ_avg = N / h
            
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
            (3,) average membrane stress [σ_xx, σ_yy, σ_xy]
        """
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
        
        # Force resultants
        N = self._A_matrix @ epsilon_0 + self._B_matrix @ kappa
        
        # Average stress through thickness
        sigma_avg = N / self.thickness
        
        return sigma_avg

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute tangent stiffness matrix for nonlinear analysis of composite.
        
        The tangent stiffness matrix is:
        
            K_T = K_0 + K_L + K_σ
            
        where:
        - K_0: Initial (linear) stiffness matrix (from ABD formulation)
        - K_L: Large displacement stiffness (from nonlinear strain terms)
        - K_σ: Geometric (stress) stiffness matrix
        
        Parameters
        ----------
        sigma : np.ndarray, optional
            Current stress state for geometric stiffness. If None, computed
            from current displacements. Shape (3,) for membrane stress
            [σ_xx, σ_yy, σ_xy].
        transform_to_global : bool, optional
            Transform result to global coordinates. Default True.
            
        Returns
        -------
        np.ndarray
            (18, 18) tangent stiffness matrix
        """
        # Get linear stiffness using composite formulation
        if self._has_coupling:
            K_0 = self._K_with_coupling()
        else:
            K_0 = self._K_symmetric_laminate()
        
        if not self.nonlinear:
            if transform_to_global:
                T = self.T()
                return T.T @ K_0 @ T
            return K_0
        
        # Compute stress if not provided
        if sigma is None:
            T = self.T()
            u_local = T @ self._current_displacements
            sigma = self.compute_membrane_stress_from_displacement(u_local)
        
        # Large displacement stiffness K_L using A matrix
        K_L = np.zeros((18, 18))
        
        # Use A matrix normalized for constitutive relation
        Cm_raw = self._A_matrix / self.thickness
        
        area = self.area()
        
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            B_NL = self._compute_B_NL(r, s)
            B_L = self._compute_B_L(r, s)
            
            B_m_L = B_L[[0, 1, 3], :]
            B_m_NL = B_NL[[0, 1, 3], :]
            
            k_L_local = (B_m_L.T @ Cm_raw @ B_m_NL + B_m_NL.T @ Cm_raw @ B_m_L) * w * area * self.thickness
            K_L += k_L_local
        
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
        Compute internal force vector for nonlinear analysis of composite.
        
        Parameters
        ----------
        transform_to_global : bool, optional
            Transform result to global coordinates. Default True.
            
        Returns
        -------
        np.ndarray
            (18,) internal force vector
        """
        T = self.T()
        u_local = T @ self._current_displacements
        
        f_int = np.zeros(18)
        
        # Constitutive matrices from laminate
        A_mat = self._A_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix
        B_coup = self._B_matrix
        
        # DOF indices
        mem_dofs = np.array([0, 1, 6, 7, 12, 13])
        bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
        
        area = self.area()
        
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            # Get strains
            epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
            
            # Force and moment resultants using ABD
            N = A_mat @ epsilon_0 + B_coup @ kappa
            M = B_coup @ epsilon_0 + D_mat @ kappa
            
            # Membrane contribution
            B_m = self.B_m(r, s)
            f_int[mem_dofs] += B_m[:, mem_dofs].T @ N * w * area
            
            # Bending contribution
            B_kappa = self.B_kappa(r, s)
            f_int[bend_dofs] += B_kappa[:, bend_dofs].T @ M * w * area
            
            # Shear contribution
            B_gamma = self.B_gamma(r, s)
            gamma = B_gamma[:, bend_dofs] @ u_local[bend_dofs]
            Q = Cs_mat @ gamma
            f_int[bend_dofs] += B_gamma[:, bend_dofs].T @ Q * w * area
        
        if transform_to_global:
            f_int = T.T @ f_int
        
        return f_int

    def compute_strain_energy(self) -> float:
        """
        Compute total strain energy stored in the composite element.
        
        Returns
        -------
        float
            Total strain energy (Joules)
        """
        T = self.T()
        u_local = T @ self._current_displacements
        
        U = 0.0
        
        # Constitutive matrices from laminate
        A_mat = self._A_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix
        B_coup = self._B_matrix
        
        bend_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
        
        area = self.area()
        
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
            
            # Membrane + coupling strain energy: 0.5 * (ε₀ᵀAε₀ + 2ε₀ᵀBκ + κᵀDκ)
            U += 0.5 * (epsilon_0 @ A_mat @ epsilon_0 + 
                       2 * epsilon_0 @ B_coup @ kappa + 
                       kappa @ D_mat @ kappa) * w * area
            
            # Shear strain energy
            B_gamma = self.B_gamma(r, s)
            gamma = B_gamma[:, bend_dofs] @ u_local[bend_dofs]
            U += 0.5 * gamma @ Cs_mat @ gamma * w * area
        
        return U

    @property
    def M(self) -> np.ndarray:
        """
        Compute element mass matrix for laminate.

        The mass is computed by integrating the density through
        the thickness of each ply. Uses the same approach as the parent
        MITC3 class but with laminate-specific mass properties.

        Returns
        -------
        np.ndarray
            18x18 consistent mass matrix
        """
        # Compute total mass per unit area (ρ*h equivalent)
        mass_per_area = sum(
            ply.material.rho * ply.thickness
            for ply in self.laminate.plies
        )

        # Compute rotational inertia (ρ*h³/12 equivalent for laminate)
        rotational_inertia = sum(
            ply.material.rho * (ply.z_top**3 - ply.z_bottom**3) / 3
            for ply in self.laminate.plies
        )

        M_local = np.zeros((18, 18))
        area = self.area()

        # Translational and rotational inertia
        for (r, s), w in zip(self._gauss_points, self._gauss_weights):
            N = self._compute_N(r, s)  # 6x18 shape function matrix

            # Translational mass (u, v, w)
            for i in range(3):
                for j in range(3):
                    val_t = N[0, 6*i] * N[0, 6*j] * mass_per_area * w * area
                    for k in range(3):
                        M_local[6*i+k, 6*j+k] += val_t

                    # Rotational (θx, θy, θz)
                    val_r = N[0, 6*i] * N[0, 6*j] * rotational_inertia * w * area
                    for k in range(3, 6):
                        M_local[6*i+k, 6*j+k] += val_r

        # Transform to global coordinates
        T = self.T()
        return T.T @ M_local @ T

    def __repr__(self) -> str:
        return (
            f"MITC3Composite(nodes={self.node_ids}, "
            f"laminate={self.laminate}, "
            f"nonlinear={self.nonlinear})"
        )
