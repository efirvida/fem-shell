"""
MITC4 Composite Shell Element Implementation.

This module extends the MITC4+ shell element for laminated composite materials
using Classical Lamination Theory (CLT). The element supports:

- Multi-layer composite laminates with arbitrary fiber orientations
- ABD stiffness matrix formulation
- Membrane-bending coupling (asymmetric laminates)
- Layer-by-layer stress recovery
- Failure criteria evaluation (Tsai-Wu, Hashin)

References
----------
- Ko, Y., Lee, P.S., and Bathe, K.J. (2016). "A new MITC4+ shell element."
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
"""

from typing import Dict, List, Tuple

import numpy as np

from fem_shell.constitutive.failure import evaluate_ply_failure, stress_transformation_matrix
from fem_shell.core.laminate import Laminate, compute_Qbar
from fem_shell.core.material import OrthotropicMaterial
from fem_shell.elements.MITC4 import MITC4


class MITC4Composite(MITC4):
    """
    MITC4+ shell element with laminated composite support.

    This element extends MITC4 to handle multi-layer composite laminates
    using Classical Lamination Theory (CLT). The ABD stiffness matrices
    are computed from the laminate definition and used for constitutive
    behavior.

    Parameters
    ----------
    node_coords : np.ndarray
        Array of nodal coordinates in global system [4x3]
    node_ids : Tuple[int, int, int, int]
        Global node IDs for element connectivity
    laminate : Laminate
        Laminate definition with ply stack and material properties
    kx_mod : float, optional
        Stiffness modification factor in x-direction, by default 1.0
    ky_mod : float, optional
        Stiffness modification factor in y-direction, by default 1.0
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
    **Key Differences from MITC4:**

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
    >>> element = MITC4Composite(coords, node_ids, laminate)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        node_ids: Tuple[int, int, int, int],
        laminate: Laminate,
        kx_mod: float = 1.0,
        ky_mod: float = 1.0,
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

        # Store modifiers for curvature calculation
        self._kx_mod = kx_mod
        self._ky_mod = ky_mod

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
        self.element_type = "MITC4Composite"

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
        Ex = equiv_props["Ex_membrane"]
        Ey = equiv_props["Ey_membrane"]
        Gxy = equiv_props["Gxy_membrane"]
        nuxy = equiv_props["nuxy_membrane"]

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

    def K(self) -> np.ndarray:
        """
        Compute element stiffness matrix for laminated composite.

        This overrides the parent method to include membrane-bending
        coupling when the laminate is asymmetric (B ≠ 0).

        Returns
        -------
        np.ndarray
            24x24 element stiffness matrix
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

        # Symmetrize and stabilize
        K = 0.5 * (K + K.T)
        K += 1e-10 * np.eye(self.dofs_count)

        return K

    def _K_symmetric_laminate(self) -> np.ndarray:
        """
        Compute stiffness matrix for symmetric laminate (B = 0).

        Uses ABD matrices directly without thickness multiplication
        since CLT already integrates through thickness.
        """
        K = np.zeros((24, 24))

        # Constitutive matrices (already integrated through thickness)
        A_mat = self._A_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix

        # Gauss integration
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)

            # Strain-displacement matrices (all 24-DOF format)
            B_m = self.B_m(r, s)  # 3x24
            B_kappa = self.B_kappa(r, s)  # 3x24
            B_gamma = self.B_gamma_MITC4(r, s)  # 2x24

            # Membrane stiffness: Bm^T @ A @ Bm (A already includes thickness integration)
            K += B_m.T @ A_mat @ B_m * detJ

            # Bending stiffness: Bκ^T @ D @ Bκ (D already includes h³/12 factor)
            K += B_kappa.T @ D_mat @ B_kappa * detJ

            # Shear stiffness: Bγ^T @ Cs @ Bγ
            K += B_gamma.T @ Cs_mat @ B_gamma * detJ

        # Apply drilling stiffness
        K = self._add_drilling_stiffness(K)

        return K

    def _K_with_coupling(self) -> np.ndarray:
        """
        Compute stiffness matrix with membrane-bending coupling.

        For asymmetric laminates:
        K = ∫[Bm^T·A·Bm + Bm^T·B·Bκ + Bκ^T·B·Bm + Bκ^T·D·Bκ + Bγ^T·Cs·Bγ]dA

        Returns local stiffness matrix (transformation to global is done in K()).
        """
        K = np.zeros((24, 24))

        # Constitutive matrices
        A_mat = self._A_matrix
        B_mat = self._B_matrix
        D_mat = self._D_matrix
        Cs_mat = self._Cs_matrix

        # Gauss integration
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)

            # Strain-displacement matrices (all 24-DOF format)
            B_m = self.B_m(r, s)  # 3x24
            B_kappa = self.B_kappa(r, s)  # 3x24
            B_gamma = self.B_gamma_MITC4(r, s)  # 2x24

            # Membrane stiffness: Bm^T @ A @ Bm
            K += B_m.T @ A_mat @ B_m * detJ

            # Bending stiffness: Bκ^T @ D @ Bκ
            K += B_kappa.T @ D_mat @ B_kappa * detJ

            # Shear stiffness: Bγ^T @ Cs @ Bγ
            K += B_gamma.T @ Cs_mat @ B_gamma * detJ

            # Coupling terms: Bm^T @ B @ Bκ and Bκ^T @ B @ Bm
            K_mb = B_m.T @ B_mat @ B_kappa * detJ
            K += K_mb
            K += K_mb.T

        # Apply drilling stiffness
        K = self._add_drilling_stiffness(K)

        return K

    def _add_drilling_stiffness(self, K: np.ndarray) -> np.ndarray:
        """Add small drilling stiffness to θz DOFs."""
        # Get representative stiffness value
        diag_avg = np.mean(np.abs(np.diag(K)[np.diag(K) != 0]))
        drilling_stiffness = 1e-6 * diag_avg if diag_avg > 0 else 1e-3

        # Drilling DOFs: θz at each node (positions 5, 11, 17, 23)
        drilling_dofs = [5, 11, 17, 23]
        for dof in drilling_dofs:
            K[dof, dof] += drilling_stiffness

        return K

    def compute_midplane_strains(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mid-plane strains and curvatures at a point.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]

        Returns
        -------
        epsilon_0 : np.ndarray
            Mid-plane strains [εxx, εyy, γxy]
        kappa : np.ndarray
            Curvatures [κxx, κyy, κxy]
        """
        # Compute strains using full 24-DOF B matrices
        B_m = self.B_m(r, s)  # 3x24
        epsilon_0 = B_m @ u_local

        B_kappa = self.B_kappa(r, s)  # 3x24
        kappa = B_kappa @ u_local

        return epsilon_0, kappa

    def compute_ply_strains(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """
        Compute strains in each ply at specified through-thickness positions.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]
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

                results.append(
                    {
                        "ply_index": k,
                        "angle": ply.angle,
                        "z": z,
                        "epsilon_laminate": epsilon_lam.copy(),
                        "epsilon_ply": epsilon_ply.copy(),
                    }
                )

        return results

    def compute_ply_stresses(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """
        Compute stresses in each ply at specified through-thickness positions.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]
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

                results.append(
                    {
                        "ply_index": k,
                        "angle": ply.angle,
                        "z": z,
                        "sigma_laminate": sigma_lam.copy(),
                        "sigma_ply": sigma_ply.copy(),
                    }
                )

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

        return np.array([[c2, s2, s * c], [s2, c2, -s * c], [-2 * s * c, 2 * s * c, c2 - s2]])

    def evaluate_failure(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        criterion: str = "tsai-wu",
    ) -> List[Dict]:
        """
        Evaluate failure criteria for all plies.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]
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
            k = stress_data["ply_index"]
            ply = self.laminate.plies[k]

            if ply.strength is None:
                raise ValueError(
                    f"Ply {k} (angle={ply.angle}°) has no strength properties. "
                    "Define StrengthProperties for failure analysis."
                )

            sigma_ply = stress_data["sigma_ply"]

            # Evaluate failure criterion
            failure_result = evaluate_ply_failure(sigma_ply, ply.strength, criterion)

            results.append(
                {
                    "ply_index": k,
                    "angle": ply.angle,
                    "z": stress_data["z"],
                    "sigma_ply": sigma_ply.copy(),
                    "failure_result": failure_result,
                    "failed": failure_result.failed,
                    "failure_index": failure_result.failure_index,
                    "mode": failure_result.mode,
                }
            )

        return results

    def get_critical_ply(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        criterion: str = "tsai-wu",
    ) -> Dict:
        """
        Find the ply with the highest failure index.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]
        criterion : str
            Failure criterion

        Returns
        -------
        Dict
            Failure data for the critical ply
        """
        failure_results = self.evaluate_failure(u_local, r, s, criterion)

        return max(failure_results, key=lambda x: x["failure_index"])

    def compute_stress_resultants(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute membrane forces and bending moments per unit length.

        Parameters
        ----------
        u_local : np.ndarray
            Local element displacement vector (24,)
        r, s : float
            Parametric coordinates [-1, 1]

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

    def mass(self) -> np.ndarray:
        """
        Compute element mass matrix for laminate.

        The mass is computed by integrating the density through
        the thickness of each ply. Uses the same approach as the parent
        MITC4 class but with laminate-specific mass properties.

        Returns
        -------
        np.ndarray
            24x24 consistent mass matrix
        """
        # Compute total mass per unit area (ρ*h equivalent)
        mass_per_area = sum(ply.material.rho * ply.thickness for ply in self.laminate.plies)

        # Compute rotational inertia (ρ*h³/12 equivalent for laminate)
        rotational_inertia = sum(
            ply.material.rho * (ply.z_top**3 - ply.z_bottom**3) / 3 for ply in self.laminate.plies
        )

        M_local = np.zeros((24, 24))

        # Use same approach as parent class
        for r, s in self._gauss_points:
            _, detJ = self.J(r, s)

            # Build shape function matrix (6x24) for mass matrix
            N_vals = self._shape_functions(r, s)  # [N0, N1, N2, N3]
            N = np.zeros((6, 24))
            for i in range(4):
                for dof in range(6):
                    N[dof, 6 * i + dof] = N_vals[i]

            # Translational mass (u, v, w)
            M_trans = mass_per_area * (N[:3].T @ N[:3]) * detJ

            # Rotational inertia contributions
            M_rot = np.zeros((24, 24))
            for dof in [3, 4, 5]:  # θx, θy, θz
                M_rot += rotational_inertia * np.outer(N[dof], N[dof]) * detJ

            M_local += M_trans + M_rot

        # Transform to global coordinates
        T = self.T()
        M = T.T @ M_local @ T

        return M

    def __repr__(self) -> str:
        return (
            f"MITC4Composite(nodes={self.node_ids}, "
            f"laminate={self.laminate}, "
            f"nonlinear={self.nonlinear})"
        )
