"""
MITC4+ Composite Shell Element Implementation.

This module extends the MITC4+ shell element for laminated composite materials
using Classical Lamination Theory (CLT). The element supports:

- Multi-layer composite laminates with arbitrary fiber orientations
- ABD stiffness matrix formulation
- Membrane-bending coupling (asymmetric laminates)
- Layer-by-layer stress recovery
- Failure criteria evaluation (Tsai-Wu, Hashin)

The composite element inherits the full MITC4+ formulation (bubble enrichment,
MITC shear interpolation, static condensation) from the parent class, overriding
only the constitutive and material-abstraction methods.

References
----------
- Ko, Y., Lee, P.S., and Bathe, K.J. (2016). "A new MITC4+ shell element."
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
"""

from typing import Dict, List, Optional, Tuple

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
    The element inherits the full MITC4+ formulation with bubble enrichment
    and static condensation for shear locking prevention.

    For laminates with coupling (B != 0):
    K = int[Bm^T*A*Bm + Bm^T*B*Bk + Bk^T*B*Bm + Bk^T*D*Bk + Bg^T*Cs*Bg]dA

    For symmetric laminates (B = 0), the coupling terms vanish.
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
        """Create equivalent orthotropic material for parent class."""
        ref_material = laminate.plies[0].material
        Ex = equiv_props["Ex_membrane"]
        Ey = equiv_props["Ey_membrane"]
        Gxy = equiv_props["Gxy_membrane"]
        nuxy = equiv_props["nuxy_membrane"]
        _, G23, G13 = ref_material.G

        return OrthotropicMaterial(
            name=f"Equivalent_{laminate}",
            E=(Ex, Ey, Ey),
            G=(Gxy, G23, G13),
            nu=(nuxy, nuxy * Ey / Ex, nuxy),
            rho=ref_material.rho,
            shear_correction_factor=laminate.shear_correction_factor,
        )

    # =========================================================================
    # ABD MATRIX PROPERTIES
    # =========================================================================

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
        """Bending stiffness matrix (3x3) [N*m]."""
        return self._D_matrix

    @property
    def has_coupling(self) -> bool:
        """True if laminate has membrane-bending coupling (B != 0)."""
        return self._has_coupling

    # =========================================================================
    # CONSTITUTIVE MATRIX OVERRIDES
    # =========================================================================

    def Cm(self) -> np.ndarray:
        """Membrane constitutive matrix from CLT (A matrix)."""
        return self._A_matrix

    def Cb(self) -> np.ndarray:
        """Bending constitutive matrix from CLT (D matrix)."""
        return self._D_matrix

    def Cs(self) -> np.ndarray:
        """Transverse shear constitutive matrix from CLT."""
        return self._Cs_matrix

    # =========================================================================
    # VIRTUAL METHOD OVERRIDES FOR MATERIAL ABSTRACTION
    # =========================================================================

    def _drilling_stiffness_factor(self) -> float:
        """Drilling stabilization using laminate A matrix trace."""
        return np.trace(self._A_matrix) / 3.0 * self.thickness * 0.15

    def _mass_per_area(self) -> float:
        """Ply-integrated mass per unit area."""
        return sum(ply.material.rho * ply.thickness for ply in self.laminate.plies)

    def _rotational_inertia(self) -> float:
        """Ply-integrated rotational inertia per unit area."""
        return sum(
            ply.material.rho * (ply.z_top**3 - ply.z_bottom**3) / 3 for ply in self.laminate.plies
        )

    def _membrane_constitutive_raw(self) -> np.ndarray:
        """Membrane stress-strain matrix (A / h) for nonlinear methods."""
        return self._A_matrix / self.thickness

    @property
    def _has_membrane_bending_coupling(self) -> bool:
        return self._has_coupling

    def _coupling_stiffness(self) -> np.ndarray:
        """Membrane-bending coupling stiffness with bubble condensation.

        Assembles in the extended 26-DOF space (24 nodal + 2 bubble rotations)
        so the bubble-enriched bending field is properly coupled to the membrane,
        then statically condenses back to 24 DOFs.
        """
        B_mat = self._B_matrix
        K_ext = np.zeros((26, 26))

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)

            # Membrane B-matrix extended to 26 DOFs (zero bubble columns)
            Bm_24 = self.B_m_MITC4_plus(xi, eta)  # 3×24
            Bm_ext = np.zeros((3, 26))
            Bm_ext[:, :24] = Bm_24

            # Extended bending B-matrix including bubble (3×26)
            Bk_nodal = self.B_kappa(xi, eta)  # 3×24
            Bk_bubble = self.B_kappa_bubble(xi, eta)  # 3×2
            Bk_ext = np.zeros((3, 26))
            Bk_ext[:, :24] = Bk_nodal
            Bk_ext[:, 24:] = Bk_bubble

            K_mb = Bm_ext.T @ B_mat @ Bk_ext * w * detJ
            K_ext += K_mb + K_mb.T

        # Static condensation of bubble DOFs (last 2)
        K_uu = K_ext[:24, :24]
        K_uq = K_ext[:24, 24:]
        K_qq = K_ext[24:, 24:]

        if np.linalg.matrix_rank(K_qq) == K_qq.shape[0]:
            K_cond = K_uu - K_uq @ np.linalg.solve(K_qq, K_uq.T)
        else:
            K_cond = K_uu

        return K_cond

    # =========================================================================
    # STRESS RECOVERY (COMPOSITE-SPECIFIC)
    # =========================================================================

    def compute_midplane_strains(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mid-plane strains and curvatures at a point."""
        B_m = self.B_m(r, s)
        epsilon_0 = B_m @ u_local

        B_kappa = self.B_kappa(r, s)
        kappa = B_kappa @ u_local

        return epsilon_0, kappa

    def compute_ply_strains(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """Compute strains in each ply at specified through-thickness positions."""
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
        results = []

        for k, ply in enumerate(self.laminate.plies):
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
                epsilon_lam = epsilon_0 + z * kappa
                T_eps = self._strain_transformation_matrix(ply.angle)
                epsilon_ply = T_eps @ epsilon_lam
                results.append({
                    "ply_index": k,
                    "angle": ply.angle,
                    "z": z,
                    "epsilon_laminate": epsilon_lam.copy(),
                    "epsilon_ply": epsilon_ply.copy(),
                })

        return results

    def compute_ply_stresses(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """Compute stresses in each ply at specified through-thickness positions."""
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
        results = []

        for k, ply in enumerate(self.laminate.plies):
            Qbar = compute_Qbar(ply.material, ply.angle)

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
                epsilon_lam = epsilon_0 + z * kappa
                sigma_lam = Qbar @ epsilon_lam
                T_stress = stress_transformation_matrix(ply.angle)
                sigma_ply = T_stress @ sigma_lam
                results.append({
                    "ply_index": k,
                    "angle": ply.angle,
                    "z": z,
                    "sigma_laminate": sigma_lam.copy(),
                    "sigma_ply": sigma_ply.copy(),
                })

        return results

    def _strain_transformation_matrix(self, theta_deg: float) -> np.ndarray:
        """Strain transformation matrix: laminate -> ply coordinates."""
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        c2, s2 = c**2, s**2
        return np.array([
            [c2, s2, s * c],
            [s2, c2, -s * c],
            [-2 * s * c, 2 * s * c, c2 - s2],
        ])

    def compute_transverse_shear_stresses(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        z_positions: str = "mid",
    ) -> List[Dict]:
        """Recover transverse shear stresses through the thickness via equilibrium.

        Uses 3D equilibrium integration of in-plane bending stresses to obtain
        the piecewise-quadratic distribution of tau_xz and tau_yz. This satisfies
        traction-free boundary conditions at the top and bottom surfaces.

        Parameters
        ----------
        u_local : np.ndarray
            Local displacement vector.
        r, s : float
            Parametric coordinates for evaluation.
        z_positions : str
            Which z-positions per ply: 'mid', 'top', 'bottom', or 'all'.

        Returns
        -------
        List[Dict]
            Per-ply results with keys: ply_index, angle, z, tau_xz, tau_yz.
        """
        # Transverse shear strains and resultants from FSDT
        B_g = self.B_gamma_MITC4(r, s)
        gamma = B_g @ u_local  # [gamma_xz, gamma_yz]
        V = self.laminate.Cs @ gamma  # [V_xz, V_yz]

        D_11 = self.laminate.D[0, 0]
        D_22 = self.laminate.D[1, 1]

        # Build cumulative Phi function for each ply
        cum_11 = 0.0
        cum_22 = 0.0
        ply_data = []
        for ply in self.laminate.plies:
            Qbar = compute_Qbar(ply.material, ply.angle)
            Q11 = Qbar[0, 0]
            Q22 = Qbar[1, 1]
            z_bot = ply.z_bottom
            z_top = ply.z_top

            ply_data.append((cum_11, Q11, cum_22, Q22, z_bot, z_top))
            cum_11 += Q11 * (z_top**2 - z_bot**2) / 2
            cum_22 += Q22 * (z_top**2 - z_bot**2) / 2

        results = []
        for k, ply in enumerate(self.laminate.plies):
            c11, Q11, c22, Q22, z_bot, z_top = ply_data[k]

            if z_positions == "mid":
                z_list = [(z_bot + z_top) / 2]
            elif z_positions == "top":
                z_list = [z_top]
            elif z_positions == "bottom":
                z_list = [z_bot]
            elif z_positions == "all":
                z_list = [z_bot, (z_bot + z_top) / 2, z_top]
            else:
                raise ValueError(f"Unknown z_positions: {z_positions}")

            for z in z_list:
                Phi_11 = c11 + Q11 / 2 * (z**2 - z_bot**2)
                Phi_22 = c22 + Q22 / 2 * (z**2 - z_bot**2)

                tau_xz = -Phi_11 / D_11 * V[0] if D_11 > 0 else 0.0
                tau_yz = -Phi_22 / D_22 * V[1] if D_22 > 0 else 0.0

                results.append({
                    "ply_index": k,
                    "angle": ply.angle,
                    "z": z,
                    "tau_xz": float(tau_xz),
                    "tau_yz": float(tau_yz),
                })

        return results

    def evaluate_failure(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        criterion: str = "tsai-wu",
    ) -> List[Dict]:
        """Evaluate failure criteria for all plies."""
        stresses = self.compute_ply_stresses(u_local, r, s, z_positions="mid")
        results = []

        for stress_data in stresses:
            k = stress_data["ply_index"]
            ply = self.laminate.plies[k]

            if ply.strength is None:
                raise ValueError(
                    f"Ply {k} (angle={ply.angle}) has no strength properties. "
                    "Define StrengthProperties for failure analysis."
                )

            sigma_ply = stress_data["sigma_ply"]
            failure_result = evaluate_ply_failure(sigma_ply, ply.strength, criterion)
            results.append({
                "ply_index": k,
                "angle": ply.angle,
                "z": stress_data["z"],
                "sigma_ply": sigma_ply.copy(),
                "failure_result": failure_result,
                "failed": failure_result.failed,
                "failure_index": failure_result.failure_index,
                "mode": failure_result.mode,
            })

        return results

    def get_critical_ply(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
        criterion: str = "tsai-wu",
    ) -> Dict:
        """Find the ply with the highest failure index."""
        failure_results = self.evaluate_failure(u_local, r, s, criterion)
        return max(failure_results, key=lambda x: x["failure_index"])

    def compute_stress_resultants(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute membrane forces and bending moments per unit length."""
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
        N = self._A_matrix @ epsilon_0 + self._B_matrix @ kappa
        M = self._B_matrix @ epsilon_0 + self._D_matrix @ kappa
        return N, M

    def compute_membrane_stress_from_displacement(
        self,
        u_local: np.ndarray,
        r: float = 0.0,
        s: float = 0.0,
    ) -> np.ndarray:
        """Compute average membrane stress using ABD relation."""
        epsilon_0, kappa = self.compute_midplane_strains(u_local, r, s)
        N = self._A_matrix @ epsilon_0 + self._B_matrix @ kappa
        return N / self.thickness

    # =========================================================================
    # NONLINEAR ANALYSIS OVERRIDES (coupling-aware)
    # =========================================================================

    def _get_local_stiffness(self) -> np.ndarray:
        """Local stiffness including membrane-bending coupling for asymmetric laminates."""
        K = self.k_m() + self._k_bending_shear() + self.k_drill()
        if self._has_membrane_bending_coupling:
            K += self._coupling_stiffness()
        return K

    def compute_tangent_stiffness(
        self,
        sigma: Optional[np.ndarray] = None,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Tangent stiffness with coupling for asymmetric composite laminates.

        For Total Lagrangian formulation:
            K_T = K_0 + K_L + K_sigma
        where K_L is the initial displacement stiffness from GL strain nonlinearity.
        """
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

        # Initial displacement stiffness K_L (membrane contribution)
        K_L = np.zeros((24, 24))
        Cm_raw = self._membrane_constitutive_raw()

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            B_L = self._compute_B_L(xi, eta)
            B_NL = self._compute_B_NL(xi, eta)
            B_m_L = B_L[[0, 1, 3], :]
            B_m_NL = B_NL[[0, 1, 3], :]

            K_L += (
                (B_m_L.T @ Cm_raw @ B_m_NL + B_m_NL.T @ Cm_raw @ B_m_L + B_m_NL.T @ Cm_raw @ B_m_NL)
                * w
                * detJ
                * self.thickness
            )

        K_sigma = self.compute_geometric_stiffness(sigma, transform_to_global=False)
        K_T = K0 + K_L + K_sigma
        K_T = 0.5 * (K_T + K_T.T)

        if transform_to_global:
            T = self.T()
            K_T = T.T @ K_T @ T

        return K_T

    def compute_internal_forces(
        self,
        transform_to_global: bool = True,
    ) -> np.ndarray:
        """Internal forces with coupling for asymmetric composite laminates.

        Uses the same strategy as MITC3Composite:
        - Bending/shear: linear (bubble-condensed)
        - Membrane: GL strain integration when nonlinear
        - Coupling: linear (B matrix couples membrane and bending)
        - Drilling: stabilization restoring force
        """
        T = self.T()
        u_local = T @ self._current_displacements

        if not self.nonlinear:
            K_local = self._get_local_stiffness()
            f_int = K_local @ u_local
        else:
            f_int = np.zeros(24)

            # Bending + shear (linear, bubble-condensed)
            f_int += self._k_bending_shear() @ u_local

            # Drilling (linear)
            f_int += self.k_drill() @ u_local

            # Coupling contribution (linear)
            if self._has_membrane_bending_coupling:
                f_int += self._coupling_stiffness() @ u_local

            # Membrane (GL strain integration)
            Cm_raw = self._membrane_constitutive_raw()
            for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
                _, detJ = self.J(xi, eta)
                E_GL = self.compute_green_lagrange_strain(xi, eta)
                eps_m = np.array([E_GL[0, 0], E_GL[1, 1], 2 * E_GL[0, 1]])
                sigma_m = Cm_raw @ eps_m

                B_L = self._compute_B_L(xi, eta)
                B_NL = self._compute_B_NL(xi, eta)
                B_total = B_L[[0, 1, 3], :] + B_NL[[0, 1, 3], :]

                f_int += B_total.T @ sigma_m * w * detJ * self.thickness

        if transform_to_global:
            f_int = T.T @ f_int

        return f_int

    def compute_strain_energy(self) -> float:
        """Strain energy with coupling for asymmetric composite laminates.

        For nonlinear analysis, uses Green-Lagrange strains for membrane energy.
        Bending/shear and coupling remain linear in the local frame.
        """
        T = self.T()
        u_local = T @ self._current_displacements

        # Bending + shear energy (linear, bubble-condensed)
        U = 0.5 * u_local @ (self._k_bending_shear() + self.k_drill()) @ u_local

        # Coupling energy (linear)
        if self._has_membrane_bending_coupling:
            U += 0.5 * u_local @ self._coupling_stiffness() @ u_local

        # Membrane energy — GL strains when nonlinear
        Cm_raw = self._membrane_constitutive_raw()

        for (xi, eta), w in zip(self._gauss_points, self._gauss_weights):
            _, detJ = self.J(xi, eta)
            if self.nonlinear:
                E_GL = self.compute_green_lagrange_strain(xi, eta)
                eps_m = np.array([E_GL[0, 0], E_GL[1, 1], 2 * E_GL[0, 1]])
            else:
                Bm = self.B_m(xi, eta)
                eps_m = Bm @ u_local

            U += 0.5 * eps_m @ Cm_raw @ eps_m * w * detJ * self.thickness

        return float(U)

    def __repr__(self) -> str:
        return (
            f"MITC4Composite(nodes={self.node_ids}, "
            f"laminate={self.laminate}, "
            f"nonlinear={self.nonlinear})"
        )
