"""
Linear Dynamic FSI Solver.

This module provides the LinearDynamicFSISolver for fluid-structure interaction
problems using preCICE coupling.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import meshio
import numpy as np
from petsc4py import PETSc
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fem_shell.core.bc import BoundaryConditionManager
from fem_shell.core.mesh import MeshModel
from fem_shell.postprocess.stress_recovery import StressRecovery, StressType
from fem_shell.solvers.linear import LinearDynamicSolver

from .base import Adapter, ForceClipper, NewmarkCoefficients

_console = Console(highlight=False)

# =============================================================================
# Module Constants
# =============================================================================

# Solver configuration thresholds
# 200k DOFs fits comfortably in MUMPS memory on a single HPC node and gives
# 1 back-substitution per solve instead of hundreds of CG/GAMG iterations.
_DOF_THRESHOLD_DIRECT_SOLVER = 200_000

# Solver tolerances
_DIRECT_SOLVER_RTOL = 1e-12
_DIRECT_SOLVER_ATOL = 1e-14
_ITERATIVE_SOLVER_RTOL = 1e-6
_ITERATIVE_SOLVER_ATOL = 1e-10
_ITERATIVE_SOLVER_MAX_IT = 1000


class LinearDynamicFSISolver(LinearDynamicSolver):
    """
    Linear dynamic solver for Fluid-Structure Interaction (FSI) problems.

    Inherits from LinearDynamicSolver and adds functionality for FSI problems
    using preCICE coupling.

    Parameters
    ----------
    mesh : MeshModel
        The finite element mesh.
    fem_model_properties : Dict
        Configuration dictionary containing solver and element properties.

    Configuration
    -------------
    The fem_model_properties dictionary should contain:
        solver.coupling.participant : str
            Name of this participant in preCICE.
        solver.coupling.config_file : str
            Path to preCICE configuration XML.
        solver.coupling.coupling_mesh : str
            Name of the coupling mesh.
        solver.coupling.write_data : str or list
            Data field(s) to write to preCICE.
        solver.coupling.read_data : str or list
            Data field(s) to read from preCICE.
        solver.coupling_boundaries : list
            Node set names for coupling interface.
        solver.force_max_cap : float, optional
            Maximum force magnitude per node (for clipping).
        solver.force_ramp_time : float, optional
            Time over which to ramp forces from 0 to 1.
        solver.solver_type : str, optional
            Linear solver type: "auto", "direct", "iterative". Default: "auto".
        solver.damping.enabled : bool, optional
            Enable Rayleigh damping. Default: True.
        solver.damping.eta_m : float, optional
            Mass-proportional Rayleigh damping coefficient.
        solver.damping.eta_k : float, optional
            Stiffness-proportional Rayleigh damping coefficient.
        solver.damping.zeta : float, optional
            Target damping ratio for auto-computation. Default: 0.02.
        solver.damping.mode_i : int, optional
            First reference mode for auto-computation. Default: 1.
        solver.damping.mode_j : int, optional
            Second reference mode for auto-computation. Default: 2.
        solver.force_max_magnitude : float, optional
            Absolute upper bound on total force magnitude (divergence check).
        solver.force_jump_factor : float, optional
            Relative spike detector threshold. Default: 1000.
    """

    def __init__(self, mesh: MeshModel, fem_model_properties: Dict):
        super().__init__(mesh, fem_model_properties)

        # Extract coupling configuration
        coupling_cfg = fem_model_properties["solver"]["coupling"]

        self.precice_participant = Adapter(
            participant=coupling_cfg["participant"],
            config_file=coupling_cfg["config_file"],
            coupling_mesh=coupling_cfg["coupling_mesh"],
            write_data=coupling_cfg["write_data"],
            read_data=coupling_cfg["read_data"],
        )
        self._prepared = False
        self._init_solver_config()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_solver_config(self) -> None:
        """Initialize solver configuration parameters.

        Reads damping configuration from a nested ``damping`` dict with
        fallback to flat ``eta_m``/``eta_k`` keys for backward compatibility.
        Also reads ``solver_type`` and force sanity-check parameters.
        """
        # --- Damping configuration ---
        damping_cfg = self.solver_params.get("damping") or {}

        self._damping_enabled: bool = damping_cfg.get("enabled", True)
        # Auto mode: compute coefficients from modal analysis at assembly time
        self._damping_auto: bool = (
            self._damping_enabled
            and damping_cfg.get("eta_m") is None
            and damping_cfg.get("eta_k") is None
            and bool(damping_cfg)  # only auto when damping section is present
        )
        self._damping_cfg: dict = damping_cfg

        if not self._damping_enabled:
            self._eta_m = 0.0
            self._eta_k = 0.0
        elif self._damping_auto:
            # Placeholder — will be overwritten in solve() via _compute_rayleigh_auto
            self._eta_m = 0.0
            self._eta_k = 0.0
        else:
            # Manual mode: read from nested damping dict, fall back to flat keys
            self._eta_m = float(
                damping_cfg["eta_m"]
                if damping_cfg.get("eta_m") is not None
                else self.solver_params.get("eta_m", 0.0)
            )
            self._eta_k = float(
                damping_cfg["eta_k"]
                if damping_cfg.get("eta_k") is not None
                else self.solver_params.get("eta_k", 0.0)
            )

        # --- Solver type configuration ---
        self._solver_type = self.solver_params.get("solver_type", "auto")

        # --- Force sanity checks ---
        self._force_ramp_time = float(self.solver_params.get("force_ramp_time", 0.0))

        _fmax = self.solver_params.get("force_max_magnitude", None)
        self._force_max_magnitude: Optional[float] = float(_fmax) if _fmax is not None else None
        self._force_jump_factor: float = float(self.solver_params.get("force_jump_factor", 1000.0))
        self._max_force_seen: float = 0.0

        # On restart, CFD may produce transient force spikes while it re-stabilizes.
        # During the grace period, force spikes are clamped instead of aborting.
        self._restart_force_grace_windows: int = int(
            self.solver_params.get("restart_force_grace_windows", 5)
        )
        self._restart_grace_remaining: int = 0

        # On restart from latest checkpoint, force ramps must not be re-applied.

        # --- Probe monitoring ---
        self._probe_node_ids: list[int] = []
        self._probe_file: Optional[str] = None
        self._skip_ramps: bool = False

    # =========================================================================
    # Solver Setup
    # =========================================================================

    def _setup_solver(self) -> None:
        """Configure PETSc linear solver based on solver_type and problem size."""
        self._solver = PETSc.KSP().create(self.comm)
        pc = self._solver.getPC()
        opts = PETSc.Options()

        dof_count = self.domain.dofs_count
        solver_type = self._solver_type

        # Auto-select solver based on problem size
        if solver_type == "auto":
            solver_type = "direct" if dof_count < _DOF_THRESHOLD_DIRECT_SOLVER else "iterative"

        if solver_type == "direct":
            self._configure_direct_solver(pc)
        else:
            self._configure_iterative_solver(pc, opts)

        self._solver.setFromOptions()
        self._keff_last: Optional[PETSc.Mat] = None  # tracks last operator to skip redundant setOperators
        self._setup_mass_solver()

    @staticmethod
    def _has_mumps() -> bool:
        """Check if MUMPS is available in the current PETSc installation."""
        try:
            test = PETSc.Mat().createAIJ(size=[2, 2], comm=PETSc.COMM_SELF)
            test.setUp()
            test.setValue(0, 0, 1.0)
            test.setValue(1, 1, 1.0)
            test.assemble()
            test.getOrdering("natural")
            fmat = PETSc.Mat().createAIJ(size=[2, 2], comm=PETSc.COMM_SELF)
            fmat.setUp()
            fmat.assemble()
            try:
                ksp = PETSc.KSP().create(PETSc.COMM_SELF)
                ksp.setType("preonly")
                ksp_pc = ksp.getPC()
                ksp_pc.setType("lu")
                ksp_pc.setFactorSolverType("mumps")
                ksp.setOperators(test)
                rhs = test.createVecRight()
                rhs.set(1.0)
                sol = test.createVecRight()
                ksp.solve(rhs, sol)
                available = ksp.getConvergedReason() >= 0
                ksp.destroy()
                rhs.destroy()
                sol.destroy()
            except PETSc.Error:
                available = False
            test.destroy()
            fmat.destroy()
            return available
        except Exception:
            return False

    def _configure_direct_solver(self, pc: PETSc.PC) -> None:
        """Configure direct LU solver - best for small/medium problems.

        Uses MUMPS if available, otherwise falls back to PETSc native LU.
        """
        self._solver.setType("preonly")
        pc.setType("lu")
        if self._has_mumps():
            pc.setFactorSolverType("mumps")
        else:
            pc.setFactorSolverType("petsc")
        self._solver.setTolerances(rtol=_DIRECT_SOLVER_RTOL, atol=_DIRECT_SOLVER_ATOL, max_it=1)

    def _configure_iterative_solver(self, pc: PETSc.PC, opts: PETSc.Options) -> None:
        """Configure iterative solver with GAMG preconditioner.

        Uses conservative aggregation settings suitable for anisotropic shell
        meshes (high aspect ratios, mixed edge lengths).
        """
        self._solver.setType("cg")
        pc.setType("gamg")
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 2
        opts["pc_gamg_threshold"] = 0.005
        opts["pc_gamg_square_graph"] = 1
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_ksp_max_it"] = 4
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "lu"
        self._solver.setTolerances(
            rtol=_ITERATIVE_SOLVER_RTOL,
            atol=_ITERATIVE_SOLVER_ATOL,
            max_it=_ITERATIVE_SOLVER_MAX_IT,
        )

    def _solve_linear_system(
        self, K_eff: PETSc.Mat, F_eff: PETSc.Vec
    ) -> tuple[PETSc.Vec, int, int]:
        """Solve K_eff * u = F_eff with automatic fallback on divergence.

        Fallback chain (iterative solver only):
          1. Primary CG+GAMG solve
          2. BCGS+GAMG with threshold=0 (handles indefinite PC)
          3. Direct LU (PETSc native, then MUMPS if available)

        Returns (u_new, ksp_iterations, ksp_converged_reason).
        """
        u_new = K_eff.createVecRight()
        # Only update operators if K_eff changed since the last call.  Within a
        # time window K_eff is constant, so this avoids redundant LU
        # re-factorizations for direct solvers and GAMG hierarchy rebuilds for
        # iterative solvers on every IQN-ILS coupled iteration.
        if K_eff is not self._keff_last:
            self._solver.setOperators(K_eff)
            self._keff_last = K_eff
        self._solver.solve(F_eff, u_new)

        ksp_its = self._solver.getIterationNumber()
        ksp_reason = self._solver.getConvergedReason()

        is_iterative = self._solver.getType() != "preonly"
        if ksp_reason >= 0 or not is_iterative:
            return u_new, ksp_its, ksp_reason

        # --- Fallback 1: BCGS+GAMG threshold=0 ---
        # CG requires SPD preconditioner; GAMG aggregation can produce
        # indefinite actions (reason=-8).  BCGS has no such requirement.
        _console.print(Panel(
            f"[yellow]KSP diverged (reason={ksp_reason})[/yellow]\n"
            f"CG+GAMG failed — switching to [bold]BCGS + GAMG(threshold=0)[/bold]",
            title="⚠ Solver Fallback 1/2",
            border_style="yellow",
        ))
        fb1 = PETSc.KSP().create(self.comm)
        fb1.setType("bcgs")
        fb1_pc = fb1.getPC()
        fb1_pc.setType("gamg")
        opts = PETSc.Options()
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 2
        opts["pc_gamg_threshold"] = 0.0
        opts["pc_gamg_square_graph"] = 1
        opts["pc_gamg_sym_graph"] = True
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_ksp_max_it"] = 4
        opts["mg_coarse_ksp_type"] = "preonly"
        opts["mg_coarse_pc_type"] = "lu"
        fb1.setTolerances(
            rtol=_ITERATIVE_SOLVER_RTOL,
            atol=_ITERATIVE_SOLVER_ATOL,
            max_it=_ITERATIVE_SOLVER_MAX_IT,
        )
        fb1.setOperators(K_eff)
        fb1.setFromOptions()

        u_new.zeroEntries()
        fb1.solve(F_eff, u_new)
        ksp_its = fb1.getIterationNumber()
        ksp_reason = fb1.getConvergedReason()
        fb1.destroy()

        if ksp_reason >= 0:
            _console.print(
                f"  [green]✓ BCGS+GAMG converged[/green] │ "
                f"reason={ksp_reason}  its={ksp_its}"
            )
            return u_new, ksp_its, ksp_reason

        # --- Fallback 2: Direct LU solver ---
        # Try without specifying MUMPS first (PETSc native LU always exists).
        # If the build has MUMPS, try that as a second attempt.
        _console.print(Panel(
            f"[red]BCGS+GAMG also diverged (reason={ksp_reason})[/red]\n"
            f"Attempting [bold]direct LU factorization[/bold]",
            title="⚠ Solver Fallback 2/2",
            border_style="red",
        ))
        for solver_pkg in (None, "mumps"):
            pkg_label = solver_pkg or "petsc-native"
            if solver_pkg == "mumps" and not self._has_mumps():
                continue
            fb2 = PETSc.KSP().create(self.comm)
            fb2.setType("preonly")
            fb2_pc = fb2.getPC()
            fb2_pc.setType("lu")
            if solver_pkg is not None:
                fb2_pc.setFactorSolverType(solver_pkg)
            fb2.setTolerances(rtol=_DIRECT_SOLVER_RTOL, atol=_DIRECT_SOLVER_ATOL, max_it=1)
            fb2.setOperators(K_eff)
            fb2.setFromOptions()
            try:
                u_new.zeroEntries()
                fb2.solve(F_eff, u_new)
                ksp_its = fb2.getIterationNumber()
                ksp_reason = fb2.getConvergedReason()
                fb2.destroy()
                if ksp_reason >= 0:
                    _console.print(
                        f"  [green]✓ Direct LU ({pkg_label}) converged[/green] │ "
                        f"reason={ksp_reason}"
                    )
                    return u_new, ksp_its, ksp_reason
            except PETSc.Error as exc:
                _console.print(f"  [yellow]✗ LU ({pkg_label}) unavailable: {exc}[/yellow]")
                fb2.destroy()

        _console.print(Panel(
            "[bold red]ALL solver fallbacks exhausted — solution may be invalid[/bold red]",
            title="✗ Solver Failure",
            border_style="red bold",
        ))
        return u_new, 0, -1

    def _setup_mass_solver(self) -> None:
        """Setup mass solver with diagonal preconditioner for lumped mass."""
        self._m_solver = PETSc.KSP().create(self.comm)
        self._m_solver.setType("preonly")
        m_pc = self._m_solver.getPC()
        m_pc.setType("jacobi")
        self._m_solver.setTolerances(rtol=_DIRECT_SOLVER_RTOL, max_it=1)
        self._m_solver.setFromOptions()

    def lump_mass_matrix(self, M: PETSc.Mat) -> PETSc.Mat:
        """Convert mass matrix M to lumped (diagonal) form via row-sum technique.

        The critical implementation detail: ``Mat.createAIJ`` must be called
        with ``nnz=1`` so PETSc preallocates one non-zero per row BEFORE
        ``setDiagonal`` is called.  Without preallocation, ``setDiagonal``
        fails silently and the returned matrix is all zeros, which makes the
        Newmark effective stiffness K_eff ≈ K (no mass term) and the time
        integration degenerates to quasi-static.

        A mass floor of 1e-4 × global_max is also applied via PETSc-native
        ``pointwiseMax`` to handle tip-closure elements whose thickness → 0.
        """
        diag = PETSc.Vec().createMPI(M.getSize()[0], comm=M.getComm())
        M.getRowSum(diag)  # row sum = lumped mass per DOF

        # Apply floor using PETSc operations (avoids numpy-view mutability issues).
        _, m_max = diag.max()  # returns (global_index, global_max_value)
        if m_max > 0.0:
            floor_vec = diag.duplicate()
            floor_vec.set(m_max * 1e-4)
            diag.pointwiseMax(diag, floor_vec)
            floor_vec.destroy()

        # nnz=1 preallocates exactly one non-zero per row (the diagonal).
        # Without this, setDiagonal silently discards all insertions.
        M_lumped = PETSc.Mat().createAIJ(size=M.getSize(), comm=M.getComm(), nnz=1)
        M_lumped.setUp()
        M_lumped.setDiagonal(diag)
        M_lumped.assemble()
        return M_lumped

    # =========================================================================
    # Matrix Assembly Helpers
    # =========================================================================

    def _create_damping_matrix(self, K: PETSc.Mat, M: PETSc.Mat) -> Optional[PETSc.Mat]:
        """Create Rayleigh damping matrix C = η_m·M + η_k·K.

        Returns None if both coefficients are zero.
        """
        if self._eta_m == 0.0 and self._eta_k == 0.0:
            return None

        C = K.copy()
        C.scale(self._eta_k)
        C.axpy(self._eta_m, M)
        return C

    def _build_effective_stiffness(
        self,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        coeffs: NewmarkCoefficients,
        C_red: Optional[PETSc.Mat] = None,
    ) -> PETSc.Mat:
        """Build the Newmark effective stiffness matrix.

        Assembles K_eff for the implicit Newmark-β scheme:

            K_eff = [K] + a₀·[M] + a₁·[C]

        where a₀ = 1/(β·dt²) and a₁ = γ/(β·dt).

        Parameters
        ----------
        K_red : PETSc.Mat
            Reduced elastic stiffness matrix.
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        coeffs : NewmarkCoefficients
            Precomputed Newmark integration coefficients.
        C_red : PETSc.Mat, optional
            Reduced Rayleigh damping matrix.

        Returns
        -------
        PETSc.Mat
            Effective stiffness matrix ready for KSP solve.
        """
        K_eff = K_red.copy()
        K_eff.axpy(coeffs.a0, M_red)
        if C_red is not None:
            K_eff.axpy(coeffs.a1, C_red)
        return K_eff

    def _compute_effective_force(
        self,
        F_new_red: PETSc.Vec,
        u: PETSc.Vec,
        v: PETSc.Vec,
        a: PETSc.Vec,
        M_red: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        coeffs: NewmarkCoefficients,
    ) -> PETSc.Vec:
        """Compute the Newmark effective force vector.

        Assembles the RHS of the Newmark implicit system:

            F_eff = {F_new} + [M]·(a₀·u + a₂·v + a₃·a) + [C]·(a₁·u + a₄·v + a₅·a)

        Parameters
        ----------
        F_new_red : PETSc.Vec
            Reduced external force vector at time t+dt.
        u, v, a : PETSc.Vec
            Current displacement, velocity, acceleration (reduced).
        M_red : PETSc.Mat
            Reduced lumped mass matrix.
        C_red : PETSc.Mat, optional
            Reduced damping matrix.
        coeffs : NewmarkCoefficients
            Precomputed integration coefficients.

        Returns
        -------
        PETSc.Vec
            Effective force vector for the Newmark solve.
        """
        # Mass contribution: M·(a₀·u + a₂·v + a₃·a)
        temp_vec = u.duplicate()
        u.copy(temp_vec)
        temp_vec.scale(coeffs.a0)
        temp_vec.axpy(coeffs.a2, v)
        temp_vec.axpy(coeffs.a3, a)

        temp_result = M_red.createVecRight()
        M_red.mult(temp_vec, temp_result)

        F_eff = F_new_red.duplicate()
        F_new_red.copy(F_eff)
        F_eff.axpy(1.0, temp_result)

        # Damping contribution: C·(a₁·u + a₄·v + a₅·a)
        if C_red is not None:
            temp_vec_c = u.duplicate()
            u.copy(temp_vec_c)
            temp_vec_c.scale(coeffs.a1)
            temp_vec_c.axpy(coeffs.a4, v)
            temp_vec_c.axpy(coeffs.a5, a)

            temp_result_c = C_red.createVecRight()
            C_red.mult(temp_vec_c, temp_result_c)
            F_eff.axpy(1.0, temp_result_c)

            temp_vec_c.destroy()
            temp_result_c.destroy()

        temp_vec.destroy()
        temp_result.destroy()

        return F_eff

    def _compute_rayleigh_auto(self) -> Tuple[float, float]:
        """Compute Rayleigh damping coefficients automatically from modal analysis.

        Uses the two-point method: given damping ratios ζ_i, ζ_j at two
        natural frequencies ω_i, ω_j, solves the system:

            α (η_k) = 2·(ζ_i·ω_i − ζ_j·ω_j) / (ω_i² − ω_j²)
            β (η_m) = 2·ω_i·ω_j·(ζ_j·ω_i − ζ_i·ω_j) / (ω_i² − ω_j²)

        Returns
        -------
        Tuple[float, float]
            (eta_k, eta_m)
        """
        from slepc4py import SLEPc

        cfg = self._damping_cfg
        zeta = float(cfg.get("zeta", 0.02))
        zeta_i = float(cfg["zeta_1"]) if cfg.get("zeta_1") is not None else zeta
        zeta_j = float(cfg["zeta_2"]) if cfg.get("zeta_2") is not None else zeta
        mode_i = int(cfg.get("mode_i", 1))
        mode_j = int(cfg.get("mode_j", 2))
        num_modes = int(cfg.get("num_modes", max(mode_j + 2, 6)))

        if mode_i == mode_j:
            raise ValueError("mode_i and mode_j must be different for Rayleigh auto-computation.")

        K_dup = self.K.copy()
        M_dup = self.M.copy()
        F_tmp = K_dup.createVecRight()
        F_tmp.set(0.0)

        bc_tmp = BoundaryConditionManager(K_dup, F_tmp, M_dup, self.domain.dofs_per_node)
        bc_tmp.apply_dirichlet(self.dirichlet_conditions)
        K_red, F_red, M_red = bc_tmp.reduced_system

        eps = SLEPc.EPS().create(self.comm)
        eps.setOperators(K_red, M_red)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)
        ksp_st = st.getKSP()
        ksp_st.setType("preonly")
        pc_st = ksp_st.getPC()
        pc_st.setType("lu")
        pc_st.setFactorSolverType("petsc")

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(0.0)
        eps.setDimensions(num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-8, max_it=500)
        try:
            eps.solve()
        except Exception as exc:
            K_red.destroy()
            if M_red is not None:
                M_red.destroy()
            F_red.destroy()
            K_dup.destroy()
            M_dup.destroy()
            F_tmp.destroy()
            raise RuntimeError(f"Rayleigh auto: SLEPc eigenvalue solve failed — {exc}.") from exc

        nconv = eps.getConverged()
        raw_eigs = sorted(eps.getEigenvalue(i).real for i in range(nconv))
        positive_eigs = [lam for lam in raw_eigs if lam > 1e-8]

        K_red.destroy()
        if M_red is not None:
            M_red.destroy()
        F_red.destroy()
        K_dup.destroy()
        M_dup.destroy()
        F_tmp.destroy()

        needed = max(mode_i, mode_j)
        if len(positive_eigs) < needed:
            raise RuntimeError(
                f"Rayleigh auto: modal solve found only {len(positive_eigs)} "
                f"positive eigenvalues but mode {needed} is required."
            )

        omega_i = float(np.sqrt(positive_eigs[mode_i - 1]))
        omega_j = float(np.sqrt(positive_eigs[mode_j - 1]))

        denom = omega_i**2 - omega_j**2
        if abs(denom) < 1e-12:
            raise ValueError(
                f"Rayleigh auto: modes {mode_i} and {mode_j} share the same "
                f"natural frequency (ω ≈ {omega_i:.4e} rad/s)."
            )

        alpha = 2.0 * (zeta_i * omega_i - zeta_j * omega_j) / denom  # η_k
        beta = 2.0 * omega_i * omega_j * (zeta_j * omega_i - zeta_i * omega_j) / denom  # η_m

        if alpha < 0.0 or beta < 0.0:
            _console.print(
                f"  [yellow bold]⚠ Rayleigh auto: negative coefficient "
                f"η_k={alpha:.3e}  η_m={beta:.3e}[/yellow bold]"
            )

        if self._is_primary_rank():
            f_i = omega_i / (2.0 * np.pi)
            f_j = omega_j / (2.0 * np.pi)
            print(
                f"  [Rayleigh auto] Mode {mode_i}: f={f_i:.3f} Hz "
                f"(ω={omega_i:.3f} rad/s), ζ={zeta_i:.4f}",
                flush=True,
            )
            print(
                f"  [Rayleigh auto] Mode {mode_j}: f={f_j:.3f} Hz "
                f"(ω={omega_j:.3f} rad/s), ζ={zeta_j:.4f}",
                flush=True,
            )
            print(f"  [Rayleigh auto] η_k (stiffness) = {alpha:.4e} s", flush=True)
            print(f"  [Rayleigh auto] η_m (mass)      = {beta:.4e} 1/s", flush=True)

        return alpha, beta

    # =========================================================================
    # Utility
    # =========================================================================

    def _is_primary_rank(self) -> bool:
        """Always True in serial."""
        return True

    def _export_interface_debug_data(
        self,
        step: int,
        time: float,
        forces_raw: Optional[np.ndarray],
        forces_applied: np.ndarray,
        directory: str = "debug_interface",
    ):
        """Export interface debugging data (points + vectors) to VTU."""
        # Only proceed if enabled
        if not self.solver_params.get("debug_interface", False):
            return

        # Only proceed if interface coordinates are available
        if not hasattr(self.precice_participant, "interface_coordinates"):
            return

        out_dir = Path(directory)
        out_dir.mkdir(exist_ok=True, parents=True)

        points = self.precice_participant.interface_coordinates
        n_points = len(points)

        # Prepare point data
        point_data = {}

        if forces_raw is not None:
            point_data["Force CFD Raw"] = (
                forces_raw.reshape(-1, 3) if forces_raw.ndim == 1 else forces_raw
            )

        if forces_applied is not None:
            point_data["Force Applied"] = (
                forces_applied.reshape(-1, 3) if forces_applied.ndim == 1 else forces_applied
            )

        if not point_data:
            return

        # Create vertices for point cloud visualization
        # Note: meshio needs cells. For points, use 'vertex' cells.
        cells = [("vertex", np.arange(n_points).reshape(-1, 1))]

        mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)

        # Write VTU file (points only)
        # Using VTU instead of VTP because Paraview handles UnstructuredGrid of vertices well
        # and it's consistent with other outputs
        filename = f"interface_{step:06d}.vtu"
        full_path = out_dir / filename
        mesh.write(str(full_path))

        self._update_debug_pvd(out_dir, "interface_forces.pvd", filename, time)

    def _update_debug_pvd(self, folder: Path, pvd_name: str, filename: str, time: float):
        """Update PVD file for debug stream."""
        pvd_path = folder / pvd_name

        header = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
  <Collection>
"""
        footer = """  </Collection>
</VTKFile>"""

        entry = f'    <DataSet timestep="{time}" group="" part="0" file="{filename}"/>\n'

        if not pvd_path.exists():
            with open(pvd_path, "w") as f:
                f.write(header + entry + footer)
        else:
            # Read existing content
            with open(pvd_path, "r") as f:
                lines = f.readlines()

            # Remove footer lines to append new entry
            valid_lines = [
                line for line in lines if "</Collection>" not in line and "</VTKFile>" not in line
            ]

            # Append new entry and footer
            with open(pvd_path, "w") as f:
                f.writelines(valid_lines)
                f.write(entry)
                f.write(footer)

    def _expand_interface_forces_to_full(
        self,
        interface_forces: np.ndarray,
    ) -> np.ndarray:
        """
        Expand interface forces to full mesh array for visualization.

        Parameters
        ----------
        interface_forces : np.ndarray
            Forces at interface nodes, shape (n_interface_indices, dim).

        Returns
        -------
        np.ndarray
            Full force array, shape (n_nodes, 3) for VTU visualization.
        """
        n_nodes = self.domain.mesh.node_count
        full_forces = np.zeros((n_nodes, 3), dtype=np.float64)
        dim = interface_forces.shape[1]

        # Get the proper node indices from the adapter
        interface_node_indices = self.precice_participant.interface_node_indices

        # Map interface forces to their corresponding mesh node indices
        for i, node_idx in enumerate(interface_node_indices):
            if node_idx < n_nodes:
                full_forces[node_idx, :dim] = interface_forces[i, :]

        return full_forces

    def _compute_stress_fields(self, u_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute stress and strain fields for checkpoint VTU export."""
        sr = StressRecovery(self.domain, u_full)
        has_shell = any(sr._is_shell(e) for e in self.domain._element_map.values())
        has_solid = any(sr._is_solid(e) for e in self.domain._element_map.values())

        out: Dict[str, np.ndarray] = {}
        if has_shell and not has_solid:
            out.update(sr.compute_nodal_stresses_all_layers_dict(stress_type=StressType.TOTAL))
            out.update(sr.compute_nodal_strains_all_layers_dict())
        elif has_solid and not has_shell:
            result = sr.compute_nodal_stresses()
            out.update(result.to_dict())
            out.update({f"strain_{k}": v for k, v in sr.compute_nodal_strains().to_dict().items()})
        else:
            out.update(sr.compute_nodal_stresses_all_layers_dict(stress_type=StressType.TOTAL))
            out.update(sr.compute_nodal_strains_all_layers_dict())
        return out

    # =========================================================================
    # CSV Report & Probes
    # =========================================================================

    def _init_probes(self) -> None:
        """Resolve probe node IDs from solver_params['probes'].

        Accepts a list of ``[x, y, z]`` coordinates.  Each coordinate is
        matched to the nearest mesh node.  Results are written once per
        converged time window to ``<output_folder>/probes.csv``.
        """
        probe_cfg = self.solver_params.get("probes")
        if not probe_cfg:
            return

        coords = np.array([[n.x, n.y, n.z] for n in self.domain.mesh.nodes])
        resolved: list[int] = []
        probe_table = Table(show_header=True, box=None, padding=(0, 1))
        probe_table.add_column("#", style="bold")
        probe_table.add_column("Target (x, y, z)")
        probe_table.add_column("Node ID", justify="right")
        probe_table.add_column("Distance", justify="right")
        for i, pt in enumerate(probe_cfg):
            pt = np.asarray(pt, dtype=float)
            dists = np.linalg.norm(coords - pt, axis=1)
            idx = int(np.argmin(dists))
            node = self.domain.mesh.nodes[idx]
            resolved.append(idx)
            probe_table.add_row(
                str(i),
                f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})",
                str(node.id),
                f"{dists[idx]:.4e}",
            )
        self._probe_node_ids = resolved
        _console.print(Panel(probe_table, title="Probe Monitoring", border_style="cyan"))

        output_folder = self.solver_params.get("output_folder", "results")
        self._probe_file = str(Path(output_folder) / "probes.csv")

    def _log_structural_report(
        self,
        t: float,
        time_step: int,
        u_full: np.ndarray,
        v_full: np.ndarray,
        a_full: np.ndarray,
        stress_fields: Dict[str, np.ndarray],
        applied_force_mag: float,
    ) -> None:
        """Append one row to ``structural_report.csv`` with key mechanical metrics.

        Written once per converged time window (rank 0 only).

        Columns
        -------
        Time, TimeStep, Max Displacement, Max Displacement Node,
        Max Velocity, Max Acceleration,
        Max VonMises TOP/MID/BOT + node, Max Sigma1 TOP,
        Applied Force Magnitude.
        """
        if not self._is_primary_rank():
            return

        output_folder = self.solver_params.get("output_folder", "results")
        csv_path = Path(output_folder) / "structural_report.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        nodes = self.domain.mesh.nodes
        n_nodes = len(nodes)
        dofs_per_node = u_full.size // n_nodes

        # Displacement magnitude per node
        u_mat = u_full.reshape(n_nodes, dofs_per_node)[:, :3]
        u_mag = np.linalg.norm(u_mat, axis=1)
        max_disp_idx = int(np.argmax(u_mag))
        max_disp = float(u_mag[max_disp_idx])
        max_disp_node = nodes[max_disp_idx].id

        # Velocity / acceleration magnitudes
        v_mat = v_full.reshape(n_nodes, dofs_per_node)[:, :3]
        a_mat = a_full.reshape(n_nodes, dofs_per_node)[:, :3]
        max_vel = float(np.max(np.linalg.norm(v_mat, axis=1)))
        max_acc = float(np.max(np.linalg.norm(a_mat, axis=1)))

        # Stress peaks — TOP/MID/BOT von Mises + TOP sigma_1
        def _peak(key: str) -> tuple[float, int]:
            arr = stress_fields.get(key)
            if arr is None:
                return 0.0, -1
            idx = int(np.argmax(np.abs(arr)))
            return float(arr[idx]), nodes[idx].id if idx < n_nodes else idx

        vm_top, vm_top_nd = _peak("TOP_von_mises")
        vm_mid, vm_mid_nd = _peak("MID_von_mises")
        vm_bot, vm_bot_nd = _peak("BOT_von_mises")
        s1_top, s1_top_nd = _peak("TOP_sigma_1")

        # Position of max displacement node
        nd = nodes[max_disp_idx]
        max_pos = f"{nd.x:.4f};{nd.y:.4f};{nd.z:.4f}"

        write_header = not csv_path.exists()
        try:
            with open(csv_path, "a") as f:
                if write_header:
                    f.write(
                        "Time [s],TimeStep,"
                        "Max Disp [m],Max Disp Node,Max Disp Pos (x;y;z),"
                        "Max Vel [m/s],Max Acc [m/s2],"
                        "VonMises TOP [Pa],VonMises TOP Node,"
                        "VonMises MID [Pa],VonMises MID Node,"
                        "VonMises BOT [Pa],VonMises BOT Node,"
                        "Sigma1 TOP [Pa],Sigma1 TOP Node,"
                        "Applied Force [N]\n"
                    )
                f.write(
                    f"{t:.6f},{time_step},"
                    f"{max_disp:.6e},{max_disp_node},{max_pos},"
                    f"{max_vel:.6e},{max_acc:.6e},"
                    f"{vm_top:.6e},{vm_top_nd},"
                    f"{vm_mid:.6e},{vm_mid_nd},"
                    f"{vm_bot:.6e},{vm_bot_nd},"
                    f"{s1_top:.6e},{s1_top_nd},"
                    f"{applied_force_mag:.6e}\n"
                )
        except Exception as e:
            _console.print(f"  [yellow]⚠ Failed to write structural report: {e}[/yellow]")

    def _log_probe_data(
        self,
        t: float,
        time_step: int,
        u_full: np.ndarray,
        v_full: np.ndarray,
        stress_fields: Dict[str, np.ndarray],
    ) -> None:
        """Append probe data for all monitored nodes to ``probes.csv``.

        One row per time step.  For each probe node: displacement (3),
        velocity magnitude, von Mises TOP.
        """
        if not self._probe_node_ids or not self._is_primary_rank():
            return

        nodes = self.domain.mesh.nodes
        n_nodes = len(nodes)
        dofs_per_node = u_full.size // n_nodes
        u_mat = u_full.reshape(n_nodes, dofs_per_node)
        v_mat = v_full.reshape(n_nodes, dofs_per_node)

        vm_top = stress_fields.get("TOP_von_mises")

        csv_path = self._probe_file
        write_header = not Path(csv_path).exists()

        try:
            with open(csv_path, "a") as f:
                if write_header:
                    cols = ["Time [s]", "TimeStep"]
                    for i, nid in enumerate(self._probe_node_ids):
                        nd = nodes[nid]
                        tag = f"P{i}(n{nd.id})"
                        cols.extend([
                            f"{tag} Ux [m]", f"{tag} Uy [m]", f"{tag} Uz [m]",
                            f"{tag} |V| [m/s]",
                            f"{tag} VonMises TOP [Pa]",
                        ])
                    f.write(",".join(cols) + "\n")

                parts = [f"{t:.6f}", str(time_step)]
                for nid in self._probe_node_ids:
                    ux, uy, uz = float(u_mat[nid, 0]), float(u_mat[nid, 1]), float(u_mat[nid, 2])
                    vmag = float(np.linalg.norm(v_mat[nid, :3]))
                    vm = float(vm_top[nid]) if vm_top is not None else 0.0
                    parts.extend([
                        f"{ux:.6e}", f"{uy:.6e}", f"{uz:.6e}",
                        f"{vmag:.6e}", f"{vm:.6e}",
                    ])
                f.write(",".join(parts) + "\n")
        except Exception as e:
            _console.print(f"  [yellow]⚠ Failed to write probe data: {e}[/yellow]")

    def solve(self):
        """Perform dynamic analysis using improved Newmark-β method."""

        _console.print()
        _console.rule("[bold]FSI Dynamic Analysis — Structural Solver[/bold]", style="blue")

        # =====================================================================
        # Phase 1: Matrix assembly
        # =====================================================================
        print("  [1/7] Assembling stiffness matrix...", flush=True)
        self.K = self.domain.assemble_stiffness_matrix()

        print("  [2/7] Assembling mass matrix...", flush=True)
        self.M = self.domain.assemble_mass_matrix()
        self.M = self.lump_mass_matrix(self.M)

        # --- mass diagnostic: sum translational DOFs of M_lumped ---
        _m_diag = self.M.createVecRight()
        self.M.getDiagonal(_m_diag)
        _diag_arr = _m_diag.array
        _dofs_per_node = self.domain.dofs_per_node
        _m_total = (
            _diag_arr[0::_dofs_per_node].sum()
            + _diag_arr[1::_dofs_per_node].sum()
            + _diag_arr[2::_dofs_per_node].sum()
        ) / 3.0
        print(f"        Total blade mass (M_lumped translational DOFs): {_m_total:.4e} kg", flush=True)
        _m_diag.destroy()

        # =====================================================================
        # Phase 2: Damping matrix
        # =====================================================================
        C = None
        if not self._damping_enabled:
            print("  [3/7] Rayleigh damping: disabled", flush=True)
        elif self._damping_auto:
            print("  [3/7] Rayleigh damping: auto-computing from modal analysis...", flush=True)
            self._eta_k, self._eta_m = self._compute_rayleigh_auto()
            print(
                f"        Creating Rayleigh damping (η_m={self._eta_m:.4e}, η_k={self._eta_k:.4e})...",
                flush=True,
            )
            C = self._create_damping_matrix(self.K, self.M)
        elif self._eta_m != 0.0 or self._eta_k != 0.0:
            print(
                f"  [3/7] Creating Rayleigh damping (η_m={self._eta_m}, η_k={self._eta_k})...",
                flush=True,
            )
            C = self._create_damping_matrix(self.K, self.M)
        else:
            print("  [3/7] Rayleigh damping: disabled (zero coefficients)", flush=True)

        force_temp = PETSc.Vec().createMPI(self.domain.dofs_count, comm=self.comm)
        force_temp.set(0.0)
        self.F = force_temp

        # =====================================================================
        # Phase 3: Boundary conditions
        # =====================================================================
        print("  [4/7] Applying boundary conditions...", flush=True)
        bc_manager = BoundaryConditionManager(self.K, self.F, self.M, self.domain.dofs_per_node)
        bc_manager.apply_dirichlet(self.dirichlet_conditions)
        print(
            f"        Fixed: {len(bc_manager.fixed_dofs)} DOFs, Free: {len(bc_manager.free_dofs)} DOFs",
            flush=True,
        )

        t = self.solver_params.get("start_time", 0.0)
        step = 0
        time_step = int(round(t / self.solver_params.get("time_step", 1.0)))

        # =====================================================================
        # Phase 4: preCICE initialization
        # =====================================================================
        print("  [5/7] Initializing preCICE coupling...", flush=True)
        self.precice_participant.initialize(
            self.domain,
            self.model_properties["solver"]["coupling_boundaries"],
            tuple(bc_manager.fixed_dofs.keys()),
        )

        self.dt = self.precice_participant.dt
        K_red, F_red, M_red = bc_manager.reduced_system
        C_red = bc_manager.reduce_matrix(C) if C is not None else None

        self.free_dofs = bc_manager.free_dofs

        # Tell GAMG to aggregate at the node level (dofs_per_node DOFs per block).
        # Critical for shell elements (6 DOFs/node) where GAMG must coarsen all
        # DOFs of a node together to avoid indefinite preconditioners.
        K_red.setBlockSize(self.domain.dofs_per_node)

        # =====================================================================
        # Phase 5: Solver setup + effective stiffness
        # =====================================================================
        print("  [6/7] Setting up linear solver...", flush=True)
        if not self._prepared:
            self._setup_solver()
            self._prepared = True

        beta = self.solver_params["beta"]
        gamma = self.solver_params["gamma"]
        coeffs = NewmarkCoefficients.from_newmark_params(beta, gamma, self.dt)

        K_eff = self._build_effective_stiffness(K_red, M_red, coeffs, C_red)

        # --- K_eff composition diagnostic ---
        _diag_K = K_red.createVecRight()
        _diag_M = K_red.createVecRight()
        _diag_Keff = K_red.createVecRight()
        K_red.getDiagonal(_diag_K)
        M_red.getDiagonal(_diag_M)
        K_eff.getDiagonal(_diag_Keff)
        print(
            f"        K_eff composition:  ||K_diag||={_diag_K.norm():.4e}"
            f"  a0*||M_diag||={coeffs.a0 * _diag_M.norm():.4e}"
            f"  ||K_eff_diag||={_diag_Keff.norm():.4e}",
            flush=True,
        )
        _diag_K.destroy(); _diag_M.destroy(); _diag_Keff.destroy()

        # IMPORTANT: For direct solvers (preonly+LU), PETSc factorizes the
        # preconditioner matrix (2nd argument).  Passing K_red as P would make
        # LU solve K_red·u = F instead of K_eff·u = F — ignoring mass/damping.
        # Always use K_eff for both operator and preconditioner.  For iterative
        # solvers, K_eff is also a valid (and correct) preconditioner; using
        # K_red as an approximate PC for GAMG is an optional optimization that
        # should be handled inside _solve_linear_system if needed.
        self._solver.setOperators(K_eff)
        self._keff_last = K_eff  # mark as already configured; _solve_linear_system will skip setOperators

        # =====================================================================
        # Phase 6: Initial conditions
        # =====================================================================
        print("  [7/7] Setting initial conditions...", flush=True)
        u = K_red.createVecRight()
        v = K_red.createVecRight()

        residual = F_red.duplicate()
        residual.copy(F_red)
        K_red.mult(u, residual)
        residual.scale(-1.0)
        residual.axpy(1.0, F_red)

        a = M_red.createVecRight()
        self._m_solver.setOperators(M_red)
        self._m_solver.solve(residual, a)

        # Try to restore from checkpoint if start_from='latestTime'
        checkpoint_state = self._try_restore_checkpoint()
        starting_from_zero = True

        # Validate checkpoint dimensionality if present
        if checkpoint_state is not None and "u_red" in checkpoint_state:
            stored_dofs = len(checkpoint_state["u_red"])
            current_dofs = len(u.array)
            if stored_dofs != current_dofs:
                print(
                    f"  ⚠️ Checkpoint DOF mismatch ({stored_dofs} vs {current_dofs}). "
                    "Incompatible mesh/BCs. Checkpoint ignored.",
                    flush=True,
                )
                checkpoint_state = None

        if checkpoint_state is not None:
            starting_from_zero = False
            print(f"  ✓ Restored from checkpoint at t = {checkpoint_state['t']:.6f} s", flush=True)
            t = checkpoint_state["t"]
            time_step = checkpoint_state["time_step"]
            if "u_red" in checkpoint_state:
                u.array[:] = checkpoint_state["u_red"]
                v.array[:] = checkpoint_state["v_red"]
                a.array[:] = checkpoint_state["a_red"]
            else:
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                print(
                    "  ↳ Using checkpoint time only; state vectors reinitialized to zero",
                    flush=True,
                )

            if self.solver_params.get("reset_state_on_restart", False):
                u.array[:] = 0.0
                v.array[:] = 0.0
                a.array[:] = 0.0
                print(
                    "  ↳ State vectors reset to zero; time/time_step preserved from checkpoint",
                    flush=True,
                )

            ckpt_max_force = checkpoint_state.get("max_force_seen", None)
            if ckpt_max_force is not None:
                self._max_force_seen = float(ckpt_max_force)
                print(
                    f"  ↳ Force baseline restored: |F|_max = {self._max_force_seen:.3e} N",
                    flush=True,
                )

            # For true restart with state vectors, avoid re-ramping and enable
            # temporary force spike grace to let CFD recover.
            if "u_red" in checkpoint_state:
                self._skip_ramps = True
                self._restart_grace_remaining = self._restart_force_grace_windows
                if self._restart_grace_remaining > 0:
                    print(
                        f"  ↳ Force grace period: {self._restart_grace_remaining} windows"
                        " (clamp instead of abort)",
                        flush=True,
                    )

        # Write initial state at t=0 with undeformed mesh if configured
        if (
            self._checkpoint_manager is not None
            and self.solver_params.get("write_initial_state", True)
            and starting_from_zero
            and t == 0
        ):
            u_full_init = bc_manager.expand_solution(u).array.copy()
            zeros_full = np.zeros((self.domain.mesh.node_count, 3), dtype=np.float64)
            initial_fields = {
                "F_AERO": zeros_full,
                "F_TOTAL": zeros_full.copy(),
            }
            initial_fields.update(self._compute_stress_fields(u_full_init))
            self._handle_checkpoint(
                t=0.0,
                time_step=0,
                dt=self.dt,
                u_red=u.array.copy(),
                v_red=v.array.copy(),
                a_red=a.array.copy(),
                u_full=u_full_init,
                v_full=bc_manager.expand_state_vector(v).array.copy(),
                a_full=bc_manager.expand_state_vector(a).array.copy(),
                extra_fields=initial_fields,
            )

        # =====================================================================
        # Force configuration
        # =====================================================================
        force_max_cap = self.solver_params.get("force_max_cap", None)
        force_clipper = ForceClipper(force_max_cap=force_max_cap)
        ramp_time = self._force_ramp_time

        cfg_table = Table(show_header=False, box=None, padding=(0, 1))
        cfg_table.add_column("Key", style="bold")
        cfg_table.add_column("Value")
        cfg_table.add_row("dt", f"{self.dt:.6f} s")
        cfg_table.add_row("Newmark", f"β={beta:.2f}  γ={gamma:.2f}")
        if self._eta_m != 0.0 or self._eta_k != 0.0:
            cfg_table.add_row("Damping", f"η_m={self._eta_m:.4e}  η_k={self._eta_k:.4e}")
        cfg_table.add_row("Solver", self._solver_type)
        if force_max_cap:
            cfg_table.add_row("Force cap", f"{force_max_cap:.2e} N")
        if ramp_time > 0:
            cfg_table.add_row("Ramp", f"linear  T={ramp_time:.4f} s")
        if self._force_max_magnitude is not None:
            cfg_table.add_row("Divergence limit", f"{self._force_max_magnitude:.2e} N")
        _console.print(Panel(cfg_table, title="FSI Solver Configuration", border_style="blue"))

        # Initialize probe monitoring (resolves nearest nodes)
        self._init_probes()

        # Store interface DOFs reference for checkpoint handling
        self._interface_dofs = self.precice_participant.interface_dofs

        # Get interface mesh dimensions
        mesh_dim = self.precice_participant.mesh_dimensions

        # =====================================================================
        # Main coupling loop
        # =====================================================================
        while self.precice_participant.is_coupling_ongoing:
            step += 1

            if self.precice_participant.requires_writing_checkpoint:
                self.precice_participant.store_checkpoint((u, v, a, t))

            # Read coupling data from preCICE
            data = self.precice_participant.read_data()
            data_raw = data.copy() if data is not None else None
            interface_dofs = self.precice_participant.interface_dofs

            # --- Mapping conservation diagnostic ---
            if data is not None and data.ndim == 2:
                _diag_sum = np.sum(data, axis=0)
                _diag_mag = np.linalg.norm(_diag_sum)
                print(
                    f"---[preciceAdapter] READ \"Force\" (solid, {data.shape[0]} pts): "
                    f"max|nodal|={np.max(np.linalg.norm(data, axis=1)):.4e}  "
                    f"|sum|={_diag_mag:.4e}  "
                    f"sum=({_diag_sum[0]:.4e}, {_diag_sum[1]:.4e}, {_diag_sum[2]:.4e})",
                    flush=True,
                )

            # ==================================================================
            # Force analysis: raw CFD forces
            # ==================================================================
            if data.ndim == 2:
                n_nodes = data.shape[0]
                raw_force_x = np.sum(data[:, 0])
                raw_force_y = np.sum(data[:, 1])
                raw_force_z = np.sum(data[:, 2]) if data.shape[1] >= 3 else 0.0
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data, axis=1))
            else:
                n_nodes = len(data) // mesh_dim
                data_2d = data.reshape(-1, mesh_dim)
                raw_force_x = np.sum(data_2d[:, 0])
                raw_force_y = np.sum(data_2d[:, 1])
                raw_force_z = np.sum(data_2d[:, 2]) if mesh_dim >= 3 else 0.0
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))

            # ==================================================================
            # Force sanity checks: detect CFD divergence
            # ==================================================================
            force_total_mag = raw_force_mag
            force_jump_exceeded = (
                self._max_force_seen > 0.0
                and force_total_mag > self._force_jump_factor * self._max_force_seen
            )
            if force_jump_exceeded:
                if self._restart_grace_remaining > 0:
                    clamp_scale = self._max_force_seen / force_total_mag
                    data = data * clamp_scale
                    if data.ndim == 2:
                        raw_force_x = np.sum(data[:, 0])
                        raw_force_y = np.sum(data[:, 1])
                        raw_force_z = np.sum(data[:, 2]) if data.shape[1] >= 3 else 0.0
                        raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                        raw_max_nodal = np.max(np.linalg.norm(data, axis=1))
                    else:
                        data_2d = data.reshape(-1, mesh_dim)
                        raw_force_x = np.sum(data_2d[:, 0])
                        raw_force_y = np.sum(data_2d[:, 1])
                        raw_force_z = np.sum(data_2d[:, 2]) if mesh_dim >= 3 else 0.0
                        raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                        raw_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))
                    force_total_mag = raw_force_mag
                    print(
                        f"  ⚠ Force clamped (grace {self._restart_grace_remaining}): "
                        f"|F| = {force_total_mag:.3e} N",
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        f"Diverged force detected: |F| = {force_total_mag:.3e} N is "
                        f"{force_total_mag / self._max_force_seen:.1f}× the running "
                        f"max. CFD likely diverged."
                    )
            if (
                self._force_max_magnitude is not None
                and force_total_mag > self._force_max_magnitude
            ):
                if self._restart_grace_remaining > 0:
                    clamp_scale = self._force_max_magnitude / force_total_mag
                    data = data * clamp_scale
                    force_total_mag = self._force_max_magnitude
                    print(
                        f"  ⚠ Force clamped to limit (grace {self._restart_grace_remaining}): "
                        f"|F| = {force_total_mag:.3e} N",
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        f"Force exceeds limit: |F| = {force_total_mag:.3e} N > "
                        f"{self._force_max_magnitude:.3e} N"
                    )

            # Recompute raw force diagnostics after potential restart clamping.
            if data.ndim == 2:
                n_nodes = data.shape[0]
                raw_force_x = np.sum(data[:, 0])
                raw_force_y = np.sum(data[:, 1])
                raw_force_z = np.sum(data[:, 2]) if data.shape[1] >= 3 else 0.0
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data, axis=1))
            else:
                n_nodes = len(data) // mesh_dim
                data_2d = data.reshape(-1, mesh_dim)
                raw_force_x = np.sum(data_2d[:, 0])
                raw_force_y = np.sum(data_2d[:, 1])
                raw_force_z = np.sum(data_2d[:, 2]) if mesh_dim >= 3 else 0.0
                raw_force_mag = np.sqrt(raw_force_x**2 + raw_force_y**2 + raw_force_z**2)
                raw_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))
            force_total_mag = raw_force_mag

            # ==================================================================
            # Force clipping
            # ==================================================================
            data_clipped, clip_diags = force_clipper.apply(data)

            # ==================================================================
            # Force ramping (physical time based) — linear ramp
            # ==================================================================
            # NOTE: Linear ramp f(t)=t/T gives O(t) growth near origin,
            # ensuring non-zero applied forces from the first time step.
            # A cosine ramp 0.5*(1-cos(πt/T)) has O(t²) near origin,
            # which can produce forces so small that the KSP solver
            # converges with u=0 (ATOL convergence), causing preCICE
            # IQN-ILS to receive identical zero displacements and crash.
            t_target = t + self.dt
            if self._skip_ramps:
                ramp_factor = 1.0
            elif ramp_time > 0:
                ramp_factor = min(t_target / ramp_time, 1.0)
            else:
                ramp_factor = 1.0

            data_ramped = data_clipped * ramp_factor
            if not force_jump_exceeded and ramp_factor >= 1.0:
                self._max_force_seen = max(self._max_force_seen, force_total_mag)

            # ==================================================================
            # Applied forces (after clipping + ramping)
            # ==================================================================
            if data_ramped.ndim == 2:
                applied_force_x = np.sum(data_ramped[:, 0])
                applied_force_y = np.sum(data_ramped[:, 1])
                applied_force_z = np.sum(data_ramped[:, 2]) if data_ramped.shape[1] >= 3 else 0.0
                applied_force_mag = np.sqrt(
                    applied_force_x**2 + applied_force_y**2 + applied_force_z**2
                )
                applied_max_nodal = np.max(np.linalg.norm(data_ramped, axis=1))
            else:
                data_2d = data_ramped.reshape(-1, mesh_dim)
                applied_force_x = np.sum(data_2d[:, 0])
                applied_force_y = np.sum(data_2d[:, 1])
                applied_force_z = np.sum(data_2d[:, 2]) if mesh_dim >= 3 else 0.0
                applied_force_mag = np.sqrt(
                    applied_force_x**2 + applied_force_y**2 + applied_force_z**2
                )
                applied_max_nodal = np.max(np.linalg.norm(data_2d, axis=1))

            # ==================================================================
            # Logging — per-iteration status table
            # ==================================================================
            iter_table = Table(show_header=False, box=None, padding=(0, 1))
            iter_table.add_column("Key", style="dim")
            iter_table.add_column("Value")
            iter_table.add_row("Time Window", f"[bold cyan]{time_step + 1}[/bold cyan]")
            iter_table.add_row("Iteration", f"[bold]{step}[/bold]")
            iter_table.add_row("Target time", f"{t_target:.6f} s")

            # Force details
            iter_table.add_row("Raw |F|", f"{raw_force_mag:.4e} N  ({n_nodes} nodes)")
            iter_table.add_row(
                "F components",
                f"Fx={applied_force_x:.4e}  Fy={applied_force_y:.4e}  "
                f"Fz={applied_force_z:.4e}",
            )
            if force_max_cap is not None and clip_diags["n_clipped"] > 0:
                iter_table.add_row(
                    "Clipping",
                    f"[yellow]{clip_diags['n_clipped']}/{n_nodes} nodes "
                    f"at {force_max_cap:.2e} N[/yellow]",
                )
            if ramp_time > 0 and ramp_factor < 1.0:
                iter_table.add_row("Ramp", f"{ramp_factor * 100:.1f}%")
            iter_table.add_row("Applied |F|", f"{applied_force_mag:.4e} N")

            # Use ramped data for assembly
            data = data_ramped

            self._export_interface_debug_data(step, t_target, data_raw, data)

            F_new = self.F.copy()
            F_new.setValues(interface_dofs, data)
            F_new_red = bc_manager.reduce_vector(F_new)

            # ==================================================================
            # Effective force (with damping contribution)
            # ==================================================================
            F_eff = self._compute_effective_force(F_new_red, u, v, a, M_red, C_red, coeffs)

            # Solve for displacement
            u_new, ksp_its, ksp_reason = self._solve_linear_system(K_eff, F_eff)

            # Compute solution response for this iteration
            max_disp_iter = u_new.norm(PETSc.NormType.INFINITY)

            # Add solver result to iteration table
            reason_style = "green" if ksp_reason > 0 else "red bold"
            iter_table.add_row(
                "KSP",
                f"its={ksp_its}  reason=[{reason_style}]{ksp_reason}[/{reason_style}]",
            )
            iter_table.add_row("max|u|", f"{max_disp_iter:.4e} m")
            _console.print(Panel(
                iter_table,
                title=f"TW {time_step + 1} · Iteration {step}",
                border_style="dim",
                expand=False,
            ))

            self.precice_participant.write_data(bc_manager.expand_solution(u_new).array)

            # preCICE advance
            self.precice_participant.advance(self.dt)

            if self.precice_participant.requires_reading_checkpoint:
                u, v, a, t = self.precice_participant.retrieve_checkpoint()
            else:
                # Time window completed - increment physical time step counter
                time_step += 1

                # Update acceleration and velocity using Newmark coefficients
                a_new = coeffs.a0 * (u_new - u) - coeffs.a2 * v - coeffs.a3 * a
                v_new = v + coeffs.a6 * a + coeffs.a7 * a_new
                u, v, a = u_new, v_new, a_new
                t += self.dt

                # Log final state after time window completion
                max_disp = u.norm(PETSc.NormType.INFINITY)
                max_vel = v.norm(PETSc.NormType.INFINITY)
                max_acc = a.norm(PETSc.NormType.INFINITY)

                conv_table = Table(show_header=False, box=None, padding=(0, 1))
                conv_table.add_column("Key", style="bold")
                conv_table.add_column("Value")
                conv_table.add_row("Time", f"{t:.6f} s")
                conv_table.add_row("Iterations", str(step - (time_step - 1)))
                conv_table.add_row("max |u|", f"{max_disp:.4e} m")
                conv_table.add_row("max |v|", f"{max_vel:.4e} m/s")
                conv_table.add_row("max |a|", f"{max_acc:.4e} m/s²")
                _console.print(Panel(
                    conv_table,
                    title=f"✓ TW {time_step} Converged",
                    border_style="green bold",
                    expand=False,
                ))

                # Prepare for checkpoint
                u_expanded = bc_manager.expand_solution(u)

                # Prepare force fields for checkpoint
                force_fields = {}
                if data_raw is not None:
                    force_fields["F_AERO_RAW"] = self._expand_interface_forces_to_full(
                        data_raw.reshape(-1, mesh_dim)
                    )

                if data is not None:
                    mapped_forces = self._expand_interface_forces_to_full(
                        data.reshape(-1, mesh_dim)
                    )
                    force_fields["F_AERO"] = mapped_forces
                    force_fields["F_TOTAL"] = mapped_forces.copy()

                force_fields.update(self._compute_stress_fields(u_expanded.array.copy()))

                # CSV reports (rank 0 only)
                u_full_arr = u_expanded.array.copy()
                v_full_arr = bc_manager.expand_state_vector(v).array.copy()
                a_full_arr = bc_manager.expand_state_vector(a).array.copy()

                self._log_structural_report(
                    t=t,
                    time_step=time_step,
                    u_full=u_full_arr,
                    v_full=v_full_arr,
                    a_full=a_full_arr,
                    stress_fields=force_fields,
                    applied_force_mag=applied_force_mag,
                )
                self._log_probe_data(
                    t=t,
                    time_step=time_step,
                    u_full=u_full_arr,
                    v_full=v_full_arr,
                    stress_fields=force_fields,
                )

                # Handle checkpoint writing if enabled
                self._handle_checkpoint(
                    t=t,
                    time_step=time_step,
                    dt=self.dt,
                    max_force_seen=self._max_force_seen,
                    u_red=u.array.copy(),
                    v_red=v.array.copy(),
                    a_red=a.array.copy(),
                    u_full=u_full_arr,
                    v_full=v_full_arr,
                    a_full=a_full_arr,
                    extra_fields=force_fields,
                )

                if self._restart_grace_remaining > 0:
                    self._restart_grace_remaining -= 1

        # Get final clipping statistics
        clip_stats = force_clipper.get_statistics()

        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_column("Key", style="bold")
        summary_table.add_column("Value")
        summary_table.add_row("Final time", f"{t:.6f} s")
        summary_table.add_row("Time steps", str(time_step))
        summary_table.add_row("Total iterations", str(step))
        if time_step > 0:
            summary_table.add_row("Avg iters/step", f"{step / time_step:.2f}")
        if clip_stats["clipped_fraction"] > 0:
            summary_table.add_row(
                "Clipping", f"{100 * clip_stats['clipped_fraction']:.2f}% nodes clipped"
            )
        _console.print(
            Panel(summary_table, title="FSI Simulation Completed", border_style="green")
        )

        # Flush async checkpoints and create PVD index
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.finalize(timeout=60.0)

        self.u = bc_manager.expand_solution(u)
        self.v = bc_manager.expand_state_vector(v)
        self.a = bc_manager.expand_state_vector(a)

        return self.u, self.v, self.a
