"""
StressStiffenedFSISolver — incremental linearization with per-step K_G update.

Physical motivation
-------------------
A standard ``LinearDynamicFSISolver`` assembles the elastic stiffness K once at
t = 0 and keeps it frozen throughout the simulation:

    K_eff = K + a₀·M + a₁·C                (constant for all time)

This neglects *stress stiffening* (geometric nonlinearity): under large membrane
tension (e.g. aerodynamic loading on a flexible blade) the structure is stiffer
than the linearized K predicts, leading to over-estimated tip deformations.

``StressStiffenedFSISolver`` adds a geometric stiffness term K_G that is updated
from the deformed state at the end of each converged time window:

    K_eff^{n+1} = K + K_G(σ^n) + a₀·M + a₁·C

where σ^n is the membrane stress field recovered from the displacement u^n.
This is a *linearized incremental* (also called "frozen-tangent") approach:

* ONE linear solve per time step — no Newton-Raphson inner iterations.
* ONE K_eff factorization per ``update_interval`` converged steps.
* Converges to the correct nonlinear solution as Δt → 0.
* Compatible with direct (MUMPS) and iterative (GAMG) solvers.

Cost relative to base solver
-----------------------------
Let T = MUMPS factorization time, t_s = time per back-substitution,
t_KG = K_G assembly time (one element loop — fast), n = total time steps.

Base:            T + n·t_s
StressStiffened: n/N · (T + t_KG) + n·t_s      (N = update_interval)

For small meshes (≤ 50 k DOFs) where T ≈ 10 · t_s and N = 1:
  Overhead ≈ 10× factorizations — wall-time roughly doubles.
For N = 5: overhead ≈ 2×.
For N = 10: overhead ≈ marginal.

Configuration
-------------
.. code-block:: yaml

    solver:
      type: StressStiffenedDynamicFSI
      geometric_stiffness:
        update_interval: 1   # K_G rebuilt every N converged steps (default 1)

References
----------
- Ko, Y., Lee, P.S., and Bathe, K.J. (2017). "The MITC4+ shell element in
  geometric nonlinear analysis." Computers & Structures, 185, 1-14.
- Bathe, K.J. (1996). Finite Element Procedures. §6.3 Updated Lagrangian.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from petsc4py import PETSc

from fem_shell.postprocess.stress_recovery import StressLocation, StressRecovery, StressType

from .linear_dynamic import LinearDynamicFSISolver

if TYPE_CHECKING:
    from fem_shell.core.bc import BoundaryConditionManager
    from .base import NewmarkCoefficients

_logger = logging.getLogger(__name__)


class StressStiffenedFSISolver(LinearDynamicFSISolver):
    """Implicit Newmark FSI solver with per-step geometric stiffness update.

    Inherits all FSI coupling, checkpointing, force clipping/ramping, and
    damping logic from ``LinearDynamicFSISolver``.  The only difference is
    the ``_post_convergence_hook`` override that rebuilds K_eff with an
    updated K_G = ∫ B_G^T · σ_m · B_G dA assembled from the membrane
    stress of the last converged displacement.

    Parameters
    ----------
    domain : MeshAssembler
        FEM domain (mesh + elements + DOF map).
    solver_params : dict
        Solver configuration dictionary.  In addition to the base-class
        keys, recognises:

        ``geometric_stiffness.update_interval`` : int, default 1
            Rebuild K_G every *N* converged time windows.
            1 = every step (maximum accuracy).
            N > 1 = amortize factorization cost over N steps.
    """

    def __init__(self, domain, solver_params: dict) -> None:
        super().__init__(domain, solver_params)
        gs_cfg = solver_params.get("geometric_stiffness", {})
        self._kg_update_interval: int = max(1, int(gs_cfg.get("update_interval", 1)))
        self._K_G_red: Optional[PETSc.Mat] = None
        _logger.info(
            "StressStiffenedFSISolver: K_G update every %d converged step(s).",
            self._kg_update_interval,
        )

    # ------------------------------------------------------------------
    # Convergence hook
    # ------------------------------------------------------------------

    def _post_convergence_hook(
        self,
        u: PETSc.Vec,
        time_step: int,
        K_eff: PETSc.Mat,
        K_red: PETSc.Mat,
        M_red: PETSc.Mat,
        C_red: Optional[PETSc.Mat],
        coeffs: "NewmarkCoefficients",
        bc_manager: "BoundaryConditionManager",
    ) -> Optional[PETSc.Mat]:
        """Rebuild K_eff with K_G from the current deformed state.

        Called once per *converged* time window by the base-class
        ``solve()`` loop.  Returns a NEW PETSc.Mat so that
        ``_solve_linear_system`` automatically refactorizes.

        The update is skipped when ``time_step % update_interval != 0``
        to amortize factorization cost.
        """
        if time_step % self._kg_update_interval != 0:
            return None

        # ------------------------------------------------------------------
        # 1. Recover membrane stress field from converged displacement.
        # ------------------------------------------------------------------
        # Expand reduced DOFs to full system for StressRecovery.
        u_full = bc_manager.expand_solution(u).array.copy()

        sr = StressRecovery(self.domain, u_full)

        # Element-centroid membrane stresses [σxx, σyy, τxy] per element.
        elem_result = sr.compute_element_stresses(
            location=StressLocation.MIDDLE,
            stress_type=StressType.MEMBRANE,
        )

        # Build stress_field dict expected by assemble_geometric_stiffness:
        #   {elem_id: np.array([σxx, σyy, σxy])}
        stress_field: dict[int, np.ndarray] = {}
        for elem_id in self.domain._element_map:
            sigma_voigt = np.array(
                [
                    elem_result.sigma_xx[elem_id],
                    elem_result.sigma_yy[elem_id],
                    elem_result.sigma_xy[elem_id],
                ],
                dtype=float,
            )
            if np.max(np.abs(sigma_voigt)) > 1e-20:
                stress_field[elem_id] = sigma_voigt

        if not stress_field:
            # No significant stress yet (e.g. t ≈ 0 with zero load).
            # Keep K_eff unchanged to avoid a no-op refactorization.
            _logger.debug(
                "StressStiffened step %d: stress field is zero — skipping K_G update.",
                time_step,
            )
            return None

        # ------------------------------------------------------------------
        # 2. Assemble global K_G from stress field and reduce to free DOFs.
        # ------------------------------------------------------------------
        try:
            K_G_full = self.domain.assemble_geometric_stiffness(stress_field=stress_field)
        except Exception as exc:
            _logger.warning(
                "StressStiffened step %d: K_G assembly failed (%s) — keeping previous K_eff.",
                time_step,
                exc,
            )
            return None

        K_G_red = bc_manager.reduce_matrix(K_G_full)
        K_G_full.destroy()

        # ------------------------------------------------------------------
        # 3. Build new K_eff = K + K_G + a₀·M + a₁·C.
        # ------------------------------------------------------------------
        K_eff_new = K_red.copy()
        K_eff_new.axpy(1.0, K_G_red)
        K_eff_new.axpy(coeffs.a0, M_red)
        if C_red is not None:
            K_eff_new.axpy(coeffs.a1, C_red)

        # Dispose previous K_G to free PETSc memory.
        if self._K_G_red is not None:
            self._K_G_red.destroy()
        self._K_G_red = K_G_red

        # Norm of K_G for diagnostic logging.
        kg_norm = K_G_red.norm(PETSc.NormType.FROBENIUS)
        k_norm = K_red.norm(PETSc.NormType.FROBENIUS)
        _logger.info(
            "StressStiffened step %d: ||K_G||_F / ||K||_F = %.3e",
            time_step,
            kg_norm / k_norm if k_norm > 0 else float("nan"),
        )

        # Return the new matrix — identity != K_eff triggers refactorization
        # inside _solve_linear_system.
        return K_eff_new
