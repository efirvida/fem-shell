"""Nonlinear static solver — PETSc SNES Newton-Raphson via Rust assembler."""

import logging
from time import perf_counter
from typing import List, Optional

import numpy as np

from aeroelast.core.mesh import MeshModel
from aeroelast.solvers.solver import Solver

_log = logging.getLogger(__name__)


class StaticNonlinearSolver(Solver):
    """
    Nonlinear static solver using PETSc SNES (Newton-Raphson) via Rust.

    Solves R(u) = F_int(u) - F_ext = 0 through geometric nonlinearity
    using the tangent stiffness K_T(u) and internal forces F_int(u).

    Assembly and the SNES loop run entirely in Rust/PETSc.
    The Python layer orchestrates BCs, post-processing and convergence
    parameters forwarded from the YAML ``solver`` block.

    YAML solver parameters (all optional)
    --------------------------------------
    atol : float
        Absolute residual tolerance for SNES (default: 1e-10).
    rtol : float
        Relative residual tolerance for SNES (default: 1e-8).
    stol : float
        Step-length tolerance for SNES (default: 1e-8).
    max_it : int
        Maximum Newton iterations (default: 50).
    continuation : bool
        Enable adaptive load continuation fallback when a full-load solve
        diverges (default: True).
    continuation_steps : int
        Initial number of continuation load increments (default: 8).
    continuation_max_steps : int
        Maximum number of continuation substeps allowed (default: 64).
    diagnostics : bool
        Emit detailed diagnostics from Python and Rust SNES callbacks
        (default: False).
    diagnostics_every : int
        Log every N callback evaluations when diagnostics are enabled
        (default: 1).
    """

    _DEFAULT_ATOL: float = 1e-10
    _DEFAULT_RTOL: float = 1e-8
    _DEFAULT_STOL: float = 1e-8
    _DEFAULT_MAX_IT: int = 50
    _DEFAULT_CONTINUATION: bool = True
    _DEFAULT_CONTINUATION_STEPS: int = 8
    _DEFAULT_CONTINUATION_MAX_STEPS: int = 64
    _DEFAULT_DIAGNOSTICS: bool = False
    _DEFAULT_DIAGNOSTICS_EVERY: int = 1

    def __init__(self, mesh: MeshModel, fem_model_properties: dict):
        super().__init__(mesh, fem_model_properties)
        self.u: Optional[np.ndarray] = None
        self._iterations: int = 0
        self._residual_norm: float = 0.0
        self._converged_reason: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> np.ndarray:
        """
        Run the nonlinear static solve via SNES.

        Returns
        -------
        np.ndarray
            Full displacement vector (n_dofs,).

        Raises
        ------
        RuntimeError
            If the Rust assembler is not available or SNES diverges.
        """
        import _aeroelast  # noqa: PLC0415 — optional Rust extension

        if self.domain._rust is None:
            raise RuntimeError(
                "StaticNonlinearSolver requires the Rust assembler. "
                "Make sure _aeroelast is built and the mesh is assembled."
            )

        f_ext = self._build_f_ext()
        dirichlet_dofs = self._collect_dirichlet_dofs()
        params = self.solver_params if isinstance(self.solver_params, dict) else {}

        atol = float(params.get("atol", self._DEFAULT_ATOL))
        rtol = float(params.get("rtol", self._DEFAULT_RTOL))
        stol = float(params.get("stol", self._DEFAULT_STOL))
        max_it = int(params.get("max_it", self._DEFAULT_MAX_IT))
        continuation = bool(params.get("continuation", self._DEFAULT_CONTINUATION))
        continuation_steps = max(
            1,
            int(params.get("continuation_steps", self._DEFAULT_CONTINUATION_STEPS)),
        )
        continuation_max_steps = max(
            continuation_steps,
            int(params.get("continuation_max_steps", self._DEFAULT_CONTINUATION_MAX_STEPS)),
        )
        diagnostics = bool(params.get("diagnostics", self._DEFAULT_DIAGNOSTICS))
        diagnostics_every = max(
            1,
            int(params.get("diagnostics_every", self._DEFAULT_DIAGNOSTICS_EVERY)),
        )

        if diagnostics:
            _log.info(
                (
                    "Nonlinear diagnostics enabled: n_dof=%d, atol=%.1e, rtol=%.1e, "
                    "stol=%.1e, max_it=%d, continuation=%s, continuation_steps=%d, "
                    "continuation_max_steps=%d, diagnostics_every=%d"
                ),
                self.domain.dofs_count,
                atol,
                rtol,
                stol,
                max_it,
                continuation,
                continuation_steps,
                continuation_max_steps,
                diagnostics_every,
            )

        def _solve_single(
            f_rhs: np.ndarray,
            x0: Optional[np.ndarray],
        ) -> tuple[np.ndarray, int, float, int]:
            # Backward-compatible call path for older _aeroelast binaries
            # that do not expose the optional x0 keyword yet.
            try:
                return _aeroelast.nonlinear_static_solve_coo(
                    self.domain._rust,
                    f_rhs,
                    dirichlet_dofs,
                    atol,
                    rtol,
                    stol,
                    max_it,
                    x0=x0,
                    diagnostics=diagnostics,
                    diagnostics_every=diagnostics_every,
                )
            except TypeError:
                return _aeroelast.nonlinear_static_solve_coo(
                    self.domain._rust,
                    f_rhs,
                    dirichlet_dofs,
                    atol,
                    rtol,
                    stol,
                    max_it,
                )

        t0_full = perf_counter()
        u_arr, iters, res_norm, conv_reason = _solve_single(f_ext, None)
        t1_full = perf_counter()

        if diagnostics:
            _log.info(
                "[Full load] SNES finished in %.3fs: reason=%d, iters=%d, |R|=%.3e",
                t1_full - t0_full,
                conv_reason,
                iters,
                res_norm,
            )

        if conv_reason <= 0 and continuation:
            _log.warning(
                "SNES full-load solve diverged (reason=%d, iters=%d, |R|=%.3e). "
                "Retrying with adaptive load continuation...",
                conv_reason,
                iters,
                res_norm,
            )

            lam = 0.0
            dlam = 1.0 / float(continuation_steps)
            min_dlam = 1.0 / float(continuation_max_steps)
            u_prev = np.zeros_like(f_ext)
            cont_total_iters = 0
            substep = 0

            last_fail: Optional[tuple[int, int, float, float]] = None

            while lam < 1.0 - 1e-14:
                substep += 1
                lam_try = min(1.0, lam + dlam)
                f_try = f_ext * lam_try

                if diagnostics:
                    _log.info(
                        ("[Continuation step %d] lambda %.6f -> %.6f (dlam=%.6f), |F|=%.3e"),
                        substep,
                        lam,
                        lam_try,
                        dlam,
                        float(np.linalg.norm(f_try)),
                    )

                t0_step = perf_counter()
                u_try, it_try, res_try, reason_try = _solve_single(f_try, u_prev)
                t1_step = perf_counter()
                cont_total_iters += it_try

                if diagnostics:
                    _log.info(
                        (
                            "[Continuation step %d] SNES finished in %.3fs: "
                            "reason=%d, iters=%d, |R|=%.3e"
                        ),
                        substep,
                        t1_step - t0_step,
                        reason_try,
                        it_try,
                        res_try,
                    )

                if reason_try > 0:
                    lam = lam_try
                    u_prev = np.asarray(u_try, dtype=np.float64)
                    dlam = max(min(2.0 * dlam, 1.0 - lam), min_dlam)
                    if diagnostics:
                        _log.info(
                            ("[Continuation step %d] accepted lambda=%.6f, next dlam=%.6f"),
                            substep,
                            lam,
                            dlam,
                        )
                    continue

                last_fail = (reason_try, it_try, res_try, lam_try)
                if diagnostics:
                    _log.warning(
                        ("[Continuation step %d] rejected lambda=%.6f; halving dlam %.6f -> %.6f"),
                        substep,
                        lam_try,
                        dlam,
                        dlam * 0.5,
                    )
                dlam *= 0.5
                if dlam < min_dlam:
                    if diagnostics:
                        _log.error(
                            "[Continuation] min dlam reached (dlam=%.6f < min=%.6f)",
                            dlam,
                            min_dlam,
                        )
                    break

            if lam >= 1.0 - 1e-14:
                u_arr = u_prev
                iters = cont_total_iters
                res_norm = float(
                    np.linalg.norm(self.domain._rust.assemble_fint(u_arr, True) - f_ext)
                )
                conv_reason = 2
                _log.info(
                    "Adaptive continuation converged at full load: substeps<=%d, total_iters=%d",
                    continuation_max_steps,
                    iters,
                )
            elif last_fail is not None:
                conv_reason, iters, res_norm, lam_fail = last_fail
                _log.error(
                    "Adaptive continuation failed at lambda=%.6f (reason=%d, iters=%d, |R|=%.3e)",
                    lam_fail,
                    conv_reason,
                    iters,
                    res_norm,
                )

        self.u = np.asarray(u_arr, dtype=np.float64)
        self._iterations = iters
        self._residual_norm = res_norm
        self._converged_reason = conv_reason

        if conv_reason > 0:
            _log.info(
                "SNES converged in %d iterations, |R|=%.3e, reason=%d",
                iters,
                res_norm,
                conv_reason,
            )
        else:
            _log.warning(
                "SNES finished without convergence: iters=%d, |R|=%.3e, reason=%d",
                iters,
                res_norm,
                conv_reason,
            )

        if conv_reason <= 0:
            raise RuntimeError(
                "SNES diverged in StaticNonlinearSolver: "
                f"reason={conv_reason}, iterations={iters}, residual={res_norm:.3e}"
            )

        return self.u

    def print_solver_info(self) -> None:
        """Print nonlinear solver statistics to stdout."""
        print("\n--- Nonlinear Solver (SNES / Newton-Raphson) ---")
        print(f"  Converged reason : {self._converged_reason}")
        print(f"  Iterations       : {self._iterations}")
        print(f"  Final |R|        : {self._residual_norm:.3e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_f_ext(self) -> np.ndarray:
        """Assemble the external force vector from body + nodal loads."""
        n = self.domain.dofs_count
        f = np.zeros(n, dtype=np.float64)

        # Distributed/body loads
        for force in self.body_forces:
            fe = self.domain.assemble_load_vector(force)
            f += np.asarray(fe.getArray(), dtype=np.float64)

        # Concentrated nodal loads
        for load in self.nodal_loads:
            dofs = np.asarray(load.dofs, dtype=np.int64)
            vals = np.asarray(load.force, dtype=np.float64)
            if len(dofs) != len(vals):
                raise ValueError(
                    "Nodal load dof/value length mismatch in nonlinear solver: "
                    f"{len(dofs)} != {len(vals)}"
                )
            f[dofs] += vals

        return f

    def _collect_dirichlet_dofs(self) -> np.ndarray:
        """Collect all constrained DOF indices from Dirichlet BCs."""
        dofs: List[int] = []
        for bc in self.dirichlet_conditions:
            dofs.extend(bc.dofs)
        return np.array(dofs, dtype=np.int64)
